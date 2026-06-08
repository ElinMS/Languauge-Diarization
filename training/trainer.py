"""
trainer.py
──────────
Full training loop with:
  • Mixed-precision (fp16) via torch.cuda.amp
  • Cosine LR schedule with linear warmup
  • Gradient clipping
  • TensorBoard logging
  • Periodic checkpointing (keep last N)
"""

import os
import math
import time
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# PyTorch 2.x amp API (falls back to legacy for older versions)
try:
    from torch.amp import GradScaler, autocast
    AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_DEVICE = None

from models.language_detector import LanguageBoundaryDetector
from data.feature_extraction import DualFrontend, SpecAugment
from training.losses import MultiTaskLoss, MultiTaskLoss as _MTL
from training.metrics import (
    MetricMeter,
    frame_language_metrics,
    aggregate_boundary_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler: cosine with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Transformer-style LR schedule (Vaswani 2017)."""
    if step == 0:
        step = 1
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


# Support both PyTorch < 2.0 (_LRScheduler) and >= 2.0 (LRScheduler)
_BaseLRScheduler = getattr(
    torch.optim.lr_scheduler, "LRScheduler",
    torch.optim.lr_scheduler._LRScheduler
)

class CosineWarmupScheduler(_BaseLRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            max(self.min_lr, base_lr * scale)
            for base_lr in self.base_lrs
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg, train_loader: DataLoader, val_loader: DataLoader):
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] using device: {self.device}")

        # ── frontend ────────────────────────────────────────────────────────
        self.frontend = DualFrontend(
            sample_rate=cfg.audio.sample_rate,
            n_mels=cfg.audio.n_mels,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            win_length=cfg.audio.win_length,
            f_min=cfg.audio.f_min,
            f_max=cfg.audio.f_max,
            normalize=cfg.audio.normalize,
            wavegram_channels=cfg.audio.wavegram_channels,
            wavegram_kernel=cfg.audio.wavegram_kernel,
        ).to(self.device)

        self.spec_aug = SpecAugment().to(self.device)

        # ── model ───────────────────────────────────────────────────────────
        m = cfg.model
        self.model = LanguageBoundaryDetector(
            input_dim=m.input_dim,
            encoder_dim=m.encoder_dim,
            num_layers=m.num_layers,
            num_heads=m.num_heads,
            ff_expansion_factor=m.ff_expansion_factor,
            conv_kernel_size=m.conv_kernel_size,
            dropout=m.dropout,
            num_languages=m.num_languages,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Trainer] model params: {total_params / 1e6:.2f} M")

        # ── loss ────────────────────────────────────────────────────────────
        lc = cfg.training.loss
        self.criterion = MultiTaskLoss(
            ctc_weight=lc.ctc_weight,
            frame_cls_weight=lc.frame_cls_weight,
            boundary_weight=lc.boundary_weight,
            blank_idx=m.num_languages,
        ).to(self.device)

        # ── optimiser ───────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.frontend.parameters()),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=(0.9, 0.98),
        )

        steps_per_epoch = len(train_loader)
        total_steps     = steps_per_epoch * cfg.training.epochs

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=cfg.training.warmup_steps,
            total_steps=total_steps,
            min_lr=cfg.training.scheduler.min_lr,
        )

        # ── mixed precision ─────────────────────────────────────────────────
        self.use_amp = cfg.training.mixed_precision and self.device.type == "cuda"
        # PyTorch 2.x GradScaler requires device arg; older versions do not
        try:
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            self.scaler = GradScaler(enabled=self.use_amp)

        # ── logging / checkpointing ─────────────────────────────────────────
        self.ckpt_dir = Path(cfg.training.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=str(self.ckpt_dir / "tb_logs"))
        self.global_step   = 0
        self.best_val_f1   = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _audio_to_feats(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (B, T_samples) → feats (B, T_frames, n_mels)"""
        feats = self.frontend(audio)
        return feats

    def _compute_frame_lens(self, audio_lens: torch.Tensor) -> torch.Tensor:
        """Convert sample counts to frame counts (matches frontend hop)."""
        hop = self.cfg.audio.hop_length
        return torch.div(audio_lens, hop, rounding_mode="floor") + 1

    def _save_checkpoint(self, epoch: int, val_metrics: dict):
        tag  = f"epoch{epoch:03d}_f1{val_metrics.get('boundary_f1', 0):.3f}"
        path = self.ckpt_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "epoch":       epoch,
                "model":       self.model.state_dict(),
                "frontend":    self.frontend.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "scheduler":   self.scheduler.state_dict(),
                "val_metrics": val_metrics,
            },
            path,
        )
        print(f"  [ckpt] saved → {path}")

        # keep last N
        keep = self.cfg.training.keep_last_n
        all_ckpts = sorted(glob.glob(str(self.ckpt_dir / "ckpt_*.pt")))
        for old in all_ckpts[:-keep]:
            os.remove(old)

    # ── train one epoch ──────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.frontend.train()
        meter = MetricMeter()
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            audio      = batch["audio"].to(self.device)           # (B, T_samp)
            audio_lens = batch["audio_lens"].to(self.device)

            # move label tensors
            targets = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k not in ("audio", "audio_lens", "switch_times", "duration_sec")
            }

            # ── forward ──────────────────────────────────────────────────
            amp_ctx = ({"device_type": AMP_DEVICE} if AMP_DEVICE else {})
            with autocast(**amp_ctx, enabled=self.use_amp):
                feats      = self._audio_to_feats(audio)          # (B, T_f, F)
                feats      = self.spec_aug(feats)
                frame_lens = self._compute_frame_lens(audio_lens)
                model_out  = self.model(feats, frame_lens)
                loss_dict  = self.criterion(model_out, targets)
                loss       = loss_dict["loss"]

            # ── backward ─────────────────────────────────────────────────
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            # ── log ──────────────────────────────────────────────────────
            meter.update(
                {k: v.item() for k, v in loss_dict.items()},
                n=audio.size(0),
            )

            if step % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch} | step {step}/{len(self.train_loader)} "
                    f"| loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s"
                )
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr",   lr,           self.global_step)

        return meter.mean()

    # ── validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        self.frontend.eval()
        meter      = MetricMeter()
        all_pred_b = []   # boundary time lists
        all_gt_b   = []
        all_pred_f = []   # flat frame label arrays
        all_gt_f   = []

        for batch in self.val_loader:
            audio      = batch["audio"].to(self.device)
            audio_lens = batch["audio_lens"].to(self.device)
            targets = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k not in ("audio", "audio_lens", "switch_times", "duration_sec")
            }

            amp_ctx = ({"device_type": AMP_DEVICE} if AMP_DEVICE else {})
            with autocast(**amp_ctx, enabled=self.use_amp):
                feats      = self._audio_to_feats(audio)
                frame_lens = self._compute_frame_lens(audio_lens)
                model_out  = self.model(feats, frame_lens)
                loss_dict  = self.criterion(model_out, targets)

            meter.update(
                {k: v.item() for k, v in loss_dict.items()},
                n=audio.size(0),
            )

            # ── boundary F1 ──────────────────────────────────────────────
            enc_hop = 0.01 * self.model.subsampling
            bnd_probs = torch.sigmoid(model_out["boundary_logits"])  # (B, T_enc)
            thr = self.cfg.model.boundary_threshold

            for b in range(audio.size(0)):
                pred_frames = (bnd_probs[b] >= thr).nonzero(as_tuple=True)[0]
                pred_times  = (pred_frames.float() * enc_hop).tolist()
                # ground-truth: from original (non-device) boundary mask
                gt_bnd    = batch["boundary_mask"][b]   # float tensor (T_full,)
                gt_frames = gt_bnd.nonzero(as_tuple=True)[0]
                gt_times  = (gt_frames.float() * 0.01).tolist()
                all_pred_b.append(pred_times)
                all_gt_b.append(gt_times)

            # ── frame accuracy ────────────────────────────────────────────
            pred_ids = model_out["frame_logits"].argmax(dim=-1)  # (B, T_enc)
            lbl_sub  = _MTL._subsample_labels(
                targets["frame_labels"], pred_ids.size(1)
            )
            all_pred_f.append(pred_ids.cpu().numpy().flatten())
            all_gt_f.append(lbl_sub.cpu().numpy().flatten())

        bnd_metrics   = aggregate_boundary_metrics(
            all_pred_b, all_gt_b,
            tolerance_s=self.cfg.eval.boundary_tolerance_ms / 1000,
        )
        frame_mets = frame_language_metrics(
            import_np_concat(all_pred_f),
            import_np_concat(all_gt_f),
        )

        val_metrics = {**meter.mean(), **bnd_metrics, **frame_mets}

        # TensorBoard
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

        return val_metrics

    # ── main train loop ──────────────────────────────────────────────────────

    def fit(self):
        cfg = self.cfg.training
        for epoch in range(1, cfg.epochs + 1):
            print(f"\n{'='*60}")
            print(f"  Epoch {epoch}/{cfg.epochs}")
            print(f"{'='*60}")

            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.validate(epoch)

            print(f"\n  [Train] {_fmt(train_metrics)}")
            print(f"  [Val]   {_fmt(val_metrics)}")

            # checkpoint
            if epoch % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics)

            # best model
            bnd_f1 = val_metrics.get("boundary_f1", 0.0)
            if bnd_f1 > self.best_val_f1:
                self.best_val_f1 = bnd_f1
                torch.save(
                    {"epoch": epoch, "model": self.model.state_dict(),
                     "frontend": self.frontend.state_dict()},
                    self.ckpt_dir / "best_model.pt",
                )
                print(f"  ★ New best boundary F1: {bnd_f1:.4f}")

        self.writer.close()
        print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def import_np_concat(arrays):
    import numpy as np
    return np.concatenate(arrays) if arrays else np.array([])


def _fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in d.items())
