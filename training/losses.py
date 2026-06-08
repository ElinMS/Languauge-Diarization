"""
losses.py
─────────
Multi-task loss for language boundary detection.

Three losses are combined with configurable weights:

  L_total = w_ctc  · L_ctc
          + w_frame · L_frame_cls
          + w_bnd  · L_boundary

L_ctc        — CTC loss on the language-ID sequence (handles alignment)
L_frame_cls  — Cross-entropy on per-frame language labels (dense supervision)
L_boundary   — Weighted binary cross-entropy on boundary frames
               (heavily positive-weighted because switches are rare)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Parameters
    ----------
    ctc_weight       : weight for CTC loss
    frame_cls_weight : weight for frame-level classification loss
    boundary_weight  : weight for boundary detection loss
    boundary_pos_weight : positive-class weight for BCEWithLogits
                          (set > 1 to penalise missed boundaries more)
    blank_idx        : CTC blank token index (= num_languages by convention)
    """

    def __init__(
        self,
        ctc_weight: float       = 0.3,
        frame_cls_weight: float = 0.4,
        boundary_weight: float  = 0.3,
        boundary_pos_weight: float = 10.0,
        blank_idx: int          = 10,
    ):
        super().__init__()
        self.w_ctc   = ctc_weight
        self.w_frame = frame_cls_weight
        self.w_bnd   = boundary_weight

        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
        self.ce_loss  = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        pos_w = torch.tensor([boundary_pos_weight])
        self.register_buffer("pos_weight", pos_w)

    # ------------------------------------------------------------------
    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        targets:   Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        model_out : output dict from LanguageBoundaryDetector.forward()
            frame_logits    : (B, T_enc, L)
            boundary_logits : (B, T_enc)
            ctc_log_probs   : (B, T_enc, L+1)
            enc_lengths     : (B,)

        targets : batch dict from collate_fn (partially subsampled)
            frame_labels    : (B, T_enc_orig)  — will be subsampled here
            boundary_mask   : (B, T_enc_orig)
            language_seq    : (B, S)           — CTC targets
            seq_lens        : (B,)             — valid lengths in language_seq

        Returns
        -------
        dict with individual and total losses
        """
        frame_logits    = model_out["frame_logits"]     # (B, T_enc, L)
        boundary_logits = model_out["boundary_logits"]  # (B, T_enc)
        ctc_log_probs   = model_out["ctc_log_probs"]    # (B, T_enc, L+1)
        enc_lengths     = model_out["enc_lengths"]      # (B,)

        B, T_enc, L = frame_logits.shape

        # ── Subsample frame labels to match encoder output length ──────────
        # The encoder applies 4× subsampling; we need to downsample targets.
        frame_labels_orig = targets["frame_labels"]     # (B, T_full)
        bnd_mask_orig     = targets["boundary_mask"]    # (B, T_full)

        frame_labels = self._subsample_labels(frame_labels_orig, T_enc)  # (B, T_enc)
        bnd_mask     = self._subsample_boundary(bnd_mask_orig,   T_enc)  # (B, T_enc)

        # ── 1. Frame-level cross-entropy ───────────────────────────────────
        # frame_logits : (B, T_enc, L) → (B·T_enc, L)
        fl_flat  = frame_logits.reshape(-1, L)
        lbl_flat = frame_labels.reshape(-1)
        loss_frame = self.ce_loss(fl_flat, lbl_flat)

        # ── 2. Boundary binary cross-entropy ──────────────────────────────
        # bnd_mask is float 0/1; ignore padding where frame_label == -1
        valid      = (frame_labels != -1).float()          # (B, T_enc)
        bnd_target = bnd_mask.float()

        loss_bnd = F.binary_cross_entropy_with_logits(
            boundary_logits,
            bnd_target,
            weight=valid,
            pos_weight=self.pos_weight.to(boundary_logits.device),
            reduction="sum",
        ) / valid.sum().clamp(min=1)

        # ── 3. CTC loss ────────────────────────────────────────────────────
        # ctc_log_probs : (B, T_enc, L+1) → (T_enc, B, L+1) for CTCLoss
        log_probs  = ctc_log_probs.permute(1, 0, 2)        # (T, B, vocab)
        lang_seq   = targets["language_seq"]               # (B, S)
        seq_lens   = targets["seq_lens"]                   # (B,)

        # Flatten targets removing padding (-1)
        ctc_targets = []
        ctc_tlens   = []
        for b in range(B):
            slen = seq_lens[b].item()
            seq  = lang_seq[b, :slen]
            ctc_targets.append(seq)
            ctc_tlens.append(slen)

        ctc_targets_flat = torch.cat(ctc_targets)           # (sum_S,)
        ctc_tlens_tensor = torch.tensor(
            ctc_tlens, dtype=torch.long, device=log_probs.device
        )

        loss_ctc = self.ctc_loss(
            log_probs,
            ctc_targets_flat,
            enc_lengths,
            ctc_tlens_tensor,
        )

        # ── Weighted total ─────────────────────────────────────────────────
        total = (
            self.w_ctc   * loss_ctc
          + self.w_frame * loss_frame
          + self.w_bnd   * loss_bnd
        )

        return {
            "loss":       total,
            "loss_ctc":   loss_ctc.detach(),
            "loss_frame": loss_frame.detach(),
            "loss_bnd":   loss_bnd.detach(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _subsample_labels(labels: torch.Tensor, T_enc: int) -> torch.Tensor:
        """
        Nearest-neighbour downsampling of integer label tensors.
        labels : (B, T_full)  →  (B, T_enc)
        """
        B, T_full = labels.shape
        if T_full == T_enc:
            return labels
        # select every k-th frame
        idx = torch.linspace(0, T_full - 1, T_enc, device=labels.device).long()
        return labels[:, idx]

    @staticmethod
    def _subsample_boundary(mask: torch.Tensor, T_enc: int) -> torch.Tensor:
        """
        Downsample boundary mask (float) with max-pooling so that any
        boundary within a 4-frame window stays marked.
        mask : (B, T_full) float  →  (B, T_enc) float
        """
        B, T_full = mask.shape
        if T_full == T_enc:
            return mask
        factor = max(1, T_full // T_enc)
        # pad so T_full is divisible
        pad = (factor - T_full % factor) % factor
        m   = F.pad(mask, (0, pad))                     # (B, T_full+pad)
        m   = m.view(B, -1, factor)                     # (B, T_enc', factor)
        m   = m.max(dim=-1).values                      # (B, T_enc')
        return m[:, :T_enc]
