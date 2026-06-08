"""
language_detector.py
────────────────────
Full model: Conformer encoder + three task heads

  ┌─────────────┐
  │  Log-Mel    │  (B, T_frames, n_mels)
  └──────┬──────┘
         │ Conv2D subsampling  (÷4)
  ┌──────▼──────┐
  │  Conformer  │  12 × ConformerBlock  →  (B, T//4, D)
  └──────┬──────┘
         ├──────────────────────────────────────────┐
         │                                          │
  ┌──────▼──────┐                         ┌────────▼────────┐
  │  Frame CLS  │  per-frame lang ID      │ Boundary Head   │ sigmoid → P(switch)
  └─────────────┘  (B, T//4, num_langs)   └─────────────────┘ (B, T//4, 1)
         │
  ┌──────▼──────┐
  │  CTC Head   │  log-softmax → CTC loss
  └─────────────┘  (B, T//4, num_langs+1)

All three outputs are returned together so that the multi-task loss
can be computed in one forward pass.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock, Conv2dSubsampling


class ConformerEncoder(nn.Module):
    """
    Stack of N ConformerBlocks preceded by Conv2D subsampling.
    """

    def __init__(
        self,
        input_dim: int,          # n_mels
        encoder_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, encoder_dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=encoder_dim,
                num_heads=num_heads,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,                        # (B, T, n_mels)
        padding_mask: Optional[torch.Tensor] = None,  # (B, T//4) bool
    ) -> torch.Tensor:
        """
        Returns encoder output: (B, T//4, encoder_dim)
        """
        x = self.subsampling(x)                 # (B, T//4, D)

        # build attention mask: (B, 1, 1, T//4) → broadcast over heads & query
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T//4)

        for block in self.blocks:
            x = block(x, attn_mask)

        return x                                # (B, T//4, D)


class LanguageBoundaryDetector(nn.Module):
    """
    Full multi-task model for language boundary detection.

    Outputs (per forward pass)
    --------------------------
    frame_logits     : (B, T_enc, num_languages)   — cross-entropy target
    boundary_logits  : (B, T_enc)                   — BCEWithLogits target
    ctc_logits       : (B, T_enc, num_languages+1)  — CTC target (+1 for blank)
    """

    def __init__(
        self,
        input_dim: int        = 80,
        encoder_dim: int      = 256,
        num_layers: int       = 12,
        num_heads: int        = 4,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float        = 0.1,
        num_languages: int    = 10,
    ):
        super().__init__()

        # ── Conformer encoder ──────────────────────────────────────────────
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_expansion_factor=ff_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # ── Task heads ─────────────────────────────────────────────────────

        # 1. Frame-level language classifier
        self.frame_cls_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 2, num_languages),
        )

        # 2. Boundary detection head (binary: switch / no-switch)
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 4, 1),
        )

        # 3. CTC head (num_languages + 1 blank token)
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, num_languages + 1),
        )

        self.num_languages  = num_languages
        self.encoder_dim    = encoder_dim
        self.subsampling    = 4   # from Conv2dSubsampling

    def make_padding_mask(
        self, frame_lens: torch.Tensor, subsampled_len: int
    ) -> torch.Tensor:
        """
        Creates a boolean padding mask for the subsampled time axis.

        Returns
        -------
        mask : (B, T_enc) — True at padding positions
        """
        enc_lens = torch.div(frame_lens, self.subsampling, rounding_mode="floor")
        enc_lens = enc_lens.clamp(max=subsampled_len)
        B = frame_lens.size(0)
        mask = torch.arange(subsampled_len, device=frame_lens.device).unsqueeze(0)
        mask = mask.expand(B, -1) >= enc_lens.unsqueeze(1)
        return mask  # (B, T_enc), True = padding

    def forward(
        self,
        feats: torch.Tensor,                         # (B, T_frames, n_mels)
        frame_lens: Optional[torch.Tensor] = None,   # (B,) original frame counts
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = feats.shape

        # build padding mask
        padding_mask = None
        if frame_lens is not None:
            T_enc = (T + self.subsampling - 1) // self.subsampling
            padding_mask = self.make_padding_mask(frame_lens, T_enc)

        # ── encode ──────────────────────────────────────────────────────────
        enc = self.encoder(feats, padding_mask)      # (B, T_enc, D)
        T_enc = enc.size(1)

        # ── heads ───────────────────────────────────────────────────────────
        frame_logits    = self.frame_cls_head(enc)          # (B, T_enc, L)
        boundary_logits = self.boundary_head(enc).squeeze(-1)  # (B, T_enc)
        ctc_log_probs   = F.log_softmax(self.ctc_head(enc), dim=-1)  # (B, T_enc, L+1)

        return {
            "frame_logits":    frame_logits,      # raw logits for CE loss
            "boundary_logits": boundary_logits,   # raw logits for BCE loss
            "ctc_log_probs":   ctc_log_probs,     # log-probs for CTC loss
            "encoder_out":     enc,               # for downstream inspection
            "enc_lengths": (
                torch.div(frame_lens, self.subsampling, rounding_mode="floor")
                .clamp(max=T_enc)
                if frame_lens is not None
                else torch.full((B,), T_enc, device=feats.device)
            ),
        }

    @torch.no_grad()
    def predict_boundaries(
        self,
        feats: torch.Tensor,                  # (1, T, n_mels) — single clip
        hop_sec: float = 0.01,
        threshold: float = 0.5,
        smooth_window: int = 5,
    ) -> Dict:
        """
        Inference helper.

        Returns
        -------
        dict with:
          switch_times_sec  : list[float]  — predicted switch timestamps
          frame_lang_ids    : list[int]    — per-encoder-frame language ID
          boundary_probs    : np.ndarray   — (T_enc,) sigmoid probs
        """
        import numpy as np

        out = self.forward(feats)
        frame_logits    = out["frame_logits"][0]       # (T_enc, L)
        boundary_logits = out["boundary_logits"][0]    # (T_enc,)

        # per-frame language id
        frame_lang_ids = frame_logits.argmax(dim=-1).cpu().tolist()

        # boundary probabilities with temporal smoothing
        boundary_probs = torch.sigmoid(boundary_logits).cpu().numpy()

        # simple moving-average smoothing
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            boundary_probs = np.convolve(boundary_probs, kernel, mode="same")

        # find switch frames
        switch_frames = np.where(boundary_probs >= threshold)[0].tolist()

        # convert frames → seconds (accounting for 4× subsampling)
        enc_hop_sec = hop_sec * self.subsampling
        switch_times_sec = [f * enc_hop_sec for f in switch_frames]

        return {
            "switch_times_sec": switch_times_sec,
            "frame_lang_ids":   frame_lang_ids,
            "boundary_probs":   boundary_probs,
        }
