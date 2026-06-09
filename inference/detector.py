"""
detector.py  —  inference/detector.py
──────────────────────────────────────
Streaming inference pipeline that:
  1. Loads a trained checkpoint
  2. Accepts an audio file (any format, any length)
  3. Slides a window over the audio
  4. Returns precise timestamps of language switches

Output format
─────────────
  {
    "segments": [
        {"start": 0.0,  "end": 4.32, "language": "en", "lang_id": 0},
        {"start": 4.32, "end": 9.10, "language": "fr", "lang_id": 1},
        ...
    ],
    "switch_times_sec": [4.32, 9.10, ...],
    "boundary_probs":   np.ndarray  (T_enc_total,),
    "frame_lang_ids":   list[int]
  }
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio

from data.feature_extraction import DualFrontend
from data.dataset import ID2LANG
from models.language_detector import LanguageBoundaryDetector


class LanguageBoundaryInference:
    """
    Streaming language-boundary detector.

    Parameters
    ----------
    checkpoint_path  : path to best_model.pt or any epoch checkpoint
    device           : "cuda" or "cpu"
    chunk_size_sec   : audio window fed to the model at once (seconds)
    step_size_sec    : stride of the sliding window (seconds)
    boundary_thr     : sigmoid threshold for declaring a boundary
    smooth_window    : moving-average frames for boundary probability smoothing
    min_seg_dur_sec  : minimum segment duration; shorter segments are merged
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        sample_rate: int = 16_000,
        n_mels: int = 80,
        hop_length: int = 160,
        encoder_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.0,
        num_languages: int = 5,
        input_dim: int = 208,
        wavegram_channels: int = 128,
        wavegram_kernel: int = 1024,
        chunk_size_sec: float = 30.0,
        step_size_sec: float = 5.0,
        boundary_thr: float = 0.5,
        smooth_window: int = 5,
        min_seg_dur_sec: float = 0.5,
    ):
        self.device         = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sr             = sample_rate
        self.hop            = hop_length
        self.chunk_samp     = int(chunk_size_sec * sample_rate)
        self.step_samp      = int(step_size_sec  * sample_rate)
        self.boundary_thr   = boundary_thr
        self.smooth_window  = smooth_window
        self.min_seg_dur    = min_seg_dur_sec
        self.enc_hop_sec    = (hop_length / sample_rate) * 4  # 4× subsampling

        # ── frontend ────────────────────────────────────────────────────────
        self.frontend = DualFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            normalize=True,
            wavegram_channels=wavegram_channels,
            wavegram_kernel=wavegram_kernel,
        ).to(self.device)
        self.frontend.eval()

        # ── model ───────────────────────────────────────────────────────────
        self.model = LanguageBoundaryDetector(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_expansion_factor=ff_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            num_languages=num_languages,
        ).to(self.device)
        self.model.eval()

        # ── load weights ────────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "frontend" in ckpt:
            self.frontend.load_state_dict(ckpt["frontend"])
        print(f"[Inference] loaded checkpoint: {checkpoint_path}")

    # ─────────────────────────────────────────────────────────────────────────

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load any audio file → mono float32 @ self.sr"""
        wav, sr = torchaudio.load(str(path))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(dim=0).numpy().astype(np.float32)
        return wav

    @torch.no_grad()
    def _process_chunk(self, chunk: np.ndarray) -> Dict:
        """Run model on a single chunk. Returns raw logits & probs."""
        audio_t = torch.from_numpy(chunk).unsqueeze(0).float().to(self.device)  # (1, T) float32
        feats   = self.frontend(audio_t)                                  # (1, T_f, F)
        out     = self.model(feats)

        frame_logits    = out["frame_logits"][0]        # (T_enc, L)
        boundary_logits = out["boundary_logits"][0]     # (T_enc,)

        lang_ids   = frame_logits.argmax(dim=-1).cpu().numpy()  # (T_enc,)
        bnd_probs  = torch.sigmoid(boundary_logits).cpu().numpy()  # (T_enc,)

        return {"lang_ids": lang_ids, "bnd_probs": bnd_probs}

    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, audio_path: Union[str, Path]) -> Dict:
        """
        Main entry point.

        Parameters
        ----------
        audio_path : path to any audio file

        Returns
        -------
        result dict — see module docstring for schema
        """
        wav = self._load_audio(audio_path)
        total_samp  = len(wav)
        total_sec   = total_samp / self.sr

        # ── sliding window over audio ────────────────────────────────────
        all_lang_ids  = []
        all_bnd_probs = []

        pos = 0
        while pos < total_samp:
            end  = min(pos + self.chunk_samp, total_samp)
            chunk = wav[pos:end]

            # zero-pad if shorter than chunk_size (last chunk)
            if len(chunk) < self.chunk_samp:
                chunk = np.concatenate(
                    [chunk, np.zeros(self.chunk_samp - len(chunk), dtype=np.float32)]
                )

            result = self._process_chunk(chunk)

            # only keep frames that correspond to actual audio (not padding)
            valid_samp   = end - pos
            valid_frames = int(math.ceil(valid_samp / self.hop / 4))  # /4 subsampling
            valid_frames = min(valid_frames, len(result["lang_ids"]))

            all_lang_ids.extend(result["lang_ids"][:valid_frames].tolist())
            all_bnd_probs.extend(result["bnd_probs"][:valid_frames].tolist())

            pos += self.step_samp

        # ── smooth boundary probs ────────────────────────────────────────
        bnd_probs = np.array(all_bnd_probs)
        if self.smooth_window > 1:
            kernel    = np.ones(self.smooth_window) / self.smooth_window
            bnd_probs = np.convolve(bnd_probs, kernel, mode="same")

        # ── detect switch frames ─────────────────────────────────────────
        above_thr = bnd_probs >= self.boundary_thr  # (T_enc,)

        # non-maximum suppression: keep only the first frame in each contiguous run
        switch_frames = []
        in_run = False
        for i, flag in enumerate(above_thr):
            if flag and not in_run:
                switch_frames.append(i)
                in_run = True
            elif not flag:
                in_run = False

        switch_times_sec = [f * self.enc_hop_sec for f in switch_frames]

        # ── build language segments ──────────────────────────────────────
        segments = self._build_segments(
            all_lang_ids, switch_times_sec, total_sec
        )
        segments = self._merge_short_segments(segments)

        return {
            "segments":        segments,
            "switch_times_sec": [s["start"] for s in segments[1:]],
            "boundary_probs":  bnd_probs,
            "frame_lang_ids":  all_lang_ids,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _build_segments(
        self,
        frame_lang_ids: List[int],
        switch_times:   List[float],
        total_sec:      float,
    ) -> List[Dict]:
        """Convert switch times → segment list with majority-vote language."""
        boundaries = [0.0] + switch_times + [total_sec]
        segments   = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i + 1]

            # majority vote over encoder frames in this segment
            s_frame = int(start / self.enc_hop_sec)
            e_frame = int(end   / self.enc_hop_sec)
            seg_ids = frame_lang_ids[s_frame:e_frame]

            if not seg_ids:
                lang_id = 0
            else:
                lang_id = int(np.bincount(seg_ids).argmax())

            segments.append({
                "start":    round(start, 3),
                "end":      round(end,   3),
                "lang_id":  lang_id,
                "language": ID2LANG.get(lang_id, f"lang_{lang_id}"),
            })

        return segments

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments shorter than min_seg_dur into their neighbour."""
        if not segments:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            dur = seg["end"] - seg["start"]
            if dur < self.min_seg_dur:
                # merge into previous
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg)

        return merged

    # ─────────────────────────────────────────────────────────────────────────

    def pretty_print(self, result: Dict):
        """Print a human-readable timeline of detected language segments."""
        print("\n── Language Boundary Detection Results ──────────────────")
        for seg in result["segments"]:
            bar_len = max(1, int((seg["end"] - seg["start"]) * 4))
            bar     = "█" * bar_len
            print(
                f"  {seg['start']:6.2f}s → {seg['end']:6.2f}s  "
                f"[{seg['language'].upper():>3}]  {bar}"
            )
        switches = result["switch_times_sec"]
        print(f"\n  Total language switches: {len(switches)}")
        if switches:
            print(f"  Switch times (s):  {[round(t,2) for t in switches]}")
        print("─" * 56)
