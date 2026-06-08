"""
utils/audio.py
──────────────
Audio I/O helpers used across the pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


def load_audio(
    path: Union[str, Path],
    target_sr: int = 16_000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load any audio file, resample, and convert to mono if needed.

    Returns
    -------
    (waveform_np, sample_rate)  —  float32 numpy array, values in [-1, 1]
    """
    wav, sr = torchaudio.load(str(path))

    if sr != target_sr:
        wav = F.resample(wav, sr, target_sr)
        sr  = target_sr

    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0).numpy().astype(np.float32), sr


def trim_silence(
    wav: np.ndarray,
    sr:  int,
    top_db: float = 30.0,
) -> np.ndarray:
    """Trim leading and trailing silence using an energy threshold."""
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        return trimmed
    except ImportError:
        return wav


def pad_or_trim(wav: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or trim to exactly target_samples."""
    if len(wav) >= target_samples:
        return wav[:target_samples]
    pad = target_samples - len(wav)
    return np.concatenate([wav, np.zeros(pad, dtype=wav.dtype)])


def chunk_audio(
    wav: np.ndarray,
    sr:  int,
    chunk_sec: float,
    step_sec:  Optional[float] = None,
) -> list:
    """
    Split audio into (possibly overlapping) chunks.

    Returns list of (start_sec, end_sec, chunk_array).
    """
    if step_sec is None:
        step_sec = chunk_sec

    chunk_samp = int(chunk_sec * sr)
    step_samp  = int(step_sec  * sr)
    chunks     = []

    pos = 0
    while pos < len(wav):
        end   = min(pos + chunk_samp, len(wav))
        chunk = wav[pos:end]
        chunks.append((pos / sr, end / sr, chunk))
        pos  += step_samp

    return chunks


def save_audio(
    wav: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    sr:  int = 16_000,
):
    """Save waveform to a .wav file."""
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.cpu().float(), sr)
