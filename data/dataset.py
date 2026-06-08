import random
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Windows DataLoader spawn fix: use 0 workers on Windows
_NUM_WORKERS_TRAIN = 0 if sys.platform == "win32" else 4
_NUM_WORKERS_VAL   = 0 if sys.platform == "win32" else 2

# We will use simple integer IDs for L1, L2, etc. based on the rttms
# The config specifies 10 languages, let's just make it up to 10
LANGUAGES = ["en", "fr", "de", "es", "hi", "zh", "ar", "ru", "pt", "ja"]
LANG2ID   = {lang: idx for idx, lang in enumerate(LANGUAGES)}
ID2LANG   = {
    0: "English",
    1: "French",
    2: "German",
    3: "Spanish",
    4: "Hindi",
    5: "Mandarin",
    6: "Arabic",
    7: "Russian",
    8: "Portuguese",
    9: "Japanese",
}

# hop length in seconds (must match config)
HOP_SEC = 0.01   # 10 ms per frame


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def frames_for_duration(duration_sec: float) -> int:
    return int(math.ceil(duration_sec / HOP_SEC))


def build_frame_labels(
    segments: List[Tuple[float, float, int]],  # (start_sec, end_sec, lang_id)
    total_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-frame language label array and boundary mask.
    """
    frame_labels  = np.zeros(total_frames, dtype=np.int32)
    boundary_mask = np.zeros(total_frames, dtype=bool)

    # In case there's no language (silence), we'll default to 0 (L1), 
    # but the rttms should be dense.
    
    # We sort segments to find boundaries
    segments = sorted(segments, key=lambda x: x[0])
    
    for i, (start, end, lang_id) in enumerate(segments):
        s_frame = max(0, int(start / HOP_SEC))
        e_frame = min(int(end   / HOP_SEC), total_frames)
        if s_frame < e_frame:
            frame_labels[s_frame:e_frame] = lang_id
            if i > 0 and s_frame > 0:
                boundary_mask[s_frame] = True   # first frame of new language

    return frame_labels, boundary_mask

def parse_rttm(rttm_path: Path) -> List[Tuple[float, float, int]]:
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "LANGUAGE":
                start_time = float(parts[3])
                duration = float(parts[4])
                lang_label = parts[7]
                
                lang_id = LANG2ID.get(lang_label, 0)
                segments.append((start_time, start_time + duration, lang_id))
    return segments

# ─────────────────────────────────────────────────────────────────────────────
#  Real Multilingual Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RealMultilingualDataset(Dataset):
    """
    Builds clips by sampling random windows from long .wav files and their corresponding .rttms.
    """

    def __init__(
        self,
        data_dir: str,
        num_samples: int,
        sample_rate: int = 16_000,
        target_duration: float = 15.0,
        augment: bool = False,
        seed: int = 42,
    ):
        self.data_dir      = Path(data_dir)
        self.num_samples   = num_samples
        self.sr            = sample_rate
        self.target_dur    = target_duration
        self.augment       = augment
        self.seed          = seed
        
        self.wav_dir = self.data_dir / "wav"
        self.rttm_dir = self.data_dir / "rttm"
        
        self.file_list = []
        self.file_info = {}
        
        if self.wav_dir.exists() and self.rttm_dir.exists():
            wav_files = sorted(self.wav_dir.glob("*.wav"))
            for wav_path in wav_files:
                file_id = wav_path.stem
                rttm_path = self.rttm_dir / f"{file_id}.rttm"
                if rttm_path.exists():
                    self.file_list.append(file_id)
                    info = torchaudio.info(str(wav_path))
                    duration = info.num_frames / info.sample_rate
                    segments = parse_rttm(rttm_path)
                    self.file_info[file_id] = {
                        "wav_path": str(wav_path),
                        "duration": duration,
                        "segments": segments,
                    }
        else:
            print(f"[warn] wav or rttm directory not found in {self.data_dir}")

    def __len__(self):
        return self.num_samples

    def _make_rng(self, idx: int) -> random.Random:
        return random.Random(self.seed + idx)

    def __getitem__(self, idx: int) -> Dict:
        if not self.file_list:
            raise RuntimeError(f"No audio files found in {self.data_dir}")
            
        rng = self._make_rng(idx)
        
        file_id = rng.choice(self.file_list)
        info = self.file_info[file_id]
        
        total_duration = info["duration"]
        max_start = max(0.0, total_duration - self.target_dur)
        start_sec = rng.uniform(0, max_start)
        end_sec = min(start_sec + self.target_dur, total_duration)
        actual_dur = end_sec - start_sec
        
        # Load audio chunk
        frame_offset = int(start_sec * self.sr)
        num_frames = int(actual_dur * self.sr)
        
        try:
            wav, sr = torchaudio.load(info["wav_path"], frame_offset=frame_offset, num_frames=num_frames)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            audio = wav.mean(dim=0).numpy() # mono
        except Exception:
            # fallback: silence
            audio = np.zeros(int(self.target_dur * self.sr))
            actual_dur = self.target_dur
            
        if len(audio) < int(self.target_dur * self.sr):
            pad = int(self.target_dur * self.sr) - len(audio)
            audio = np.concatenate([audio, np.zeros(pad)])
            
        # Extract segments overlapping with the chunk
        chunk_segments = []
        for s_start, s_end, lang_id in info["segments"]:
            if s_end <= start_sec or s_start >= end_sec:
                continue
            # clip to chunk window
            c_start = max(0.0, s_start - start_sec)
            c_end = min(self.target_dur, s_end - start_sec)
            chunk_segments.append((c_start, c_end, lang_id))
            
        if not chunk_segments:
            chunk_segments = [(0.0, self.target_dur, 0)]

        # ── optional simple augmentation ─────────────────────────────────────
        if self.augment:
            # additive Gaussian noise
            noise_scale = rng.uniform(0.0, 0.005)
            audio = audio + np.random.randn(len(audio)) * noise_scale
            # random amplitude scaling
            audio = audio * rng.uniform(0.8, 1.2)

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return self._package(audio, chunk_segments)

    def _package(self, audio: np.ndarray, segments_meta: list) -> Dict:
        total_frames = frames_for_duration(len(audio) / self.sr)
        frame_labels, boundary_mask = build_frame_labels(segments_meta, total_frames)

        # CTC target: unique consecutive language IDs (no blank)
        segments_meta = sorted(segments_meta, key=lambda x: x[0])
        lang_seq = [segments_meta[0][2]]
        for _, _, lid in segments_meta[1:]:
            if lid != lang_seq[-1]:
                lang_seq.append(lid)

        switch_times = [seg[0] for seg in segments_meta[1:] if seg[2] != segments_meta[0][2]]

        return {
            "audio":         torch.from_numpy(audio),                        # (T_samples,)
            "frame_labels":  torch.from_numpy(frame_labels.astype(np.int64)),# (T_frames,) int64 for cross entropy
            "boundary_mask": torch.from_numpy(boundary_mask.astype(np.float32)),# (T_frames,) float32 for BCE loss
            "language_seq":  torch.tensor(lang_seq, dtype=torch.long),
            "switch_times":  switch_times,   # list[float] — kept as metadata
            "duration_sec":  len(audio) / self.sr,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pads audio and frame-level labels to the longest sample in the batch.
    """
    # sort by length (longest first, helps RNN-style layers if any)
    batch = sorted(batch, key=lambda x: len(x["audio"]), reverse=True)

    max_audio  = max(len(b["audio"])        for b in batch)
    max_frames = max(len(b["frame_labels"]) for b in batch)
    max_seq    = max(len(b["language_seq"]) for b in batch)

    audios         = []
    frame_labels   = []
    boundary_masks = []
    lang_seqs      = []
    audio_lens     = []
    frame_lens     = []
    seq_lens       = []

    for b in batch:
        a_len = len(b["audio"])
        f_len = len(b["frame_labels"])
        s_len = len(b["language_seq"])

        # pad audio
        pad_a = max_audio - a_len
        audios.append(torch.nn.functional.pad(b["audio"], (0, pad_a)))

        # pad frame labels (with -1 = ignore index)
        pad_f = max_frames - f_len
        frame_labels.append(
            torch.nn.functional.pad(b["frame_labels"], (0, pad_f), value=-1)
        )
        boundary_masks.append(
            torch.nn.functional.pad(b["boundary_mask"].float(), (0, pad_f))
        )

        # pad language sequence (with -1)
        pad_s = max_seq - s_len
        lang_seqs.append(
            torch.nn.functional.pad(b["language_seq"], (0, pad_s), value=-1)
        )

        audio_lens.append(a_len)
        frame_lens.append(f_len)
        seq_lens.append(s_len)

    return {
        "audio":         torch.stack(audios),                           # (B, T_audio)
        "frame_labels":  torch.stack(frame_labels),                     # (B, T_frames)
        "boundary_mask": torch.stack(boundary_masks),                   # (B, T_frames)
        "language_seq":  torch.stack(lang_seqs),                        # (B, S)
        "audio_lens":    torch.tensor(audio_lens,  dtype=torch.long),
        "frame_lens":    torch.tensor(frame_lens,  dtype=torch.long),
        "seq_lens":      torch.tensor(seq_lens,    dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from the OmegaConf config.

    Data layout expected under cfg.data.cache_dir:
        Dataset/
            train/  wav/  rttm/
            eval/   wav/  rttm/
            test/   wav/  rttm/

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    cache_dir = cfg.data.cache_dir
    sr        = cfg.audio.sample_rate
    syn_cfg   = cfg.data.synthetic

    train_ds = RealMultilingualDataset(
        data_dir=os.path.join(cache_dir, "train"),
        num_samples=syn_cfg.num_train_samples,
        sample_rate=sr,
        target_duration=syn_cfg.output_sample_duration,
        augment=True,
        seed=42,
    )

    val_ds = RealMultilingualDataset(
        data_dir=os.path.join(cache_dir, "eval"),
        num_samples=syn_cfg.num_val_samples,
        sample_rate=sr,
        target_duration=syn_cfg.output_sample_duration,
        augment=False,
        seed=999,
    )

    test_ds = RealMultilingualDataset(
        data_dir=os.path.join(cache_dir, "test"),
        num_samples=syn_cfg.num_val_samples,   # same size budget as val
        sample_rate=sr,
        target_duration=syn_cfg.output_sample_duration,
        augment=False,
        seed=1337,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS_TRAIN,
        collate_fn=collate_fn,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(_NUM_WORKERS_TRAIN > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS_VAL,
        collate_fn=collate_fn,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(_NUM_WORKERS_VAL > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS_VAL,
        collate_fn=collate_fn,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(_NUM_WORKERS_VAL > 0),
    )

    return train_loader, val_loader, test_loader


def build_test_dataloader(cfg) -> DataLoader:
    """
    Convenience helper: returns only the test DataLoader.
    Useful for standalone evaluation after training.
    """
    _, _, test_loader = build_dataloaders(cfg)
    return test_loader
