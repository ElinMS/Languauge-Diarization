"""
scripts/prepare_data.py
────────────────────────
Downloads and pre-processes multilingual audio data from HuggingFace
into the MonolingualStore cache layout:

    data/cache/
        en/  *.wav
        fr/  *.wav
        de/  *.wav
        ...

Supports:
  • google/fleurs          — high-quality read speech, 102 languages
  • mozilla-foundation/common_voice_13_0  — crowd-sourced, 100+ langs
  • facebook/voxpopuli    — parliamentary speech, 23 EU langs

Usage
─────
    python scripts/prepare_data.py --dataset fleurs --langs en fr de es hi
    python scripts/prepare_data.py --dataset common_voice --langs en fr de --max_per_lang 2000
    python scripts/prepare_data.py --dataset all --langs en fr de es hi zh ar ru pt ja
"""

import os
import sys
import argparse
import uuid
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import numpy as np

TARGET_SR   = 16_000
CACHE_ROOT  = Path("data/cache")

# ── HuggingFace dataset identifiers per source ──────────────────────────────
FLEURS_NAME     = "google/fleurs"
CV_NAME         = "mozilla-foundation/common_voice_13_0"
VOXPOPULI_NAME  = "facebook/voxpopuli"

# FLEURS uses locale codes like "en_us", "fr_fr" etc.
FLEURS_LOCALE = {
    "en": "en_us", "fr": "fr_fr", "de": "de_de",
    "es": "es_419", "hi": "hi_in", "zh": "cmn_hans_cn",
    "ar": "ar_eg",  "ru": "ru_ru", "pt": "pt_br",
    "ja": "ja_jp",
}

# CommonVoice uses lang codes directly
CV_LOCALE = {
    "en": "en", "fr": "fr", "de": "de", "es": "es",
    "hi": "hi", "zh-CN": "zh-CN", "ar": "ar", "ru": "ru",
    "pt": "pt", "ja": "ja",
}


def resample_to_np(audio_array, orig_sr: int) -> np.ndarray:
    """Resample any array to TARGET_SR and return float32."""
    if orig_sr == TARGET_SR:
        return audio_array.astype(np.float32)
    try:
        import librosa
        return librosa.resample(
            audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR
        )
    except ImportError:
        import torchaudio, torch
        t = torch.from_numpy(audio_array.astype(np.float32))
        if t.dim() == 1:
            t = t.unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_sr, TARGET_SR)
        return t.squeeze(0).numpy()


def save_clip(wav: np.ndarray, lang: str, idx: int):
    out_dir = CACHE_ROOT / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    fname   = out_dir / f"{lang}_{idx:06d}.wav"
    sf.write(str(fname), wav, TARGET_SR)


# ── FLEURS ──────────────────────────────────────────────────────────────────

def prepare_fleurs(langs, max_per_lang: int, split: str = "train"):
    from datasets import load_dataset
    for lang in langs:
        locale = FLEURS_LOCALE.get(lang)
        if locale is None:
            print(f"  [FLEURS] no locale mapping for '{lang}', skipping")
            continue
        print(f"  [FLEURS] downloading {lang} ({locale}) …")
        try:
            ds = load_dataset(
                FLEURS_NAME, locale,
                split=split,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [FLEURS] failed: {e}")
            continue

        count = 0
        for i, sample in enumerate(ds):
            if count >= max_per_lang:
                break
            audio = sample["audio"]
            wav   = resample_to_np(np.array(audio["array"]), audio["sampling_rate"])
            # skip very short clips
            if len(wav) < TARGET_SR * 0.5:
                continue
            save_clip(wav, lang, i)
            count += 1

        print(f"    → saved {count} clips for {lang}")


# ── CommonVoice ─────────────────────────────────────────────────────────────

def prepare_common_voice(langs, max_per_lang: int, split: str = "train"):
    from datasets import load_dataset
    for lang in langs:
        locale = CV_LOCALE.get(lang, lang)
        print(f"  [CommonVoice] downloading {lang} ({locale}) …")
        try:
            ds = load_dataset(
                CV_NAME, locale,
                split=split,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [CommonVoice] failed: {e}")
            continue

        count = 0
        for i, sample in enumerate(ds):
            if count >= max_per_lang:
                break
            audio = sample["audio"]
            wav   = resample_to_np(np.array(audio["array"]), audio["sampling_rate"])
            if len(wav) < TARGET_SR * 0.5:
                continue
            save_clip(wav, lang, i + 100_000)   # offset to avoid name clash with FLEURS
            count += 1

        print(f"    → saved {count} clips for {lang}")


# ── VoxPopuli ────────────────────────────────────────────────────────────────

def prepare_voxpopuli(langs, max_per_lang: int, split: str = "train"):
    from datasets import load_dataset
    SUPPORTED = {"en","fr","de","es","pt","it","pl","nl","fi","hu","cs","ro","sk"}
    for lang in langs:
        if lang not in SUPPORTED:
            print(f"  [VoxPopuli] '{lang}' not in supported set, skipping")
            continue
        print(f"  [VoxPopuli] downloading {lang} …")
        try:
            ds = load_dataset(
                VOXPOPULI_NAME, lang,
                split=split,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [VoxPopuli] failed: {e}")
            continue

        count = 0
        for i, sample in enumerate(ds):
            if count >= max_per_lang:
                break
            audio = sample["audio"]
            wav   = resample_to_np(np.array(audio["array"]), audio["sampling_rate"])
            if len(wav) < TARGET_SR * 0.5:
                continue
            save_clip(wav, lang, i + 200_000)
            count += 1

        print(f"    → saved {count} clips for {lang}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["fleurs", "common_voice", "voxpopuli", "all"],
        default="fleurs",
        help="Which dataset to download",
    )
    p.add_argument(
        "--langs", nargs="+",
        default=["en", "fr", "de", "es", "hi"],
        help="Language codes to download",
    )
    p.add_argument(
        "--max_per_lang", type=int, default=5000,
        help="Maximum clips per language",
    )
    p.add_argument(
        "--split", default="train",
        help="Dataset split: train / validation / test",
    )
    p.add_argument(
        "--cache_dir", default="data/cache",
        help="Root output directory for cached WAV files",
    )
    return p.parse_args()


def main():
    args = parse_args()
    global CACHE_ROOT
    CACHE_ROOT = Path(args.cache_dir)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Preparing multilingual audio cache")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Languages    : {args.langs}")
    print(f"  Max per lang : {args.max_per_lang}")
    print(f"  Output dir   : {CACHE_ROOT.resolve()}")
    print(f"{'='*60}\n")

    if args.dataset in ("fleurs", "all"):
        prepare_fleurs(args.langs, args.max_per_lang, args.split)

    if args.dataset in ("common_voice", "all"):
        prepare_common_voice(args.langs, args.max_per_lang, args.split)

    if args.dataset in ("voxpopuli", "all"):
        prepare_voxpopuli(args.langs, args.max_per_lang, args.split)

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Cache summary:")
    total = 0
    for lang_dir in sorted(CACHE_ROOT.iterdir()):
        if lang_dir.is_dir():
            n = len(list(lang_dir.glob("*.wav")))
            total += n
            print(f"    {lang_dir.name:>5}  :  {n:>6} clips")
    print(f"  {'TOTAL':>5}  :  {total:>6} clips")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
