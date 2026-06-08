# Conformer Language Boundary Detector

A **Conformer-based** model that detects *language switches* in multilingual audio as anomalies, pinpointing the exact timestamp (in seconds) of each switch.

---

## Architecture Overview

```
Raw Audio (16 kHz)
       │
  ┌────▼────────────┐
  │ Log-Mel Frontend │  80-dim filterbanks, 10ms hop → (B, T, 80)
  └────┬────────────┘
       │ SpecAugment (training only)
  ┌────▼────────────────────────────────────────────┐
  │  Conv2D Subsampling  (2×Conv3×3 stride-2)       │
  │  → 4× time compression  (B, T//4, 256)          │
  └────┬────────────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────────────┐
  │  Conformer Encoder  — 12 × ConformerBlock         │
  │                                                   │
  │  Each block:                                      │
  │    FF (½-step, SiLU)                              │
  │    → MHSA + RoPE (4 heads)                        │
  │    → Depthwise Conv Module + GLU + BatchNorm      │
  │    → FF (½-step, SiLU)                            │
  │    → LayerNorm                                    │
  └────┬─────────────────────────────────────────────┘
       │  encoder output  (B, T//4, 256)
       ├──────────────┬──────────────────┐
       │              │                  │
  ┌────▼───┐   ┌──────▼──────┐   ┌──────▼──────┐
  │ Frame  │   │  Boundary   │   │   CTC Head  │
  │  CLS   │   │    Head     │   │  (lang seq) │
  │ Head   │   │ (sigmoid)   │   │             │
  └────────┘   └─────────────┘   └─────────────┘
 (B,T,L) CE   (B,T,1) BCE         (B,T,L+1) CTC
```

### Loss Function (multi-task)
```
L_total = 0.3 · L_CTC  +  0.4 · L_FrameCE  +  0.3 · L_BoundaryBCE
```
Boundary BCE uses a **10× positive class weight** — language switches are rare events.

---

## Project Structure

```
language diarization/
│
├── configs/
│   └── conformer_config.yaml      ← all hyperparameters
│
├── data/
│   ├── feature_extraction.py      ← Log-Mel + SpecAugment
│   └── dataset.py                 ← MonolingualStore + Synthetic code-switching
│
├── models/
│   ├── conformer.py               ← RoPE · MHSA · ConvModule · ConformerBlock · Subsampling
│   └── language_detector.py       ← Full model + predict_boundaries()
│
├── training/
│   ├── losses.py                  ← MultiTaskLoss (CTC + CE + BCE)
│   ├── metrics.py                 ← Boundary F1 · Frame Accuracy · LER · MetricMeter
│   └── trainer.py                 ← Training loop, mixed-precision, cosine-warmup LR
│
├── inference/
│   └── detector.py                ← Sliding-window streaming inference + NMS
│
├── utils/
│   ├── audio.py                   ← load · trim · chunk · save helpers
│   └── visualization.py           ← Timeline · loss curves · confusion matrix plots
│
├── scripts/
│   └── prepare_data.py            ← Download FLEURS / CommonVoice / VoxPopuli
│
├── train.py                       ← Training entry point
├── inference.py                   ← CLI inference entry point
└── requirements.txt
```

---

## Step-by-Step Workflow

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download multilingual data
```bash
# Download FLEURS for 5 languages (~5000 clips each)
python scripts/prepare_data.py \
    --dataset fleurs \
    --langs en fr de es hi zh ar ru pt ja \
    --max_per_lang 5000

# Optionally add CommonVoice
python scripts/prepare_data.py \
    --dataset common_voice \
    --langs en fr de es --max_per_lang 3000
```

This builds the cache layout:
```
data/cache/
    en/  000000.wav … 004999.wav
    fr/  000000.wav …
    ...
```

### Step 3 — Train
```bash
python train.py

# Override any config value on the command line:
python train.py training.batch_size=8 training.epochs=30
python train.py model.num_layers=6 model.encoder_dim=128   # lightweight
```

Checkpoints are saved to `checkpoints/`. Best model by boundary F1 → `checkpoints/best_model.pt`.

TensorBoard:
```bash
tensorboard --logdir checkpoints/tb_logs
```

### Step 4 — Inference
```bash
python inference.py \
    --audio path/to/audio.wav \
    --checkpoint checkpoints/best_model.pt \
    --plot output/timeline.png \
    --json_out output/result.json
```

**Console output:**
```
── Language Boundary Detection Results ─────────────────
    0.00s →  4.32s  [ EN]  ████████████████
    4.32s →  9.10s  [ FR]  ███████████████████
    9.10s → 13.55s  [ DE]  █████████████████

  Total language switches: 2
  Switch times (s):  [4.32, 9.10]
────────────────────────────────────────────────────────
```

**JSON output:**
```json
{
  "segments": [
    {"start": 0.0,  "end": 4.32, "language": "en", "lang_id": 0},
    {"start": 4.32, "end": 9.10, "language": "fr", "lang_id": 1},
    {"start": 9.10, "end": 13.55,"language": "de", "lang_id": 2}
  ],
  "switch_times_sec": [4.32, 9.10]
}
```

---

## Configuration Reference (`configs/conformer_config.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| audio | sample_rate | 16000 | Input sample rate |
| audio | n_mels | 80 | Log-mel bins |
| audio | hop_length | 160 | 10ms frame shift |
| model | encoder_dim | 256 | Conformer hidden size |
| model | num_layers | 12 | Number of Conformer blocks |
| model | num_heads | 4 | MHSA heads |
| model | conv_kernel_size | 31 | Depthwise conv kernel |
| model | num_languages | 10 | Target language count |
| model | boundary_threshold | 0.5 | Sigmoid decision threshold |
| training | batch_size | 16 | Training batch size |
| training | epochs | 50 | Total training epochs |
| training | learning_rate | 1e-3 | Peak LR |
| training | warmup_steps | 10000 | LR warmup steps |
| training | mixed_precision | true | fp16 training |

---

## Key Design Decisions

### Why Conformer?
- **Conv Module** captures local acoustic patterns (phonemes, intonation) that signal language change.
- **Self-Attention** with **RoPE** captures long-range context (a language being spoken consistently for several seconds).
- **4× subsampling** reduces computation while keeping 40ms resolution — enough for clean boundary localisation.

### Why Synthetic Code-Switching?
Real code-switching data is scarce and hard to annotate. We build unlimited synthetic clips by concatenating monolingual segments with known, exact switch timestamps — giving perfect ground truth for boundary supervision.

### Three-Head Multi-task Learning
| Head | Loss | Role |
|------|------|------|
| Frame CLS | CrossEntropy | Dense per-frame language identity |
| Boundary | WeightedBCE | Explicit boundary anomaly detection |
| CTC | CTC | Sequence-level alignment — regularises encoder |

### Anomaly Framing
The **Boundary Head** treats each language switch as a binary anomaly. Positive class weight = 10× compensates for the high class imbalance (most frames are *not* boundaries).

### Streaming Inference
The sliding window (30s chunks, 5s stride) allows processing arbitrarily long audio. Non-maximum suppression collapses boundary probability runs to single timestamps.

---

## Metrics

| Metric | Description |
|--------|-------------|
| `boundary_f1` | F1 with ±200ms tolerance window |
| `boundary_precision` | Precision of predicted switch times |
| `boundary_recall` | Recall of ground-truth switch times |
| `frame_accuracy` | Per-frame language classification accuracy |
| `macro_f1` | Macro-averaged F1 across all languages |
| `language_error_rate` | Fraction of frames with wrong language ID |

---

## Supported Languages (default)

| Code | Language  |
|------|-----------|
| en   | English   |
| fr   | French    |
| de   | German    |
| es   | Spanish   |
| hi   | Hindi     |
| zh   | Mandarin  |
| ar   | Arabic    |
| ru   | Russian   |
| pt   | Portuguese|
| ja   | Japanese  |

Add more by extending `LANGUAGES` in `data/dataset.py` and updating `configs/conformer_config.yaml`.

---

## Next Steps

- [ ] **SSM variant**: Replace self-attention with a Mamba/S4 state-space layer for linear-time scaling
- [ ] **Real code-switching data**: Add SEAME, Miami, CS-English-Hindi corpora
- [ ] **Speaker diarization fusion**: Combine with speaker embeddings (ECAPA-TDNN) for joint speaker + language diarization
- [ ] **CRF boundary decoder**: Replace sigmoid with a linear-chain CRF for coherent sequence labelling
- [ ] **Export to ONNX**: For production deployment
