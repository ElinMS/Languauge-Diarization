"""
test_forward_pass.py
====================
Smoke test — verifies the full pipeline (DualFrontend → LanguageBoundaryDetector)
runs end-to-end with a dummy waveform WITHOUT needing any real data or checkpoint.

Usage
-----
  python test_forward_pass.py          # CPU test (quick)
  python test_forward_pass.py --cuda   # GPU test

Expected output (all PASSED):
  [1/6] PASSED  DualFrontend forward pass
  [2/6] PASSED  LanguageBoundaryDetector forward pass
  [3/6] PASSED  Output shapes
  [4/6] PASSED  MultiTaskLoss forward pass
  [5/6] PASSED  Inference helper (predict_boundaries)
  [6/6] PASSED  Padding mask with variable lengths
  ✅  All tests passed!
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

DEVICE   = "cuda" if ("--cuda" in sys.argv and torch.cuda.is_available()) else "cpu"
B        = 2        # batch size
T_SAMP   = 16000    # 1 second of audio @ 16 kHz
N_LANGS  = 10

print(f"\n{'='*55}")
print(f"  Language Boundary Detector — Smoke Test")
print(f"  Device : {DEVICE.upper()}")
print(f"{'='*55}\n")

passed = 0
failed = 0

def check(name, fn, step):
    global passed, failed
    try:
        fn()
        print(f"  [{step}/6] PASSED  {name}")
        passed += 1
    except Exception as e:
        print(f"  [{step}/6] FAILED  {name}")
        print(f"            -> {type(e).__name__}: {e}")
        failed += 1

# ── imports ────────────────────────────────────────────────────────────────────
from data.feature_extraction import DualFrontend, SpecAugment
from models.language_detector import LanguageBoundaryDetector
from training.losses import MultiTaskLoss

# ── shared objects ─────────────────────────────────────────────────────────────
frontend = DualFrontend().to(DEVICE)
model    = LanguageBoundaryDetector(
    input_dim    = 208,   # 80 mel + 128 wavegram
    encoder_dim  = 64,    # small for speed
    num_layers   = 2,
    num_heads    = 4,
    num_languages= N_LANGS,
).to(DEVICE)
criterion = MultiTaskLoss(blank_idx=N_LANGS)
spec_aug  = SpecAugment()

dummy_wav = torch.randn(B, T_SAMP).to(DEVICE)
feats_ref = None   # filled in test 1

# ── Test 1: DualFrontend ───────────────────────────────────────────────────────
def t1():
    global feats_ref
    feats = frontend(dummy_wav)              # (B, T_frames, 208)
    assert feats.ndim == 3, f"Expected 3D, got {feats.ndim}D"
    assert feats.shape[0] == B
    assert feats.shape[2] == 208, f"Expected 208 features, got {feats.shape[2]}"
    feats_ref = feats.detach()

check("DualFrontend forward pass", t1, 1)

# ── Test 2: Model forward ──────────────────────────────────────────────────────
out_ref = None
def t2():
    global out_ref
    feats = feats_ref if feats_ref is not None else frontend(dummy_wav)
    out   = model(feats)
    assert "frame_logits"    in out
    assert "boundary_logits" in out
    assert "ctc_log_probs"   in out
    assert "enc_lengths"     in out
    out_ref = {k: v.detach() for k, v in out.items()}

check("LanguageBoundaryDetector forward pass", t2, 2)

# ── Test 3: Output shapes ──────────────────────────────────────────────────────
def t3():
    assert out_ref is not None, "Test 2 must pass first"
    T_enc = out_ref["frame_logits"].shape[1]
    assert out_ref["frame_logits"].shape    == (B, T_enc, N_LANGS),   \
        f"frame_logits shape wrong: {out_ref['frame_logits'].shape}"
    assert out_ref["boundary_logits"].shape == (B, T_enc),             \
        f"boundary_logits shape wrong: {out_ref['boundary_logits'].shape}"
    assert out_ref["ctc_log_probs"].shape   == (B, T_enc, N_LANGS+1), \
        f"ctc_log_probs shape wrong: {out_ref['ctc_log_probs'].shape}"
    print(f"            T_enc={T_enc}, feat_dim=208 OK", end="")

check("Output shapes", t3, 3)

# ── Test 4: Loss ───────────────────────────────────────────────────────────────
def t4():
    feats      = frontend(dummy_wav)
    T_frames   = feats.shape[1]
    frame_lens = torch.tensor([T_frames, T_frames // 2], dtype=torch.long).to(DEVICE)
    out        = model(feats, frame_lens)

    T_enc    = out["frame_logits"].shape[1]
    T_full   = T_frames

    targets = {
        "frame_labels":  torch.randint(0, N_LANGS, (B, T_full)).to(DEVICE),
        "boundary_mask": torch.zeros(B, T_full).to(DEVICE),
        "language_seq":  torch.randint(0, N_LANGS, (B, 3)).to(DEVICE),
        "seq_lens":      torch.tensor([3, 2], dtype=torch.long).to(DEVICE),
    }
    # put a few boundaries in
    targets["boundary_mask"][:, T_full // 2] = 1.0

    loss_dict = criterion(out, targets)
    assert "loss" in loss_dict
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN!"
    assert loss_dict["loss"].item() > 0

check("MultiTaskLoss forward pass", t4, 4)

# ── Test 5: predict_boundaries helper ─────────────────────────────────────────
def t5():
    feats = frontend(dummy_wav[:1])   # single sample
    result = model.predict_boundaries(feats)
    assert "switch_times_sec"  in result
    assert "frame_lang_ids"    in result
    assert "boundary_probs"    in result
    assert isinstance(result["frame_lang_ids"], list)

check("predict_boundaries inference helper", t5, 5)

# ── Test 6: Padding mask with variable lengths ─────────────────────────────────
def t6():
    feats      = frontend(dummy_wav)
    T_frames   = feats.shape[1]
    frame_lens = torch.tensor([T_frames, T_frames // 3], dtype=torch.long).to(DEVICE)
    out        = model(feats, frame_lens)
    enc_lens   = out["enc_lengths"]
    assert enc_lens[0] > enc_lens[1], \
        f"Longer input should have longer enc_length: {enc_lens.tolist()}"

check("Padding mask with variable lengths", t6, 6)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"{'='*55}")
if failed == 0:
    print(f"  [OK] All {passed} tests passed!")
else:
    print(f"  [!!] {failed} test(s) FAILED, {passed} passed")
print(f"{'='*55}\n")
sys.exit(0 if failed == 0 else 1)
