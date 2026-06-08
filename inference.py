"""
inference.py
────────────
Command-line inference script.

Usage
─────
  python inference.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pt

  # with custom threshold
  python inference.py --audio interview.mp3 --checkpoint checkpoints/best_model.pt \
      --threshold 0.4 --smooth_window 7

  # save visualisation
  python inference.py --audio clip.wav --checkpoint checkpoints/best_model.pt \
      --plot output/timeline.png
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(__file__))

from omegaconf import OmegaConf
from inference.detector import LanguageBoundaryInference
from utils.visualization import plot_timeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Conformer Language Boundary Detector — Inference"
    )
    p.add_argument("--audio",      required=True,  help="Path to input audio file")
    p.add_argument("--checkpoint", required=True,  help="Path to model checkpoint (.pt)")
    p.add_argument("--config",     default="configs/conformer_config.yaml",
                   help="Config YAML (default: configs/conformer_config.yaml)")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Boundary detection threshold (default from config)")
    p.add_argument("--smooth_window", type=int, default=None,
                   help="Moving-average smoothing window (frames)")
    p.add_argument("--chunk_sec",  type=float, default=None,
                   help="Sliding window chunk size in seconds")
    p.add_argument("--step_sec",   type=float, default=None,
                   help="Sliding window step in seconds")
    p.add_argument("--plot",       default=None,
                   help="Save timeline plot to this path (optional)")
    p.add_argument("--json_out",   default=None,
                   help="Save result JSON to this path (optional)")
    p.add_argument("--device",     default="cuda",
                   help="Device: cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ── config ───────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    inf_cfg = cfg.inference
    m_cfg   = cfg.model
    a_cfg   = cfg.audio

    threshold    = args.threshold    or m_cfg.boundary_threshold
    smooth_win   = args.smooth_window or m_cfg.temporal_smoothing_window
    chunk_sec    = args.chunk_sec    or inf_cfg.chunk_size_sec
    step_sec     = args.step_sec     or inf_cfg.step_size_sec

    # ── detector ─────────────────────────────────────────────────────────────
    detector = LanguageBoundaryInference(
        checkpoint_path      = args.checkpoint,
        device               = args.device,
        sample_rate          = a_cfg.sample_rate,
        n_mels               = a_cfg.n_mels,
        hop_length           = a_cfg.hop_length,
        encoder_dim          = m_cfg.encoder_dim,
        num_layers           = m_cfg.num_layers,
        num_heads            = m_cfg.num_heads,
        ff_expansion_factor  = m_cfg.ff_expansion_factor,
        conv_kernel_size     = m_cfg.conv_kernel_size,
        num_languages        = m_cfg.num_languages,
        input_dim            = m_cfg.input_dim,
        wavegram_channels    = a_cfg.wavegram_channels,
        wavegram_kernel      = a_cfg.wavegram_kernel,
        chunk_size_sec       = chunk_sec,
        step_size_sec        = step_sec,
        boundary_thr         = threshold,
        smooth_window        = smooth_win,
        min_seg_dur_sec      = cfg.eval.min_segment_duration,
    )

    # ── run ──────────────────────────────────────────────────────────────────
    print(f"\n[Inference] Processing: {args.audio}")
    result = detector.predict(args.audio)
    detector.pretty_print(result)

    # ── optional: JSON output ─────────────────────────────────────────────
    if args.json_out:
        out_data = {
            "segments":         result["segments"],
            "switch_times_sec": result["switch_times_sec"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"[Inference] JSON saved → {args.json_out}")

    # ── optional: timeline plot ───────────────────────────────────────────
    if args.plot:
        import torchaudio, os
        info = torchaudio.info(args.audio)
        dur  = info.num_frames / info.sample_rate
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        plot_timeline(
            result,
            audio_duration_sec=dur,
            title=f"Language Boundaries — {os.path.basename(args.audio)}",
            save_path=args.plot,
            show=False,
        )


if __name__ == "__main__":
    main()
