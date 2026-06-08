"""
train.py
────────
Main entry point for training the Conformer language boundary detector.

Usage
─────
  # default config
  python train.py

  # override any value via dotlist CLI
  python train.py training.batch_size=8 training.epochs=30
  python train.py model.num_layers=6 model.encoder_dim=128
"""

import sys
import os

# ── CRITICAL: Windows multiprocessing fix ────────────────────────────────────
# PyTorch DataLoader spawns worker processes on Windows.
# Without this guard the module re-imports itself → infinite loop.
# All training code must live inside  if __name__ == "__main__":
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    # "spawn" is the Windows default; setting it explicitly prevents subtle bugs
    multiprocessing.set_start_method("spawn", force=True)

    # make project root importable when called from any working directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from omegaconf import OmegaConf
    from data.dataset import build_dataloaders
    from training.trainer import Trainer

    def main():
        # ── load config ──────────────────────────────────────────────────────
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs", "conformer_config.yaml",
        )
        cfg = OmegaConf.load(cfg_path)

        # allow CLI overrides: python train.py key=value
        if len(sys.argv) > 1:
            cli_overrides = OmegaConf.from_dotlist(sys.argv[1:])
            cfg = OmegaConf.merge(cfg, cli_overrides)

        print("=" * 60)
        print("  Conformer Language Boundary Detector — Training")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

        # ── data ─────────────────────────────────────────────────────────────
        print("\n[Data] Building dataloaders …")
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
        print(f"  Train batches : {len(train_loader)}")
        print(f"  Val   batches : {len(val_loader)}")
        print(f"  Test  batches : {len(test_loader)}")

        # ── train ────────────────────────────────────────────────────────────
        trainer = Trainer(cfg, train_loader, val_loader)
        trainer.fit()

    main()
