"""
utils/visualization.py
───────────────────────
Plotting helpers for:
  • Language boundary timelines
  • Boundary probability heat-maps
  • Per-frame language probability matrices
  • Training loss curves
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Colour palette — one distinct colour per language (up to 12)
LANG_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC", "#D37295", "#A0CBE8",
]


def plot_timeline(
    result: Dict,
    audio_duration_sec: float,
    title: str = "Language Boundary Detection",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Stacked visualisation:
      Top    — coloured segment timeline
      Bottom — boundary probability curve

    Parameters
    ----------
    result            : output dict from LanguageBoundaryInference.predict()
    audio_duration_sec: total audio length in seconds
    """
    segments   = result["segments"]
    bnd_probs  = result["boundary_probs"]   # (T_enc,)
    enc_hop    = audio_duration_sec / max(len(bnd_probs), 1)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={"height_ratios": [1, 2]},
        sharex=True,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── top: language timeline ────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_yticks([])
    ax0.set_ylabel("Language", fontsize=9)

    lang_ids_seen = {}
    for seg in segments:
        lid   = seg["lang_id"]
        color = LANG_COLORS[lid % len(LANG_COLORS)]
        lang_ids_seen[lid] = (seg["language"], color)
        ax0.barh(
            0,
            width=seg["end"] - seg["start"],
            left=seg["start"],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )
        # label in middle of segment if wide enough
        mid = (seg["start"] + seg["end"]) / 2
        width = seg["end"] - seg["start"]
        if width > 0.5:
            ax0.text(
                mid, 0, seg["language"].upper(),
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )

    # vertical switch lines
    for sw in result.get("switch_times_sec", []):
        ax0.axvline(sw, color="red", lw=1.2, ls="--", alpha=0.8)

    # legend
    patches = [
        mpatches.Patch(color=c, label=lang.upper())
        for lid, (lang, c) in sorted(lang_ids_seen.items())
    ]
    ax0.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.7)
    ax0.set_xlim(0, audio_duration_sec)
    ax0.set_ylim(-0.5, 0.5)

    # ── bottom: boundary probability curve ────────────────────────────────
    ax1 = axes[1]
    t   = np.arange(len(bnd_probs)) * enc_hop
    ax1.fill_between(t, bnd_probs, alpha=0.4, color="#E15759", label="P(boundary)")
    ax1.plot(t, bnd_probs, color="#E15759", lw=1.2)
    ax1.axhline(0.5, color="black", lw=0.8, ls=":", label="threshold=0.5")

    for sw in result.get("switch_times_sec", []):
        ax1.axvline(sw, color="red", lw=1.0, ls="--", alpha=0.7)

    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_ylabel("P(language switch)", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, audio_duration_sec)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] saved → {save_path}")
    if show:
        plt.show()
    return fig


def plot_loss_curves(
    train_losses: List[float],
    val_losses:   List[float],
    metric_name:  str = "loss",
    save_path:    Optional[str] = None,
    show:         bool = True,
) -> plt.Figure:
    """Simple epoch-level loss / metric curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label=f"Train {metric_name}", marker="o", ms=4)
    ax.plot(epochs, val_losses,   label=f"Val {metric_name}",   marker="s", ms=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"Training Curves — {metric_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    save_path:   Optional[str] = None,
    show:        bool = True,
) -> plt.Figure:
    """Per-language confusion matrix (normalised by row)."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(true_labels, pred_labels,
                          labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=[c.upper() for c in class_names],
        yticklabels=[c.upper() for c in class_names],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Language")
    ax.set_ylabel("True Language")
    ax.set_title("Per-Language Confusion Matrix (row-normalised)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
