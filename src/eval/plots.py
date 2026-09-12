"""A couple of figures for the report. Nothing fancy."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.taxonomy import LABELS


def plot_confusion(gold, pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [l for l in LABELS if l in set(gold) or l in set(pred)]
    m = confusion_matrix(gold, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.replace("_", "\n") for l in labels]
    ax.set_xticklabels(short, fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_variant_bars(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [k for k in results if not k.startswith("_") and "random" in results[k]]
    if not names:
        return
    acc = [results[n]["random"]["intent"]["accuracy"] for n in names]
    f1 = [results[n]["random"]["intent"]["macro_f1"] for n in names]
    rec = [results[n]["random"]["routing"]["escalate_recall"] for n in names]
    send = [
        (results[n]["random"].get("judge") or {}).get("send_rate", 0.0) for n in names
    ]

    x = np.arange(len(names))
    w = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 1.5 * w, acc, w, label="intent acc")
    ax.bar(x - 0.5 * w, f1, w, label="macro-F1")
    ax.bar(x + 0.5 * w, rec, w, label="esc recall")
    ax.bar(x + 1.5 * w, send, w, label="send rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_title("random slice only (n=120)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")
