"""
tda_analysis.py
---------------
Applies Topological Data Analysis (persistent homology) to ESM-2 hidden state
activations to characterize and compare latent space geometry between
in-distribution and out-of-distribution protein sequences.

Core hypothesis:
    In-distribution sequences occupy topologically structured (persistent)
    regions of latent space, while OOD sequences produce geometrically
    noisier, less persistent topological features.

Author: Grold Otieno Mboya
Project: TDA Analysis of Latent Space Geometry in Foundation Models
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ripser import ripser
from persim import bottleneck
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import os
import warnings
warnings.filterwarnings("ignore")

# ── Publication-quality matplotlib style ─────────────────────────────────────
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "axes.titleweight":   "normal",
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "legend.fontsize":    7,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth":    0.8,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

C_IN  = "#3778C2"
C_OOD = "#C03B26"
C_H0  = "#3778C2"
C_H1  = "#E8A838"

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
LAYERS     = [2, 4, 6]
MAX_DIM    = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Activations ──────────────────────────────────────────────────────────
def load_activations(layer):
    in_act  = np.load(os.path.join(DATA_DIR, f"in_dist_layer{layer}.npy"))
    ood_act = np.load(os.path.join(DATA_DIR, f"ood_layer{layer}.npy"))
    return in_act, ood_act

# ── Dimensionality Reduction ──────────────────────────────────────────────────
def reduce_dimensions(in_act, ood_act, n_components=8):
    combined = StandardScaler().fit_transform(np.vstack([in_act, ood_act]))
    reduced  = PCA(n_components=n_components, random_state=42).fit_transform(combined)
    n = len(in_act)
    return reduced[:n], reduced[n:]

# ── Persistent Homology ───────────────────────────────────────────────────────
def compute_persistence(activations, label, layer):
    print(f"  Computing persistence for {label} (layer {layer})...")
    result   = ripser(activations, maxdim=MAX_DIM)
    diagrams = result['dgms']
    h0       = diagrams[0]
    h1       = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
    h0_fin   = h0[h0[:, 1] != np.inf]
    h0_pers  = (h0_fin[:, 1] - h0_fin[:, 0]) if len(h0_fin) > 0 else np.array([0.0])
    h1_pers  = (h1[:, 1] - h1[:, 0])          if len(h1) > 0     else np.array([0.0])
    return {
        "label": label, "layer": layer, "diagrams": diagrams,
        "h0_mean_pers": float(np.mean(h0_pers)),
        "h1_mean_pers": float(np.mean(h1_pers)),
        "total_pers":   float(np.sum(h0_pers) + np.sum(h1_pers)),
    }

# ── Bottleneck Distance ───────────────────────────────────────────────────────
def compute_bottleneck_distance(s_in, s_ood):
    d_h0 = bottleneck(s_in["diagrams"][0], s_ood["diagrams"][0])
    h1_in, h1_ood = s_in["diagrams"][1], s_ood["diagrams"][1]
    d_h1 = bottleneck(h1_in, h1_ood) if (len(h1_in) > 0 and len(h1_ood) > 0) else 0.0
    return {"h0": d_h0, "h1": d_h1}

# ── Persistence Diagram Plot ──────────────────────────────────────────────────
def plot_persistence_comparison(s_in, s_ood, layer):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    for ax, stats, title in zip(axes,
                                 [s_in, s_ood],
                                 ["In-distribution", "OOD (shuffled)"]):
        dgms = stats["diagrams"]
        h0   = dgms[0]
        h1   = dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2)
        h0_fin = h0[h0[:, 1] != np.inf]

        all_finite = []
        if len(h0_fin): all_finite.append(h0_fin)
        if len(h1):     all_finite.append(h1)

        if all_finite:
            all_vals = np.vstack(all_finite)
            lo = all_vals.min() * 0.92
            hi = all_vals.max() * 1.08
        else:
            lo, hi = 0, 1

        ax.plot([lo, hi], [lo, hi], color="gray", lw=0.5, ls="--", zorder=1)

        if len(h0_fin):
            ax.scatter(h0_fin[:, 0], h0_fin[:, 1],
                       s=14, c=C_H0, label=r"$H_0$",
                       zorder=3, linewidths=0, alpha=0.9)
        if len(h1):
            ax.scatter(h1[:, 0], h1[:, 1],
                       s=14, c=C_H1, label=r"$H_1$",
                       zorder=3, linewidths=0, alpha=0.9)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(
            f"{title} — layer {layer}\n"
            f"total persistence: {stats['total_pers']:.2f}",
            fontsize=8, pad=4
        )
        ax.set_xlabel("Birth", fontsize=7)
        ax.set_ylabel("Death", fontsize=7)
        if len(h0_fin) or len(h1):
            ax.legend(loc="lower right", markerscale=1.4, handletextpad=0.3)

    fig.suptitle("Persistence diagrams: ESM-2 latent representations", fontsize=9, y=1.01)
    plt.tight_layout(w_pad=2.0)
    base = os.path.join(OUTPUT_DIR, f"persistence_layer{layer}")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()
    print(f"  Saved persistence diagram → {base}.png")

# ── UMAP Visualization ────────────────────────────────────────────────────────
def plot_umap(in_act, ood_act, layer):
    combined  = np.vstack([in_act, ood_act])
    labels    = np.array(["In-distribution"] * len(in_act) +
                         ["OOD (shuffled)"]  * len(ood_act))
    embedding = umap.UMAP(n_components=2, random_state=42,
                          n_neighbors=5).fit_transform(combined)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    specs = [("In-distribution", C_IN,  "o"),
             ("OOD (shuffled)",  C_OOD, "s")]
    for lbl, color, marker in specs:
        mask = labels == lbl
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=18, label=lbl, marker=marker,
                   alpha=0.85, linewidths=0.3, edgecolors="white")

    ax.set_title(f"UMAP projection — ESM-2 layer {layer}", fontsize=9, pad=4)
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.legend(loc="best")
    plt.tight_layout()
    base = os.path.join(OUTPUT_DIR, f"umap_layer{layer}")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()
    print(f"  Saved UMAP plot → {base}.png")

# ── Summary Bar Chart ─────────────────────────────────────────────────────────
def plot_summary(all_results):
    layers = [r["layer"] for r in all_results]
    d_h0   = [r["bottleneck_h0"] for r in all_results]
    d_h1   = [r["bottleneck_h1"] for r in all_results]
    x      = np.arange(len(layers))
    w      = 0.30

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    b1 = ax.bar(x - w/2, d_h0, w, label=r"$H_0$ (connected components)",
                color=C_H0, alpha=0.85, linewidth=0)
    b2 = ax.bar(x + w/2, d_h1, w, label=r"$H_1$ (loops / cycles)",
                color=C_H1, alpha=0.85, linewidth=0)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Transformer layer", fontsize=8)
    ax.set_ylabel("Bottleneck distance", fontsize=8)
    ax.set_title(
        "Topological divergence between in-distribution and OOD\n"
        "sequences across ESM-2 layers",
        fontsize=8, pad=5
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend(fontsize=7)
    ax.set_ylim(0, max(d_h0 + [0.1]) * 1.22)
    plt.tight_layout()
    base = os.path.join(OUTPUT_DIR, "bottleneck_summary")
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".png")
    plt.close()
    print(f"\n  Saved summary chart → {base}.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TDA Analysis of ESM-2 Latent Space Geometry")
    print("In-Distribution vs Out-of-Distribution Protein Sequences")
    print("=" * 60)

    all_results = []

    for layer in LAYERS:
        print(f"\n── Layer {layer} ──────────────────────────────────────")
        in_act, ood_act = load_activations(layer)
        in_red, ood_red = reduce_dimensions(in_act, ood_act)
        print(f"  Activations loaded. Shape: {in_act.shape} → reduced to {in_red.shape}")
        s_in  = compute_persistence(in_red,  "in_dist", layer)
        s_ood = compute_persistence(ood_red, "ood",     layer)
        dist  = compute_bottleneck_distance(s_in, s_ood)

        print(f"\n  ┌─ In-distribution  │ H0 mean pers: {s_in['h0_mean_pers']:.4f}  │ H1: {s_in['h1_mean_pers']:.4f}")
        print(f"  ├─ OOD (shuffled)   │ H0 mean pers: {s_ood['h0_mean_pers']:.4f}  │ H1: {s_ood['h1_mean_pers']:.4f}")
        print(f"  └─ Bottleneck dist  │ H0: {dist['h0']:.4f}  │ H1: {dist['h1']:.4f}")

        plot_persistence_comparison(s_in, s_ood, layer)
        plot_umap(in_red, ood_red, layer)

        all_results.append({
            "layer":         layer,
            "in_h0_pers":    s_in["h0_mean_pers"],
            "ood_h0_pers":   s_ood["h0_mean_pers"],
            "in_h1_pers":    s_in["h1_mean_pers"],
            "ood_h1_pers":   s_ood["h1_mean_pers"],
            "bottleneck_h0": dist["h0"],
            "bottleneck_h1": dist["h1"],
        })

    plot_summary(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Layer':<8} {'In H0':>10} {'OOD H0':>10} {'BN-H0':>10} {'BN-H1':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"  {r['layer']:<6} {r['in_h0_pers']:>10.4f} {r['ood_h0_pers']:>10.4f} "
              f"{r['bottleneck_h0']:>10.4f} {r['bottleneck_h1']:>10.4f}")

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}/")
    print("  Formats: .pdf (submission) and .png (preview)")