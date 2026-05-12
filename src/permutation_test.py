"""
permutation_test.py
-------------------
Stratified permutation test using both Wasserstein (primary) and
bottleneck (robustness check) distances on H0 persistence diagrams.

Method: Family-stratified permutation (n=1000)
  Labels swapped within each protein family to control for
  biological diversity as a confounder.

Output: outputs/summary/permutation_test.png

Author: Grold Otieno Mboya
Project: TDA Analysis of Latent Space Geometry in Foundation Models
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ripser import ripser
from persim import wasserstein, bottleneck
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          7,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7,
    "axes.titleweight":   "normal",
    "axes.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "legend.fontsize":    6,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    "lines.linewidth":    0.7,
})

C_NULL_W = "#6BAED6"
C_NULL_B = "#AAAAAA"
C_OBS_W  = "#08519C"
C_OBS_B  = "#C03B26"

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "data"
OUTPUT_DIR     = os.path.join("outputs", "summary")
N_PERMUTATIONS = 1000
LAYERS         = [2, 4, 6]
RANDOM_SEED    = 42
FAMILIES       = ["kinase", "oxidoreductase", "transcription_factor",
                  "chaperone", "transporter"]
N_PER_FAMILY   = 12

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_layer(layer):
    in_act  = np.load(os.path.join(DATA_DIR, f"in_dist_layer{layer}.npy"))
    ood_act = np.load(os.path.join(DATA_DIR, f"ood_layer{layer}.npy"))
    return in_act, ood_act

def reduce_dimensions(in_act, ood_act, n_components=8):
    combined = StandardScaler().fit_transform(np.vstack([in_act, ood_act]))
    reduced  = PCA(n_components=n_components, random_state=42).fit_transform(combined)
    n = len(in_act)
    return reduced[:n], reduced[n:]

def get_h0_diagrams(a, b):
    dgm_a = ripser(a, maxdim=0)['dgms'][0]
    dgm_b = ripser(b, maxdim=0)['dgms'][0]
    return dgm_a, dgm_b

def get_family_indices():
    return {fam: list(range(i * N_PER_FAMILY, (i + 1) * N_PER_FAMILY))
            for i, fam in enumerate(FAMILIES)}

def sig_label(p):
    if p < 0.001: return "p < 0.001 ***"
    if p < 0.01:  return f"p = {p:.3f} **"
    if p < 0.05:  return f"p = {p:.3f} *"
    return f"p = {p:.3f} (n.s.)"

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

# ── Stratified Permutation Test ───────────────────────────────────────────────
def stratified_permutation_test(in_red, ood_red, n_perms=N_PERMUTATIONS):
    family_idx = get_family_indices()

    dgm_in, dgm_ood = get_h0_diagrams(in_red, ood_red)
    obs_w = wasserstein(dgm_in, dgm_ood)
    obs_b = bottleneck(dgm_in, dgm_ood)

    null_w, null_b = [], []
    print(f"  Running {n_perms} stratified permutations", end="", flush=True)

    for i in range(n_perms):
        if i % 100 == 0:
            print(".", end="", flush=True)

        perm_in  = in_red.copy()
        perm_ood = ood_red.copy()

        for fam, idx in family_idx.items():
            for j in idx:
                if np.random.rand() < 0.5:
                    perm_in[j], perm_ood[j] = ood_red[j].copy(), in_red[j].copy()

        dgm_a, dgm_b = get_h0_diagrams(perm_in, perm_ood)
        null_w.append(wasserstein(dgm_a, dgm_b))
        null_b.append(bottleneck(dgm_a, dgm_b))

    print(" done")
    null_w = np.array(null_w)
    null_b = np.array(null_b)

    return {
        "obs_w":  obs_w,  "null_w": null_w, "p_w": float(np.mean(null_w >= obs_w)),
        "obs_b":  obs_b,  "null_b": null_b, "p_b": float(np.mean(null_b >= obs_b)),
    }

# ── Plot Panel ────────────────────────────────────────────────────────────────
def plot_panel(null, observed, p, label, c_null, c_obs, ax):
    x_max = max(null.max(), observed) * 1.12
    ax.hist(null, bins=30, color=c_null, alpha=0.70,
            edgecolor="white", linewidth=0.25, label="Null")
    ax.axvline(observed, color=c_obs, lw=1.0, ls="--",
               label=f"Obs = {observed:.2f}")
    thr = np.percentile(null, 95)
    ax.axvspan(thr, x_max, alpha=0.10, color=c_obs, label="p < 0.05")
    ax.set_xlim(0, x_max)
    ax.set_xlabel(label, fontsize=6.5)
    ax.set_ylabel("Frequency", fontsize=6.5)
    ax.legend(fontsize=5.5, handlelength=1.2)
    return sig_label(p)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Stratified Permutation Test")
    print("Wasserstein (primary) + Bottleneck (robustness check)")
    print("=" * 60)

    fig, axes   = plt.subplots(2, 3, figsize=(8.5, 5.2))
    all_results = []

    for col, layer in enumerate(LAYERS):
        print(f"\n── Layer {layer} ──────────────────────────────────────")
        in_act, ood_act = load_layer(layer)
        in_red, ood_red = reduce_dimensions(in_act, ood_act)
        res             = stratified_permutation_test(in_red, ood_red)

        print(f"  Wasserstein  obs: {res['obs_w']:.4f}  "
              f"null: {res['null_w'].mean():.4f} ± {res['null_w'].std():.4f}  "
              f"p = {res['p_w']:.4f}  {sig_stars(res['p_w'])}")
        print(f"  Bottleneck   obs: {res['obs_b']:.4f}  "
              f"null: {res['null_b'].mean():.4f} ± {res['null_b'].std():.4f}  "
              f"p = {res['p_b']:.4f}  {sig_stars(res['p_b'])}")

        # Wasserstein row
        sl_w = plot_panel(res["null_w"], res["obs_w"], res["p_w"],
                          r"$W_1(H_0)$", C_NULL_W, C_OBS_W, axes[0, col])
        axes[0, col].set_title(f"Wasserstein — layer {layer}\n{sl_w}",
                               fontsize=7, pad=3)

        # Bottleneck row
        sl_b = plot_panel(res["null_b"], res["obs_b"], res["p_b"],
                          r"$d_B(H_0)$", C_NULL_B, C_OBS_B, axes[1, col])
        axes[1, col].set_title(f"Bottleneck — layer {layer}\n{sl_b}",
                               fontsize=7, pad=3)

        all_results.append({
            "layer":       layer,
            "obs_w":       res["obs_w"],
            "null_mean_w": float(res["null_w"].mean()),
            "null_std_w":  float(res["null_w"].std()),
            "p_w":         res["p_w"],
            "obs_b":       res["obs_b"],
            "null_mean_b": float(res["null_b"].mean()),
            "null_std_b":  float(res["null_b"].std()),
            "p_b":         res["p_b"],
        })

    fig.suptitle(
        f"Stratified permutation test (n = {N_PERMUTATIONS} permutations, "
        f"stratified by protein family)\n"
        f"Top: Wasserstein distance  ·  Bottom: Bottleneck distance",
        fontsize=7.5, y=1.01
    )
    plt.tight_layout(h_pad=2.2, w_pad=1.5)

    out_path = os.path.join(OUTPUT_DIR, "permutation_test.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\n  Saved → {out_path}")

    with open(os.path.join(OUTPUT_DIR, "permutation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SIGNIFICANCE SUMMARY")
    print("=" * 60)
    print(f"{'Layer':<6} {'W-obs':>8} {'W-null':>8} {'W-p':>8} {'W':>5} "
          f"{'B-obs':>8} {'B-null':>8} {'B-p':>8} {'B':>5}")
    print("-" * 68)
    for r in all_results:
        print(f"  {r['layer']:<4} {r['obs_w']:>8.3f} {r['null_mean_w']:>8.3f} "
              f"{r['p_w']:>8.4f} {sig_stars(r['p_w']):>5} "
              f"{r['obs_b']:>8.3f} {r['null_mean_b']:>8.3f} "
              f"{r['p_b']:>8.4f} {sig_stars(r['p_b']):>5}")

    print(f"\n✓ Figure  → outputs/summary/permutation_test.png")
    print(f"✓ Results → outputs/summary/permutation_results.json")