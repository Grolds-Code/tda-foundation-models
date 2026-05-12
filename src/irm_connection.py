"""
irm_connection.py
-----------------
Applies the CausTab gradient variance penalty (Mboya, 2026) to ESM-2
hidden state activations to identify invariant vs spurious activation
dimensions across protein family environments.

Theoretical grounding:
  Theorem 1 (Mboya, 2026): Omega(theta) = 0 at the causally invariant
  solution and > 0 at any solution relying on spurious features.

Environment definition:
  Each protein family = one environment.
  Shared causal mechanism assumption holds: real vs shuffled is a causal
  property, not a family-specific spurious one.

Reference:
  Mboya, G.O. (2026). CausTab: Gradient Variance Regularization for
  Causal Invariant Representation Learning on Tabular Data. medRxiv.
  DOI: 10.64898/2026.04.09.26350513

Author: Grold Otieno Mboya
Project: TDA Analysis of Latent Space Geometry in Foundation Models
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn as nn
import torch.optim as optim
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

C_FULL = "#555555"
C_INV  = "#3778C2"
C_SPUR = "#C03B26"

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "data"
OUTPUT_DIR  = os.path.join("outputs", "irm_connection")
LAYERS      = [2, 4, 6]
FAMILIES    = ["kinase", "oxidoreductase", "transcription_factor",
               "chaperone", "transporter"]
N_PER_FAM   = 12
N_DIMS      = 320
SUBSPACE_Q  = 0.25

# CausTab hyperparameters — Algorithm 1 (Mboya, 2026)
HIDDEN_DIM  = 64
T_TOTAL     = 200
T_ANNEAL    = 50
T_WARMUP    = 20
LAMBDA      = 100.0
LR          = 1e-3
PCA_DIMS    = 8
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_layer(layer):
    in_act  = np.load(os.path.join(DATA_DIR, f"in_dist_layer{layer}.npy"))
    ood_act = np.load(os.path.join(DATA_DIR, f"ood_layer{layer}.npy"))
    return in_act, ood_act


def get_environments(in_act, ood_act):
    """One environment per protein family — Section 3.1, Mboya (2026)."""
    envs = []
    for i in range(len(FAMILIES)):
        start = i * N_PER_FAM
        end   = start + N_PER_FAM
        x     = np.vstack([in_act[start:end], ood_act[start:end]])
        y     = np.array([1.0] * N_PER_FAM + [0.0] * N_PER_FAM)
        x     = StandardScaler().fit_transform(x)
        envs.append((
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        ))
    return envs


# ── Model ─────────────────────────────────────────────────────────────────────
class CausTabEncoder(nn.Module):
    """
    Feedforward encoder + linear head.
    Uses LayerNorm instead of BatchNorm — batch size per environment
    (24 samples) is too small for stable BatchNorm statistics.
    LayerNorm normalises per sample and works at any batch size.
    Architecture otherwise follows Section 4.1, Mboya (2026).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LayerNorm(HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(HIDDEN_DIM // 2, 1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.encoder(x)))


# ── CausTab Penalty — Equation 6 (Mboya, 2026) ───────────────────────────────
def caustab_penalty(model, envs, criterion):
    """
    Omega(theta) = (1/|theta|) * sum_j Var_e [ g^e_j(theta) ]
    Full gradient vector — not scalar dummy (Section 4.3, Mboya 2026).
    """
    env_grads = []
    for x_e, y_e in envs:
        loss_e = criterion(model(x_e), y_e)
        grads  = torch.autograd.grad(
            loss_e, model.parameters(),
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )
        grad_vec = torch.cat([g.view(-1) for g in grads if g is not None])
        env_grads.append(grad_vec)
    grad_matrix = torch.stack(env_grads, dim=0)   # (n_envs, n_params)
    return grad_matrix.var(dim=0).mean()


# ── Training — Algorithm 1 (Mboya, 2026) ─────────────────────────────────────
def train_caustab(envs, input_dim=N_DIMS):
    model     = CausTabEncoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    for t in range(1, T_TOTAL + 1):
        model.train()
        optimizer.zero_grad()

        erm_loss = torch.stack([
            criterion(model(x_e), y_e) for x_e, y_e in envs
        ]).mean()

        if t <= T_ANNEAL:
            lambda_t = 0.0
        else:
            lambda_t = LAMBDA * min(1.0, (t - T_ANNEAL) / T_WARMUP)

        if lambda_t > 0:
            penalty = caustab_penalty(model, envs, criterion)
            loss    = erm_loss + lambda_t * penalty
        else:
            loss    = erm_loss

        loss.backward()
        optimizer.step()

    return model
def train_erm_only(envs, input_dim=N_DIMS):
    """ERM-only training — captures natural gradient variance before
    invariance pressure is applied. Used for dimension importance scoring."""
    model     = CausTabEncoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    for t in range(T_ANNEAL):
        model.train()
        optimizer.zero_grad()
        loss = torch.stack([
            criterion(model(x_e), y_e) for x_e, y_e in envs
        ]).mean()
        loss.backward()
        optimizer.step()
    return model

# ── Input Gradient Variance per Dimension ────────────────────────────────────
def compute_input_dim_importance(model, envs):
    """
    Gradient of loss w.r.t. input per dimension.
    Variance across environments = spuriousness score.
    model.train() ensures LayerNorm runs in training mode for
    consistent gradient flow.
    """
    criterion       = nn.BCELoss()
    env_input_grads = []
    model.train()

    for x_e, y_e in envs:
        x_req = x_e.clone().requires_grad_(True)
        loss  = criterion(model(x_req), y_e)
        grad  = torch.autograd.grad(loss, x_req)[0]
        env_input_grads.append(
            grad.abs().mean(dim=0).detach().numpy()
        )

    # Cast to float64 before variance computation — input gradients are
    # very small (~1e-4); float32 variance flushes to zero. float64 preserves it.
    grads_matrix = np.stack(env_input_grads, axis=0).astype(np.float64)
    return grads_matrix.var(axis=0)


# ── SDI — Definition 2 (Mboya, 2026) ─────────────────────────────────────────
def compute_sdi(in_act, ood_act):
    eps         = 1e-8
    corr_ranges = []
    corr_means  = []

    for d in range(N_DIMS):
        env_corrs = []
        for i in range(len(FAMILIES)):
            s    = i * N_PER_FAM
            e    = s + N_PER_FAM
            x_d  = np.concatenate([in_act[s:e, d], ood_act[s:e, d]])
            y    = np.array([1.0] * N_PER_FAM + [0.0] * N_PER_FAM)
            corr = np.corrcoef(x_d, y)[0, 1]
            if not np.isnan(corr):
                env_corrs.append(abs(corr))
        if env_corrs:
            corr_ranges.append(max(env_corrs) - min(env_corrs))
            corr_means.append(np.mean(env_corrs))

    corr_ranges = np.array(corr_ranges)
    corr_means  = np.array(corr_means)
    med         = np.median(corr_ranges)
    spur        = corr_ranges > med
    caus        = ~spur

    rho_s   = corr_means[spur].mean()  if spur.sum() > 0 else 0.0
    rho_c   = corr_means[caus].mean()  if caus.sum() > 0 else eps
    delta_s = corr_ranges[spur].mean() if spur.sum() > 0 else 0.0
    delta_c = corr_ranges[caus].mean() if caus.sum() > 0 else 0.0

    return float((rho_s * delta_s) / (rho_c * (1.0 - delta_c) + eps))


# ── Subspace Splitting ────────────────────────────────────────────────────────
def split_subspaces(in_act, ood_act, importance):
    n_sel    = max(1, int(N_DIMS * SUBSPACE_Q))
    sorted_i = np.argsort(importance)
    inv_idx  = sorted_i[:n_sel]
    spur_idx = sorted_i[-n_sel:]
    return {
        "full":      (in_act,               ood_act),
        "invariant": (in_act[:, inv_idx],   ood_act[:, inv_idx]),
        "spurious":  (in_act[:, spur_idx],  ood_act[:, spur_idx]),
    }, inv_idx, spur_idx


# ── PCA + TDA ─────────────────────────────────────────────────────────────────
def reduce(in_sub, ood_sub):
    n_comp   = min(PCA_DIMS, in_sub.shape[1])
    combined = StandardScaler().fit_transform(
                   np.vstack([in_sub, ood_sub]))
    reduced  = PCA(n_components=n_comp,
                   random_state=RANDOM_SEED).fit_transform(combined)
    n = len(in_sub)
    return reduced[:n], reduced[n:]


def tda_distances(in_red, ood_red):
    dgm_in  = ripser(in_red,  maxdim=0)['dgms'][0]
    dgm_ood = ripser(ood_red, maxdim=0)['dgms'][0]
    return {
        "wasserstein": float(wasserstein(dgm_in, dgm_ood)),
        "bottleneck":  float(bottleneck(dgm_in,  dgm_ood)),
        "dgm_in":      dgm_in,
        "dgm_ood":     dgm_ood,
    }


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_dim_importance(all_importance):
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    for ax, layer in zip(axes, LAYERS):
        imp      = all_importance[layer]
        n_sel    = int(N_DIMS * SUBSPACE_Q)
        s        = np.argsort(imp)
        inv_thr  = imp[s[n_sel]]
        spur_thr = imp[s[-n_sel]]
        ax.hist(imp, bins=40, color=C_FULL, alpha=0.60,
                edgecolor="white", linewidth=0.2)
        ax.axvline(inv_thr,  color=C_INV,  lw=0.9, ls="--",
                   label="invariant threshold")
        ax.axvline(spur_thr, color=C_SPUR, lw=0.9, ls="--",
                   label="spurious threshold")
        ax.axvspan(0,              inv_thr,        alpha=0.10, color=C_INV)
        ax.axvspan(spur_thr, imp.max() * 1.05,     alpha=0.10, color=C_SPUR)
        ax.set_title(f"layer {layer}", fontsize=7.5, pad=3)
        ax.set_xlabel("input gradient variance across families",
                      fontsize=6.5)
        ax.set_ylabel("no. of dimensions", fontsize=6.5)
        ax.legend(fontsize=5.5)
    fig.suptitle(
        "CausTab input gradient variance per activation dimension\n"
        "blue = invariant  ·  red = spurious  (Mboya, 2026)",
        fontsize=7.5, y=1.02,
    )
    plt.tight_layout(w_pad=1.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "irm_penalty_distribution.png"))
    plt.close()
    print("  Saved → outputs/irm_connection/irm_penalty_distribution.png")


def plot_subspace_comparison(all_metrics):
    names  = ["full", "invariant", "spurious"]
    colors = [C_FULL, C_INV, C_SPUR]
    x      = np.arange(len(LAYERS))
    w      = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, metric in zip(axes, ["wasserstein", "bottleneck"]):
        for i, (name, color) in enumerate(zip(names, colors)):
            vals = [all_metrics[l][name][metric] for l in LAYERS]
            bars = ax.bar(x + (i - 1) * w, vals, w,
                          label=name, color=color,
                          alpha=0.82, linewidth=0)
            for bar in bars:
                h = bar.get_height()
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + 0.5, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {l}" for l in LAYERS])
        ax.set_xlabel("transformer layer", fontsize=7)
        lbl = r"$W_1(H_0)$" if metric == "wasserstein" else r"$d_B(H_0)$"
        ax.set_ylabel(lbl, fontsize=7)
        t = "Wasserstein" if metric == "wasserstein" else "Bottleneck"
        ax.set_title(f"{t} distance\nfull vs invariant vs spurious",
                     fontsize=7.5, pad=4)
        ax.legend(fontsize=6)
    fig.suptitle(
        "Topological separation across CausTab-defined subspaces\n"
        "invariant = low gradient variance  ·  "
        "spurious = high gradient variance",
        fontsize=7.5, y=1.02,
    )
    plt.tight_layout(w_pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "subspace_comparison.png"))
    plt.close()
    print("  Saved → outputs/irm_connection/subspace_comparison.png")


def plot_persistence_subspaces(all_metrics):
    subspaces = ["full", "invariant", "spurious"]
    titles    = ["full space", "invariant subspace", "spurious subspace"]
    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.0))
    for row, (name, title) in enumerate(zip(subspaces, titles)):
        for col, layer in enumerate(LAYERS):
            ax     = axes[row, col]
            m      = all_metrics[layer][name]
            h0_in  = m["dgm_in"][m["dgm_in"][:, 1]   != np.inf]
            h0_ood = m["dgm_ood"][m["dgm_ood"][:, 1] != np.inf]
            pts    = [p for p in [h0_in, h0_ood] if len(p)]
            if pts:
                v  = np.vstack(pts)
                lo = v.min() * 0.90
                hi = v.max() * 1.10
            else:
                lo, hi = 0, 1
            ax.plot([lo, hi], [lo, hi], color="gray",
                    lw=0.4, ls="--")
            if len(h0_in):
                ax.scatter(h0_in[:, 0],  h0_in[:, 1],
                           s=10, c=C_INV,  alpha=0.85,
                           linewidths=0, label="in-dist")
            if len(h0_ood):
                ax.scatter(h0_ood[:, 0], h0_ood[:, 1],
                           s=10, c=C_SPUR, alpha=0.85,
                           linewidths=0, label="OOD", marker="s")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(
                f"{title} — layer {layer}\n"
                r"$W_1$ = " + f"{m['wasserstein']:.1f}",
                fontsize=7, pad=3,
            )
            ax.set_xlabel("birth", fontsize=6.5)
            ax.set_ylabel("death", fontsize=6.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=5.5, markerscale=1.2)
    fig.suptitle(
        r"$H_0$ persistence diagrams: in-distribution vs OOD" + "\n"
        "rows: full · invariant · spurious  ·  "
        "columns: layers 2, 4, 6",
        fontsize=7.5, y=1.01,
    )
    plt.tight_layout(h_pad=2.2, w_pad=1.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "persistence_subspaces.png"))
    plt.close()
    print("  Saved → outputs/irm_connection/persistence_subspaces.png")


def plot_sdi(sdi_values):
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    layers  = list(sdi_values.keys())
    vals    = list(sdi_values.values())
    bars    = ax.bar(range(len(layers)), vals, color=C_INV,
                     alpha=0.80, linewidth=0, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6)
    ax.axhline(2.0, color="gray",  lw=0.6, ls="--",
               label="SDI = 2 (causal-dominant)")
    ax.axhline(5.0, color=C_SPUR, lw=0.6, ls="--",
               label="SDI = 5 (spurious-dominant)")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("Spurious Dominance Index", fontsize=7)
    ax.set_title(
        "SDI per ESM-2 layer\n(Mboya, 2026 — Definition 2)",
        fontsize=7.5, pad=4,
    )
    ax.legend(fontsize=5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sdi_per_layer.png"))
    plt.close()
    print("  Saved → outputs/irm_connection/sdi_per_layer.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CausTab IRM Connection — Mboya (2026)")
    print("Invariant Features and Topological Persistence in ESM-2")
    print("=" * 60)

    all_importance = {}
    all_metrics    = {}
    all_sdi        = {}
    summary_rows   = []

    for layer in LAYERS:
        print(f"\n── Layer {layer} ──────────────────────────────────────")
        in_act, ood_act = load_layer(layer)
        envs            = get_environments(in_act, ood_act)
        print(f"  Built {len(envs)} environments "
              f"({N_PER_FAM * 2} samples each)")

        print("  Training ERM warmup model for dimension ranking...")
        model_erm = train_erm_only(envs, input_dim=N_DIMS)
        print("  ERM training complete")

        print("  Training full CausTab model...")
        model_caustab = train_caustab(envs, input_dim=N_DIMS)
        print("  CausTab training complete")

        print("  Computing input gradient variance per dimension...")
        dim_imp = compute_input_dim_importance(model_erm, envs)
        all_importance[layer] = dim_imp
        print(f"  Variance range: {dim_imp.min():.2e} – "
              f"{dim_imp.max():.2e}  mean: {dim_imp.mean():.2e}")

        sdi            = compute_sdi(in_act, ood_act)
        all_sdi[layer] = sdi
        print(f"  SDI: {sdi:.4f}")

        subspaces, inv_idx, spur_idx = split_subspaces(
            in_act, ood_act, dim_imp)
        print(f"  Invariant dims: {len(inv_idx)}  "
              f"|  Spurious dims: {len(spur_idx)}")

        all_metrics[layer] = {}
        for name, (in_sub, ood_sub) in subspaces.items():
            in_red, ood_red          = reduce(in_sub, ood_sub)
            m                        = tda_distances(in_red, ood_red)
            all_metrics[layer][name] = m
            print(f"  [{name:<10}]  "
                  f"W1 = {m['wasserstein']:>8.3f}  "
                  f"BN = {m['bottleneck']:>8.3f}")

        inv_gt_spur = bool(
            all_metrics[layer]["invariant"]["wasserstein"] >
            all_metrics[layer]["spurious"]["wasserstein"]
        )

        summary_rows.append({
            "layer":         layer,
            "sdi":           sdi,
            "full_w":        all_metrics[layer]["full"]["wasserstein"],
            "inv_w":         all_metrics[layer]["invariant"]["wasserstein"],
            "spur_w":        all_metrics[layer]["spurious"]["wasserstein"],
            "full_b":        all_metrics[layer]["full"]["bottleneck"],
            "inv_b":         all_metrics[layer]["invariant"]["bottleneck"],
            "spur_b":        all_metrics[layer]["spurious"]["bottleneck"],
            "inv_gt_spur_w": inv_gt_spur,
        })

    # Figures
    print("\nGenerating figures...")
    plot_dim_importance(all_importance)
    plot_subspace_comparison(all_metrics)
    plot_persistence_subspaces(all_metrics)
    plot_sdi(all_sdi)

    # Save results — strip numpy arrays before serialising
    json_rows = []
    for r in summary_rows:
        json_rows.append({
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in r.items()
            if k != "dgm_in" and k != "dgm_ood"
        })
    with open(os.path.join(OUTPUT_DIR,
                           "irm_connection_results.json"), "w") as f:
        json.dump(json_rows, f, indent=2)

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Layer':<6} {'SDI':>7} {'Full W1':>10} "
          f"{'Inv W1':>10} {'Spur W1':>10}  {'Inv>Spur':>10}")
    print("-" * 60)
    for r in summary_rows:
        verdict = "YES ✓" if r["inv_gt_spur_w"] else "no"
        print(f"  {r['layer']:<4} {r['sdi']:>7.3f} "
              f"{r['full_w']:>10.3f} {r['inv_w']:>10.3f} "
              f"{r['spur_w']:>10.3f}  {verdict:>10}")

    print(f"\n✓ All outputs saved to outputs/irm_connection/")
    print("  irm_penalty_distribution.png")
    print("  subspace_comparison.png")
    print("  persistence_subspaces.png")
    print("  sdi_per_layer.png")
    print("  irm_connection_results.json")
