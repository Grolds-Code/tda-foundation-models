# Topological Probing of Latent Space Geometry in Protein Foundation Models

**Grold Otieno Mboya**  

[ORCID: 0009-0005-9102-4028](https://orcid.org/0009-0005-9102-4028) · [nixque.com](https://nixque.com) · [github.com/Grolds-Code](https://github.com/Grolds-Code)

---

## Overview

This project develops a **topological framework for evaluating the internal geometry of protein foundation models under distribution shift**. Using ESM-2 as a model system, we extract hidden state activations at multiple transformer layers, apply persistent homology to characterize their geometric structure, and test whether activation dimensions identified as invariant by the CausTab gradient variance penalty (Mboya, 2026) produce more topologically persistent representations than spurious dimensions.

The work sits at the intersection of three research areas:

- **Foundation model evaluation** — developing principled diagnostics for when a model is operating outside its training distribution, without relying on output-level confidence estimates
- **Mechanistic interpretability** — probing the internal geometry of transformer representations layer by layer to understand where and how distributional structure is encoded
- **Causal invariant learning** — extending the CausTab gradient variance framework (Mboya, 2026) from tabular classifiers to the latent spaces of biological foundation models

The central scientific question is:

> *When a protein language model encounters biologically invalid sequences, does the geometry of its internal representations change in a topologically measurable, statistically significant, and layer-dependent way — and do activation dimensions identified as causally invariant produce more stable topological structure than spurious dimensions?*

---

## Scientific Background

### Why Topological Data Analysis?

Standard approaches to distribution shift detection operate at the output level — monitoring prediction confidence, output entropy, or softmax scores. These measures are unreliable when a model confidently produces plausible-looking but incorrect outputs, a failure mode that is particularly dangerous in scientific applications.

Topological Data Analysis (TDA) operates at the level of **latent geometry** rather than output distributions. Persistent homology characterizes the shape of a point cloud — its connected components (H₀) and loop structures (H₁) — across all spatial scales simultaneously, producing a persistence diagram that encodes the full multi-scale structure of the data.

Applied to neural network activations, TDA can detect when the geometric structure of a model's internal representations has changed, independently of what the model outputs. This makes it a complementary and architecturally agnostic evaluation strategy.

### Why ESM-2?

ESM-2 (Lin et al., 2023) is a family of protein language models trained on the UniRef database using masked language modelling. Its hidden representations encode biologically meaningful information about protein structure, function, and evolutionary relationships, making it one of the most widely used foundation models in computational biology. It is an ideal model system for this study because:

1. The distinction between biologically valid and invalid sequences is sharp and principled
2. Its architecture (standard transformer encoder) is representative of the broader class of sequence foundation models
3. Its representations have known interpretable structure, enabling validation of our findings against prior mechanistic work

### Why CausTab?

The CausTab gradient variance penalty (Mboya, 2026) identifies **invariant activation dimensions** — those whose gradient signal is consistent across biological family environments — by penalising cross-environment gradient variance. Unlike the scalar IRM penalty (Arjovsky et al., 2019), CausTab operates on the full gradient vector across all model parameters, and Theorem 1 in Mboya (2026) proves that the penalty is zero at the causally invariant solution and strictly positive at any solution relying on spurious features.

Applying CausTab to foundation model activations extends this framework from tabular classifiers to latent representation analysis, creating a geometry-level diagnostic for invariant structure that operates without test-time labels.

---

## Methodology

### Full Pipeline

```
UniProt Swiss-Prot API
  ↓
build_dataset.py
  60 protein sequences × 5 biological families
  (kinase, oxidoreductase, transcription factor,
   chaperone, transporter)
  + 60 matched OOD sequences (residue-preserving shuffle)
  ↓
extract_activations.py
  ESM-2 (facebook/esm2_t6_8M_UR50D, 7.5M parameters)
  Hidden states extracted at layers 2, 4, 6
  Mean pooling over sequence length → (60, 320) per layer
  ↓
tda_analysis.py
  StandardScaler + PCA (320 → 8 dims)
  Vietoris-Rips persistent homology (ripser)
  H₀ (connected components) + H₁ (loops/cycles)
  Bottleneck distance + Wasserstein distance
  UMAP visualisation
  ↓
permutation_test.py
  Family-stratified permutation test (n=1000)
  Wasserstein (primary) + Bottleneck (robustness check)
  Controls for biological family as confounder
  ↓
irm_connection.py
  CausTab encoder training — Algorithm 1 (Mboya, 2026)
  ERM warmup (Ta=50) + gradient variance penalty (λ=100)
  Input gradient variance per dimension across environments
  Subspace splitting: invariant (bottom 25%) vs spurious (top 25%)
  TDA comparison across full / invariant / spurious subspaces
  Spurious Dominance Index (SDI) per layer
```

### In-Distribution vs Out-of-Distribution Sequences

**In-distribution sequences** are biologically valid proteins drawn from five functionally distinct families in UniProt Swiss-Prot (reviewed entries), covering diverse evolutionary origins, lengths (56–395 residues), and biochemical functions.

**OOD sequences** are generated by randomly shuffling the amino acid order of each in-distribution sequence, preserving residue composition while destroying all positional and structural information. This is the standard negative control for protein language models: the OOD sequences are guaranteed to be biologically meaningless while being statistically matched to their in-distribution counterparts in amino acid composition.

### Statistical Testing

The permutation test controls for a critical confounder: **biological family diversity**. A naive pooled permutation test would conflate two sources of variation — the in-dist vs OOD signal (our target) and the between-family biological diversity (a confounder). By permuting labels within each family rather than across the entire pool, the test isolates exactly the signal of interest: topological difference between real and shuffled sequences of the same biological type.

---

## Results

### 1. Topological Distribution Shift is Highly Significant

Wasserstein distance between in-distribution and OOD persistence diagrams is statistically significant at p < 0.001 at all three layers under the stratified permutation test:

| Layer | Observed W₁ | Null mean ± std | p-value | Significance |
|-------|-------------|-----------------|---------|--------------|
| 2     | 106.53      | 46.34 ± 10.52   | < 0.001 | ***          |
| 4     | 239.61      | 76.92 ± 24.26   | < 0.001 | ***          |
| 6     | 136.15      | 40.56 ± 12.17   | < 0.001 | ***          |

The bottleneck distance confirms significance at layer 2 (p = 0.039) but not layers 4 and 6, consistent with the known limitation of bottleneck distance as a worst-case rather than aggregate metric. The Wasserstein result is primary.

### 2. Layer 4 is the Critical Detection Layer

The topological divergence between distributions peaks at the middle transformer layer:

| Layer | In-dist H₀ persistence | OOD H₀ persistence | Bottleneck H₀ |
|-------|------------------------|---------------------|---------------|
| 2     | 8.18                   | 6.31                | 11.11         |
| **4** | **7.05**               | **2.46**            | **25.14**     |
| 6     | 8.29                   | 5.98                | 2.99          |

OOD H₀ persistence collapses by 61% between layer 2 and layer 4 (6.31 → 2.46), while in-distribution persistence remains stable (8.18 → 7.05). This asymmetric collapse indicates that ESM-2's middle layers are most sensitive to biological validity — the representations of shuffled sequences lose their cluster structure precisely where the model's learned biological knowledge is most concentrated.

### 3. H₁ Features Emerge in OOD Representations at Deeper Layers

Higher-order topological features (loops/cycles, H₁) are absent in in-distribution representations but present in OOD representations at layers 4 and 6. This suggests that biologically invalid sequences produce irregular, non-manifold geometric structure in deeper layers — a topological signature of inputs that the model's learned representations cannot organise coherently.

### 4. CausTab IRM Connection

Applying the CausTab gradient variance penalty (Mboya, 2026) to rank activation dimensions:

| Layer | SDI    | Full W₁ | Invariant W₁ | Spurious W₁ | Inv > Spur |
|-------|--------|---------|--------------|-------------|------------|
| 2     | 0.789  | 106.53  | 54.37        | 53.50       | YES ✓      |
| 4     | 1.003  | 239.61  | 130.02       | 135.42      | no         |
| 6     | 0.942  | 136.15  | 91.85        | 80.83       | YES ✓      |

All three SDI values fall below 2.0, placing this dataset in the **causal-dominant regime** by Definition 2 of Mboya (2026). This is scientifically coherent: the in-dist vs OOD distinction is a strong causal property of the sequences (biological validity), not a spurious family-level correlation. In the causal-dominant regime, CausTab theory predicts that the distinction between invariant and spurious subspaces will be subtle — which is exactly what we observe, with layers 2 and 6 confirming the directional hypothesis and layer 4 showing a small reversal within the expected noise range for SDI ≈ 1.

The gradient variance ranges from 2.30e-12 to 7.22e-10 across layers — small but real, reflecting the compressed scale of mean-pooled transformer activations.

---

## Connection to Prior Work

This project directly extends two prior publications by the same author:

**Mboya, G.O. (2026). CausTab: Gradient Variance Regularization for Causal Invariant Representation Learning on Tabular Data.** *medRxiv.* DOI: [10.64898/2026.04.09.26350513](https://doi.org/10.64898/2026.04.09.26350513)

The CausTab framework is applied here to foundation model activations rather than tabular classifiers. The theoretical guarantees (Theorem 1 and Proposition 1) carry over directly: the gradient variance penalty is zero at the causally invariant solution regardless of the data modality. The SDI diagnostic is computed using the same Definition 2, enabling direct comparison of the current protein language model setting with the NHANES and synthetic settings reported in Mboya (2026).

**Mboya, G.O. (2026). Amortized Variational Inference for Scalable Bayesian Tensor Factorization in Spatial Transcriptomics.** *Zenodo.* DOI: [10.5281/zenodo.19852038](https://doi.org/10.5281/zenodo.19852038)

The amortized inference framework developed for spatial transcriptomics provides methodological precedent for scalable probabilistic analysis of high-dimensional biological data. The present work applies similar principles — dimensionality reduction followed by structured geometric analysis — to foundation model representations.

---

## Implications for Foundation Model Evaluation

A core challenge in deploying foundation models for scientific applications is knowing when a model is operating outside its training regime. Standard output-level diagnostics (confidence scores, entropy) are unreliable when a model confidently produces structurally plausible but scientifically invalid outputs.

This work proposes **topological probing of intermediate representations** as a complementary evaluation strategy with three advantages over output-level approaches:

1. **Layer-specificity** — the analysis reveals which transformer layers encode distributional sensitivity, providing mechanistic insight rather than a scalar alarm
2. **Model-agnosticism** — the TDA framework operates on any set of activation vectors and requires no architectural assumptions or access to training data
3. **Theoretical grounding** — the bottleneck and Wasserstein distances between persistence diagrams are mathematically well-defined metrics with stability guarantees, unlike heuristic uncertainty scores

The finding that layer 4 is the critical detection layer in ESM-2 is directly actionable: it suggests that monitoring the topological structure of middle-layer representations may provide earlier and more reliable distributional alarms than monitoring output distributions.

---

## Repository Structure

```
tda-foundation-models/
├── src/
│   ├── build_dataset.py          # UniProt API fetching, OOD generation
│   ├── extract_activations.py    # ESM-2 inference, hidden state extraction
│   ├── tda_analysis.py           # Persistent homology, UMAP, bottleneck/Wasserstein
│   ├── permutation_test.py       # Stratified permutation test (W₁ + dB)
│   └── irm_connection.py         # CausTab penalty, SDI, subspace TDA
├── data/                         # Activation .npy files + sequences.json (git-ignored)
├── outputs/
│   ├── persistence_diagrams/     # H₀/H₁ persistence diagrams per layer (.png)
│   ├── umap/                     # UMAP projections per layer (.png)
│   ├── summary/                  # Bottleneck summary, permutation test (.png + .json)
│   └── irm_connection/           # CausTab subspace analysis (.png + .json)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and create environment
git clone https://github.com/Grolds-Code/tda-foundation-models.git
cd tda-foundation-models
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt

# 2. Build dataset (fetches from UniProt — requires internet)
python src/build_dataset.py

# 3. Extract ESM-2 activations (~31MB model download on first run)
python src/extract_activations.py

# 4. TDA analysis and visualisation
python src/tda_analysis.py

# 5. Statistical significance testing (~25 min on CPU)
python src/permutation_test.py

# 6. CausTab IRM connection and subspace analysis
python src/irm_connection.py
```

**Hardware requirements:** Runs fully on CPU. Tested on Intel Core i7 vPro, 32GB RAM, Windows 11. No GPU required. Total runtime approximately 45–60 minutes end to end.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` (CPU build) | ESM-2 inference, CausTab encoder training |
| `transformers` | HuggingFace model loading |
| `ripser` | Vietoris-Rips persistent homology |
| `persim` | Persistence diagram distances (Wasserstein, bottleneck) |
| `umap-learn` | Dimensionality reduction for visualisation |
| `scikit-learn` | PCA, StandardScaler |
| `matplotlib` | Publication-quality figures |
| `numpy`, `scipy`, `pandas` | Numerical computing |
| `requests` | UniProt REST API |

Full pinned dependencies: `requirements.txt`

---

## Figures

All figures are saved at 300 DPI in `outputs/`. Key outputs:

- **`outputs/persistence_diagrams/`** — Side-by-side H₀/H₁ persistence diagrams for in-distribution vs OOD at each layer. Points far from the diagonal represent topologically persistent features.
- **`outputs/umap/`** — UMAP projections of ESM-2 activations coloured by distribution. Visual separation increases from layer 2 to layer 4.
- **`outputs/summary/bottleneck_summary.png`** — Bottleneck distances across layers showing the layer 4 peak.
- **`outputs/summary/permutation_test.png`** — Null distributions (Wasserstein top row, bottleneck bottom row) with observed statistics. All Wasserstein panels show observed >> null.
- **`outputs/irm_connection/subspace_comparison.png`** — Wasserstein and bottleneck distances for full, invariant, and spurious subspaces across layers.
- **`outputs/irm_connection/sdi_per_layer.png`** — Spurious Dominance Index per layer with causal-dominant and spurious-dominant thresholds from Mboya (2026).

---

## Citation

If you use this code, methodology, or findings, please cite:

```bibtex
@misc{mboya2026tda,
  author    = {Mboya, Grold Otieno},
  title     = {Topological Probing of Latent Space Geometry in Protein
               Foundation Models under Distribution Shift},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Grolds-Code/tda-foundation-models}
}

@article{mboya2026caustab,
  author  = {Mboya, Grold Otieno},
  title   = {{CausTab}: Gradient Variance Regularization for Causal
             Invariant Representation Learning on Tabular Data},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.04.09.26350513}
}
```

---

## Author

**Grold Otieno Mboya**  
Department of Epidemiology and Biostatistics  
Jaramogi Oginga Odinga University of Science and Technology, Kenya  
[groldotieno97@gmail.com](mailto:groldotieno97@gmail.com) · [nixque.com](https://nixque.com)

*Research interests: mechanistic interpretability, topological data analysis, causal invariant learning, foundation model evaluation, AI for science.*

---

## References

Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. (2019). Invariant risk minimization. *arXiv:1907.02893.*

Lin, Z., Akin, H., Rao, R., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.

Mboya, G.O. (2026). CausTab: Gradient variance regularization for causal invariant representation learning on tabular data. *medRxiv.* DOI: 10.64898/2026.04.09.26350513

Mboya, G.O. (2026). Amortized variational inference for scalable Bayesian tensor factorization in spatial transcriptomics. *Zenodo.* DOI: 10.5281/zenodo.19852038

Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., and Harrington, H.A. (2017). A roadmap for the computation of persistent homology. *EPJ Data Science*, 6(1), 17.

Peters, J., Bühlmann, P., and Meinshausen, N. (2016). Causal inference by using invariant prediction. *Journal of the Royal Statistical Society: Series B*, 78(5), 947–1012.

Edelsbrunner, H. and Harer, J. (2010). *Computational Topology: An Introduction.* American Mathematical Society.

---

*Last updated: May 2026*