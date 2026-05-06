 # Topological Analysis of Latent Space Geometry in Protein Foundation Models

**Grold Otieno Mboya** · [ORCID](https://orcid.org/0009-0005-9102-4028) · [nixque.com](https://nixque.com)

---

## Overview

This project applies **Topological Data Analysis (TDA)** — specifically persistent homology — to characterize the internal geometry of protein foundation models under distribution shift. Using ESM-2 as a model system, we extract hidden state activations at multiple transformer layers and compare their topological structure between biologically valid (in-distribution) and scrambled (out-of-distribution) protein sequences.

The central question is:

> *When a foundation model encounters inputs outside its training distribution, does the geometry of its latent representations change in a topologically measurable way — and at which layer does this signal emerge most strongly?*

This work sits at the intersection of **foundation model evaluation**, **mechanistic interpretability**, and **topological data analysis**, with direct implications for building robust, auditable AI systems for scientific applications.

---

## Key Findings

Persistent homology applied to ESM-2 (8M) hidden states reveals that distribution shift produces a measurable and layer-dependent topological signal:

| Layer | In-dist H₀ persistence | OOD H₀ persistence | Bottleneck distance (H₀) |
|-------|------------------------|---------------------|--------------------------|
| 2     | 15.61                  | 13.67               | 6.19                     |
| 4     | 16.61                  | 9.38                | **13.91**                |
| 6     | 15.15                  | 12.01               | 8.22                     |

- **Layer 4 is the critical detection layer** — topological divergence between distributions peaks at the middle transformer layers, suggesting that intermediate representations are most sensitive to sequence validity
- **OOD sequences exhibit emergent H₁ features** (loops/cycles) at layers 4 and 6 that are absent in in-distribution sequences, indicating structurally irregular latent geometry for biologically meaningless inputs
- **In-distribution sequences maintain consistently higher persistence** across all layers, reflecting stable, well-separated cluster structure in latent space

---

## Methodology

```
Protein sequences (in-dist + OOD)
        │
        ▼
ESM-2 (facebook/esm2_t6_8M_UR50D)
  Hidden states at layers 2, 4, 6
        │
        ▼
Mean pooling → per-sequence vectors (320-dim)
        │
        ▼
PCA reduction (320 → 8 dims)
        │
        ▼
Vietoris-Rips persistent homology (ripser)
  H₀: connected components
  H₁: loops / cycles
        │
        ▼
Bottleneck distance (in-dist vs OOD per layer)
+ UMAP visualization
```

**In-distribution sequences:** Biologically valid protein sequences drawn from diverse functional families.

**OOD sequences:** Amino acid compositions preserved but order randomly shuffled — destroying biological structure while maintaining residue statistics.

---

## Repository Structure

```
tda-foundation-models/
├── src/
│   ├── extract_activations.py   # ESM-2 inference + hidden state extraction
│   └── tda_analysis.py          # Persistent homology + visualization
├── notebooks/                   # Interactive exploration (coming)
├── data/                        # Activation .npy files (git-ignored)
├── outputs/                     # Figures and results (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/Grolds-Code/tda-foundation-models.git
cd tda-foundation-models
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt

# 2. Extract ESM-2 activations
python src/extract_activations.py

# 3. Run TDA analysis and generate figures
python src/tda_analysis.py
```

On first run, `extract_activations.py` will download the ESM-2 8M model weights (~31 MB) from HuggingFace. All subsequent runs use the local cache.

**Hardware:** Runs fully on CPU. Tested on Intel Core i7 vPro with 32 GB RAM. No GPU required.

---

## Dependencies

Core dependencies:

```
torch          — ESM-2 inference (CPU build)
transformers   — HuggingFace model loading
ripser         — Vietoris-Rips persistent homology
persim         — Persistence diagram distances (bottleneck)
umap-learn     — Dimensionality reduction for visualization
scikit-learn   — PCA, preprocessing
matplotlib     — Publication-quality figures
numpy / scipy / pandas
```

Full pinned dependencies: see `requirements.txt`.

---

## Connection to AI Safety and Foundation Model Evaluation

A core challenge in deploying foundation models for scientific applications is knowing *when* a model is operating outside the regime it was trained on. Standard approaches rely on output-level uncertainty estimates, which can be unreliable when a model confidently produces plausible-looking but incorrect outputs.

This work proposes **topological probing of intermediate representations** as a complementary evaluation strategy — one that operates at the level of latent geometry rather than output distributions. The bottleneck distance between persistence diagrams provides a mathematically grounded, model-agnostic metric for detecting representational anomalies layer by layer.

This connects directly to the author's prior work on Invariant Risk Minimization ([medRxiv, 2026](https://doi.org/10.64898/2026.04.09.26350513)) and Amortized Variational Inference for spatial transcriptomics ([Zenodo, 2026](https://doi.org/10.5281/zenodo.19852038)).

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{mboya2026tda,
  author    = {Mboya, Grold Otieno},
  title     = {Topological Analysis of Latent Space Geometry in Protein Foundation Models},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Grolds-Code/tda-foundation-models}
}
```

---

## Author

**Grold Otieno Mboya**  
Research Fellow-Fatima Institute  
Technical AI Safety Researcher | Co-Founder, NIXQUE LTD  
[groldotieno97@gmail.com](mailto:groldotieno97@gmail.com) · [nixque.com](https://nixque.com) · [GitHub](https://github.com/Grolds-Code)

---

*Last updated: May 2026*
