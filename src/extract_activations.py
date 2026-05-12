"""
extract_activations.py
----------------------
Loads ESM-2 (8M parameter variant) and extracts hidden state activations
from in-distribution and out-of-distribution protein sequences.

Sequences are loaded from data/sequences.json (built by build_dataset.py).

Author: Grold Otieno Mboya
Project: TDA Analysis of Latent Space Geometry in Foundation Models
"""

import torch
import numpy as np
import json
import os
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME        = "facebook/esm2_t6_8M_UR50D"
DATA_DIR          = "data"
SEQUENCES_FILE    = os.path.join(DATA_DIR, "sequences.json")
LAYERS_TO_EXTRACT = [2, 4, 6]

# ── Load Sequences ────────────────────────────────────────────────────────────
def load_sequences() -> tuple:
    with open(SEQUENCES_FILE, "r") as f:
        dataset = json.load(f)

    in_seqs  = [entry["sequence"] for entry in dataset["in_dist"]]
    ood_seqs = [entry["sequence"] for entry in dataset["ood"]]
    families = [entry["family"]   for entry in dataset["in_dist"]]
    meta     = dataset["metadata"]

    print(f"Loaded {len(in_seqs)} in-distribution sequences")
    print(f"Loaded {len(ood_seqs)} OOD sequences")
    print(f"Families: {meta['families']}")
    print(f"Source: {meta['source']}\n")

    return in_seqs, ood_seqs, families

# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model(model_name: str):
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {n_params:,} parameters\n")
    return tokenizer, model

# ── Activation Extraction ─────────────────────────────────────────────────────
def extract_hidden_states(sequences: list, tokenizer, model, label: str) -> dict:
    layer_activations = {layer: [] for layer in LAYERS_TO_EXTRACT}

    print(f"Extracting activations [{label}] — {len(sequences)} sequences...")
    for seq in tqdm(sequences, desc=label):
        inputs = tokenizer(seq, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        for layer_idx in LAYERS_TO_EXTRACT:
            rep = hidden_states[layer_idx][0].mean(dim=0).numpy()
            layer_activations[layer_idx].append(rep)

    for layer_idx in LAYERS_TO_EXTRACT:
        layer_activations[layer_idx] = np.array(layer_activations[layer_idx])

    return layer_activations

# ── Save Activations ──────────────────────────────────────────────────────────
def save_activations(activations: dict, label: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    for layer_idx, arr in activations.items():
        path = os.path.join(DATA_DIR, f"{label}_layer{layer_idx}.npy")
        np.save(path, arr)
        print(f"  Saved {label} layer {layer_idx}: shape {arr.shape} → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ESM-2 Activation Extraction")
    print("=" * 60 + "\n")

    in_seqs, ood_seqs, families = load_sequences()
    tokenizer, model            = load_model(MODEL_NAME)

    in_acts  = extract_hidden_states(in_seqs,  tokenizer, model, label="in_dist")
    ood_acts = extract_hidden_states(ood_seqs, tokenizer, model, label="ood")

    print("\nSaving activations...")
    save_activations(in_acts,  "in_dist")
    save_activations(ood_acts, "ood")

    # Save family labels for stratified analysis
    labels_path = os.path.join(DATA_DIR, "family_labels.json")
    with open(labels_path, "w") as f:
        json.dump(families, f)
    print(f"  Saved family labels → {labels_path}")

    print(f"\n✓ Extraction complete.")
    print(f"  {len(in_seqs)} sequences × {len(LAYERS_TO_EXTRACT)} layers extracted")
    print(f"  Activation shape per layer: {in_acts[LAYERS_TO_EXTRACT[0]].shape}")