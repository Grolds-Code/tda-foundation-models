"""
extract_activations.py
----------------------
Loads ESM-2 (8M parameter variant) and extracts hidden state activations
from in-distribution and out-of-distribution protein sequences.

Author: Grold Otieno Mboya
Project: TDA Analysis of Latent Space Geometry in Foundation Models
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"   # 8M params — runs well on CPU
SAVE_DIR   = "data"
LAYERS_TO_EXTRACT = [2, 4, 6]              # early, middle, final transformer layers

# ── Protein Sequences ─────────────────────────────────────────────────────────
# IN-DISTRIBUTION: real, biologically valid protein sequences
IN_DIST_SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTLDH",
    "ACDEFGHIKLMNPQRSTVWY" * 5,
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPNEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV",
    "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
    "MKLVLSLSLLVLVTIVCLAGSSHHHHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVTTTYTGPGTTTATYTTGPPTTTTTTTSATTTAATTTGGTATTTTTTTTTTT",
    "GSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVTTTYTGPGTTTATYTTGPPTTTTTTTSATTTAATTTGGTATTTTTTTTTTTGTATTTTTTTTTTTTTTATTTTTTTTTTTTTTTTTTT",
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
    "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
]

# OUT-OF-DISTRIBUTION: shuffled/scrambled sequences (biologically invalid)
import random
random.seed(42)

def shuffle_sequence(seq):
    s = list(seq)
    random.shuffle(s)
    return "".join(s)

OOD_SEQUENCES = [shuffle_sequence(seq) for seq in IN_DIST_SEQUENCES]

# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model(model_name: str):
    print(f"Loading tokenizer and model: {model_name}")
    print("(This will download ~31MB on first run, then cache locally)\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()  # inference mode — no gradient tracking needed
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    return tokenizer, model

# ── Activation Extraction ─────────────────────────────────────────────────────
def extract_hidden_states(sequences: list, tokenizer, model, label: str) -> dict:
    """
    Runs each sequence through ESM-2 and collects hidden states
    from the specified layers.

    Returns a dict: {layer_index: numpy array of shape (n_sequences, hidden_dim)}
    """
    layer_activations = {layer: [] for layer in LAYERS_TO_EXTRACT}

    print(f"Extracting activations for {label} sequences...")
    for seq in tqdm(sequences, desc=label):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states is a tuple: one tensor per layer (including embedding layer)
        # each tensor shape: (1, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states

        for layer_idx in LAYERS_TO_EXTRACT:
            # Mean-pool across sequence length → single vector per sequence
            layer_rep = hidden_states[layer_idx][0].mean(dim=0).numpy()
            layer_activations[layer_idx].append(layer_rep)

    # Stack into arrays
    for layer_idx in LAYERS_TO_EXTRACT:
        layer_activations[layer_idx] = np.array(layer_activations[layer_idx])

    return layer_activations

# ── Save Activations ──────────────────────────────────────────────────────────
def save_activations(activations: dict, label: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for layer_idx, arr in activations.items():
        path = os.path.join(save_dir, f"{label}_layer{layer_idx}.npy")
        np.save(path, arr)
        print(f"  Saved {label} layer {layer_idx}: shape {arr.shape} → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer, model = load_model(MODEL_NAME)

    # Extract for both distributions
    in_dist_acts  = extract_hidden_states(IN_DIST_SEQUENCES,  tokenizer, model, label="in_dist")
    ood_acts      = extract_hidden_states(OOD_SEQUENCES,      tokenizer, model, label="ood")

    # Save
    print("\nSaving activations to disk...")
    save_activations(in_dist_acts, "in_dist", SAVE_DIR)
    save_activations(ood_acts,     "ood",     SAVE_DIR)

    print("\n✓ Activation extraction complete.")
    print(f"  Layers extracted: {LAYERS_TO_EXTRACT}")
    print(f"  Sequences per distribution: {len(IN_DIST_SEQUENCES)}")
    print(f"  Files saved in: {SAVE_DIR}/")