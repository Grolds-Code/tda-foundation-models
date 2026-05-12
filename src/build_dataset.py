"""
build_dataset.py
----------------
Fetches biologically diverse protein sequences from UniProt REST API
and constructs matched OOD sequences via residue-preserving shuffling.

Sequence families sampled:
  - Kinases              (signal transduction)
  - Oxidoreductases      (metabolic enzymes)
  - Transcription factors(gene regulation)
  - Chaperones           (protein folding)
  - Transporters         (membrane proteins)

Author: Grold Otieno Mboya
Project: TDA Analysis of Latent Space Geometry in Foundation Models
"""

import requests
import random
import json
import time
import os
import numpy as np
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
SAVE_DIR          = "data"
SEQUENCES_FILE    = os.path.join(SAVE_DIR, "sequences.json")
TARGET_PER_FAMILY = 12
MIN_LENGTH        = 50
MAX_LENGTH        = 400
RANDOM_SEED       = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── UniProt Query Definitions ─────────────────────────────────────────────────
# Uses UniProt keyword IDs — these are stable and always valid
# KW-0418 = Kinase, KW-0560 = Oxidoreductase, KW-0804 = Transcription factor
# KW-0143 = Chaperone, KW-0813 = Transport
FAMILIES = [
    ("kinase",               "reviewed:true AND keyword:KW-0418 AND length:[50 TO 400]"),
    ("oxidoreductase",       "reviewed:true AND keyword:KW-0560 AND length:[50 TO 400]"),
    ("transcription_factor", "reviewed:true AND keyword:KW-0804 AND length:[50 TO 400]"),
    ("chaperone",            "reviewed:true AND keyword:KW-0143 AND length:[50 TO 400]"),
    ("transporter",          "reviewed:true AND keyword:KW-0813 AND length:[50 TO 400]"),
]

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"

# ── Fetch Sequences from UniProt ──────────────────────────────────────────────
def fetch_sequences(family_name: str, query: str, n: int) -> list:
    params = {
        "query":  query,
        "format": "fasta",
        "size":   min(n * 3, 50),
    }

    try:
        resp = requests.get(UNIPROT_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [WARNING] Failed to fetch {family_name}: {e}")
        return []

    sequences   = []
    current_id  = None
    current_seq = []

    for line in resp.text.splitlines():
        if line.startswith(">"):
            if current_id and current_seq:
                seq = "".join(current_seq)
                if MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                    sequences.append({
                        "id":       current_id,
                        "sequence": seq,
                        "family":   family_name,
                        "length":   len(seq),
                    })
            current_id  = line.split("|")[1] if "|" in line else line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip())

    # Handle last entry
    if current_id and current_seq:
        seq = "".join(current_seq)
        if MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            sequences.append({
                "id":       current_id,
                "sequence": seq,
                "family":   family_name,
                "length":   len(seq),
            })

    random.shuffle(sequences)
    return sequences[:n]

# ── OOD Generation ────────────────────────────────────────────────────────────
def make_ood_sequence(seq: str) -> str:
    chars = list(seq)
    random.shuffle(chars)
    return "".join(chars)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Building protein sequence dataset from UniProt")
    print("=" * 60)

    all_in_dist = []
    all_ood     = []

    for family_name, query in FAMILIES:
        print(f"\nFetching {family_name} sequences...")
        seqs = fetch_sequences(family_name, query, TARGET_PER_FAMILY)
        print(f"  Retrieved {len(seqs)} sequences")

        for entry in seqs:
            all_in_dist.append(entry)
            ood_entry             = dict(entry)
            ood_entry["sequence"] = make_ood_sequence(entry["sequence"])
            ood_entry["family"]   = f"{family_name}_ood"
            all_ood.append(ood_entry)

        time.sleep(0.5)

    print(f"\n{'─'*60}")
    print(f"Total in-distribution sequences : {len(all_in_dist)}")
    print(f"Total OOD sequences             : {len(all_ood)}")

    from collections import Counter
    counts = Counter(s["family"] for s in all_in_dist)
    print("\nFamily breakdown:")
    for fam, count in counts.items():
        print(f"  {fam:<25} {count:>3} sequences")

    lengths = [s["length"] for s in all_in_dist]
    print(f"\nSequence length — min: {min(lengths)}  max: {max(lengths)}  "
          f"mean: {np.mean(lengths):.1f}  median: {np.median(lengths):.1f}")

    dataset = {
        "in_dist": all_in_dist,
        "ood":     all_ood,
        "metadata": {
            "total_sequences": len(all_in_dist),
            "families":        [f[0] for f in FAMILIES],
            "min_length":      MIN_LENGTH,
            "max_length":      MAX_LENGTH,
            "random_seed":     RANDOM_SEED,
            "source":          "UniProt Swiss-Prot (reviewed)",
        }
    }

    with open(SEQUENCES_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Dataset saved to {SEQUENCES_FILE}")
    print(f"  {len(all_in_dist)} in-distribution + {len(all_ood)} OOD sequences ready")