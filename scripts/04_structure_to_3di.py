#!/usr/bin/env python3
"""
04_structure_to_3di.py  (FIXED VERSION)

Converts the structure model into 3Di tokens and cuts out the SAME window
used in step 02 -- matched by residue NUMBER, not by array position.

What was wrong in the first version: it assumed the 3Di array started at
protein residue 1. The model file C3RTM2_12_2001_... covers residues 12-2001,
so array index 0 is residue 12, and the window came out shifted by 11
residues. The length check passed anyway because both windows were 1001
long, which proved nothing. This version:
  1. reads each residue's number and amino acid from the PDB itself,
  2. checks those amino acids against your UniProt sequence,
  3. refuses to continue if they disagree anywhere in the window.

Usage:
    python scripts\\04_structure_to_3di.py --pdb data\\C3RTM2_12_2001_7w9t.1.A.pdb
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "target.yaml"
WT_FULL_FASTA = ROOT / "data" / "vgsc_wildtype.fasta"
OUT_3DI_FULL = ROOT / "data" / "vgsc_3di_full.txt"
OUT_3DI_WINDOW = ROOT / "data" / "window_3di_tokens.txt"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_fasta_seq(path):
    lines = Path(path).read_text().strip().splitlines()
    return "".join(lines[1:])


def main(pdb_path, chain_id):
    try:
        import mini3di
        from Bio.PDB import PDBParser
    except ImportError:
        raise SystemExit("Run: pip install mini3di biopython")

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    position = cfg["mutation"]["position"]
    window = cfg["scoring"]["window_residues"]
    full_seq = read_fasta_seq(WT_FULL_FASTA)

    struct = PDBParser(QUIET=True).get_structure("vgsc", pdb_path)
    chains = {c.get_id(): c for c in struct.get_chains()}
    print(f"Chains in file: {list(chains)}  -> using chain {chain_id}")
    chain = chains[chain_id]

    # Same residue filter mini3di uses internally (residues that have a CA).
    residues = [r for r in chain.get_residues() if "CA" in r]
    resnums = [r.get_id()[1] for r in residues]
    aas = [THREE_TO_ONE.get(r.get_resname(), "X") for r in residues]

    states = mini3di.Encoder().encode_chain(chain)
    tokens = mini3di.Encoder().build_sequence(states)
    assert len(tokens) == len(residues), "3Di count != residue count"
    print(f"Structure covers residues {resnums[0]}-{resnums[-1]} "
          f"({len(residues)} residues)")

    # Check structure's own sequence against the UniProt sequence.
    mismatches = [
        (n, a, full_seq[n - 1]) for n, a in zip(resnums, aas)
        if n - 1 < len(full_seq) and full_seq[n - 1] != a
    ]
    print(f"Residues where structure != UniProt sequence: {len(mismatches)}")
    if mismatches:
        print("  first few (number, structure, uniprot):", mismatches[:5])

    num_to_token = dict(zip(resnums, tokens))
    num_to_aa = dict(zip(resnums, aas))

    start = max(1, position - window)
    end = min(len(full_seq), position + window)
    missing = [n for n in range(start, end + 1) if n not in num_to_token]
    if missing:
        raise SystemExit(
            f"STOP: structure has no coordinates for {len(missing)} residues "
            f"inside the window {start}-{end} (e.g. {missing[:5]}). "
            f"Do not score with this structure."
        )
    window_mismatch = [n for n in range(start, end + 1)
                       if num_to_aa[n] != full_seq[n - 1]]
    if window_mismatch:
        raise SystemExit(
            f"STOP: structure and UniProt sequence disagree at "
            f"{len(window_mismatch)} positions inside the window "
            f"(e.g. {window_mismatch[:5]}). Numbering is not aligned."
        )

    window_tokens = "".join(num_to_token[n] for n in range(start, end + 1))
    print(f"\nOK: window {start}-{end}: {len(window_tokens)} tokens, "
          f"aligned by residue number and verified against the sequence.")
    print(f"Residue {position} in structure: {num_to_aa[position]} "
          f"(expected {cfg['mutation']['wildtype_residue']})")

    OUT_3DI_FULL.write_text("".join(tokens))
    OUT_3DI_WINDOW.write_text(window_tokens)
    print(f"Saved {OUT_3DI_WINDOW}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", required=True)
    p.add_argument("--chain", default="A")
    a = p.parse_args()
    main(a.pdb, a.chain)
