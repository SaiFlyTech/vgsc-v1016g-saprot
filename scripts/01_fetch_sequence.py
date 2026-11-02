#!/usr/bin/env python3
"""
01_fetch_sequence.py

Step 01 of the pipeline (per the reference deck): fetch the Aedes aegypti
VGSC protein sequence (gene AAEL023266, AaegL5 numbering).

Primary source: UniProt REST API (well-documented, stable endpoint).
VectorBase is the deck's named source for the AaegL5 gene model specifically
— if the UniProt record's numbering doesn't match AaegL5 (check this; isoform
numbering can differ between databases), fetch the sequence directly from
VectorBase (vectorbase.org) by searching "AAEL023266" and download its FASTA
manually, then point --manual-fasta at that file instead.

Requires internet access — run on your own machine, not in a sandboxed
environment with restricted outbound network access.

Usage:
    python scripts/01_fetch_sequence.py
    # or, if you downloaded the sequence from VectorBase by hand:
    python scripts/01_fetch_sequence.py --manual-fasta path/to/downloaded.fasta
"""
import argparse
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "target.yaml"
OUT_PATH = ROOT / "data" / "vgsc_wildtype.fasta"

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
AEDES_AEGYPTI_TAXID = "7159"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def fetch_from_uniprot(gene_id: str):
    params = {
        "query": f"gene:{gene_id} AND organism_id:{AEDES_AEGYPTI_TAXID}",
        "format": "fasta",
    }
    resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        raise RuntimeError(
            f"No UniProt record found for gene {gene_id} in Aedes aegypti. "
            f"Try fetching from VectorBase directly instead (see --manual-fasta)."
        )
    return text


def parse_fasta(fasta_text: str):
    lines = fasta_text.strip().splitlines()
    header = lines[0]
    seq = "".join(lines[1:])
    return header, seq


def main(manual_fasta: str | None):
    cfg = load_config()
    gene_id = cfg["gene_id"]
    position = cfg["mutation"]["position"]
    expected_wt = cfg["mutation"]["wildtype_residue"]

    if manual_fasta:
        print(f"Using manually-downloaded FASTA: {manual_fasta}")
        with open(manual_fasta) as f:
            fasta_text = f.read()
    else:
        print(f"Querying UniProt for gene {gene_id} (Aedes aegypti)...")
        fasta_text = fetch_from_uniprot(gene_id)

    header, seq = parse_fasta(fasta_text)
    print(f"\nFetched: {header}")
    print(f"Length: {len(seq)} residues")

    # Critical check from the deck: "Verify seq[position] == 'from' residue.
    # Fails loudly or not at all."
    idx = position - 1  # 1-based biological numbering -> 0-based string index
    if idx < 0 or idx >= len(seq):
        sys.exit(
            f"ERROR: position {position} is out of range for a sequence of "
            f"length {len(seq)}. This sequence is NOT a correct match for "
            f"AaegL5 numbering — do not proceed. Try VectorBase directly."
        )

    actual = seq[idx]
    if actual != expected_wt:
        sys.exit(
            f"ERROR: expected residue '{expected_wt}' at position {position}, "
            f"found '{actual}' instead. This sequence's numbering does not "
            f"match AaegL5 — do not proceed with this sequence. This is "
            f"exactly the silent-error risk the deck flags: different "
            f"databases/isoforms can number the same protein differently. "
            f"Fetch the sequence from VectorBase directly (search AAEL023266) "
            f"and re-run with --manual-fasta."
        )

    print(f"\n✓ Verified: position {position} is '{actual}' as expected.")

    ROOT.joinpath("data").mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(f">{gene_id}_AaegL5_VGSC_wildtype\n{seq}\n")
    print(f"\nSaved verified wild-type sequence to {OUT_PATH}")

    # Update config to record that verification passed
    cfg["sequence_source"]["verified_wildtype_match"] = True
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("Updated config/target.yaml: verified_wildtype_match = true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual-fasta",
        default=None,
        help="Path to a FASTA file you downloaded by hand from VectorBase, "
             "if the UniProt record doesn't match AaegL5 numbering.",
    )
    args = parser.parse_args()
    main(args.manual_fasta)
