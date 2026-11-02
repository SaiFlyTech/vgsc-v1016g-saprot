#!/usr/bin/env python3
"""
05_prepare_all_mutations.py

Builds ONE file (data/mutations_input.json) holding, for every mutation in
config/mutations.yaml:
  - the amino-acid window and the 3Di window around it, aligned by residue
    NUMBER (same method as the fixed step 04),
  - the position of the mutation inside that window.

It refuses to continue if, for any mutation, the residue at that position in
your sequence or in the structure is not the wild-type residue the paper
says it should be (a numbering mismatch). That check is the whole point:
scoring the wrong residue would give a confident-looking, meaningless number.

Usage (run from the project folder):
    python scripts\\05_prepare_all_mutations.py --pdb data\\C3RTM2_12_2001_7w9t.1.A.pdb
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MUT_PATH = ROOT / "config" / "mutations.yaml"
WT_FASTA = ROOT / "data" / "vgsc_wildtype.fasta"
OUT_JSON = ROOT / "data" / "mutations_input.json"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def main(pdb_path, chain_id, skip_failed=False):
    try:
        import mini3di
        from Bio.PDB import PDBParser
    except ImportError:
        raise SystemExit("Run: pip install mini3di biopython")

    cfg = yaml.safe_load(MUT_PATH.read_text())
    W = cfg["window_residues"]
    full_seq = "".join(WT_FASTA.read_text().strip().splitlines()[1:])

    chain = {c.get_id(): c for c in
             PDBParser(QUIET=True).get_structure("v", pdb_path).get_chains()}[chain_id]
    residues = [r for r in chain.get_residues() if "CA" in r]
    resnums = [r.get_id()[1] for r in residues]
    aas = [THREE_TO_ONE.get(r.get_resname(), "X") for r in residues]
    enc = mini3di.Encoder()
    tokens = enc.build_sequence(enc.encode_chain(chain))
    assert len(tokens) == len(residues)
    num_to_tok = dict(zip(resnums, tokens))
    num_to_aa = dict(zip(resnums, aas))
    first, last = resnums[0], resnums[-1]
    print(f"Sequence length {len(full_seq)}; structure covers {first}-{last}\n")

    out, problems = [], []
    print(f"{'mutation':8} {'label':14} {'seq has':8} {'struct has':11} {'window':12} status")
    for m in cfg["mutations"]:
        # "seq_position" lets you map a paper number onto this UniProt entry once YOU have verified it
        pos, wt = m.get("seq_position", m["position"]), m["wt"]
        seq_res = full_seq[pos - 1] if pos <= len(full_seq) else "?"
        st_res = num_to_aa.get(pos, "?")
        start, end = max(first, pos - W), min(last, pos + W)
        missing = [n for n in range(start, end + 1) if n not in num_to_tok]
        disagree = [n for n in range(start, end + 1)
                    if n in num_to_aa and num_to_aa[n] != full_seq[n - 1]]
        ok = (seq_res == wt and st_res == wt and not missing and not disagree)
        print(f"{m['name']:8} {m['label']:14} {seq_res:8} {st_res:11} "
              f"{start}-{end:<7} {'OK' if ok else 'PROBLEM'}")
        if not ok:
            problems.append(f"{m['name']}: expected {wt} at {pos}, sequence has "
                            f"{seq_res}, structure has {st_res}; "
                            f"{len(missing)} residues missing, "
                            f"{len(disagree)} sequence/structure disagreements in window")
            continue
        out.append({
            "name": m["name"], "label": m["label"], "group": m.get("group", "aedes"), "position": pos, "paper_position": m["position"],
            "wt": wt, "mut": m["mut"], "window_start": start, "window_end": end,
            "local_index": pos - start,   # 0-based index of the mutation in the window
            "aa_window": full_seq[start - 1:end],
            "struct_window": "".join(num_to_tok[n] for n in range(start, end + 1)),
        })

    if problems and skip_failed:
        print("\nSKIPPED (failed the check, not scored):")
        for p_ in problems:
            print("  -", p_)
        problems = []
    if problems:
        print("\nSTOP. These mutations failed the check, so NOTHING was written:")
        for p in problems:
            print("  -", p)
        print("\nThis usually means the numbering in the paper differs from this "
              "UniProt entry. Paste this output to me; do not score anything yet.\n(Or add --skip-failed to prepare only the ones that passed.)")
        raise SystemExit(1)

    OUT_JSON.write_text(json.dumps({"mutations": out}))
    print(f"\nAll {len(out)} mutations passed. Wrote {OUT_JSON}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", required=True)
    p.add_argument("--chain", default="A")
    p.add_argument("--skip-failed", action="store_true")
    a = p.parse_args()
    main(a.pdb, a.chain, a.skip_failed)
