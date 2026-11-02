#!/usr/bin/env python3
"""
02_mutate_and_window.py

Step 02 (part A) of the pipeline: given the verified wild-type sequence
(from step 01), generate the V1016G mutant sequence, then extract a
window of ±N residues around the mutation position — because VGSC is
~2,100 residues and SaProt's context window can't take the full protein
(per the deck, slides 3-4).

This step is pure local logic — no internet needed. Runs entirely on
the wild-type FASTA that 01_fetch_sequence.py already verified and saved.

Usage:
    python scripts/02_mutate_and_window.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "target.yaml"
WT_FASTA = ROOT / "data" / "vgsc_wildtype.fasta"
OUT_WT_WINDOW = ROOT / "data" / "window_wildtype.fasta"
OUT_MUT_WINDOW = ROOT / "data" / "window_mutant.fasta"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_fasta(path: Path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    header = lines[0].lstrip(">")
    seq = "".join(lines[1:])
    return header, seq


def mutate(seq: str, position: int, wildtype: str, mutant: str) -> str:
    idx = position - 1
    if seq[idx] != wildtype:
        raise ValueError(
            f"Sequence mismatch: expected '{wildtype}' at position {position}, "
            f"found '{seq[idx]}'. Re-run step 01 — this sequence wasn't "
            f"properly verified."
        )
    return seq[:idx] + mutant + seq[idx + 1:]


def extract_window(seq: str, position: int, window: int):
    """Returns (windowed_sequence, new_position_within_window, start, end)."""
    idx = position - 1
    start = max(0, idx - window)
    end = min(len(seq), idx + window + 1)
    windowed = seq[start:end]
    new_position_1based = idx - start + 1
    return windowed, new_position_1based, start, end


def main():
    cfg = load_config()
    if not cfg["sequence_source"]["verified_wildtype_match"]:
        raise SystemExit(
            "config/target.yaml says verified_wildtype_match is still false. "
            "Run 01_fetch_sequence.py first (and successfully) before this step."
        )

    mut_cfg = cfg["mutation"]
    position = mut_cfg["position"]
    wt_res = mut_cfg["wildtype_residue"]
    mut_res = mut_cfg["mutant_residue"]
    window = cfg["scoring"]["window_residues"]

    header, full_seq = parse_fasta(WT_FASTA)
    print(f"Loaded wild-type sequence: {len(full_seq)} residues")

    mutant_seq = mutate(full_seq, position, wt_res, mut_res)
    print(f"Generated mutant sequence: {mut_cfg['name']}")

    wt_window, new_pos, start, end = extract_window(full_seq, position, window)
    mut_window, new_pos_mut, _, _ = extract_window(mutant_seq, position, window)
    assert new_pos == new_pos_mut, "Window extraction mismatch between WT and mutant"

    print(
        f"\nWindow: residues {start + 1}-{end} of the full protein "
        f"({len(wt_window)} residues, mutation now at local position {new_pos})"
    )

    ROOT.joinpath("data").mkdir(exist_ok=True)
    with open(OUT_WT_WINDOW, "w") as f:
        f.write(f">{header}_window_{start+1}-{end}_wildtype\n{wt_window}\n")
    with open(OUT_MUT_WINDOW, "w") as f:
        f.write(f">{header}_window_{start+1}-{end}_{mut_cfg['name']}\n{mut_window}\n")

    print(f"\nSaved:")
    print(f"  {OUT_WT_WINDOW}")
    print(f"  {OUT_MUT_WINDOW}")
    print(
        f"\nLocal mutation position within window: {new_pos} "
        f"(needed for step 03's scoring call)"
    )


if __name__ == "__main__":
    main()
