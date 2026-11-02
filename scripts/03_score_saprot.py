#!/usr/bin/env python3
"""
03_score_saprot.py

Step 03 of the pipeline: zero-shot SaProt scoring of V1016G.

score = log P(mutant residue | context) - log P(wildtype residue | context)

This is the masked-marginal scoring method the deck specifies: mask the
mutation position in the windowed sequence, get the model's probability
distribution over amino acids at that position, and compare the mutant vs.
wild-type log-probabilities. No training happens — only a forward pass
through SaProt's pretrained weights.

IMPORTANT — this step needs things the sandbox this was built in does not
have:
  1. A GPU (SaProt-650M is a 650M-parameter model; CPU inference will be
     very slow, and the deck specifies it should run on a single GPU).
  2. Internet access to download the model weights from HuggingFace
     (westlake-repl/SaProt_650M_AF2) on first run.
  3. The 3Di structural tokens from step 04 (Foldseek) — this script
     currently only has placeholder logic for combining AA + 3Di; see the
     TODO marker below. I could not generate real 3Di tokens here since that
     needs the AlphaFold structure (network) and the Foldseek binary
     (external tool, not pip-installable), neither available in this
     sandbox.

Run this on your own machine (with GPU) once steps 01, 02, and 04 (Foldseek
tokenization) are done for real.

Usage:
    python scripts/03_score_saprot.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "target.yaml"
WT_WINDOW = ROOT / "data" / "window_wildtype.fasta"
MUT_WINDOW = ROOT / "data" / "window_mutant.fasta"
STRUCT_TOKENS = ROOT / "data" / "window_3di_tokens.txt"  # produced by step 04


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_fasta(path: Path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    return lines[0].lstrip(">"), "".join(lines[1:])


def combine_aa_3di(aa_seq: str, tokens_3di: str) -> str:
    """
    SaProt's input format interleaves each amino acid with its structural
    token, e.g. 'VqGhLm...' where each AA is paired with a lowercase 3Di
    letter. Exact formatting conventions (separators, casing) should be
    checked against the SaProt repo / model card before running for real:
    https://github.com/westlake-repl/SaProt
    """
    if len(aa_seq) != len(tokens_3di):
        raise ValueError(
            f"AA sequence ({len(aa_seq)}) and 3Di token string "
            f"({len(tokens_3di)}) must be the same length — one 3Di token "
            f"per residue."
        )
    return "".join(aa + tok for aa, tok in zip(aa_seq, tokens_3di))


def score_mutation(model, tokenizer, combined_seq: str, position_1based: int,
                    wildtype_residue: str, mutant_residue: str) -> float:
    """
    Placeholder for the actual masked-marginal scoring call.
    Real implementation needs:
      1. Tokenize combined_seq with SaProt's tokenizer
      2. Mask the token(s) at `position_1based`
      3. Run a forward pass, get logits at that position
      4. log_softmax -> log P(mutant) and log P(wildtype)
      5. Return their difference

    This function is NOT implemented here — it needs the actual SaProt
    model loaded (torch + transformers + the real checkpoint), which this
    sandbox can't download/run. Fill this in on your own machine using the
    inference example in the SaProt repo's README.
    """
    raise NotImplementedError(
        "Load the real SaProt-650M checkpoint here (see SaProt GitHub repo "
        "for the exact HuggingFace model id and inference snippet) and "
        "implement the masked-marginal scoring call described in this "
        "function's docstring."
    )


def main():
    cfg = load_config()
    mut = cfg["mutation"]

    if not WT_WINDOW.exists() or not MUT_WINDOW.exists():
        raise SystemExit(
            "Missing windowed sequences — run 02_mutate_and_window.py first."
        )
    if not STRUCT_TOKENS.exists():
        raise SystemExit(
            "Missing 3Di structure tokens — run 04_foldseek_tokenize.py "
            "first (needs the AlphaFold structure and the Foldseek binary)."
        )

    _, wt_seq = parse_fasta(WT_WINDOW)
    with open(STRUCT_TOKENS) as f:
        tokens_3di = f.read().strip()

    combined = combine_aa_3di(wt_seq, tokens_3di)

    # Position of the mutation within the window, 1-based — computed in
    # step 02 and printed there; re-derive it here from window size/position
    # rather than hardcoding, to avoid silent drift between steps.
    window = cfg["scoring"]["window_residues"]
    full_pos = mut["position"]
    # mirrors the clipping logic in 02_mutate_and_window.py's extract_window
    local_pos = min(full_pos - 1, window) + 1  # approximate; recompute exactly
    # NOTE: for a precise re-derivation, read the actual start offset step 02
    # printed/saved, rather than recomputing the clip here. Left as a TODO —
    # don't trust this approximation without double-checking against step
    # 02's printed "local mutation position."

    print(
        "This script's scoring call is NOT implemented (see script "
        "docstring) — it needs the real SaProt model running on a GPU "
        "machine with the actual 3Di tokens from Foldseek. Everything up "
        "to this point (windowing, combining AA+3Di) is wired up and ready "
        "for you to plug the real model call into."
    )

    print(f"\nWould score: {mut['name']} at local window position {local_pos}")
    print(f"Combined AA+3Di input length: {len(combined)}")


if __name__ == "__main__":
    main()
