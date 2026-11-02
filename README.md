# Pyrethroid-resistance predictor — *Aedes aegypti* sodium channel

A zero-shot pipeline that scores how likely a mutation in the *Aedes aegypti*
voltage-gated sodium channel is to confer pyrethroid resistance, using the
**SaProt** protein language model. No training; the model scores mutations
directly from its pretrained knowledge of natural proteins.

**Start here:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — the full project
write-up and current results, written for a non-specialist reader.

## Headline result

Judged against **Type I pyrethroids**, the model ranks every resistant mutation
above every harmless one (ranking 1.00, accuracy 12/13, p = 0.0006), and beats
every conservation and substitution baseline by a wide margin. Full numbers,
caveats and method are in `PROJECT_STATUS.md`.

## What's in here

```
config/     the mutation labels and target definition
  db1_labels.yaml     the laboratory-measured mutations (the labelled set)
  mutations.yaml      the earlier confirmed-mutation list
  target.yaml         the target protein / settings
scripts/    the pipeline, in run order (01–16) plus the Colab cells
data/       inputs, the built databases, and all results
PROJECT_STATUS.md     the project document (share this with the client)
requirements.txt
```

### The databases (built, in `data/`)

| File | What it is |
| --- | --- |
| `db1.json` | Database 1 — labelled mutations (the only trainable/testable set) |
| `db2.json` | Database 2 — every amino acid at each of the 24 positions |
| `db3_observed.json` / `db3_known_pairs.json` | Database 3 — combinations |
| `db2_lab_shortlist.csv` | The 42 candidate mutations to put to the laboratory |
| `combined_db*.json` | Final per-mutation scores |
| `baseline_report.json` | The evaluation (ranking, accuracy, confidence intervals) |
| `embedding_model_report.json` | The "does training help" test result |

## How to run it

Most steps run on an ordinary laptop. Only the scoring step needs a GPU, which a
free Google Colab notebook provides.

```bash
pip install -r requirements.txt     # plus torch + transformers for scoring

# 1. build and verify the databases (local)
python scripts/10_build_db1.py
python scripts/11_build_db2.py
python scripts/12_build_db3.py --tier observed

# 2. prepare scoring windows with structure (local; needs the PDB)
python scripts/13_prepare_windows.py --db db1 --pdb data/C3RTM2_12_2001_7w9t.1.A.pdb

# 3. score on a GPU: upload data/jobs_db1.json to Colab,
#    run scripts/colab_score_db.py with DB = "db1", download scores_db1.json

# 4. combine and evaluate (local)
python scripts/14_combine_scores.py --db db1
python scripts/15_baseline_model.py
```

The baseline comparison (`06`, `07`) and the training test (`colab_extract_embeddings.py`
→ `16_train_embeddings.py`) follow the same pattern. See `PROJECT_STATUS.md` for
what each produces.

## The model

One model does all the scoring: **SaProt (650M)**, HuggingFace
`westlake-repl/SaProt_650M_AF2`. Structure tokens come from a SWISS-MODEL
homology model (template 7w9t) via `mini3di`. Everything else — BLOSUM62 and the
conservation baselines — exists only to measure SaProt against.

## Status

All four databases built and scored; evaluation and baselines complete; training
tested and documented as not yet helpful at this data size. Next phase: trialling
**VenusREM** (the current ProteinGym leader) as a drop-in replacement. See the
"What comes next" section of `PROJECT_STATUS.md`.
