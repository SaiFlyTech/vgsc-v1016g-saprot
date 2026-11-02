"""
11_build_db2.py  -  build DATABASE 2, every amino acid at every position.

Reads  data/db1.json   (for the verified position map)
Writes data/db2.json

For each verified position, this generates all 19 substitutions away from the
wild-type residue. With 24 positions that is 456 variants.

Nothing here has a laboratory label and nothing ever will, short of an
experiment nobody has run. Two things it is for:

  1. A per-position scale. Knowing where a substitution sits among the 19
     alternatives at its own position turns an uninterpretable number into a
     percentile, which is the only honest basis for a confidence value.

  2. Checking the model knows more than the position. If all 19 alternatives at
     a position score alike, the model has only learned that the position
     matters, not which change matters. That is a real failure mode and this
     set is what detects it.

Cost note: a masked forward pass returns a probability for every token at once,
so all 19 alternatives at a position come from ONE pass. The whole set is 24
forward passes, which is seconds on a T4.

Entries that already carry a label in database 1 are flagged `in_db1: true` so
they can be excluded when this set is used for calibration. Calibrating on the
same points that set the scale would be circular.

Run on Windows, no GPU:
    python scripts\\11_build_db2.py

Needs: nothing beyond the standard library
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB1 = ROOT / "data" / "db1.json"
OUT = ROOT / "data" / "db2.json"
AA = "ACDEFGHIKLMNPQRSTVWY"

if not DB1.exists():
    raise SystemExit("data/db1.json is missing. Run scripts/10_build_db1.py first.")

db1 = json.load(open(DB1))
positions = {int(k): v for k, v in db1["positions"].items()}

# every substitution that already has a label or a field report, by (position, mut)
known = {}
for r in db1["labelled"] + db1["field_only"]:
    if r["kind"] == "single":
        known[(r["position"], r["mut"])] = r["name"]

records = []
for pos in sorted(positions):
    info = positions[pos]
    wt = info["wt"]
    for mut in AA:
        if mut == wt:
            continue
        key = (pos, mut)
        records.append(dict(
            name=f"{wt}{pos}{mut}",
            kind="single",
            position=pos,
            seq_position=info["seq"],
            wt=wt,
            mut=mut,
            segment=info["segment"],
            site=info["site"],
            in_db1=key in known,
            db1_name=known.get(key, ""),
        ))

n_known = sum(r["in_db1"] for r in records)
print(f"positions: {len(positions)}")
print(f"variants generated: {len(records)}   ({len(positions)} x 19)")
print(f"  already labelled or field-reported in database 1: {n_known}")
print(f"  never reported anywhere: {len(records) - n_known}")
print(f"forward passes needed to score all of them: {len(positions)}")

print("\nper position:")
print(f"  {'pos':>5} {'wt':>3} {'segment':<18} {'site':<5} {'known':>5}")
for pos in sorted(positions):
    info = positions[pos]
    k = sum(r["in_db1"] for r in records if r["position"] == pos)
    print(f"  {pos:>5} {info['wt']:>3} {info['segment']:<18} {info['site']:<5} {k:>5}")

sites = Counter(positions[p]["site"] for p in positions)
print(f"\nreceptor sites: " + ", ".join(f"{k} {v}" for k, v in sorted(sites.items())))

json.dump(dict(
    reference=db1["reference"],
    window_residues=db1["window_residues"],
    positions=db1["positions"],
    variants=records,
    note="No laboratory labels. Calibration and diagnostic set only. "
         "Exclude in_db1 entries when using this set to calibrate.",
), open(OUT, "w"), indent=1)
print(f"\nwritten: {OUT.relative_to(ROOT)}")
