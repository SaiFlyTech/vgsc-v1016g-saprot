"""
10_build_db1.py  -  build and validate DATABASE 1, the labelled set.

Reads  config/db1_labels.yaml
Writes data/db1.json

What it checks before writing anything:
  1. every position in the position map carries the wild-type residue the YAML claims
  2. every single mutation points at a position that exists in the map
  3. every combination member names a single that exists
  4. no mutation is to the same residue as the wild type

What it derives:
  overall label, used for binary training, from the per-class outcomes:
      resistant  - any class measured as resistant
      no_effect  - every tested class measured as no_effect
      hold       - the paper reports an effect but not which class, or the
                   evidence grade is too weak to count as a label
  Positions are grouped, because mutations at the same position must never be
  split across a train/test boundary. The script prints which positions carry
  more than one label.

Run on Windows, no GPU:
    python scripts\\10_build_db1.py

Needs: biopython, pyyaml
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "db1_labels.yaml"
OUT = ROOT / "data" / "db1.json"
AA = set("ACDEFGHIKLMNPQRSTVWY")

cfg = yaml.safe_load(open(CFG))
ref = str(next(SeqIO.parse(ROOT / "data" / "vgsc_wildtype.fasta", "fasta")).seq).upper()
print(f"reference: {len(ref)} residues")

# ---------------------------------------------------------------- 1) positions
posmap, errors = {}, []
for pos, d in cfg["positions"].items():
    seq, wt = d["seq"], d["wt"]
    if not 1 <= seq <= len(ref):
        errors.append(f"position {pos}: seq_position {seq} is outside the sequence")
        continue
    if ref[seq - 1] != wt:
        errors.append(f"position {pos}: YAML says {wt} at seq {seq}, sequence has {ref[seq-1]}")
        continue
    posmap[pos] = dict(seq=seq, wt=wt, segment=d.get("segment", ""), site=d.get("site", "none"),
                       flank=ref[max(0, seq - 6):seq + 5])
print(f"positions verified: {len(posmap)} of {len(cfg['positions'])}")

# ----------------------------------------------------------------- 2) singles
def overall_label(e):
    """Collapse the per-class outcomes into one training label. Documented in the header."""
    classes = [e.get("type_I", "not_tested"), e.get("type_II", "not_tested")]
    if "resistant" in classes:
        return "resistant"
    if "unknown" in classes:
        # the paper measured an effect but did not say which class
        if e["evidence"] == "oocyte_other_species":
            return "resistant"          # source states reduced pyrethroid sensitivity
        if e["evidence"] == "oocyte" and e.get("vs_singles") == "stronger":
            return "resistant"
        return "hold"                   # review_only, or too vague to use
    tested = [c for c in classes if c != "not_tested"]
    if tested and all(c == "no_effect" for c in tested):
        return "no_effect"
    return "hold"


records = []
for group, key in (("aedes", "singles_aedes"), ("other_species", "singles_other_species")):
    for e in cfg.get(key) or []:
        p = e["position"]
        if p not in posmap:
            errors.append(f"{e['name']}: position {p} is not in the verified map")
            continue
        wt, mut = posmap[p]["wt"], e["mut"]
        if mut not in AA:
            errors.append(f"{e['name']}: {mut} is not an amino acid")
            continue
        if mut == wt:
            errors.append(f"{e['name']}: mutant residue equals wild type ({wt})")
            continue
        records.append(dict(
            name=e["name"], kind="single", group=group, position=p,
            seq_position=posmap[p]["seq"], wt=wt, mut=mut,
            members=[e["name"]],
            label=overall_label(e),
            type_I=e.get("type_I", "not_tested"), type_II=e.get("type_II", "not_tested"),
            ddt=e.get("ddt", "not_tested"),
            evidence=e["evidence"], channel=e.get("channel", "unstated"),
            species=e.get("species", "aedes_aegypti"),
            source=e.get("source", ""), note=(e.get("note") or "").strip(),
        ))

by_name = {r["name"]: r for r in records}

# ------------------------------------------------------------ 3) combinations
for e in cfg.get("combinations") or []:
    edits, missing = [], []
    for m in e["members"]:
        src = by_name.get(m)
        if src is None:
            missing.append(m)
            continue
        edits.append(dict(seq_position=src["seq_position"], wt=src["wt"], mut=src["mut"],
                          position=src["position"], name=m))
    if missing:
        errors.append(f"{e['name']}: member(s) not found among the singles: {', '.join(missing)}")
        continue
    if len({d["seq_position"] for d in edits}) != len(edits):
        errors.append(f"{e['name']}: two members act on the same position")
        continue
    records.append(dict(
        name=e["name"], kind="combination", group="aedes",
        positions=[d["position"] for d in edits], edits=edits, members=e["members"],
        label=overall_label(e),
        type_I=e.get("type_I", "not_tested"), type_II=e.get("type_II", "not_tested"),
        ddt=e.get("ddt", "not_tested"),
        vs_singles=e.get("vs_singles", "unknown"),
        extends_class=e.get("extends_class", ""),
        evidence=e["evidence"], channel=e.get("channel", "unstated"),
        source=e.get("source", ""), note=(e.get("note") or "").strip(),
    ))

# ------------------------------------------------- 4) field-only, not labels
field = []
for e in cfg.get("field_only") or []:
    p = e["position"]
    if p not in posmap:
        errors.append(f"{e['name']}: position {p} is not in the verified map")
        continue
    if e["mut"] == posmap[p]["wt"]:
        errors.append(f"{e['name']}: mutant residue equals wild type")
        continue
    field.append(dict(name=e["name"], kind="single", group="field_only", position=p,
                      seq_position=posmap[p]["seq"], wt=posmap[p]["wt"], mut=e["mut"],
                      members=[e["name"]], label="none",
                      evidence=e["evidence"], species=e.get("species", "aedes_aegypti"),
                      source=e.get("source", ""), note=(e.get("note") or "").strip()))

# ------------------------------------------------------------------- report
if errors:
    print("\nFAILED. Nothing was written. Fix these first:")
    for e in errors:
        print("  -", e)
    sys.exit(1)

labels = Counter(r["label"] for r in records)
kinds = Counter(r["kind"] for r in records)
print(f"\nlabelled entries: {len(records)}   ({kinds['single']} single, {kinds['combination']} combination)")
print(f"  resistant {labels['resistant']}   no_effect {labels['no_effect']}   hold {labels['hold']}")
print(f"field-only entries (not labels): {len(field)}")

print("\nTrainable set, measured in Aedes aegypti channels only:")
tr = [r for r in records if r["group"] == "aedes" and r["label"] in ("resistant", "no_effect")]
trs = [r for r in tr if r["kind"] == "single"]
print(f"  singles      {len(trs):3d}  ({sum(r['label']=='resistant' for r in trs)} resistant, "
      f"{sum(r['label']=='no_effect' for r in trs)} no_effect)")
trc = [r for r in tr if r["kind"] == "combination"]
print(f"  combinations {len(trc):3d}  ({sum(r['label']=='resistant' for r in trc)} resistant, "
      f"{sum(r['label']=='no_effect' for r in trc)} no_effect)")
print(f"  distinct positions among the singles: {len({r['position'] for r in trs})}")

shared = defaultdict(list)
for r in trs:
    shared[r["position"]].append(f"{r['name']}={r['label']}")
multi = {p: v for p, v in shared.items() if len(v) > 1}
if multi:
    print("\n  Positions carrying more than one label. These MUST stay in the same fold:")
    for p, v in sorted(multi.items()):
        print(f"    {p}: {', '.join(v)}")

print("\nPer-class counts among the Aedes singles (what a two-output model would train on):")
for cls in ("type_I", "type_II"):
    c = Counter(r[cls] for r in trs)
    print(f"  {cls:8} resistant {c['resistant']}   no_effect {c['no_effect']}   not_tested {c['not_tested']}")

held = [r["name"] for r in records if r["label"] == "hold"]
if held:
    print(f"\nHeld aside, scored but never counted as a label: {', '.join(held)}")

OUT.parent.mkdir(exist_ok=True)
json.dump(dict(
    reference=cfg["reference"], window_residues=cfg.get("window_residues", 500),
    positions={str(k): v for k, v in posmap.items()},
    labelled=records, field_only=field,
    observed_haplotypes=cfg.get("observed_haplotypes", {}),
    linkage_notes=cfg.get("linkage_notes", []),
), open(OUT, "w"), indent=1)
print(f"\nwritten: {OUT.relative_to(ROOT)}")
