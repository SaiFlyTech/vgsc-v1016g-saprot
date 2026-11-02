"""
12_build_db3.py  -  build DATABASE 3, combinations.

Reads  data/db1.json, data/db2.json
Writes data/db3.json

Three subsets, because they differ enormously in size and in usefulness.
Choose with --tier:

  observed      The haplotypes actually seen in wild populations (India and
                Taiwan), plus the 8 combinations that have been measured in
                oocytes. Dozens of entries. This is the subset where a
                prediction is immediately useful, because no laboratory has one.

  known_pairs   Every pair of substitutions that has been reported anywhere,
                taken two at a time across different positions. Hundreds of
                entries. This is the epistasis landscape among real alleles.
                DEFAULT.

  all_pairs     Every amino acid at every position, paired with every amino
                acid at every other position. Around 100,000 entries. The full
                landscape, and the only one that needs real compute.

Why triples are not offered: all triples across these positions come to several
million, and the two surveys between them found seven and nine haplotypes.
The combinations that exist in nature are in the dozens, not the millions,
because these positions sit on one chromosome and do not assort freely.
Enumerating the arithmetically possible set would be work spent on mosquitoes
that have never existed. The observed triples are included in `observed`.

Scoring cost: a second change is scored against a window that already carries
the first, so the count of forward passes is (number of backgrounds) x
(number of other positions), and each pass yields all 19 alternatives. Both
orders of a pair are scored on purpose: the gap between the two numbers is the
model's own estimate of how much the two positions interact, which can be
checked against the 8 measured combinations.

Run on Windows, no GPU:
    python scripts\\12_build_db3.py
    python scripts\\12_build_db3.py --tier observed
    python scripts\\12_build_db3.py --tier all_pairs

Needs: nothing beyond the standard library
"""
import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "db3.json"
AA = "ACDEFGHIKLMNPQRSTVWY"

ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["observed", "known_pairs", "all_pairs"], default="known_pairs")
a = ap.parse_args()

db1 = json.load(open(ROOT / "data" / "db1.json"))
positions = {int(k): v for k, v in db1["positions"].items()}
singles = {r["name"]: r for r in db1["labelled"] + db1["field_only"] if r["kind"] == "single"}


def edit(name):
    r = singles[name]
    return dict(name=name, position=r["position"], seq_position=r["seq_position"],
                wt=r["wt"], mut=r["mut"])


records = []

# ------------------------------------------------------------------ observed
if a.tier == "observed":
    # the 8 measured combinations, carried over with their labels
    for r in db1["labelled"]:
        if r["kind"] == "combination":
            records.append(dict(name=r["name"], members=r["members"], edits=r["edits"],
                                origin="measured", label=r["label"],
                                vs_singles=r.get("vs_singles", "unknown"),
                                type_I=r.get("type_I"), type_II=r.get("type_II"),
                                source=r.get("source", "")))
    # the field haplotypes, which have no label
    seen = {tuple(sorted(r["members"])) for r in records}
    for country, haps in (db1.get("observed_haplotypes") or {}).items():
        for h in haps:
            members = h.get("members") or []
            if len(members) < 2:
                continue                      # wild type and single-mutation haplotypes
            key = tuple(sorted(members))
            if key in seen:
                # observed in the field AND measured in oocytes - note it
                for r in records:
                    if tuple(sorted(r["members"])) == key:
                        r.setdefault("observed_in", []).append(f"{country}:{h['code']}")
                continue
            seen.add(key)
            missing = [m for m in members if m not in singles]
            if missing:
                print(f"  skipped {h['code']}: unknown member(s) {', '.join(missing)}")
                continue
            records.append(dict(name="+".join(members), members=members,
                                edits=[edit(m) for m in members],
                                origin="field_haplotype", label="none",
                                observed_in=[f"{country}:{h['code']}"],
                                count=h.get("count"), freq=h.get("freq", ""),
                                note=h.get("note", "")))

# --------------------------------------------------------------- known_pairs
elif a.tier == "known_pairs":
    names = sorted(singles)
    for x, y in combinations(names, 2):
        if singles[x]["position"] == singles[y]["position"]:
            continue                          # cannot carry two changes at one residue
        records.append(dict(name=f"{x}+{y}", members=[x, y],
                            edits=[edit(x), edit(y)], origin="known_pair", label="none"))

# ----------------------------------------------------------------- all_pairs
else:
    pos_list = sorted(positions)
    for p, q in combinations(pos_list, 2):
        wp, wq = positions[p]["wt"], positions[q]["wt"]
        for mp in AA:
            if mp == wp:
                continue
            for mq in AA:
                if mq == wq:
                    continue
                records.append(dict(
                    name=f"{wp}{p}{mp}+{wq}{q}{mq}", members=[f"{wp}{p}{mp}", f"{wq}{q}{mq}"],
                    edits=[dict(name=f"{wp}{p}{mp}", position=p, seq_position=positions[p]["seq"], wt=wp, mut=mp),
                           dict(name=f"{wq}{q}{mq}", position=q, seq_position=positions[q]["seq"], wt=wq, mut=mq)],
                    origin="all_pair", label="none"))

# -------------------------------------------------------------------- report
origins = Counter(r["origin"] for r in records)
sizes = Counter(len(r["members"]) for r in records)
print(f"tier: {a.tier}")
print(f"entries: {len(records)}")
for k, v in sorted(origins.items()):
    print(f"  {k:<16} {v}")
print("  by size: " + ", ".join(f"{k} changes: {v}" for k, v in sorted(sizes.items())))

labelled = [r for r in records if r.get("label") in ("resistant", "no_effect")]
if labelled:
    c = Counter(r["label"] for r in labelled)
    vs = Counter(r.get("vs_singles") for r in labelled)
    print(f"\n  with a measured label: {len(labelled)}  "
          f"(resistant {c['resistant']}, no_effect {c['no_effect']})")
    print(f"  stronger than their strongest single: {vs['stronger']}   "
          f"same: {vs['same']}")

# how many forward passes this tier needs
backgrounds = {tuple(sorted(e["seq_position"] for e in r["edits"][:-1])) for r in records}
n_pos = len(positions)
if a.tier == "all_pairs":
    passes = n_pos * 19 * (n_pos - 1)
    print(f"\nforward passes to score: about {passes:,}  "
          f"(each pass yields all 19 alternatives at its position)")
    print(f"  at roughly 0.2 s a pass on a T4 that is about {passes*0.2/3600:.1f} hours")
else:
    passes = sum(len(r["edits"]) for r in records)
    print(f"\nforward passes to score: at most {passes:,}, fewer once backgrounds are reused")

print("\nFull enumeration, for scale:")
print(f"  pairs across all {n_pos} positions:  {math.comb(n_pos,2)*19*19:,}")
print(f"  triples:                            {math.comb(n_pos,3)*19**3:,}")
print(f"  every amino acid at every position:  {20**n_pos:.3g}")
print("  haplotypes actually found in the two surveys: 7 (India) and 9 (Taiwan)")

json.dump(dict(
    reference=db1["reference"], window_residues=db1["window_residues"],
    positions=db1["positions"], tier=a.tier, combinations=records,
    linkage_notes=db1.get("linkage_notes", []),
), open(OUT, "w"), indent=1)
print(f"\nwritten: {OUT.relative_to(ROOT)}")
