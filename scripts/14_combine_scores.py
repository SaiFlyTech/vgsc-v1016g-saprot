"""
14_combine_scores.py  -  turn raw job scores into one score per variant.

Reads  data/scores_<db>.json  (downloaded from Colab) and data/<db>.json
Writes data/combined_<db>.json

For a single mutation the score is read straight off the masked pass at its
position. Because a pass returns every amino acid at once, one pass serves
every mutation at that position.

For a combination:
    additive  = sum of the member scores, each with no background
    joint     = sum of score(member | all the other members applied)
    epistasis = joint - additive
Negative epistasis means the model thinks the combination is more unusual than
its parts suggest, which is the direction that should correspond to synergy.

For database 2 it also computes, for every variant, its rank and percentile
among the 19 alternatives at its own position. That percentile is what turns a
raw score into something reportable.

Run on Windows, no GPU:
    python scripts\\14_combine_scores.py --db db1

Needs: nothing beyond the standard library
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, choices=["db1", "db2", "db3"])
a = ap.parse_args()

scores_path = ROOT / "data" / f"scores_{a.db}.json"
if not scores_path.exists():
    raise SystemExit(f"{scores_path.name} is missing. Score the jobs in Colab first "
                     f"with scripts/colab_score_db.py, then put the file in data\\")
raw = json.load(open(scores_path))
db = json.load(open(ROOT / "data" / f"{a.db}.json"))

# ---- index the no-background passes by position: these give every single score
base = {}
for r in raw.values():
    if not r["background"] and not r["background_outside_window"]:
        base[r["seq_position"]] = r["logp"]
print(f"jobs read: {len(raw)}   positions with a clean no-background pass: {len(base)}")


def single_score(seq_position, wt, mut):
    lp = base.get(seq_position)
    if lp is None or mut not in lp or wt not in lp:
        return None
    return lp[mut] - lp[wt]


# ---- conditional scores, keyed by (variant, scored member)
cond = {(r["variant"], r["scored"]): r for r in raw.values()
        if r["background"] or r["background_outside_window"]}

out = []

if a.db == "db2":
    by_pos = defaultdict(list)
    for v in db["variants"]:
        s = single_score(v["seq_position"], v["wt"], v["mut"])
        if s is None:
            continue
        rec = dict(name=v["name"], position=v["position"], seq_position=v["seq_position"],
                   wt=v["wt"], mut=v["mut"], segment=v["segment"], site=v["site"],
                   in_db1=v["in_db1"], db1_name=v["db1_name"], score=s)
        by_pos[v["position"]].append(rec)
    for pos, group in by_pos.items():
        group.sort(key=lambda r: r["score"])          # lowest = most unusual
        n = len(group)
        for i, r in enumerate(group):
            r["rank_at_position"] = i + 1
            r["of_at_position"] = n
            r["percentile"] = round(100.0 * i / max(n - 1, 1), 1)
            out.append(r)
    out.sort(key=lambda r: (r["position"], r["rank_at_position"]))

    print(f"\nvariants scored: {len(out)} across {len(by_pos)} positions")
    print("\nMost unusual change at each position, and where the known ones sit:")
    print(f"  {'pos':>5} {'most unusual':>13} {'score':>7}   known mutations (percentile)")
    for pos in sorted(by_pos):
        g = by_pos[pos]
        known = [f"{r['db1_name'] or r['name']} {r['percentile']:.0f}%" for r in g if r["in_db1"]]
        print(f"  {pos:>5} {g[0]['name']:>13} {g[0]['score']:7.2f}   "
              + (", ".join(known) if known else "-"))

else:
    entries = (db["labelled"] + db["field_only"]) if a.db == "db1" else db["combinations"]
    for r in entries:
        if r.get("kind") == "single" or (a.db == "db1" and r.get("kind") == "single"):
            e = r
            s = single_score(e["seq_position"], e["wt"], e["mut"])
            out.append(dict(name=r["name"], kind="single", label=r.get("label", "none"),
                            group=r.get("group", "aedes"), position=r["position"],
                            type_I=r.get("type_I"), type_II=r.get("type_II"), ddt=r.get("ddt"),
                            evidence=r.get("evidence"), score=s, additive=s,
                            joint=s, epistasis=0.0, members=[r["name"]],
                            source=r.get("source", "")))
            continue

        # combination
        members, add, joint, miss, forced = r["members"], 0.0, 0.0, [], False
        for e in r["edits"]:
            s1 = single_score(e["seq_position"], e["wt"], e["mut"])
            c = cond.get((r["name"], e["name"]))
            if s1 is None or c is None or c["score"] is None:
                miss.append(e["name"])
                continue
            if c["background_outside_window"]:
                forced = True
            add += s1
            joint += c["score"]
        if miss:
            print(f"  skipped {r['name']}: no score for {', '.join(miss)}")
            continue
        out.append(dict(name=r["name"], kind="combination", label=r.get("label", "none"),
                        group=r.get("group", r.get("origin", "aedes")),
                        positions=[e["position"] for e in r["edits"]],
                        type_I=r.get("type_I"), type_II=r.get("type_II"),
                        vs_singles=r.get("vs_singles", "unknown"),
                        members=members, additive=add, joint=joint,
                        epistasis=joint - add, score=joint,
                        interaction_forced_zero=forced,
                        observed_in=r.get("observed_in", []), source=r.get("source", "")))

    singles = [r for r in out if r["kind"] == "single" and r["score"] is not None]
    combos = [r for r in out if r["kind"] == "combination"]
    print(f"\nsingles scored: {len(singles)}   combinations scored: {len(combos)}")

    if singles:
        print(f"\n{'variant':12} {'group':14} {'label':10} {'score':>7}  Type I / Type II")
        for r in sorted(singles, key=lambda r: r["score"]):
            print(f"{r['name']:12} {r['group']:14} {r['label']:10} {r['score']:7.2f}  "
                  f"{str(r.get('type_I')):<10} {str(r.get('type_II'))}")

    if combos:
        print(f"\n{'combination':24} {'label':10} {'additive':>9} {'joint':>8} "
              f"{'epistasis':>10}  vs singles")
        for r in sorted(combos, key=lambda r: r["epistasis"]):
            flag = " (interaction forced to zero)" if r.get("interaction_forced_zero") else ""
            print(f"{r['name']:24} {r['label']:10} {r['additive']:9.2f} {r['joint']:8.2f} "
                  f"{r['epistasis']:10.2f}  {r.get('vs_singles','')}{flag}")

        meas = [r for r in combos if r.get("vs_singles") in ("stronger", "same")]
        if meas:
            st = [r["epistasis"] for r in meas if r["vs_singles"] == "stronger"]
            sa = [r["epistasis"] for r in meas if r["vs_singles"] == "same"]
            if st and sa:
                pairs = [(p, q) for p in st for q in sa]
                auc = sum(1.0 if p < q else 0.5 if p == q else 0.0 for p, q in pairs) / len(pairs)
                print(f"\nDoes epistasis separate the measured combinations?")
                print(f"  {len(st)} measured as stronger than their strongest single, "
                      f"{len(sa)} measured as no stronger")
                print(f"  P(stronger has more negative epistasis) = {auc:.2f} "
                      f"over {len(pairs)} comparisons; 0.50 = coin flip")

outp = ROOT / "data" / f"combined_{a.db}.json"
json.dump(dict(db=a.db, variants=out), open(outp, "w"), indent=1)
print(f"\nwritten: {outp.relative_to(ROOT)}")
