"""
13_prepare_windows.py  -  turn a database into scoring jobs with 3Di structure tokens.

Reads  data/db1.json | data/db2.json | data/db3.json  and the PDB
Writes data/jobs_<db>.json

A JOB is one masked forward pass: a window of combined tokens with one position
masked, plus any background mutations already applied to that window. One job
returns a probability for every amino acid at the masked position, so a job
covers all 19 alternatives there at once.

  single mutation      -> 1 job per position (not per mutation)
  combination of k     -> k jobs, each scoring one member with the other k-1
                          members applied to the window as background

Scoring a member against the rest, rather than in a fixed order, keeps the
result order-free:
     additive  = sum of the single scores, with no background
     joint     = sum over members of score(member | all the others)
     epistasis = joint - additive
Epistasis is the model's own estimate of how much the positions interact. It can
be checked directly against the eight combinations that have oocyte results.

THE WINDOW LIMIT, which matters for database 3.
The window is +/- 500 residues, so two positions more than about 1,000 residues
apart cannot appear in the same window. When that happens the background is
simply absent and the model must report zero interaction, for mechanical
reasons rather than biological ones. Those jobs are flagged
`background_outside_window` and the script counts them, because treating a
forced zero as a prediction of no interaction would be wrong.

Every job is checked before it is written: the window must carry the wild-type
residue the database claims, at the scored position and at every background
position, in the sequence AND in the structure.

Run on Windows, no GPU:
    python scripts\\13_prepare_windows.py --db db1 --pdb data\\C3RTM2_12_2001_7w9t.1.A.pdb
    python scripts\\13_prepare_windows.py --db db2 --pdb data\\C3RTM2_12_2001_7w9t.1.A.pdb
    python scripts\\13_prepare_windows.py --db db3 --pdb data\\C3RTM2_12_2001_7w9t.1.A.pdb

Needs: mini3di, biopython
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                "TYR": "Y", "VAL": "V"}

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, choices=["db1", "db2", "db3"])
ap.add_argument("--pdb", required=True)
ap.add_argument("--chain", default="A")
a = ap.parse_args()

try:
    import mini3di
    from Bio.PDB import PDBParser
except ImportError:
    raise SystemExit("Run: pip install mini3di biopython")

db = json.load(open(ROOT / "data" / f"{a.db}.json"))
W = db["window_residues"]
full = "".join((ROOT / "data" / "vgsc_wildtype.fasta").read_text().strip().splitlines()[1:])

chain = {c.get_id(): c for c in
         PDBParser(QUIET=True).get_structure("v", a.pdb).get_chains()}[a.chain]
residues = [r for r in chain.get_residues() if "CA" in r]
resnums = [r.get_id()[1] for r in residues]
enc = mini3di.Encoder()
tokens = enc.build_sequence(enc.encode_chain(chain))
assert len(tokens) == len(residues), "3Di token count does not match residue count"
num_to_tok = dict(zip(resnums, tokens))
num_to_aa = dict(zip(resnums, (THREE_TO_ONE.get(r.get_resname(), "X") for r in residues)))
first, last = resnums[0], resnums[-1]
print(f"sequence {len(full)} residues; structure covers {first}-{last}; window +/-{W}")

# ---------------------------------------------------------- collect the variants
# Each variant becomes a list of (scored_edit, background_edits).
variants = []
if a.db == "db1":
    for r in db["labelled"] + db["field_only"]:
        if r["kind"] == "single":
            e = dict(name=r["name"], seq_position=r["seq_position"], wt=r["wt"], mut=r["mut"])
            variants.append(dict(variant=r["name"], kind="single", label=r.get("label", "none"),
                                 group=r.get("group", "aedes"), edits=[e]))
        else:
            variants.append(dict(variant=r["name"], kind="combination", label=r.get("label", "none"),
                                 group=r.get("group", "aedes"), edits=r["edits"],
                                 vs_singles=r.get("vs_singles", "unknown")))
elif a.db == "db2":
    for v in db["variants"]:
        variants.append(dict(variant=v["name"], kind="single", label="none", group="db2",
                             in_db1=v["in_db1"],
                             edits=[dict(name=v["name"], seq_position=v["seq_position"],
                                         wt=v["wt"], mut=v["mut"])]))
else:
    for r in db["combinations"]:
        variants.append(dict(variant=r["name"], kind="combination", label=r.get("label", "none"),
                             group=r.get("origin", "db3"), edits=r["edits"],
                             vs_singles=r.get("vs_singles", "unknown")))

# ------------------------------------------------------------- build the jobs
jobs, problems, outside = [], [], 0
seen_single_pos = {}

for v in variants:
    for i, scored in enumerate(v["edits"]):
        p = scored["seq_position"]
        bg = [e for j, e in enumerate(v["edits"]) if j != i]

        # a single mutation needs only ONE job per position: the pass returns
        # every amino acid at that position, so later mutations reuse it
        if not bg:
            if p in seen_single_pos:
                continue
            seen_single_pos[p] = True

        start, end = max(first, p - W), min(last, p + W)
        if any(n not in num_to_tok for n in range(start, end + 1)):
            problems.append(f"{v['variant']}: gap in the structure inside window {start}-{end}")
            continue
        if full[p - 1] != scored["wt"] or num_to_aa.get(p) != scored["wt"]:
            problems.append(f"{v['variant']}: expected {scored['wt']} at {p}, sequence has "
                            f"{full[p-1]}, structure has {num_to_aa.get(p,'?')}")
            continue

        aa = list(full[start - 1:end])
        applied, dropped = [], []
        for e in bg:
            q = e["seq_position"]
            if not start <= q <= end:
                dropped.append(e["name"])
                continue
            if aa[q - start] != e["wt"]:
                problems.append(f"{v['variant']}: background {e['name']} expected "
                                f"{e['wt']} at {q}, window has {aa[q-start]}")
                break
            aa[q - start] = e["mut"]
            applied.append(e["name"])
        else:
            if dropped:
                outside += 1
            jobs.append(dict(
                job_id=f"{v['variant']}|{scored['name']}",
                variant=v["variant"], kind=v["kind"], label=v["label"], group=v["group"],
                scored=scored["name"], seq_position=p, wt=scored["wt"], mut=scored["mut"],
                background=applied, background_outside_window=dropped,
                window_start=start, window_end=end, local_index=p - start,
                aa_window="".join(aa),
                struct_window="".join(num_to_tok[n] for n in range(start, end + 1)),
            ))

# ------------------------------------------------------------------- report
if problems:
    print(f"\nSTOP. {len(problems)} job(s) failed the wild-type check, so nothing was written:")
    for p_ in problems[:20]:
        print("  -", p_)
    if len(problems) > 20:
        print(f"  ... and {len(problems)-20} more")
    raise SystemExit(1)

kinds = Counter(j["kind"] for j in jobs)
print(f"\nvariants: {len(variants)}")
print(f"jobs (forward passes): {len(jobs)}   "
      f"({kinds['single']} with no background, {kinds['combination']} conditional)")
print(f"  distinct positions covered with no background: {len(seen_single_pos)}")
if outside:
    print(f"  jobs whose background fell outside the window: {outside}")
    print("    these are forced to report zero interaction and must not be read as")
    print("    a biological prediction of independence")
wsz = {j["window_end"] - j["window_start"] + 1 for j in jobs}
print(f"  window length: {min(wsz)} to {max(wsz)} tokens")

out = ROOT / "data" / f"jobs_{a.db}.json"
json.dump(dict(db=a.db, reference=db["reference"], window_residues=W, jobs=jobs),
          open(out, "w"))
print(f"\nwritten: {out.relative_to(ROOT)}  ({out.stat().st_size/1e6:.1f} MB)")
print("Upload this file to Colab and run scripts/colab_score_db.py")
