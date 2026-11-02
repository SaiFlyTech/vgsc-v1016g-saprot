"""
07_smarter_conservation.py
A smarter version of the conservation comparison (a position-specific scoring matrix, "PSSM").

What is different from 06_conservation_baseline.py:
  1. Near-duplicate sequences are down-weighted, so one big family of look-alike proteins does not dominate.
  2. Changes nobody has seen are not all treated the same. An unseen change to a SIMILAR amino acid counts as
     more likely than an unseen change to a very DIFFERENT one (BLOSUM62-based pseudocounts, as in PSI-BLAST).

score = log( chance of the mutant letter at that position / chance of the original letter )
Lower = more unusual = more likely resistant (same direction as SaProt).

No GPU needed. Run on Windows, the same way as script 06:
  python scripts\\07_smarter_conservation.py --fasta data\\uniprotkb_taxonomy_id_50557_AND_length_2026_10_27.fasta.gz

Needs: biopython, pyyaml, numpy
"""
import argparse, gzip, math, re, statistics, sys
from pathlib import Path
import numpy as np
import yaml
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)
ap.add_argument("--ref", default=str(ROOT / "data" / "vgsc_wildtype.fasta"))
ap.add_argument("--yaml", default=str(ROOT / "config" / "mutations.yaml"))
ap.add_argument("--min-identity", type=float, default=0.60)
ap.add_argument("--min-coverage", type=float, default=0.70)
ap.add_argument("--exclude", default="Aedes aegypti")
ap.add_argument("--cluster-identity", type=float, default=0.97, help="sequences this similar share one vote")
ap.add_argument("--beta", type=float, default=10.0, help="weight of the BLOSUM pseudocounts (PSI-BLAST uses about 10)")
a = ap.parse_args()

AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {c: i for i, c in enumerate(AA)}
GAP = 20
B62 = substitution_matrices.load("BLOSUM62")
ref = str(next(SeqIO.parse(a.ref, "fasta")).seq).upper()
L = len(ref)

# ---- 1) align every sequence to the reference (same filters as script 06) ----
aligner = Align.PairwiseAligner()
aligner.substitution_matrix = B62
aligner.open_gap_score, aligner.extend_gap_score = -11, -1
aligner.mode = "global"

rows, seen = [], set()
dropped = {"excluded": 0, "duplicate": 0, "too different": 0, "too short": 0}
handle = gzip.open(a.fasta, "rt") if a.fasta.lower().endswith(".gz") else open(a.fasta)
for rec in SeqIO.parse(handle, "fasta"):
    seq = str(rec.seq).upper()
    if a.exclude and a.exclude.lower() in rec.description.lower():
        dropped["excluded"] += 1; continue
    if seq in seen:
        dropped["duplicate"] += 1; continue
    seen.add(seq)
    if len(seq) < 0.5 * L:
        dropped["too short"] += 1; continue
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq)
    aln = aligner.align(ref, seq)[0]
    row = np.full(L, GAP, dtype=np.int8)
    same = pairs = 0
    for (r0, r1), (s0, s1) in zip(*aln.aligned):
        for k in range(r1 - r0):
            c = seq[s0 + k]
            if c in IDX:
                row[r0 + k] = IDX[c]
            pairs += 1
            same += ref[r0 + k] == seq[s0 + k]
    ident, cov = (same / pairs if pairs else 0), pairs / L
    if ident < a.min_identity or cov < a.min_coverage:
        dropped["too different"] += 1; continue
    rows.append(row)

n = len(rows)
print(f"Sequences kept: {n}   dropped: {dropped}")
if n < 10:
    sys.exit("Fewer than 10 usable sequences. Check the FASTA file or lower --min-identity.")
M = np.vstack(rows)                       # n x L matrix of amino acid codes (20 = gap)

# ---- 2) weights: sequences that are near-copies of each other share one vote ----
print("Weighting near-duplicate sequences ...")
neighbours = np.zeros(n)
for i in range(n):
    both = (M != GAP) & (M[i] != GAP)
    same = (M == M[i]) & both
    ident = same.sum(1) / np.maximum(both.sum(1), 1)
    neighbours[i] = (ident >= a.cluster_identity).sum()      # includes itself
w = 1.0 / neighbours
print(f"Effective number of independent sequences: {w.sum():.0f} (of {n})\n")

# ---- 3) BLOSUM62-based pseudocounts: chance of letter b when letter a is seen ----
bg = dict(zip("ARNDCQEGHILKMFPSTWYV", [.078, .051, .045, .054, .019, .043, .063, .074, .022, .051,
                                      .091, .057, .022, .039, .052, .071, .058, .013, .032, .064]))
Q = np.zeros((20, 20))                    # Q[a][b] = P(b | a)
for ai, x in enumerate(AA):
    row = np.array([bg[y] * 2 ** (B62[x][y] / 2) for y in AA])
    Q[ai] = row / row.sum()

def column_probs(pos):                    # pos is 1-based
    col = M[:, pos - 1]
    ok = col != GAP
    wc = np.zeros(20)
    for c in range(20):
        wc[c] = w[ok & (col == c)].sum()
    neff = wc.sum()
    if neff == 0:
        return None, 0.0
    f = wc / neff
    g = f @ Q                             # what similar-letter evidence suggests
    p = (neff * f + a.beta * g) / (neff + a.beta)
    return p, neff, f

# ---- 4) score each mutation ----
muts = yaml.safe_load(open(a.yaml))["mutations"]
out = []
for m in muts:
    p = m.get("seq_position", m["position"])
    assert ref[p - 1] == m["wt"], f"{m['name']}: reference has {ref[p-1]} at {p}"
    res = column_probs(p)
    if res[0] is None:
        out.append(dict(name=m["name"], label=m["label"], group=m.get("group", "aedes"), neff=0, fwt=0, fmut=0, score=0.0, b62=B62[m["wt"]][m["mut"]]))
        continue
    pr, neff, f = res
    sc = math.log(pr[IDX[m["mut"]]] / pr[IDX[m["wt"]]])
    out.append(dict(name=m["name"], label=m["label"], group=m.get("group", "aedes"), neff=neff,
                    fwt=f[IDX[m["wt"]]], fmut=f[IDX[m["mut"]]], score=sc, b62=B62[m["wt"]][m["mut"]]))

print(f"{'mutation':8} {'label':14} {'group':14} {'indep.seqs':>10} {'orig %':>7} {'mutant %':>8} {'score':>7}")
for r in sorted(out, key=lambda r: r["score"]):
    warn = "  (!) few sequences cover this position" if r["neff"] < 10 else ""
    print(f"{r['name']:8} {r['label']:14} {r['group']:14} {r['neff']:10.0f} {100*r['fwt']:7.1f} {100*r['fmut']:8.2f} {r['score']:7.2f}{warn}")

def auc(sub, key):
    pos = [r[key] for r in sub if r["label"] == "resistant"]
    neg = [r[key] for r in sub if r["label"] == "not_resistant"]
    pr = [(p, q) for p in pos for q in neg]
    return sum(1.0 if p < q else 0.5 if p == q else 0.0 for p, q in pr) / len(pr), len(pos), len(neg)

print("\nP(resistant scores lower)  [0.50 = coin flip]")
for title, sub in (("Ae. aegypti only", [r for r in out if r["group"] == "aedes"]), ("All species", out)):
    s, np_, nn = auc(sub, "score"); b, _, _ = auc(sub, "b62")
    print(f"  {title:18} ({np_} resistant vs {nn} harmless): smarter conservation={s:.2f}  BLOSUM62={b:.2f}")
print("\nFor comparison: SaProt = 1.00 (mosquito-tested) and 0.75 (all species); the simple count scored 0.62 and 0.39.")
