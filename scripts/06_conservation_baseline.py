"""
06_conservation_baseline.py
A tougher comparison for SaProt: how often does each change show up across insect sodium channels?

For every mutation we look at the same position in many insect sodium channel proteins.
  score = log( (times the mutant letter appears + 0.5) / (times the original letter appears + 0.5) )
A very low score means the mutant letter is almost never seen in nature, so the change is "unusual".
This is the same direction as SaProt: lower = more unusual = more likely resistant.

No GPU needed. Run on your Windows machine:
  python scripts\\06_conservation_baseline.py --fasta data\\insect_nav.fasta

Needs: biopython, pyyaml  (pip install biopython pyyaml)
"""
import argparse, gzip, re, statistics, sys
from pathlib import Path
import yaml
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True, help="FASTA of insect sodium channel proteins (downloaded from UniProt)")
ap.add_argument("--ref", default=str(ROOT / "data" / "vgsc_wildtype.fasta"), help="Aedes aegypti reference (C3RTM2)")
ap.add_argument("--yaml", default=str(ROOT / "config" / "mutations.yaml"))
ap.add_argument("--min-identity", type=float, default=0.60, help="keep only close relatives (para-type channels)")
ap.add_argument("--min-coverage", type=float, default=0.70, help="share of the reference the sequence must cover")
ap.add_argument("--exclude", default="Aedes aegypti", help="drop sequences whose header contains this text (our own species, to avoid circular counts)")
a = ap.parse_args()

B62 = substitution_matrices.load("BLOSUM62")
ref = str(next(SeqIO.parse(a.ref, "fasta")).seq).upper()

aligner = Align.PairwiseAligner()
aligner.substitution_matrix = B62
aligner.open_gap_score, aligner.extend_gap_score = -11, -1
aligner.mode = "global"

# ---- 1) align every sequence to the reference and keep the close relatives ----
kept, dropped = [], {"excluded": 0, "duplicate": 0, "too different": 0, "too short": 0, "bad letters": 0}
seen = set()
fasta_handle = gzip.open(a.fasta, "rt") if a.fasta.lower().endswith(".gz") else open(a.fasta)   # accepts the .gz file straight from UniProt
for rec in SeqIO.parse(fasta_handle, "fasta"):
    seq = str(rec.seq).upper()
    if a.exclude and a.exclude.lower() in rec.description.lower():
        dropped["excluded"] += 1; continue
    if seq in seen:
        dropped["duplicate"] += 1; continue
    seen.add(seq)
    if len(seq) < 0.5 * len(ref):
        dropped["too short"] += 1; continue
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq)
    aln = aligner.align(ref, seq)[0]
    col = {}                       # reference position (1-based) -> residue in this sequence
    same = pairs = 0
    for (r0, r1), (s0, s1) in zip(*aln.aligned):
        for k in range(r1 - r0):
            col[r0 + k + 1] = seq[s0 + k]
            pairs += 1
            same += ref[r0 + k] == seq[s0 + k]
    ident, cov = (same / pairs if pairs else 0), pairs / len(ref)
    if ident < a.min_identity or cov < a.min_coverage:
        dropped["too different"] += 1; continue
    kept.append((rec.id, ident, col))

print(f"Sequences read and kept: {len(kept)}   dropped: {dropped}")
if len(kept) < 10:
    sys.exit("Fewer than 10 usable sequences. Download more insect sodium channels (see the instructions) or lower --min-identity.")
ids = [k[1] for k in kept]
print(f"Identity to Aedes reference: median {statistics.median(ids):.2f}, min {min(ids):.2f}, max {max(ids):.2f}\n")

# ---- 2) score each mutation ----
import math
muts = yaml.safe_load(open(a.yaml))["mutations"]
rows = []
for m in muts:
    p = m.get("seq_position", m["position"])
    assert ref[p - 1] == m["wt"], f"{m['name']}: reference has {ref[p-1]} at {p}"
    letters = [k[2].get(p) for k in kept if k[2].get(p) not in (None, "X")]
    n = len(letters)
    c_wt, c_mut = letters.count(m["wt"]), letters.count(m["mut"])
    cons = math.log((c_mut + 0.5) / (c_wt + 0.5))
    rows.append(dict(name=m["name"], label=m["label"], group=m.get("group", "aedes"),
                     n=n, c_wt=c_wt, c_mut=c_mut, cons=cons, b62=B62[m["wt"]][m["mut"]]))
    rows[-1]["cons_b"] = cons + 0.01 * rows[-1]["b62"]     # BLOSUM62 only breaks ties between equal counts

print(f"{'mutation':8} {'label':14} {'group':14} {'seqs':>5} {'orig':>5} {'mutant':>6} {'score':>7}")
for r in sorted(rows, key=lambda r: r["cons_b"]):
    warn = "  (!) few sequences cover this position, ignore this score" if r["n"] < 10 else ""
    print(f"{r['name']:8} {r['label']:14} {r['group']:14} {r['n']:5d} {r['c_wt']:5d} {r['c_mut']:6d} {r['cons']:7.2f}{warn}")

# ---- 3) same ranking test as SaProt ----
def auc(sub, key):
    pos = [r[key] for r in sub if r["label"] == "resistant"]
    neg = [r[key] for r in sub if r["label"] == "not_resistant"]
    pr = [(p, q) for p in pos for q in neg]
    return sum(1.0 if p < q else 0.5 if p == q else 0.0 for p, q in pr) / len(pr), len(pos), len(neg)

print("\nP(resistant scores lower)  [0.50 = coin flip]")
for title, sub in (("Ae. aegypti only", [r for r in rows if r["group"] == "aedes"]), ("All species", rows)):
    c, np_, nn = auc(sub, "cons"); cb, _, _ = auc(sub, "cons_b"); b, _, _ = auc(sub, "b62")
    print(f"  {title:18} ({np_} resistant vs {nn} harmless): conservation={c:.2f}  conservation+BLOSUM tie-break={cb:.2f}  BLOSUM62={b:.2f}")
print("\nRead it like this: if conservation beats BLOSUM62 and matches or beats SaProt (1.00 / 0.75), it is a strong, simple baseline.")
