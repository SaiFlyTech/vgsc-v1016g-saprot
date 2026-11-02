"""
15_baseline_model.py  -  the baseline model, and an honest evaluation of it.

Reads  data/combined_db1.json
Writes data/baseline_report.json

THE MODEL IS ONE THRESHOLD ON THE SaProt SCORE.
That is deliberate. With 13 labelled mutations there is room for one or two
fitted parameters, not for a classifier over 1,280-dimensional embeddings and
certainly not for fine-tuning a 650-million-parameter network. Anything larger
would fit noise and report a number nobody should believe. Every later model,
including the barcode model, has to beat this one.

TWO NUMBERS, MEASURED DIFFERENTLY, BECAUSE THEY EARN IT DIFFERENTLY.

  Ranking (AUC). Nothing is fitted, so the raw score is already an honest
  out-of-sample ranking. Reported with an exact permutation test over every
  possible relabelling, not an approximation.

  Accuracy. A threshold IS fitted, so it must be fitted without seeing the
  point it is tested on. Folds are by POSITION, never by mutation: position
  1011 holds I1011M (resistant) and I1011V (no effect), and 1016 holds V1016G
  and V1016I, so a random split would train on one and test on its neighbour
  and tell us nothing. Predictions from all folds are pooled before scoring,
  because with five negatives most single folds contain none.

The accuracy reported here will be lower than the 11-out-of-11 figure quoted
earlier in the project. That figure came from a threshold chosen after seeing
every score. This one is the number to use.

Run on Windows, no GPU:
    python scripts\\15_baseline_model.py

Needs: biopython (for the BLOSUM62 comparison). Everything else is standard library.
"""
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "combined_db1.json"
if not IN.exists():
    raise SystemExit("data/combined_db1.json is missing. Run 14_combine_scores.py --db db1 first.")

variants = json.load(open(IN))["variants"]

try:
    from Bio.Align import substitution_matrices
    B62 = substitution_matrices.load("BLOSUM62")
except Exception:
    B62 = None
    print("biopython not found, so the BLOSUM62 comparison is skipped\n")


# ----------------------------------------------------------------- statistics
def auc(pos, neg):
    """P(a resistant mutation scores LOWER than a no-effect one). 0.50 = coin flip."""
    if not pos or not neg:
        return None, 0
    pairs = [(p, q) for p in pos for q in neg]
    return sum(1.0 if p < q else 0.5 if p == q else 0.0 for p, q in pairs) / len(pairs), len(pairs)


def auc_permutation_p(scores, labels, observed, limit=300000):
    """Exact p-value by enumerating every relabelling when that is feasible."""
    n, k = len(scores), sum(labels)
    total = math.comb(n, k)
    idx = range(n)
    if total <= limit:
        hits = 0
        for combo in combinations(idx, k):
            s = set(combo)
            a, _ = auc([scores[i] for i in s], [scores[i] for i in idx if i not in s])
            hits += a >= observed - 1e-12
        return hits / total, total, True
    rng = random.Random(0)
    hits, n_draw = 0, 20000
    for _ in range(n_draw):
        combo = set(rng.sample(list(idx), k))
        a, _ = auc([scores[i] for i in combo], [scores[i] for i in idx if i not in combo])
        hits += a >= observed - 1e-12
    return hits / n_draw, total, False


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval, by bisection on the exact CDF. No scipy needed."""
    if n == 0:
        return (0.0, 1.0)
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if 1 - binom_cdf(k - 1, n, m) > alpha / 2:
                b = m
            else:
                a = m
        lo = a
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if binom_cdf(k, n, m) < alpha / 2:
                b = m
            else:
                a = m
        hi = b
    return lo, hi


def fit_threshold(train):
    """Pick the cut-off maximising Youden's J on the training fold. Lower score = resistant."""
    vals = sorted({r["score"] for r in train})
    cands = [vals[0] - 1.0] + [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] + [vals[-1] + 1.0]
    best, best_j = cands[0], -2.0
    npos = sum(r["y"] == 1 for r in train)
    nneg = len(train) - npos
    for t in cands:
        if npos == 0 or nneg == 0:
            continue
        tp = sum(r["y"] == 1 and r["score"] < t for r in train)
        tn = sum(r["y"] == 0 and r["score"] >= t for r in train)
        j = tp / npos + tn / nneg - 1
        if j > best_j:
            best, best_j = t, j
    return best


def evaluate(task, rows, feature="score"):
    """One task: ranking with a permutation p, then leave-one-position-out accuracy."""
    rows = [r for r in rows if r.get(feature) is not None]
    pos = [r[feature] for r in rows if r["y"] == 1]
    neg = [r[feature] for r in rows if r["y"] == 0]
    if not pos or not neg:
        return None
    a, npairs = auc(pos, neg)
    scores = [r[feature] for r in rows]
    labels = [r["y"] for r in rows]
    p, total, exact = auc_permutation_p(scores, labels, a)

    # leave-one-position-out
    by_pos = defaultdict(list)
    for r in rows:
        by_pos[r["position"]].append(r)
    preds = []
    for held in by_pos:
        train = [dict(score=r[feature], y=r["y"]) for q in by_pos if q != held for r in by_pos[q]]
        if not train or len({r["y"] for r in train}) < 2:
            continue
        t = fit_threshold(train)
        for r in by_pos[held]:
            preds.append((r["name"], r["y"], 1 if r[feature] < t else 0, t))
    correct = sum(y == yh for _, y, yh, _ in preds)
    lo, hi = clopper_pearson(correct, len(preds)) if preds else (0, 1)
    majority = max(Counter(r["y"] for r in rows).values()) / len(rows)

    return dict(task=task, feature=feature, n=len(rows), n_pos=len(pos), n_neg=len(neg),
                n_positions=len(by_pos), auc=a, n_pairs=npairs,
                auc_step=1.0 / npairs, perm_p=p, perm_total=total, perm_exact=exact,
                loo_correct=correct, loo_n=len(preds),
                loo_acc=correct / len(preds) if preds else None,
                loo_ci=(lo, hi), majority_baseline=majority,
                predictions=[dict(name=n, true=y, pred=yh, threshold=round(t, 3))
                             for n, y, yh, t in preds])


def show(res):
    if res is None:
        print("  not enough labels in both classes for this task")
        return
    print(f"  labels: {res['n']} ({res['n_pos']} resistant, {res['n_neg']} no effect) "
          f"across {res['n_positions']} positions")
    print(f"  ranking  P(resistant lower) = {res['auc']:.3f}   "
          f"(smallest possible step {res['auc_step']:.3f} over {res['n_pairs']} comparisons)")
    kind = "exact over all" if res["perm_exact"] else "sampled from"
    print(f"           permutation p = {res['perm_p']:.4f}   "
          f"({kind} {res['perm_total']:,} relabellings)")
    if res["loo_acc"] is None:
        print("  accuracy could not be estimated: a fold had only one class")
        return
    lo, hi = res["loo_ci"]
    print(f"  accuracy leave-one-position-out = {res['loo_correct']}/{res['loo_n']} "
          f"= {res['loo_acc']:.2f}   95% CI {lo:.2f} to {hi:.2f}")
    print(f"           always guessing the commoner class would give "
          f"{res['majority_baseline']:.2f}")
    wrong = [p["name"] for p in res["predictions"] if p["true"] != p["pred"]]
    if wrong:
        print(f"           got wrong: {', '.join(wrong)}")


# ------------------------------------------------------------------ the tasks
singles = [r for r in variants if r["kind"] == "single" and r.get("score") is not None]
aedes = [r for r in singles if r["group"] == "aedes"]

TASKS = {
    "overall, resistant vs no effect": lambda r: 1 if r["label"] == "resistant" else (0 if r["label"] == "no_effect" else None),
    "Type I pyrethroids":              lambda r: 1 if r.get("type_I") == "resistant" else (0 if r.get("type_I") == "no_effect" else None),
    "Type II pyrethroids":             lambda r: 1 if r.get("type_II") == "resistant" else (0 if r.get("type_II") == "no_effect" else None),
}

report = {}
print("=" * 72)
print("BASELINE MODEL: one threshold on the SaProt score")
print("Trained and tested only on mutations measured in Aedes aegypti channels")
print("=" * 72)

for task, fn in TASKS.items():
    rows = []
    for r in aedes:
        y = fn(r)
        if y is None:
            continue
        rows.append(dict(name=r["name"], position=r["position"], score=r["score"], y=y,
                         mut=r.get("mut"), wt=r.get("wt")))
    print(f"\n--- {task} ---")
    res = evaluate(task, rows)
    show(res)
    report[task] = res

    if B62 is not None:
        for r in rows:
            name = r["name"].replace("_AaNavS2", "")
            try:
                r["blosum"] = float(B62[name[0]][name[-1]])
            except Exception:
                r["blosum"] = None
        rb = evaluate(task + " (BLOSUM62)", rows, feature="blosum")
        if rb:
            print(f"  BLOSUM62 comparison: ranking {rb['auc']:.3f}, "
                  f"accuracy {rb['loo_acc']:.2f}" if rb["loo_acc"] is not None
                  else f"  BLOSUM62 comparison: ranking {rb['auc']:.3f}")
            report[task + " (BLOSUM62)"] = rb

# -------------------------------------------- held-out: other species, combos
print("\n" + "=" * 72)
print("HELD-OUT SETS, never used to fit anything")
print("=" * 72)

other = [r for r in singles if r["group"] == "other_species" and r["label"] == "resistant"]
negs = [r for r in aedes if r["label"] == "no_effect"]
if other and negs:
    a, npairs = auc([r["score"] for r in other], [r["score"] for r in negs])
    print(f"\nOther insects' channels: {len(other)} resistant vs the "
          f"{len(negs)} Aedes no-effect controls")
    print(f"  ranking P(resistant lower) = {a:.3f} over {npairs} comparisons")
    below = sum(all(r["score"] < n["score"] for n in negs) for r in other)
    print(f"  {below} of {len(other)} score below every no-effect control")
    report["other_species"] = dict(auc=a, n_pairs=npairs, n_pos=len(other), n_neg=len(negs),
                                   below_all=below)

combos = [r for r in variants if r["kind"] == "combination" and r.get("epistasis") is not None]
meas = [r for r in combos if r.get("vs_singles") in ("stronger", "same")]
if meas:
    st = [r["epistasis"] for r in meas if r["vs_singles"] == "stronger"]
    sa = [r["epistasis"] for r in meas if r["vs_singles"] == "same"]
    print(f"\nMeasured combinations: {len(st)} stronger than their strongest single, "
          f"{len(sa)} no stronger")
    if st and sa:
        a, npairs = auc(st, sa)
        print(f"  ranking P(stronger has more negative epistasis) = {a:.3f} "
              f"over {npairs} comparisons")
        report["combinations"] = dict(auc=a, n_pairs=npairs, n_stronger=len(st), n_same=len(sa))
    forced = [r["name"] for r in combos if r.get("interaction_forced_zero")]
    if forced:
        print(f"  interaction forced to zero by the window limit: {', '.join(forced)}")

# ------------------------------------------------------------------- the rule
best = report.get("overall, resistant vs no effect")
if best and best["loo_acc"] is not None:
    print("\n" + "=" * 72)
    print("WHAT TO QUOTE")
    print("=" * 72)
    ts = sorted({p["threshold"] for p in best["predictions"]})
    print(f"Ranking is the strong result: {best['auc']:.2f}, permutation p = {best['perm_p']:.4f}.")
    print(f"Accuracy is the weak one: {best['loo_acc']:.2f} "
          f"(95% CI {best['loo_ci'][0]:.2f} to {best['loo_ci'][1]:.2f}) on "
          f"{best['loo_n']} held-out mutations.")
    print(f"Thresholds chosen across folds ranged {min(ts):.2f} to {max(ts):.2f}; "
          f"the spread is itself a warning about how little data fixes it.")
    print("With this many labels the interval is wide enough that a yes/no verdict")
    print("should not be the product. A ranking with a percentile is defensible.")

json.dump(report, open(ROOT / "data" / "baseline_report.json", "w"), indent=1, default=str)
print(f"\nwritten: data/baseline_report.json")
