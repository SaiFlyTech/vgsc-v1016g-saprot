# ---- COLAB CELL: (1) does the result survive different window sizes?  (2) leave-one-out accuracy ----
# Run AFTER the model-load cell (needs `tok`, `model`, `vocab`) and with the 10-mutation mutations_input.json in place.
import json, statistics, torch

data = json.load(open("mutations_input.json"))["mutations"]
WINDOWS = [50, 100, 250, 500]

def score(m, w):
    aa, st = m["aa_window"], m["struct_window"].lower()
    i = m["local_index"]
    lo, hi = max(0, i - w), min(len(aa), i + w + 1)      # shrink the stored window around the mutation
    aa, st, i = aa[lo:hi], st[lo:hi], i - lo
    assert aa[i] == m["wt"], m["name"]
    toks = [a + s for a, s in zip(aa, st)]
    toks[i] = "#" + st[i]
    enc = tok(" ".join(toks), return_tensors="pt").to(model.device)
    with torch.no_grad():
        lp = torch.log_softmax(model(**enc).logits[0, i + 1], dim=-1)
    return lp[vocab[m["mut"] + st[i]]].item() - lp[vocab[m["wt"] + st[i]]].item()

res = {w: {m["name"]: score(m, w) for m in data} for w in WINDOWS}
lab = {m["name"]: m["label"] for m in data}
pos = [n for n in lab if lab[n] == "resistant"]
neg = [n for n in lab if lab[n] == "not_resistant"]

print(f"{'mutation':8} {'label':14}" + "".join(f"{'w='+str(w):>9}" for w in WINDOWS))
for m in data:
    n = m["name"]
    print(f"{n:8} {lab[n]:14}" + "".join(f"{res[w][n]:9.2f}" for w in WINDOWS))

print("\n(1) P(resistant scores lower), by window size  [0.50 = coin flip]")
for w in WINDOWS:
    pairs = [(res[w][p], res[w][q]) for p in pos for q in neg]
    auc = sum(1.0 if a < b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
    print(f"  +/-{w:<4} {auc:.2f}")

print("\n(2) Leave-one-out decision accuracy: cut-off = midpoint of the two group averages of the OTHER mutations")
for w in WINDOWS:
    right = 0
    wrong = []
    for n in pos + neg:
        p_ = [res[w][x] for x in pos if x != n]
        q_ = [res[w][x] for x in neg if x != n]
        cut = (statistics.mean(p_) + statistics.mean(q_)) / 2
        pred = "resistant" if res[w][n] < cut else "not_resistant"
        if pred == lab[n]:
            right += 1
        else:
            wrong.append(n)
    print(f"  +/-{w:<4} {right}/{len(pos)+len(neg)} correct" + (f"   wrong: {', '.join(wrong)}" if wrong else ""))
