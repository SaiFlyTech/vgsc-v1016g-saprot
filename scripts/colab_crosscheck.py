# ---- COLAB CELL: cross-check the scores two ways ----
# Run AFTER the model-load cell (needs `tok`, `model`, `vocab`) and with the 9-mutation mutations_input.json in place.
# Way A = what we used so far: compare the mutant and the original amino acid, with the site's own structure letter.
# Way B = SaProt-style: add up the probability over ALL structure letters for each amino acid
#         (the structure letter is not trusted, so a wrong 3D model matters less).
import json, math, torch
from Bio.Align import substitution_matrices
B62 = substitution_matrices.load("BLOSUM62")
SLETTERS = [c for c in "abcdefghiklmnpqrstvwy" if ("A" + c) in vocab]   # structure letters present in the vocabulary

data = json.load(open("mutations_input.json"))["mutations"]
rows = []
for m in data:
    aa, st = m["aa_window"], m["struct_window"].lower()
    i = m["local_index"]
    assert aa[i] == m["wt"], m["name"]
    toks = [a + s for a, s in zip(aa, st)]
    toks[i] = "#" + st[i]
    enc = tok(" ".join(toks), return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, i + 1]
    lp = torch.log_softmax(logits, dim=-1)
    pr = lp.exp()
    A = lp[vocab[m["mut"] + st[i]]].item() - lp[vocab[m["wt"] + st[i]]].item()
    p_mut = sum(pr[vocab[m["mut"] + s]].item() for s in SLETTERS)
    p_wt = sum(pr[vocab[m["wt"] + s]].item() for s in SLETTERS)
    B = math.log(p_mut) - math.log(p_wt)
    rows.append((m["name"], m["label"], round(A, 3), round(B, 3), B62[m["wt"]][m["mut"]]))

print(f"{'mutation':8} {'label':14} {'A(ours)':>8} {'B(sum)':>8} {'BLOSUM62':>9}")
for r in sorted(rows, key=lambda r: r[2]):
    print(f"{r[0]:8} {r[1]:14} {r[2]:8} {r[3]:8} {r[4]:9}")

lab = [r for r in rows if r[1] in ("resistant", "not_resistant")]
pos = [r for r in lab if r[1] == "resistant"]; neg = [r for r in lab if r[1] == "not_resistant"]
def auc(k):
    pairs = [(p[k], n[k]) for p in pos for n in neg]
    return sum(1.0 if p < n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)
print(f"\nresistant={len(pos)}, not_resistant={len(neg)}")
print(f"P(resistant scores lower): A={auc(2):.2f}  B={auc(3):.2f}  BLOSUM62={auc(4):.2f}")
