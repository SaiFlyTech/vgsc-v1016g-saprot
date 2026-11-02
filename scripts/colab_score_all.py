# ---- COLAB CELL: score every mutation in mutations_input.json ----
# Run AFTER the model-load cell (needs `tok`, `model`, `vocab`) and after uploading mutations_input.json
import json, torch
from Bio.Align import substitution_matrices
B62 = substitution_matrices.load("BLOSUM62")

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
        lp = torch.log_softmax(model(**enc).logits[0, i + 1], dim=-1)
    saprot = lp[vocab[m["mut"] + st[i]]].item() - lp[vocab[m["wt"] + st[i]]].item()
    rows.append((m["name"], m["label"], round(saprot, 3), B62[m["wt"]][m["mut"]], m.get("group", "aedes")))

print(f"{'mutation':8} {'label':14} {'SaProt':>8} {'BLOSUM62':>9}  group")
for r in sorted(rows, key=lambda r: r[2]):
    print(f"{r[0]:8} {r[1]:14} {r[2]:8} {r[3]:9}  {r[4]}")

# Does the score separate resistant from not_resistant? (hold_aside is excluded)
def report(title, rows_):
    lab = [r for r in rows_ if r[1] in ("resistant", "not_resistant")]
    pos = [r for r in lab if r[1] == "resistant"]; neg = [r for r in lab if r[1] == "not_resistant"]
    def auc(k):  # chance a resistant one scores LOWER (more unusual) than a not_resistant one
        pairs = [(p[k], n[k]) for p in pos for n in neg]
        return sum(1.0 if p < n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)
    print(f"\n{title}: resistant={len(pos)}, not_resistant={len(neg)}")
    print(f"  P(resistant scores lower): SaProt={auc(2):.2f}  BLOSUM62={auc(3):.2f}  (0.5 = coin flip)")
report("(1) Ae. aegypti only (the tested-in-mosquito set)", [r for r in rows if r[4] == "aedes"])
report("(2) All species together", rows)

# How many other-species resistant mutations does SaProt flag? (score below EVERY not_resistant mutation)
neg_min = min(r[2] for r in rows if r[1] == "not_resistant")
others = [r for r in rows if r[4] == "other_species"]
if others:
    hit = [r[0] for r in others if r[2] < neg_min]
    miss = [r[0] for r in others if r[2] >= neg_min]
    print(f"\n(3) Other-species resistant mutations scoring below every not_resistant one ({neg_min}): {len(hit)} of {len(others)}")
    if miss:
        print("    not flagged:", ", ".join(miss))
