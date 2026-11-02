# ---- COLAB CELL: robustness test. Add noise to the model's input and see if the ranking survives. ----
# Run AFTER the model-load cell (needs `tok`, `model`, `vocab`) and with the 12-mutation mutations_input.json in place.
# Noise is added to the SURROUNDING protein only. The mutated site itself is never changed.
import json, random, statistics, torch

data = json.load(open("mutations_input.json"))["mutations"]
AA = "ACDEFGHIKLMNPQRSTVWY"
SL = sorted(set("".join(m["struct_window"].lower() for m in data)))   # 3Di letters in use
LEVELS = [0.0, 0.10, 0.20, 0.40]       # share of surrounding positions changed
SEEDS = [0, 1, 2, 3, 4]                # repeat each noisy run 5 times with different random choices

def score(m, kind, level, seed):
    rng = random.Random(seed * 1000 + sum(map(ord, m["name"])) % 997)
    aa, st = list(m["aa_window"]), list(m["struct_window"].lower())
    i = m["local_index"]
    assert aa[i] == m["wt"], m["name"]
    others = [k for k in range(len(aa)) if k != i]
    pick = set(rng.sample(others, int(level * len(others)))) if level > 0 else set()
    for k in pick:
        if kind in ("sequence", "both"):
            aa[k] = rng.choice(AA)
        if kind in ("structure", "both"):
            st[k] = rng.choice(SL)
    site_st = st[i]
    if kind == "no_structure":                     # remove the 3D shape everywhere
        toks = [a + "#" for a in aa]
        toks[i] = "##"
        site_st = "#"
    else:
        toks = [a + s for a, s in zip(aa, st)]
        toks[i] = "#" + st[i]
    enc = tok(" ".join(toks), return_tensors="pt").to(model.device)
    with torch.no_grad():
        lp = torch.log_softmax(model(**enc).logits[0, i + 1], dim=-1)
    return lp[vocab[m["mut"] + site_st]].item() - lp[vocab[m["wt"] + site_st]].item()

def rank_auc(sc, groups):
    sub = [m for m in data if m.get("group", "aedes") in groups]
    pos = [sc[m["name"]] for m in sub if m["label"] == "resistant"]
    neg = [sc[m["name"]] for m in sub if m["label"] == "not_resistant"]
    pairs = [(p, q) for p in pos for q in neg]
    return sum(1.0 if a < b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)

SETS = {"Ae. aegypti only": ("aedes",), "All species": ("aedes", "other_species")}
print(f"{'noise type':14} {'level':>6}   " + "   ".join(f"{k:>22}" for k in SETS))
print(f"{'':21}   " + "   ".join(f"{'mean P (worst)':>22}" for _ in SETS))

def line(kind, level, runs):
    cells = []
    for g in SETS.values():
        a = [rank_auc(r, g) for r in runs]
        cells.append(f"{statistics.mean(a):10.2f} ({min(a):.2f})".rjust(22))
    lvl = "all" if level is None else f"{int(level*100)}%"
    print(f"{kind:14} {lvl:>6}   " + "   ".join(cells))

# 1) no structure at all
line("no structure", None, [{m["name"]: score(m, "no_structure", 0, 0) for m in data}])
# 2) noise in the sequence, in the structure, and in both
for kind in ("sequence", "structure", "both"):
    for level in LEVELS:
        seeds = [0] if level == 0 else SEEDS
        line(kind, level, [{m["name"]: score(m, kind, level, s) for m in data} for s in seeds])
print("\nP = chance a resistant mutation scores lower than a not-resistant one (1.00 = perfect, 0.50 = coin flip).")
print("Read it like this: if the number stays near 1.00 as noise grows, the ranking is robust. If it falls toward 0.50, the model needs the clean input.")
