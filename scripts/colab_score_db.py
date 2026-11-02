# ---- COLAB CELL: score a prepared database with SaProt. Needs a T4 (or any GPU). ----
#
# Run AFTER the model-load cell, so that `tok`, `model` and `vocab` already exist.
# Upload the jobs file first, for example data/jobs_db1.json, then set DB below.
#
# One job is one masked forward pass. A masked pass returns a probability for
# EVERY token at that position, so each pass gives all 20 amino acids at once.
# That is why 456 variants in database 2 cost only 24 passes.
#
# Writes scores_<db>.json, which you download and feed to 14_combine_scores.py.

import json, time, torch
from collections import defaultdict

DB = "db1"              # "db1", "db2" or "db3"
BATCH = 8               # lower this if the GPU runs out of memory
AA = "ACDEFGHIKLMNPQRSTVWY"

jobs = json.load(open(f"jobs_{DB}.json"))["jobs"]
print(f"{len(jobs)} jobs to run for {DB}")

# group by window length so a batch needs no padding
groups = defaultdict(list)
for j in jobs:
    groups[len(j["aa_window"])].append(j)
print(f"{len(groups)} distinct window lengths")

results, t0, done = {}, time.time(), 0

for L, group in sorted(groups.items()):
    for s in range(0, len(group), BATCH):
        batch = group[s:s + BATCH]
        texts = []
        for j in batch:
            aa = list(j["aa_window"])
            st = list(j["struct_window"].lower())
            i = j["local_index"]
            assert aa[i] == j["wt"], f"{j['job_id']}: window has {aa[i]}, expected {j['wt']}"
            toks = [x + y for x, y in zip(aa, st)]
            toks[i] = "#" + st[i]                 # mask the amino acid, keep the structure
            texts.append(" ".join(toks))

        enc = tok(texts, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits

        for k, j in enumerate(batch):
            i = j["local_index"]
            site_st = j["struct_window"][i].lower()
            lp = torch.log_softmax(logits[k, i + 1], dim=-1)   # +1 skips the CLS token
            per_aa = {}
            for c in AA:
                key = c + site_st
                if key in vocab:
                    per_aa[c] = lp[vocab[key]].item()
            results[j["job_id"]] = dict(
                variant=j["variant"], scored=j["scored"], wt=j["wt"], mut=j["mut"],
                seq_position=j["seq_position"], background=j["background"],
                background_outside_window=j["background_outside_window"],
                kind=j["kind"], label=j["label"], group=j["group"],
                logp=per_aa,
                score=per_aa[j["mut"]] - per_aa[j["wt"]] if j["mut"] in per_aa else None,
            )
        done += len(batch)
        if done % 40 < BATCH or done == len(jobs):
            el = time.time() - t0
            print(f"  {done}/{len(jobs)}   {el:.0f}s elapsed, "
                  f"{el/done*(len(jobs)-done):.0f}s left")

print(f"\nfinished {len(results)} jobs in {time.time()-t0:.0f}s")

out = f"scores_{DB}.json"
json.dump(results, open(out, "w"))
print(f"written {out}")

# a quick look at the labelled singles, so you can see it worked before downloading
lab = [(r["variant"], r["score"], r["label"]) for r in results.values()
       if r["kind"] == "single" and r["label"] in ("resistant", "no_effect")]
if lab:
    print(f"\n{'variant':10} {'score':>8}  label")
    for name, sc, lb in sorted(lab, key=lambda x: x[1]):
        print(f"{name:10} {sc:8.2f}  {lb}")
    pos = [s for _, s, l in lab if l == "resistant"]
    neg = [s for _, s, l in lab if l == "no_effect"]
    if pos and neg:
        pairs = [(p, q) for p in pos for q in neg]
        auc = sum(1.0 if p < q else 0.5 if p == q else 0.0 for p, q in pairs) / len(pairs)
        print(f"\nP(resistant scores lower) = {auc:.2f} over {len(pairs)} comparisons "
              f"({len(pos)} resistant vs {len(neg)} no_effect); 0.50 = coin flip")
        print("This is the in-sample number. The honest one comes from "
              "14_combine_scores.py and 15_baseline_model.py.")
