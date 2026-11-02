# ---- COLAB CELL: extract SaProt embeddings for the labelled mutations. Needs the GPU. ----
#
# Run AFTER the model-load cell (needs `tok`, `model`, `vocab`).
# Upload jobs_db1.json AND db1.json first.
#
# For each labelled Aedes single mutation we build its window twice -- once with the
# wild-type residue in place, once with the mutant -- run the model WITHOUT masking,
# and read the 1280-d hidden vector at that position from the last layer. Those two
# vectors are the features a classifier can learn from. Nothing is trained here; this
# only pulls the representations out of the frozen model.
#
# Writes embeddings_db1.json, which you download and feed to 16_train_embeddings.py.

import json, torch

jobs = {j["seq_position"]: j for j in json.load(open("jobs_db1.json"))["jobs"]
        if not j["background"] and not j["background_outside_window"]}   # one clean window per position
db1  = json.load(open("db1.json"))
labelled = [r for r in db1["labelled"]
            if r["kind"] == "single" and r["group"] == "aedes"
            and r["label"] in ("resistant", "no_effect")]
print(f"{len(labelled)} labelled Aedes singles to embed")

def embed(aa_list, struct, i):
    toks = [a + s for a, s in zip(aa_list, struct)]           # no masking: real residue in place
    enc = tok(" ".join(toks), return_tensors="pt").to(model.device)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states[-1]   # [1, L+2, 1280]
    return hs[0, i + 1].float().cpu().tolist()                # +1 skips the CLS token

out = {}
for r in labelled:
    j = jobs[r["seq_position"]]
    st = list(j["struct_window"].lower())
    i  = j["local_index"]
    base = list(j["aa_window"]); assert base[i] == r["wt"], r["name"]
    wt_aa  = base[:]; wt_aa[i]  = r["wt"]
    mut_aa = base[:]; mut_aa[i] = r["mut"]
    out[r["name"]] = dict(
        label=r["label"], position=r["position"],
        type_I=r.get("type_I"), type_II=r.get("type_II"),
        emb_wt =embed(wt_aa,  st, i),
        emb_mut=embed(mut_aa, st, i),
    )
    print(f"  embedded {r['name']}")

json.dump(out, open("embeddings_db1.json", "w"))
print(f"\nwrote embeddings_db1.json  ({len(out)} mutations, 1280-d x 2 each)")
