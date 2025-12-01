import json
import os
import csv

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]

def load_ioi(model_name):
    # Your IOI scan files you saved earlier
    # e.g. results/copy_suppression_logitdiff_scan_pythia-70m.json
    fname = os.path.join("results", f"copy_suppression_logitdiff_scan_{model_name}.json")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"IOI file not found: {fname}")
    with open(fname, "r") as f:
        data = json.load(f)
    return data[model_name]["top_suppressor_candidates"]

def load_anti(model_name):
    # Anti-repeat scan files, e.g. results/anti_repeat_logitdiff_scan_pythia-70m.json
    fname = os.path.join("results", f"anti_repeat_logitdiff_scan_{model_name}.json")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Anti-repeat file not found: {fname}")
    with open(fname, "r") as f:
        data = json.load(f)
    return data[model_name]["top_suppressor_candidates"]

def main():
    out_path = os.path.join("results", "joint_ioi_anti_repeat_pythia.csv")
    fieldnames = ["model", "layer", "head", "delta_ioi", "delta_anti"]

    rows = []

    for model in MODELS:
        ioi_heads = load_ioi(model)
        anti_heads = load_anti(model)

        ioi_dict = {}
        anti_dict = {}

        for h in ioi_heads:
            key = (h["layer"], h["head"])
            ioi_dict[key] = h["delta_logit_diff"]

        for h in anti_heads:
            key = (h["layer"], h["head"])
            anti_dict[key] = h["delta_logit_diff"]

        all_keys = set(ioi_dict.keys()) | set(anti_dict.keys())

        for (layer, head) in sorted(all_keys):
            delta_ioi = ioi_dict.get((layer, head), 0.0)
            delta_anti = anti_dict.get((layer, head), 0.0)
            rows.append({
                "model": model,
                "layer": layer,
                "head": head,
                "delta_ioi": delta_ioi,
                "delta_anti": delta_anti,
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Wrote joint IOI + anti-repeat summary to {out_path}")

if __name__ == "__main__":
    main()

