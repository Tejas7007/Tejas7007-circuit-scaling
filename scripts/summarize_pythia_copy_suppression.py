import json
import os
import csv

RESULTS_DIR = "results"

# These are the model names you used in the filenames:
MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
]


def load_model_entry(model_name: str):
    """
    Load the JSON for one model and return:
    base_logit_diff_mean, list of suppressor heads (with layer, head, delta_logit_diff).
    """
    # Preferred filename (what you already created with cp):
    fname = os.path.join(
        RESULTS_DIR,
        f"copy_suppression_logitdiff_scan_{model_name}.json",
    )
    # Fallback: plain scan file if the above doesn't exist
    if not os.path.exists(fname):
        fname = os.path.join(RESULTS_DIR, "copy_suppression_logitdiff_scan.json")
        if not os.path.exists(fname):
            print(f"[WARN] No results file found for {model_name}")
            return None

    with open(fname, "r") as f:
        data = json.load(f)

    if model_name not in data:
        # Some files might just have a single top-level key (e.g. "pythia-70m")
        # but if that key doesn't match, just pick the first one.
        first_key = next(iter(data.keys()))
        print(
            f"[WARN] Expected key '{model_name}' not in JSON. "
            f"Using first key '{first_key}' instead."
        )
        model_key = first_key
    else:
        model_key = model_name

    entry = data[model_key]
    base = float(entry["base_logit_diff_mean"])
    suppressors = entry["top_suppressor_candidates"]

    return base, suppressors


def compute_stats(model_name: str):
    loaded = load_model_entry(model_name)
    if loaded is None:
        return None

    base, suppressors = loaded

    # Thresholds for "strength"
    strong_thresh = -0.10
    medium_thresh = -0.05

    strong = [h for h in suppressors if h["delta_logit_diff"] < strong_thresh]
    medium = [h for h in suppressors if h["delta_logit_diff"] < medium_thresh]

    def mean_layer(head_list):
        if not head_list:
            return None
        return sum(h["layer"] for h in head_list) / len(head_list)

    stats = {
        "model": model_name,
        "base_logit_diff_mean": base,
        "n_topk_suppressors": len(suppressors),
        "n_strong_0_1": len(strong),
        "n_medium_0_05": len(medium),
        "mean_layer_strong": mean_layer(strong),
        "mean_layer_medium": mean_layer(medium),
        "top3": sorted(
            suppressors, key=lambda h: h["delta_logit_diff"]
        )[:3],  # most negative Δ
    }
    return stats


def main():
    all_stats = []

    print("\n=== Copy Suppression Summary Across Pythia Models ===")
    for m in MODELS:
        stats = compute_stats(m)
        if stats is None:
            continue
        all_stats.append(stats)

        print(f"\nModel: {m}")
        print(f"  base_logit_diff_mean: {stats['base_logit_diff_mean']:.4f}")
        print(f"  top-k suppressors (in JSON): {stats['n_topk_suppressors']}")
        print(f"  strong suppressors (Δ < -0.10): {stats['n_strong_0_1']}")
        print(f"  medium suppressors (Δ < -0.05): {stats['n_medium_0_05']}")
        ml_strong = stats["mean_layer_strong"]
        ml_medium = stats["mean_layer_medium"]
        print(
            f"  mean layer (strong): "
            f"{ml_strong:.2f}" if ml_strong is not None else "  mean layer (strong): n/a"
        )
        print(
            f"  mean layer (medium): "
            f"{ml_medium:.2f}" if ml_medium is not None else "  mean layer (medium): n/a"
        )
        print("  top 3 suppressors (by Δ):")
        for h in stats["top3"]:
            print(
                f"    L{h['layer']}H{h['head']}  "
                f"Δ={h['delta_logit_diff']:.4f}"
            )

    # Also dump to CSV for plotting later
    csv_path = os.path.join(RESULTS_DIR, "pythia_copy_suppression_summary.csv")
    fieldnames = [
        "model",
        "base_logit_diff_mean",
        "n_topk_suppressors",
        "n_strong_0_1",
        "n_medium_0_05",
        "mean_layer_strong",
        "mean_layer_medium",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            writer.writerow({k: s[k] for k in fieldnames})

    print(f"\n[+] Wrote summary CSV to {csv_path}")


if __name__ == "__main__":
    main()

