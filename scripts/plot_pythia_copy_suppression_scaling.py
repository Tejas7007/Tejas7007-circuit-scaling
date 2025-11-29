import json
import os
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
]

# Map model name -> filename you actually saved
SCAN_FILES = {
    "pythia-70m":  "results/copy_suppression_logitdiff_scan_pythia-70m.json",
    "pythia-160m": "results/copy_suppression_logitdiff_scan_pythia-160m.json",
    "pythia-410m": "results/copy_suppression_logitdiff_scan_pythia-410m.json",
    "pythia-1b":   "results/copy_suppression_logitdiff_scan_pythia-1b.json",
}

def load_scan(model_name: str):
    path = SCAN_FILES[model_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scan file for {model_name}: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data[model_name]

def compute_stats():
    strong_counts = []
    medium_counts = []
    mean_layer_strong = []
    base_means = []

    for m in MODELS:
        scan = load_scan(m)
        base_means.append(scan["base_logit_diff_mean"])

        suppressors = scan["top_suppressor_candidates"]
        strong = [h for h in suppressors if h["delta_logit_diff"] < -0.10]
        medium = [h for h in suppressors if h["delta_logit_diff"] < -0.05]

        strong_counts.append(len(strong))
        medium_counts.append(len(medium))

        if strong:
            mean_layer_strong.append(sum(h["layer"] for h in strong) / len(strong))
        else:
            mean_layer_strong.append(float("nan"))

    return {
        "strong_counts": strong_counts,
        "medium_counts": medium_counts,
        "mean_layer_strong": mean_layer_strong,
        "base_means": base_means,
    }

def main():
    stats = compute_stats()
    x = np.arange(len(MODELS))

    # --- Figure 1: number of suppressors vs model size ---
    plt.figure()
    width = 0.35
    plt.bar(x - width/2, stats["strong_counts"], width, label="Δ < -0.10 (strong)")
    plt.bar(x + width/2, stats["medium_counts"], width, label="Δ < -0.05 (medium)")

    plt.xticks(x, MODELS, rotation=20)
    plt.ylabel("# suppressor heads (top-k)")
    plt.title("Copy Suppression Heads vs Model Size (IOI task)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/pythia_copy_suppression_counts.png", dpi=200)

    # --- Figure 2: mean layer of strong suppressors ---
    plt.figure()
    plt.plot(MODELS, stats["mean_layer_strong"], marker="o")
    plt.ylabel("Mean layer index (Δ < -0.10)")
    plt.title("Layer Depth of Strong IOI Suppressors vs Model Size")
    plt.tight_layout()
    plt.savefig("results/pythia_copy_suppression_layers.png", dpi=200)

    # --- Figure 3: base logit-diff mean ---
    plt.figure()
    plt.plot(MODELS, stats["base_means"], marker="o")
    plt.ylabel("Base logit-diff mean")
    plt.title("IOI Base Logit-Diff vs Model Size")
    plt.tight_layout()
    plt.savefig("results/pythia_copy_suppression_base_logitdiff.png", dpi=200)

    print("[+] Wrote plots to results/pythia_copy_suppression_*.png")

if __name__ == "__main__":
    main()

