import json
import os

# Models we’ve actually scanned
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]

# Threshold for “strong” suppressors
STRONG_THRESH = -0.05  # Δ(logit-diff) < -0.1


def load_ioi(model_name):
    """
    Load IOI copy-suppression scan results for a given model.
    We assume filenames like:
      results/copy_suppression_logitdiff_scan_pythia-70m.json
    """
    fname = f"copy_suppression_logitdiff_scan_{model_name}.json"
    path = os.path.join("results", fname)

    if not os.path.exists(path):
        print(f"[IOI] Missing file for {model_name}: {path}")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    if model_name not in data:
        print(f"[IOI] No entry for {model_name} inside {fname}")
        return None

    return data[model_name]


def load_anti_repeat(model_name):
    """
    Load anti-repeat scan results for a given model.
    We assume filenames like:
      results/anti_repeat_logitdiff_scan_pythia-70m.json
    """
    fname = f"anti_repeat_logitdiff_scan_{model_name}.json"
    path = os.path.join("results", fname)

    if not os.path.exists(path):
        print(f"[Anti-Repeat] Missing file for {model_name}: {path}")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return data


def strong_suppressors_from_ioi(ioi_entry, thresh=STRONG_THRESH):
    """Return a set of (layer, head) with Δ < thresh from IOI scan."""
    heads = ioi_entry.get("top_suppressor_candidates", [])
    return {
        (h["layer"], h["head"])
        for h in heads
        if h["delta_logit_diff"] < thresh
    }


def strong_suppressors_from_anti(ar_entry, thresh=STRONG_THRESH):
    """Return a set of (layer, head) with Δ < thresh from anti-repeat scan."""
    heads = ar_entry.get("top_suppressor_candidates", [])
    return {
        (h["layer"], h["head"])
        for h in heads
        if h["delta_logit_diff"] < thresh
    }


def main():
    print("=== IOI vs Anti-Repeat Copy-Suppression Overlap ===\n")

    for m in MODELS:
        ioi = load_ioi(m)
        ar = load_anti_repeat(m)

        if ioi is None or ar is None:
            print(f"Skipping {m} (missing data)\n")
            continue

        ioi_strong = strong_suppressors_from_ioi(ioi)
        ar_strong = strong_suppressors_from_anti(ar)

        inter = sorted(ioi_strong & ar_strong)
        union = ioi_strong | ar_strong
        jaccard = len(inter) / len(union) if union else 0.0

        print(f"Model: {m}")
        print(f"  IOI strong suppressors (Δ < {STRONG_THRESH}): {sorted(ioi_strong)}")
        print(f"  Anti-repeat strong suppressors (Δ < {STRONG_THRESH}): {sorted(ar_strong)}")
        print(f"  Intersection: {inter}")
        print(f"  Jaccard overlap: {jaccard:.3f}\n")


if __name__ == "__main__":
    main()

