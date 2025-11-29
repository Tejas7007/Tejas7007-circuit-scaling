import csv
import os
import math
from collections import defaultdict

IN_PATH = os.path.join("results", "joint_ioi_anti_repeat_pythia.csv")

THRESH = 0.05  # significance threshold for "strong-ish" suppression

def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Joint CSV not found at {IN_PATH}")

    per_model = defaultdict(list)

    with open(IN_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            layer = int(row["layer"])
            head = int(row["head"])
            delta_ioi = float(row["delta_ioi"])
            delta_anti = float(row["delta_anti"])
            per_model[model].append(
                {
                    "layer": layer,
                    "head": head,
                    "delta_ioi": delta_ioi,
                    "delta_anti": delta_anti,
                }
            )

    print("=== IOI vs Anti-Repeat Joint Analysis (Pythia) ===\n")

    for model, heads in per_model.items():
        n_heads = len(heads)
        ioi_only = 0
        anti_only = 0
        shared = 0
        none = 0

        xs = []
        ys = []

        for h in heads:
            d_i = h["delta_ioi"]
            d_a = h["delta_anti"]

            xs.append(d_i)
            ys.append(d_a)

            ioi_strong = d_i < -THRESH
            anti_strong = d_a < -THRESH

            if ioi_strong and anti_strong:
                shared += 1
            elif ioi_strong and not anti_strong:
                ioi_only += 1
            elif anti_strong and not ioi_strong:
                anti_only += 1
            else:
                none += 1

        r = pearson(xs, ys)

        print(f"Model: {model}")
        print(f"  total heads in joint table: {n_heads}")
        print(f"  IOI-only suppressors (Δ_ioi < -{THRESH}, Δ_anti >= -{THRESH}): {ioi_only}")
        print(f"  Anti-only suppressors (Δ_anti < -{THRESH}, Δ_ioi >= -{THRESH}): {anti_only}")
        print(f"  Shared suppressors (both Δ < -{THRESH}): {shared}")
        print(f"  Weak / none: {none}")
        print(f"  Pearson corr(Δ_ioi, Δ_anti): {r:.3f}\n")

if __name__ == "__main__":
    main()

