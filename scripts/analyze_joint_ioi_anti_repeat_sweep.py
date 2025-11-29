import csv
import math
import os

INPUT_PATH = os.path.join("results", "joint_ioi_anti_repeat_pythia.csv")

# You can tweak these if you want to explore more thresholds
TAUS = [0.03, 0.05, 0.07, 0.10]


def load_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find CSV at {path}")
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Adjust these keys if your CSV uses slightly different names
            r["delta_ioi"] = float(r["delta_ioi"])
            r["delta_anti"] = float(r["delta_anti"])
            r["layer"] = int(r["layer"])
            r["head"] = int(r["head"])
            rows.append(r)
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def main():
    rows = load_rows(INPUT_PATH)
    models = sorted({r["model"] for r in rows})

    print("=== Threshold Sweep: IOI vs Anti-Repeat Copy Suppression (Pythia) ===\n")

    for tau in TAUS:
        print(f"---------- τ = {tau:.2f} ----------")
        for m in models:
            sub = [r for r in rows if r["model"] == m]
            if not sub:
                continue

            n_total = len(sub)
            ioi_only = anti_only = shared = weak = 0
            xs = []
            ys = []

            for r in sub:
                di = r["delta_ioi"]
                da = r["delta_anti"]
                xs.append(di)
                ys.append(da)

                ioi_sup = di < -tau
                anti_sup = da < -tau

                if ioi_sup and not anti_sup:
                    ioi_only += 1
                elif anti_sup and not ioi_sup:
                    anti_only += 1
                elif ioi_sup and anti_sup:
                    shared += 1
                else:
                    weak += 1

            corr = pearson(xs, ys)
            corr_str = "n/a" if corr is None else f"{corr:.3f}"

            print(f"Model: {m}")
            print(f"  heads in joint table: {n_total}")
            print(f"  IOI-only (Δ_ioi < -τ, Δ_anti ≥ -τ): {ioi_only}")
            print(f"  Anti-only (Δ_anti < -τ, Δ_ioi ≥ -τ): {anti_only}")
            print(f"  Shared (both Δ < -τ): {shared}")
            print(f"  Weak / none: {weak}")
            print(f"  Pearson corr(Δ_ioi, Δ_anti): {corr_str}")
        print()

    print("[+] Done threshold sweep.")


if __name__ == "__main__":
    main()

