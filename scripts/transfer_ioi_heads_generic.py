import os
import math
import pandas as pd
import numpy as np

TABLE_PATH = "paper/tables/joint_ioi_anti_repeat_heads.csv"
OUT_PATH = "results/transfer/ioi_transfer_generic.csv"

# How many strongest IOI-like heads (by |delta_ioi|) to treat as "hero" in base
TOP_K = 20

# (family, model) pairs: (base) -> (target)
PAIR_SPECS = [
    # Within-family (sanity scaling checks)
    (("pythia", "pythia-70m"),   ("pythia", "pythia-160m")),
    (("pythia", "pythia-160m"),  ("pythia", "pythia-410m")),
    (("pythia", "pythia-410m"),  ("pythia", "pythia-1b")),

    # Cross-family comparisons at similar-ish scale
    (("pythia", "pythia-160m"),  ("gpt-neo", "gpt-neo-125M")),
    (("pythia", "pythia-160m"),  ("opt", "opt-125m")),
    (("pythia", "pythia-410m"),  ("gpt2", "gpt2-medium")),
]


def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find per-head table at {path}")
    df = pd.read_csv(path)
    required = {"family", "model", "layer", "head", "delta_ioi", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def get_model_heads(df_all: pd.DataFrame, family: str, model: str) -> pd.DataFrame:
    df = df_all[(df_all["family"] == family) & (df_all["model"] == model)].copy()
    if df.empty:
        raise ValueError(f"No rows for family={family}, model={model}")
    return df


def select_base_ioi_heads(df_model: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    IOI-like = category in {ioi_only, shared}.
    Sort by |delta_ioi| and take top_k.
    """
    mask = df_model["category"].isin(["ioi_only", "shared"])
    df_ioi = df_model[mask].copy()
    if df_ioi.empty:
        raise ValueError("No IOI-like heads (ioi_only/shared) in base model")

    df_ioi["abs_delta_ioi"] = df_ioi["delta_ioi"].abs()
    df_ioi = df_ioi.sort_values("abs_delta_ioi", ascending=False).reset_index(drop=True)
    return df_ioi.head(top_k)


def compute_transfer_for_pair(df_all, base_spec, target_spec, top_k=TOP_K):
    base_family, base_model = base_spec
    tgt_family, tgt_model = target_spec

    df_base = get_model_heads(df_all, base_family, base_model)
    df_tgt = get_model_heads(df_all, tgt_family, tgt_model)

    # Rank all heads in target by |Δ_ioi|
    df_tgt = df_tgt.copy()
    df_tgt["abs_delta_ioi"] = df_tgt["delta_ioi"].abs()
    df_tgt = df_tgt.sort_values("abs_delta_ioi", ascending=False).reset_index(drop=True)

    # Mapping: (layer, head) -> rank index (0 = strongest)
    tgt_total = len(df_tgt)
    tgt_index = {}
    for i, row in df_tgt.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        tgt_index[(layer, head)] = i

    df_base_top = select_base_ioi_heads(df_base, top_k)

    rows = []
    for _, row in df_base_top.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        base_abs = float(abs(row["delta_ioi"]))

        key = (layer, head)
        if key in tgt_index:
            tgt_rank = tgt_index[key]  # 0 is best, tgt_total-1 is worst
            tgt_frac_rank = (tgt_rank + 1) / tgt_total  # 0..1
            tgt_abs = float(abs(df_tgt.iloc[tgt_rank]["delta_ioi"]))
            missing = False
        else:
            # Head index does not exist in target (different #layers/#heads?)
            tgt_rank = math.nan
            tgt_frac_rank = math.nan
            tgt_abs = math.nan
            missing = True

        rows.append({
            "base_family": base_family,
            "base_model": base_model,
            "tgt_family": tgt_family,
            "tgt_model": tgt_model,
            "layer": layer,
            "head": head,
            "base_abs_delta_ioi": base_abs,
            "tgt_abs_delta_ioi": tgt_abs,
            "tgt_rank": tgt_rank,
            "tgt_frac_rank": tgt_frac_rank,  # 0 = very strong in target, 1 = very weak
            "missing_in_target_index": missing,
        })

    df_out = pd.DataFrame(rows)
    return df_out


def main():
    os.makedirs("results/transfer", exist_ok=True)
    df_all = load_table(TABLE_PATH)

    all_rows = []
    print("Using per-head table:", TABLE_PATH)
    print("Pairs:")
    for (b_f, b_m), (t_f, t_m) in PAIR_SPECS:
        print(f"  {b_f}/{b_m}  ->  {t_f}/{t_m}")
    print()

    for base_spec, tgt_spec in PAIR_SPECS:
        df_pair = compute_transfer_for_pair(df_all, base_spec, tgt_spec, top_k=TOP_K)
        all_rows.append(df_pair)

        base_family, base_model = base_spec
        tgt_family, tgt_model = tgt_spec

        n_heads = len(df_pair)
        n_missing = int(df_pair["missing_in_target_index"].sum())
        valid = df_pair[(~df_pair["missing_in_target_index"]) & (df_pair["tgt_frac_rank"].notna())]
        mean_frac_rank = valid["tgt_frac_rank"].mean() if not valid.empty else float("nan")

        print(f"=== {base_family}/{base_model}  →  {tgt_family}/{tgt_model} ===")
        print(f"  Base IOI-like heads (top K): {n_heads}")
        print(f"  Missing in target index    : {n_missing}")
        if not valid.empty:
            print(f"  Mean target frac rank      : {mean_frac_rank:.3f}  (0=best, 1=worst)")
        else:
            print("  No valid overlapping heads in target.")
        print()

    df_all_pairs = pd.concat(all_rows, ignore_index=True)
    df_all_pairs.to_csv(OUT_PATH, index=False)
    print(f"Wrote generic IOI transfer table to {OUT_PATH}")


if __name__ == "__main__":
    main()

