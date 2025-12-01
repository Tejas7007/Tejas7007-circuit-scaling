import os
import pandas as pd


def main():
    in_path = "results/global_ioi_circuit_gpt2-medium.csv"
    out_path = "results/ioi_hero_heads_gpt2-medium.csv"

    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"{in_path} not found. Run analyze_global_ioi_circuit_gpt2medium.py first."
        )

    df = pd.read_csv(in_path)

    # Expecting columns: type, layer, head, mean_margin_ablated, mean_margin_base, delta_margin
    required_cols = {"type", "layer", "head", "mean_margin_ablated", "mean_margin_base", "delta_margin"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Unexpected columns in {in_path}. Got: {df.columns.tolist()}"
        )

    # Keep only attention heads
    attn_df = df[df["type"] == "attn"].copy()
    if attn_df.empty:
        raise ValueError("No attention heads found in global IOI circuit CSV (type == 'attn').")

    # IOI head strength: more negative delta_margin = stronger IOI
    # (ablating head decreases margin, so it's helping IOI)
    attn_df["ioi_score"] = attn_df["delta_margin"]

    # Sort by ioi_score ascending (most negative first) and take top 10
    attn_df = attn_df.sort_values("ioi_score").reset_index(drop=True)
    top_k = 10
    top_df = attn_df.head(top_k).copy()

    # Add tags like gpt2m_ioi_0, gpt2m_ioi_1, ...
    tags = [f"gpt2m_ioi_{i}" for i in range(len(top_df))]
    top_df["tag"] = tags

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    top_df.to_csv(out_path, index=False)

    print(f"[OUT] Saved top {len(top_df)} GPT-2-Medium IOI heads to {out_path}")
    print(top_df[["layer", "head", "ioi_score", "tag"]])


if __name__ == "__main__":
    main()

