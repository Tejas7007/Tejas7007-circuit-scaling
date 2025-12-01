#!/usr/bin/env python3
"""
Functional Clustering of Attention Heads

Clusters heads across all model families using:
- IOI and Anti-repeat scores
- Layer
- Head index
- Model family (one-hot)

Produces PCA + t-SNE visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

TABLE_PATH = "paper/tables/joint_ioi_anti_repeat_heads.csv"


def load_table():
    print("[INFO] Loading joint IOI/Anti-Repeat table...")
    return pd.read_csv(TABLE_PATH)


def build_feature_matrix(df):
    print("[INFO] Building feature matrix...")

    num = df[[
        "delta_ioi", "delta_anti",
        "abs_delta_ioi", "abs_delta_anti",
        "layer", "head"
    ]].values

    enc = OneHotEncoder(sparse_output=False)
    cat = enc.fit_transform(df[["family", "model"]])

    X = np.concatenate([num, cat], axis=1)
    return StandardScaler().fit_transform(X)


def plot_pca(X, df):
    print("[INFO] Running PCA...")
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=df["abs_delta_ioi"], cmap="coolwarm", s=12)
    plt.colorbar(sc, label="abs_delta_ioi")
    plt.title("PCA of Functional Head Features")

    out = "figs/functional_clustering_heads_pca.png"
    plt.savefig(out, dpi=200)
    print("[OUT] Saved PCA:", out)
    plt.close()


def plot_tsne(X, df):
    print("[INFO] Running t-SNE (may take ~10s)...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=2000,
        init="pca"
    )

    Z = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=df["abs_delta_ioi"], cmap="coolwarm", s=10)
    plt.colorbar(sc, label="abs_delta_ioi")
    plt.title("t-SNE of Functional Head Features")

    out = "figs/functional_clustering_heads_tsne.png"
    plt.savefig(out, dpi=200)
    print("[OUT] Saved t-SNE:", out)
    plt.close()


def save_cluster_assignments(X, df):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    res = df.copy()
    res["pca_x"] = Z[:, 0]
    res["pca_y"] = Z[:, 1]

    out = "results/functional_clustering_heads_assignments.csv"
    res.to_csv(out, index=False)
    print("[OUT] Saved cluster assignments:", out)


def main():
    df = load_table()
    X = build_feature_matrix(df)
    plot_pca(X, df)
    plot_tsne(X, df)
    save_cluster_assignments(X, df)
    print("[DONE] Functional clustering complete.")


if __name__ == "__main__":
    main()

