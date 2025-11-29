from __future__ import annotations
import matplotlib.pyplot as plt

def plot_layer_migration(x_sizes, y_layers, title:str):
    plt.figure()
    plt.plot(x_sizes, y_layers, marker='o')
    plt.title(title)
    plt.xlabel('Model size (params)')
    plt.ylabel('Layer index')
    plt.tight_layout()
    plt.savefig('results/figures/layer_migration.png')
