#!/usr/bin/env python3
"""
Run all plotting scripts to generate paper-ready figures (PNG+PDF).
"""

import subprocess


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # Generic IOI transfer summary (existing script)
    try:
        run(["python", "-m", "scripts.plot_transfer_summary"])
    except Exception as e:
        print("[WARN] plot_transfer_summary failed:", e)

    # CKA real vs random
    run(["python", "-m", "scripts.plot_cka_real_vs_random_hist"])

    # Tokenization mismatch
    run(["python", "-m", "scripts.plot_tokenization_mismatch"])

    # GPT-2 IOI circuit (heads + MLPs)
    run(["python", "-m", "scripts.plot_gpt2medium_global_ioi_circuit"])

    # IOI divergence taxonomy
    run(["python", "-m", "scripts.plot_ioi_divergence_taxonomy"])

    # Relative depth alignment (scatter + hist)
    run(["python", "-m", "scripts.plot_relative_depth_alignment_ioi"])

    # IOI mass + depth
    run(["python", "-m", "scripts.plot_gpt2m_ioi_mass"])
    run(["python", "-m", "scripts.plot_gpt2m_ioi_depth_histogram"])

    # Path-patching top units
    run(["python", "-m", "scripts.plot_gpt2m_ioi_path_patching_top_units"])


if __name__ == "__main__":
    main()

