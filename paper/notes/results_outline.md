# Results Outline

## 1. IOI vs Anti-repeat Correlation Across Scale
- Refer to `corr_vs_scale_cross_family.png`.
- Key points:
  - GPT-2 family: correlation positive and increases with size (small ~0.10 → large ~0.83).
  - Pythia family: small models have **negative** correlation (70m, 160m, 410m), 1B flips to positive (~0.30).
  - OPT and GPT-Neo-125M: moderate positive correlations (~0.15–0.38).

## 2. Number of Strong Heads vs Model Size
- Refer to `strong_head_scaling.png`.
- Key points:
  - Pythia has many strong heads at all scales (∼60–75).
  - GPT-2 has fewer strong heads as size increases, but the **alignment** between IOI and anti-repeat improves.

## 3. Layer-wise Distribution of IOI-only vs Anti-only
- Refer to `layer_hist_*` figures.
- Key points:
  - GPT2-large (τ = 0.03): IOI-only heads cluster in late layers (~21), anti-only in very late layers (~30+).
  - Pythia: IOI-only and anti-only heads appear across early and mid layers; shared heads concentrated in mid layers.

## 4. Phase Space of Head Types
- Refer to `all_heads_phase_space.png` and `family_grid.png`.
- Key points:
  - IOI-only ≈ (Δ_ioi < 0, Δ_anti ≈ 0).
  - Anti-only ≈ (Δ_anti < 0, Δ_ioi ≈ 0).
  - Shared heads in lower-left quadrant (Δ_ioi < 0, Δ_anti < 0).

## 5. Hero Heads for Case Studies
- Refer to `hero_heads_for_paper.csv`.
- GPT2-large:
  - Shared: L33H11, L34H2, L34H11.
- Pythia-160m:
  - IOI-only: L0H0, L1H9, L3H0.
  - Anti-only: L3H5, L4H0, L5H1.
  - Shared: L2H4, L3H4, L4H6.
- These will be used for detailed ablations and attention visualizations.

