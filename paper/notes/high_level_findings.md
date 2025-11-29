# High-level findings (draft)

## 1. IOI vs anti-repeat correlation changes with model family and scale

- For **GPT-2**, the correlation between Δ_ioi and Δ_anti across heads is **positive and grows with scale**:
  - gpt2: corr ≈ 0.10  
  - gpt2-medium: corr ≈ 0.15  
  - gpt2-large: corr ≈ 0.83  
- For **GPT-Neo-125M** and **OPT-125m**, the correlation is **moderately positive** (≈ 0.15–0.38).
- For **Pythia**, the story is **non-monotonic**:
  - pythia-70m: corr ≈ −0.22  
  - pythia-160m: corr ≈ −0.38  
  - pythia-410m: corr ≈ −0.53  
  - pythia-1b: corr flips to **+0.30**
- So: in Pythia, increasing scale first strengthens a **tradeoff** between IOI and anti-repeat heads (more negative correlation), and only at 1B parameters do heads begin to align more strongly on both tasks.

(Use: `paper/figs/corr_vs_scale_cross_family.png`, `paper/tables/corr_vs_scale_cross_family.csv`)

---

## 2. Head counts: strong heads do not simply grow with scale

- Counting “strong” heads (|Δ_ioi| or |Δ_anti| ≥ τ, using τ=0.05) shows:
  - Pythia models have **many more strong heads** than similarly sized GPT-2 / GPT-Neo / OPT models.
  - Within Pythia, the number of strong heads **stays roughly flat or even decreases slightly** with scale from 70M → 410M → 1B.
  - GPT-2 adds parameters but does **not dramatically increase** the number of strong heads; instead, existing strong heads become **more aligned** across IOI and anti-repeat.
- Suggestive interpretation: rather than “more and more” dedicated heads, larger models may **re-use and specialize** a relatively stable population of strong heads.

(Use: `paper/figs/strong_head_scaling.png`, plus per-model layer histograms)

---

## 3. Phase space structure: IOI-only, anti-only, and shared regimes

- Plotting every head in (Δ_ioi, Δ_anti) space reveals a **structured phase space**:
  - An **IOI-only band** along the negative Δ_ioi axis near Δ_anti ≈ 0.
  - An **anti-only band** along the negative Δ_anti axis near Δ_ioi ≈ 0.
  - A **shared quadrant** where both Δ_ioi and Δ_anti are negative.
  - A large cloud of weak heads near (0, 0).
- Different families occupy this space differently:
  - GPT-2 heads cluster tightly near the axes (few extremely strong heads).
  - Pythia has a **wider spread** with more extreme IOI-only and anti-only heads, especially in mid-scale models.
  - OPT and GPT-Neo occupy a smaller but more dispersed subset of the anti-repeat direction.
- This suggests that **hero heads are not isolated anomalies** but lie at the edges of a continuous functional phase space.

(Use: `paper/figs/all_heads_phase_space.png`, `paper/figs/family_grid.png`)

---

## 4. Layer-wise structure: where do IOI-only / anti-only / shared heads live?

- In **Pythia-160M**:
  - **IOI-only heads** are concentrated in early and mid layers (e.g., L0–L4).
  - **Anti-only heads** are concentrated in slightly later layers (e.g., L3–L6, L8–L11).
  - **Shared heads** (strong on both IOI and anti-repeat) cluster in mid-layers (L2–L6).
- In **GPT-2**:
  - Most strong activity is skewed toward **later layers**, with IOI-related and anti-repeat-related behaviour overlapping more and more in gpt2-large.
- This supports a rough picture:
  - Early / mid layers in Pythia **separate IOI vs anti-repeat roles**, while
  - Larger GPT-2 models **merge** these roles into late “multi-task” heads.

(Use: all `paper/figs/layer_hist_*` plots)

---

## 5. Hero heads we will study causally

From `hero_heads_for_paper.csv` and manual inspection:

- **GPT2-large**
  - L34H2 – canonical shared head: suppresses both IOI mistakes and anti-repeat behaviour in a late layer.
- **Pythia-160M**
  - L3H0 – IOI-only hero: large Δ_ioi, negligible Δ_anti.
  - L5H1 – anti-only hero: large Δ_anti, negligible Δ_ioi.
  - L4H6 – shared hero: large on both Δ_ioi and Δ_anti.

Planned experiments (next steps):

- Single-head ablation on IOI vs anti-repeat vs matched control prompts.
- Causal tracing / activation patching to map where these heads read from and write to.
- Cross-family comparison: do GPT2-large late shared heads behave like Pythia shared mid-layer heads, or is the mechanism qualitatively different?

