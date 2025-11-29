# Paper outline (v1)

## 1. Introduction

- Motivation: IOI circuits and anti-repetition circuits are both well-studied individually, but we don’t know:
  - How they **co-exist** in the same models,
  - How they **scale** with model size and family,
  - Whether they share **mechanistic components** (heads) or compete.
- Our high-level research questions:
  1. How are IOI and anti-repeat roles distributed across attention heads in small–medium LMs?
  2. How does this distribution **change with scale and architecture**?
  3. Are there **reusable “hero heads”** that jointly implement IOI and anti-repeat behaviour?
- Contributions (short list):
  - A unified dataset + metric for measuring per-head Δ_ioi and Δ_anti across four model families.
  - Cross-family scaling results showing:
    - Non-monotonic IOI/anti correlation in Pythia (negative → more negative → positive).
    - Stable or shrinking counts of strong heads despite scale, especially in Pythia.
  - A phase-space view of attention heads that separates IOI-only, anti-only, shared, and weak regimes.
  - Identification of **hero heads** (GPT2-large, Pythia-160M) for detailed mechanistic study.
  - (Planned) Causal ablation + tracing experiments on hero heads.

## 2. Background

- IOI task and previous IOI circuit work.
- Anti-repetition / anti-copying interventions in LMs.
- Prior work on **scaling of circuits** and per-head roles (cite Anthropic / Neel).
- Overview of model families:
  - GPT-2, GPT-Neo, OPT, Pythia – sizes, training data rough facts.

## 3. Methods

### 3.1 Tasks and datasets

- IOI dataset (brief description; how we generate or sample prompts).
- Anti-repeat dataset (lists, copy-vs-non-copy prompts).
- Evaluation metrics:
  - Δ_ioi = change in IOI logit / loss when head is ablated.
  - Δ_anti = change in repeat / non-repeat behaviour when head is ablated.
- Thresholding into categories: IOI-only, anti-only, shared, weak (τ=0.05; τ=0.03 for sensitivity).

### 3.2 Models and families

- Which checkpoints we use:
  - GPT-2 (small/medium/large), GPT-Neo-125M, OPT-125m, Pythia-70M/160M/410M/1B.
- How we run per-head interventions (very high level; details later in appendix).

### 3.3 Per-head measurement and categorisation

- How Δ_ioi and Δ_anti are computed per head.
- How we aggregate by model, family, and layer.
- Definitions of:
  - “Strong head” (|Δ| ≥ τ),
  - IOI-only, anti-only, shared, weak.

## 4. Results

### 4.1 Correlation of IOI vs anti-repeat across scale

- Main figure: `corr_vs_scale_cross_family.png`.
- Table: `corr_vs_scale_cross_family.csv`.
- Story: GPT-2 correlation increases with scale; Pythia is negative → more negative → positive; Neo/OPT positive but moderate.

### 4.2 How many strong heads? Scaling of head counts

- Figure: `strong_head_scaling.png`.
- Supporting: per-model layer histograms.
- Story: Pythia has many strong heads even at small scale; counts do not explode with parameters; suggests re-use / consolidation instead of proliferation.

### 4.3 Phase-space of head roles

- Figure: `all_heads_phase_space.png` + `family_grid.png`.
- Story: clean phase-space structure with clear IOI-only / anti-only / shared bands; families occupy it differently.

### 4.4 Layer-wise organisation

- Figures: all `layer_hist_*` plots.
- Story: different layer-wise organisation of IOI-only vs anti-only vs shared heads in Pythia vs GPT-2.

### 4.5 Hero heads (case studies) – **after we run ablations**

- Table: `hero_heads_for_paper.csv`.
- Plots (future): ablation curves / causal tracing for:
  - GPT2-large L34H2.
  - Pythia-160M L3H0 (IOI-only), L5H1 (anti-only), L4H6 (shared).
- Story we want to test:
  - Are shared heads literally doing “both jobs” mechanistically, or are they routing into downstream IOI-only / anti-only heads?

## 5. Discussion

- Interpreting non-monotonic scaling in Pythia.
- What it means for “feature sharing” vs “feature competition” between tasks.
- How this relates to circuits discovered by hand in prior work.
- Speculative implications for regularization, training objectives, or safety (e.g., how easy it is to switch off harmful repeat behaviour without destroying IOI).

## 6. Limitations and future work

- Limitations:
  - Only small–medium LMs; no instruction-tuned or RLHF models yet.
  - Only IOI + anti-repeat; other behaviours (induction heads, copy suppression, etc.) might interact.
  - Current analysis mostly linear ablation; need richer causal interventions.
- Future work:
  - Extend to larger models / mixture-of-experts.
  - Add more tasks (induction, factual recall) to the phase-space picture.
  - Train toy models with controlled IOI vs anti-repeat objectives.

## 7. Appendix

- Exact prompt distributions and dataset generation code.
- Full per-model head tables.
- Extra plots at different τ.
- Implementation details for per-head ablations and causal tracing.

