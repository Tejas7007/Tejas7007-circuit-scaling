# Scaling of IOI and Anti-Repetition Circuits Across GPT-Style Model Families

## 1  Introduction

Large language models are now known to contain interpretable “circuits” implementing fairly specific behaviours, such as the *Indirect Object Identification* (IOI) task and induction-style copying. However, most mechanistic work studies one task in isolation, on a single model. We still lack a systematic picture of how multiple behaviours share or compete for the same attention heads, and how this picture changes as we scale model size and change architecture family.

In this work, we study the relationship between *IOI-suppressing* heads and *anti-repetition* heads across four small–medium GPT-style model families: **GPT-2**, **GPT-Neo**, **OPT**, and **Pythia**. For each attention head in each model, we measure two quantities:

- **Δ_ioi** – how much ablating that head harms performance on IOI prompts;
- **Δ_anti** – how much ablating that head *increases* repetition on a list-style anti-repeat task.

We then analyse how these per-head effects are distributed across heads, layers, models, and families.

Our key observations are:

1. The **correlation between Δ_ioi and Δ_anti** depends strongly on model family and scale. GPT-2’s correlation grows from mildly positive in gpt2 to strongly positive in gpt2-large, while Pythia shows a **non-monotonic pattern**: correlation becomes increasingly *negative* from 70M → 410M before flipping positive at 1B.
2. The **number of “strong” heads** (large |Δ_ioi| or |Δ_anti|) does **not simply grow with scale**. Pythia models have many strong heads even at 70M, and the count is roughly flat or slightly decreasing as we move to 1B.
3. The space of heads admits a clean **phase-space structure**: IOI-only heads lie along the Δ_ioi axis, anti-only heads along the Δ_anti axis, and “shared” heads live in a distinct joint quadrant where both metrics are strongly negative. Different families populate this phase space in recognisably different ways.
4. Within models, **layer-wise organisation** is non-trivial: in Pythia-160M, IOI-only heads concentrate earlier and anti-only heads somewhat later, with shared heads in mid-layers; in GPT-2, strong heads cluster later and become increasingly shared with scale.
5. From these distributions we identify a small set of **“hero heads”** for detailed analysis: a late shared head in gpt2-large and three complementary IOI-only / anti-only / shared heads in Pythia-160M.

Overall, our results support a picture in which *scale* and *architecture* change not just how many useful heads exist, but how **task roles are shared or disentangled** between them.

---

## 2  Background

### 2.1  IOI and anti-repetition behaviours

The **Indirect Object Identification (IOI)** task asks the model to resolve a pronoun to the correct name in sentences such as “When John and Mary went to the store, John gave a book to Mary because she had asked for it.” Previous work has identified IOI circuits in small transformers, showing how specific heads read the indirect object and steer the output logits.

Separately, **anti-repetition behaviour** is important for preventing degenerate looping on list or dialogue tasks. Models often learn heads that suppress copying the immediately previous token or phrase, especially in enumerations (“A, B, C, …”).

While both behaviours relate to *disambiguating* what to produce next, they act in quite different contexts. It is therefore natural to ask: do models implement them with **distinct heads**, or do some heads learn **joint IOI + anti-repeat roles**?

### 2.2  Model families

We study four widely-used GPT-style families:

- **GPT-2**: decoder-only transformer models (124M, 355M, 774M parameters).
- **GPT-Neo-125M**: a GPT-3-style architecture at 125M.
- **OPT-125m**: Meta’s OPT architecture at 125M parameters.
- **Pythia**: a suite of GPT-like models (70M, 160M, 410M, 1B) trained with a consistent data pipeline.

These families differ in depth, width, and training data, giving us a diverse testbed for how circuits scale.

---

## 3  Methods

### 3.1  Tasks and datasets

**IOI dataset.**  
We use a standard IOI-style dataset consisting of short stories with two names and a pronoun, where the indirect object must be chosen correctly. For each prompt we measure the model’s logit or log-probability assigned to the *correct* name vs the *incorrect* distractor, and define IOI performance accordingly.

**Anti-repeat dataset.**  
We construct list-style prompts where repetition is undesirable, e.g. “A, B, C, …” or “red, blue, green, …”. We then measure the model’s tendency to copy earlier tokens vs produce non-repeated continuations. Ablations that *increase* repetition are taken as evidence that the head is part of an anti-repeat circuit.

### 3.2  Per-head ablations

For each model, we iterate over all attention heads. Following standard mechanistic interpretability practice, we ablate a head by **zeroing its value vectors** (or equivalently, its output) at all positions while keeping everything else fixed. For each head we evaluate:

- IOI performance **with** and **without** the head;
- anti-repeat behaviour **with** and **without** the head.

This yields two signed quantities per head:

- Δ_ioi: change in IOI loss or logit margin when ablating the head;
- Δ_anti: change in repetition (or anti-repeat loss) when ablating the head.

Negative Δ values indicate that the head is **helpful** (ablating it hurts performance), while values near zero indicate weak or no effect.

These values are stored in `results/joint_ioi_anti_repeat_all.csv` and summarised per-model in `results/joint_ioi_anti_repeat_heads.csv`.

### 3.3  Categorising heads

We categorise heads based on **magnitude and selectivity** of their effects. Fixing a threshold τ (default τ = 0.05), we define:

- **IOI-only**: |Δ_ioi| ≥ τ and |Δ_anti| < τ.
- **Anti-only**: |Δ_anti| ≥ τ and |Δ_ioi| < τ.
- **Shared**: |Δ_ioi| ≥ τ and |Δ_anti| ≥ τ.
- **Weak**: both |Δ_ioi| and |Δ_anti| < τ.

We also explore a slightly lower threshold (τ = 0.03) to check robustness and to surface strong heads in larger models such as gpt2-large.

### 3.4  Aggregations and visualisations

We compute several summary statistics and plots:

- **Correlation vs scale**: for each model we compute the Pearson correlation between Δ_ioi and Δ_anti across heads, together with an approximate parameter count. (`paper/figs/corr_vs_scale_cross_family.png`)
- **Strong-head counts vs scale**: for each model we count IOI-only, anti-only, and shared heads and track how this count changes with parameter size. (`paper/figs/strong_head_scaling.png`)
- **Phase space plots**: for every head we plot (Δ_ioi, Δ_anti), coloured by category and marked by family, to visualise IOI-only, anti-only, shared, and weak regimes. (`paper/figs/all_heads_phase_space.png`, `paper/figs/family_grid.png`)
- **Layer histograms**: for each model we plot histograms of IOI-only, anti-only, and shared heads by layer. (`paper/figs/layer_hist_*`)

We then use these statistics and plots to identify **hero heads** for case studies, listed in `paper/tables/hero_heads_for_paper.csv` and summarised in `paper/notes/hero_heads.md`.

---

## 4  Results

### 4.1  IOI vs anti-repeat correlation across scale and family

Figure 1 (`corr_vs_scale_cross_family.png`) shows the correlation between Δ_ioi and Δ_anti for each model, plotted against approximate parameter count.

- For **GPT-2**, correlation is **positive and increases with scale**:
  - gpt2: ≈ 0.10  
  - gpt2-medium: ≈ 0.15  
  - gpt2-large: ≈ 0.83  
  This suggests that as GPT-2 scales, heads that help IOI increasingly also help anti-repeat, indicating **stronger sharing** between the circuits.
- **GPT-Neo-125M** and **OPT-125m** both show moderately positive correlation (~0.38 and ~0.15 respectively), closer to small GPT-2 than to gpt2-large.
- **Pythia** behaves quite differently. Correlation is **negative** at small and medium scales and becomes *more negative* as we go from 70M → 410M:
  - pythia-70m: ≈ −0.22  
  - pythia-160m: ≈ −0.38  
  - pythia-410m: ≈ −0.53  
  At 1B parameters, correlation flips to **+0.30**, indicating a qualitative change in how heads share IOI vs anti-repeat roles.

This non-monotonic pattern suggests that Pythia passes through a regime where **specialised IOI-only and anti-only heads are in tension** (strong negative correlation), before moving toward a more GPT-2-like shared regime at 1B.

### 4.2  Scaling of strong heads

We next ask whether larger models simply have **more strong heads**. Figure 2 (`strong_head_scaling.png`) plots the number of heads with |Δ_ioi| ≥ τ or |Δ_anti| ≥ τ as a function of model size.

Two patterns emerge:

1. **Pythia models already have many strong heads at small scale.** The 70M and 160M models have dozens of strong heads, and this count does not explode with scale; it is roughly flat or slightly decreasing toward 1B.
2. **GPT-2’s head count is relatively stable with scale**, even though the Δ_ioi–Δ_anti correlation increases sharply. This suggests that GPT-2 is not simply adding more specialised heads, but is instead **reusing a stable population of heads more coherently** across tasks.

In short, *scale changes how strongly heads are aligned across tasks more than it changes how many strong heads exist*.

### 4.3  Phase space of head roles

To understand individual head roles more directly, we plot every head in **(Δ_ioi, Δ_anti) space**.

Figure 3 (`all_heads_phase_space.png`) shows all heads coloured by category (IOI-only, anti-only, shared, weak) and marked by family. Figure 4 (`family_grid.png`) breaks this out by family.

Across families, we observe a clear **phase-space structure**:

- An **IOI-only band** along the negative Δ_ioi axis (Δ_anti ≈ 0).
- An **anti-only band** along the negative Δ_anti axis (Δ_ioi ≈ 0).
- A **shared quadrant** in the lower-left where both Δ_ioi and Δ_anti are negative.
- A dense cloud of **weak** heads around (0, 0).

Different families populate this phase space differently:

- **GPT-2** heads cluster near the axes, with relatively few extremely strong heads and a tight mass near the origin.
- **Pythia** heads spread further into the IOI-only and anti-only bands, especially in the 160M and 410M models, consistent with the strong negative correlations in Section 4.1.
- **OPT** and **GPT-Neo** occupy a smaller, more dispersed region, with some heads reaching far into the anti-only direction.

These patterns support the view that hero heads are **extreme points** in a continuous landscape of roles rather than isolated anomalies.

### 4.4  Layer-wise organisation

Figure 5 (`layer_hist_*`) shows, for each model, how many IOI-only, anti-only, and shared heads appear in each layer.

For **Pythia-160M**, we see:

- IOI-only heads concentrated in **earlier and mid layers** (e.g. layers 0–4).
- Anti-only heads appearing somewhat **later**, including layers 3–6 and 8–11.
- Shared heads clustering in **mid layers** (around layers 2–6), where both behaviours overlap.

For **Pythia-70M** and **Pythia-410M**, similar patterns hold, with some shifts in where the densest clusters appear.

In **GPT-2**, particularly gpt2-large (using τ = 0.03 for sensitivity), strong heads cluster toward **later layers**, and IOI-related and anti-repeat effects overlap more heavily. This matches the strong positive correlation observed earlier: late layers in large GPT-2 increasingly host **multi-task heads** that support both IOI and anti-repeat.

Overall, these histograms suggest that Pythia models prefer a **layer-wise separation** of IOI-only vs anti-only roles, while larger GPT-2 models move toward **late-layer sharing**.

### 4.5  Hero heads: IOI-only, anti-only, and shared

From the tables in `paper/tables/hero_heads_for_paper.csv` and our phase-space plots, we identify a small set of **hero heads** for closer inspection:

- **GPT2-large**
  - **L34H2** – a late shared head with moderate negative Δ_ioi and Δ_anti, representative of the strong joint correlation in gpt2-large.
- **Pythia-160M**
  - **L3H0 (IOI-only)** – large negative Δ_ioi with negligible Δ_anti, a “pure” IOI suppressor.
  - **L5H1 (anti-only)** – large negative Δ_anti with negligible Δ_ioi, a “pure” repetition suppressor.
  - **L4H6 (shared)** – strongly negative for both Δ_ioi and Δ_anti, a canonical shared head in the mid layers.

These heads sit at **different corners of the phase space**: IOI-only, anti-only, and shared. They also occupy distinct layers, making them an ideal set for future mechanistic case studies using causal tracing and path patching.

---

## 5  Discussion

Our analysis suggests that IOI and anti-repeat behaviours are implemented by a **structured population of attention heads** whose roles depend sensitively on both **scale** and **architecture family**.

In **Pythia**, heads move through a regime of strong *competition* between IOI-only and anti-only roles as we scale to a few hundred million parameters, before settling into a more **shared** regime at 1B. This could reflect a training dynamic where the model first discovers specialised solutions for each task and only later learns to compress them into shared components.

In **GPT-2**, by contrast, even small models exhibit mild positive correlation between IOI and anti-repeat roles, and gpt2-large has a strikingly high correlation. This suggests that GPT-2 learns to reuse heads aggressively: the same late-layer components support multiple behaviours, possibly improving parameter efficiency but also entangling circuits.

The phase-space and layer-wise patterns further support this picture. Pythia’s early/mid layer separation of IOI-only vs anti-only heads is reminiscent of modular feature pipelines, while GPT-2’s concentration of shared heads in later layers looks more like a **multi-task integration stage**.

These differences matter for **interpretability** and **control**. In a model with strongly separated IOI-only and anti-only heads, we might hope to modify one behaviour without affecting the other. In a model with heavily shared heads, interventions that improve one behaviour could more easily have unintended side-effects.

---

## 6  Limitations and future work

This study has several limitations:

- We focus on **small–medium** models; scaling trends may change again at much larger sizes.
- Our analysis covers only two behaviours (IOI and anti-repeat). Other circuits—induction, factual recall, safety-relevant behaviours—could interact in important ways.
- Our categorisation uses simple thresholds on |Δ_ioi| and |Δ_anti|. More sophisticated clustering or non-linear measures could reveal subtler structure.
- We rely on single-head ablations. Richer causal techniques (path patching, activation steering, fine-grained localization) are needed to fully pin down mechanisms.

Promising directions for future work include:

- Extending this analysis to **instruction-tuned** or **RLHF** models, where anti-repeat behaviour is heavily shaped by post-training.
- Adding more tasks to the phase-space view, turning (Δ_ioi, Δ_anti) into a higher-dimensional “behavioural fingerprint” per head.
- Performing **causal tracing** on the hero heads we identified (e.g. Pythia-160M L3H0, L5H1, L4H6 and GPT2-large L34H2) to map their upstream inputs and downstream logits paths.
- Training toy models with explicit IOI and anti-repeat objectives to test whether the same scaling patterns arise under controlled conditions.

---

## 7  Conclusion

We provide, to our knowledge, the first **cross-family, cross-scale** analysis of IOI and anti-repeat circuits in GPT-style language models. By measuring per-head effects on both tasks, we uncover a structured phase space of IOI-only, anti-only, shared, and weak heads, and show that how these roles are assigned depends strongly on **model family and parameter scale**.

Our findings suggest that scaling does not simply accumulate more specialised heads. Instead, models reorganise and increasingly **share circuits** across behaviours, with Pythia exhibiting a striking non-monotonic transition from negative to positive IOI/anti correlation.

We hope this work motivates further research on how circuits interact across behaviours and models, and on how to exploit these structures for safer and more controllable language models.

---

## Appendix A  Implementation details (sketch)

- All analysis code and results live in the `circuit-scaling` repository.
- Per-head Δ_ioi and Δ_anti measurements are stored in `results/joint_ioi_anti_repeat_all.csv`.
- Model-level summaries and hero-head tables are in `results/joint_ioi_anti_repeat_heads.csv` and `results/hero_heads_for_paper.csv`.
- Figures used in this draft are collected in `paper/figs/`.

