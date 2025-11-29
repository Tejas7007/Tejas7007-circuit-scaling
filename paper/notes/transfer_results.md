# IOI Head Transfer Across Scales and Model Families

We quantify how well IOI-like heads (category ∈ {ioi_only, shared}) **transfer** from a
base model to a target model. For each pair (base, target), we:

- Take the top K = 20 IOI-like heads in the base model ranked by |Δ_ioi|.
- For each such head (layer, head), look up its rank in the target model when all heads
  are sorted by |Δ_ioi|.
- Compute:
  - **mean_frac_rank**: mean relative rank in the target (0 = strongest, 1 = weakest).
  - **frac_missing**: fraction of base heads whose (layer, head) index does not exist
    in the target (e.g., different depth / width).

## Summary (from `ioi_transfer_generic.csv`)

- **Within Pythia (scaling ladder)**  
  - 70M → 160M: mean_frac_rank ≈ 0.50, moderate fraction of missing heads  
  - 160M → 410M: mean_frac_rank ≈ 0.44, substantial missing fraction  
  - 410M → 1B: mean_frac_rank ≈ 0.76, many heads missing or weak in 1B  

  Even within a single family with matched training data and architecture, only a
  subset of IOI-like heads remain strong at the **same (layer, head) index** across
  scales, and the fraction of "lost" indices grows with size.

- **Cross-family (similar parameter scale)**  
  - 160M (Pythia) → 125M (GPT–Neo): mean_frac_rank ≈ 0.54, many heads missing  
  - 160M (Pythia) → 125M (OPT): mean_frac_rank ≈ 0.70, most heads missing  
  - 410M (Pythia) → 355M (GPT-2 Medium): mean_frac_rank ≈ 0.57, most heads missing  

  IOI heads discovered in Pythia **do not** generally remain strong IOI heads at
  the same (layer, head) coordinates in GPT-2, GPT-Neo, or OPT. Both the fraction
  of missing indices and the degradation in relative rank are substantially worse
  than within-family scaling.

## Takeaways

1. IOI circuits are **not index-stable** even within a single scaling ladder.
2. Across families, IOI heads at a given (layer, head) are often either:
   - completely absent, or
   - weakly IOI-like compared to other heads in the target model.
3. Interpretability results at the granularity of specific attention heads
   **do not transfer robustly** across architectures and training pipelines.

