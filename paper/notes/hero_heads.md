# Hero heads (v1)

To ablate first:

- **GPT2-large shared head**
  - L34H2 – late-layer joint IOI + anti-repeat suppressor.

- **Pythia-160M IOI-only head**
  - L3H0 – strong IOI suppression, negligible anti-repeat effect.

- **Pythia-160M anti-only head**
  - L5H1 – strong repetition suppression, negligible IOI effect.

- **Pythia-160M shared head**
  - L4H6 – strong joint IOI + anti-repeat head; ideal for case study.

# Hero heads

## GPT2-large

- L34H2 (shared): late-layer suppressor; moderate Δ_ioi, Δ_anti. Candidate for detailed ablation / patching.
- L33H11, L34H11 (shared): similar late-layer behaviour.

## Pythia-160m

- IOI-only: L3H0, L0H0, L1H9 — "pure" IOI suppressors, no anti-repeat effect.
- Anti-only: L5H1, L3H5, L4H0 — "pure" repetition suppressors, no IOI effect.
- Shared: L4H6, L3H4, L2H4 — strong joint IOI + anti-repeat heads; ideal for case studies.

