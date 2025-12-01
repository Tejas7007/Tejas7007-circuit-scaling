set -euo pipefail
echo "[INFO] Starting full circuit-transfer analysis batch..."
echo "[INFO] Python: $(python --version || echo 'python not found')"
echo "[INFO] PWD: $(pwd)"
echo "[STEP 1] IOI head transfer Pythia → GPT-2"
python scripts/transfer_ioi_heads_pythia_gpt2.py
python scripts/transfer_ioi_heads_generic.py
echo "[STEP 2] Pythia internal baseline transfer"
python scripts/analyze_pythia_baseline_transfer.py
echo "[STEP 3] Induction scores (Pythia + GPT-2)"
python scripts/compute_induction_scores.py

echo "[STEP 3] Induction head transfer Pythia → GPT-2"
python scripts/transfer_induction_heads_generic.py
echo "[STEP 4] GPT-2 IOI global scan (heads + MLPs)"
python scripts/summarize_global_ioi_gpt2medium.py

echo "[STEP 5A] CKA alignment Pythia410M ↔ GPT-2-Medium"
python scripts/compute_cka_alignment.py
echo "[STEP 5B] Functional clustering of heads"
python scripts/functional_clustering_heads.py
echo "[STEP 5C] Learned linear head alignment"
python scripts/learned_head_alignment.py
echo "[STEP 6] Expanded divergence taxonomy for IOI"
python scripts/divergence_taxonomy_expanded.py
python scripts/plot_divergence_taxonomy_expanded.py

echo "[STEP 7] Anti-repeat inversion case studies"
python scripts/anti_repeat_inversion_case_study.py
echo "[STEP 8] Tokenization effects on IOI"
python scripts/measure_tokenization_effects.py
echo "[STEP 9] Relative-depth alignment for IOI"
python scripts/relative_depth_alignment_ioi.py
echo "[STEP 10] IOI transfer threshold sensitivity"
python scripts/threshold_sensitivity_ioi_transfer.py

echo "[STEP 11] Summarize what (if anything) transfers"
python scripts/summarize_what_transfers.py
echo "[DONE] All existing analysis scripts completed successfully."
