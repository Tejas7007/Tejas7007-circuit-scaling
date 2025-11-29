#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, math
from circuitscaling.datasets import ioi_prompts
from circuitscaling.models import load_model
from circuitscaling.scoring import scan_heads_delta_loss
from circuitscaling.utils import set_seed, dump_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['pythia-410m'])
    ap.add_argument('--n_prompts', type=int, default=200)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--max_layers', type=int, default=None, help='Limit layers for quick scans (e.g., 6)')
    ap.add_argument('--topk', type=int, default=15, help='How many top candidates to record')
    args = ap.parse_args()

    set_seed(args.seed)
    prompts = ioi_prompts(args.n_prompts)

    os.makedirs("results", exist_ok=True)
    out = {}

    for m in args.models:
        print(f"==> Scanning heads for {m} ...")
        model = load_model(m, device=args.device)
        scores = scan_heads_delta_loss(model, prompts, device=args.device, max_layers=args.max_layers)

        # Record top-k "suppressor-like" heads (most negative Δloss) and top-k "helper" heads (most positive Δloss)
        clean_scores = [((L,H), float(d)) for (L,H), d in scores if not (isinstance(d, float) and math.isnan(d))]
        neg_sorted = sorted(clean_scores, key=lambda x: x[1])[:args.topk]
        pos_sorted = sorted(clean_scores, key=lambda x: x[1], reverse=True)[:args.topk]

        out[m] = {
            "n_prompts": len(prompts),
            "top_negative_delta_loss": [{"layer":L, "head":H, "delta_loss":d} for ( (L,H), d) in neg_sorted],
            "top_positive_delta_loss": [{"layer":L, "head":H, "delta_loss":d} for ( (L,H), d) in pos_sorted],
            "note": "Δloss is global. Negative may indicate suppressor-like behavior; refine with task-specific objective next."
        }

    dump_json(out, "results/copy_suppression_scan.json")
    print("Wrote results/copy_suppression_scan.json")

if __name__ == '__main__':
    main()
