#!/usr/bin/env python
from __future__ import annotations
import argparse
from circuitscaling.datasets import sva_minimal_pairs
from circuitscaling.models import load_model
from circuitscaling.utils import set_seed, dump_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['pythia-410m'])
    ap.add_argument('--n_prompts', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=777)
    args = ap.parse_args()

    set_seed(args.seed)
    pairs = sva_minimal_pairs(args.n_prompts)
    results = {}
    for m in args.models:
        model = load_model(m)
        # TODO: compute DLA at verb position; placeholder
        results[m] = {'n_pairs': len(pairs), 'note': 'implement DLA/scoring'}

    dump_json(results, 'results/sva_stub.json')
    print('Wrote results/sva_stub.json')

if __name__ == '__main__':
    main()
