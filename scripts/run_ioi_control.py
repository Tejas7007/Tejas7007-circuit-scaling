#!/usr/bin/env python
from __future__ import annotations
import argparse
from circuitscaling.datasets import ioi_prompts
from circuitscaling.models import load_model
from circuitscaling.utils import set_seed, dump_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['gpt2-small'])
    ap.add_argument('--n_prompts', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=777)
    args = ap.parse_args()

    set_seed(args.seed)
    prompts = ioi_prompts(args.n_prompts)
    results = {}
    for m in args.models:
        model = load_model(m)
        results[m] = {'n_prompts': len(prompts), 'note': 'implement IOI control pipeline'}

    dump_json(results, 'results/ioi_stub.json')
    print('Wrote results/ioi_stub.json')

if __name__ == '__main__':
    main()
