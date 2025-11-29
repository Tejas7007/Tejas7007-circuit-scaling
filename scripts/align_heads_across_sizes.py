#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from circuitscaling.alignment import head_alignment_map
from circuitscaling.utils import dump_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', required=True, choices=['copy_suppression','sva','ioi'])
    ap.add_argument('--models', nargs='+', required=True)
    args = ap.parse_args()

    # TODO: load per-model discovered heads; stub below
    A = [(0,0),(1,3)]
    B = [(0,0),(2,5)]
    mapping = head_alignment_map(A,B)
    dump_json({'feature': args.feature, 'mapping': mapping}, 'results/alignment_stub.json')
    print('Wrote results/alignment_stub.json')

if __name__ == '__main__':
    main()
