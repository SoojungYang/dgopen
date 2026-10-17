"""Command line: `dgopen predict` and `dgopen best`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='dgopen', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict', help='per-residue dG_open with the 12-member ensemble')
    p.add_argument('--pdb', required=True)
    p.add_argument('--chain', default=None)
    p.add_argument('--out', required=True, help='output CSV')
    p.add_argument('--sse-method', default='torsion', choices=['torsion', 'psea'])
    p.add_argument('--weights-dir', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--members', action='store_true', help='also write per-member columns')
    b = sub.add_parser('best', help='Best-Vendruscolo baseline')
    b.add_argument('--pdb', required=True)
    b.add_argument('--chain', default=None)
    b.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'predict':
        from .predict import predict_structure
        df = predict_structure(a.pdb, chain=a.chain, sse_method=a.sse_method,
                               weights_dir=a.weights_dir, device=a.device,
                               return_members=a.members)
        df.to_csv(a.out, index=False, float_format='%.4f')
        meta = dict(df.attrs, output=a.out)
        print(json.dumps(meta, default=str), file=sys.stderr)
    else:
        from .baselines import best_vendruscolo
        best_vendruscolo(a.pdb, chain=a.chain).to_csv(a.out, index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
