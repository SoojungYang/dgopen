"""Predict the 13 HDX-NMR domains with dgopen and the Best-Vendruscolo baseline, and compare
both with the measured per-residue dG_open. Writes nmr_comparison.csv and nmr_rho.csv here.
Usage: python examples/hdx_nmr/run_nmr_examples.py [--device cuda]   (about 1 min on CPU)."""
import argparse, os
import pandas as pd
from scipy.stats import spearmanr
import dgopen

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--device', default=None); a = ap.parse_args()
meas = pd.read_csv(os.path.join(here, 'measured_dgopen_nmr.csv'))
rows, rho = [], []
for dom, m in meas.groupby('domain', sort=True):
    pdb = os.path.join(here, 'structures', f'{dom}.pdb')
    p = dgopen.predict_structure(pdb, device=a.device)
    b = dgopen.best_vendruscolo(pdb)[['resi', 'dg_open_kcal']].rename(columns={'dg_open_kcal': 'best_kcal'})
    t = p[['resi', 'aa1', 'exchangeable', 'after_chain_break', 'dg_open_kcal', 'dg_open_member_sd']] \
        .rename(columns={'dg_open_kcal': 'dgopen_kcal', 'dg_open_member_sd': 'dgopen_member_sd'}) \
        .merge(b, on='resi').merge(m.drop(columns=['aa1', 'domain', 'in_finetune_set']), on='resi', how='left')
    t.insert(0, 'domain', dom)
    t['in_finetune_set'] = bool(m.in_finetune_set.iloc[0])
    rows.append(t)
    s = t.dropna(subset=['dg_open_measured_kcal', 'dgopen_kcal'])
    rho.append(dict(domain=dom, n=len(s), held_out=not bool(m.in_finetune_set.iloc[0]),
                    rho_dgopen=spearmanr(s.dgopen_kcal, s.dg_open_measured_kcal).statistic,
                    rho_best=spearmanr(s.best_kcal, s.dg_open_measured_kcal).statistic))
    print(f'{dom:26s} n={len(s):3d}  rho dgopen {rho[-1]["rho_dgopen"]:+.2f}  BEST {rho[-1]["rho_best"]:+.2f}', flush=True)
out = pd.concat(rows); out.to_csv(os.path.join(here, 'nmr_comparison.csv'), index=False, float_format='%.4f')
r = pd.DataFrame(rho); r.to_csv(os.path.join(here, 'nmr_rho.csv'), index=False, float_format='%.4f')
print(f'macro rho over {len(r)} domains: dgopen {r.rho_dgopen.mean():.3f}, BEST {r.rho_best.mean():.3f}; '
      f'held-out 3: dgopen {r[r.held_out].rho_dgopen.mean():.3f}, BEST {r[r.held_out].rho_best.mean():.3f}')
