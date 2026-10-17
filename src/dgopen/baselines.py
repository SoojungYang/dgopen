"""Best-Vendruscolo phenomenological baseline (Best & Vendruscolo, Structure 2006).

ln PF_i = 0.35 N_c,i + 2.0 N_h,i ;  dG_open_i = R T ln PF_i  (kcal/mol, 25 C)
N_c: heavy atoms of residues |i - j| > 2 within 6.5 A of amide N_i.
N_h: O/N acceptors of residues |i - j| > 2 within 2.4 A of the (reconstructed) amide H_i.
Structure-only; useful as the baseline every prediction should be compared against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .featurize import parse_atoms, by_residue, reconstruct_amide_H, R_KCAL
from .pdbio import sanitize_pdb


def best_vendruscolo(pdb_path, chain=None, beta_c=0.35, beta_h=2.0, r_c=6.5, r_h=2.4,
                     temp_C=25.0, sanitize=True):
    path = sanitize_pdb(pdb_path, chain=chain)[0] if sanitize else pdb_path
    T = temp_C + 273.15
    atoms = parse_atoms(path)
    res = by_residue(atoms)
    xyz = np.array([a['xyz'] for a in atoms]); ari = np.array([a['resi'] for a in atoms])
    elem = np.array([a['elem'] for a in atoms])
    rows = []
    for i in sorted(res):
        ri = res[i]
        if ri['resn'] == 'PRO' or 'N' not in ri['atoms']:
            rows.append(dict(resi=i, resn=ri['resn'], N_c=np.nan, N_h=np.nan, dg_open_kcal=np.nan))
            continue
        N = ri['atoms']['N']
        far = np.abs(ari - i) > 2
        N_c = int(np.sum((np.linalg.norm(xyz - N, axis=1) < r_c) & far))
        CA = ri['atoms'].get('CA'); C_prev = res.get(i - 1, {}).get('atoms', {}).get('C')
        H = reconstruct_amide_H(N, C_prev, CA) if (C_prev is not None and CA is not None) \
            else N + np.array([0.0, 0.0, 1.0])
        acc = np.isin(elem, ['O', 'N']) & far
        N_h = int(np.sum((np.linalg.norm(xyz - H, axis=1) < r_h) & acc))
        rows.append(dict(resi=i, resn=ri['resn'], N_c=N_c, N_h=N_h,
                         dg_open_kcal=R_KCAL * T * (beta_c * N_c + beta_h * N_h)))
    return pd.DataFrame(rows)
