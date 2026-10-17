"""One call from a PDB file to a per-residue prediction table."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .pdbio import sanitize_pdb
from .featurize import build_inputs

_ENS = {}


def get_ensemble(weights_dir=None, device=None):
    from .ensemble import Ensemble
    key = (weights_dir, device)
    if key not in _ENS:
        _ENS[key] = Ensemble(weights_dir=weights_dir, device=device)
    return _ENS[key]


def predict_structure(pdb_path, chain=None, sse_method='torsion', weights_dir=None,
                      device=None, embedding=None, return_members=False):
    """Predict per-residue dG_open (kcal/mol) for one chain of one structure.

    Returns a DataFrame with one row per residue: resi, resn, aa1, exchangeable,
    dg_open_kcal (NaN where the amide cannot exchange: prolines and the first two
    residues), dg_open_member_sd (spread across the 12 members, not a calibrated error),
    and, if return_members, one column per member.
    `embedding` lets you pass a precomputed (L, 1280) ESM-2 650M array.
    """
    clean, report = sanitize_pdb(pdb_path, chain=chain)
    try:
        inp = build_inputs(clean, sse_method=sse_method)
    finally:
        os.remove(clean)
    ens = get_ensemble(weights_dir, device)
    if embedding is None:
        from .embed import esm2_embed
        embedding = esm2_embed(inp['sequence'], device=str(ens.device))
    mean, sd, per = ens.predict(inp['X'], inp['ca'], np.asarray(embedding, np.float32))
    ex = inp['exchangeable']
    df = pd.DataFrame(dict(resi=inp['resi'], resn=inp['features']['resn'], aa1=inp['aa1'],
                           exchangeable=ex,
                           dg_open_kcal=np.where(ex, mean, np.nan),
                           dg_open_member_sd=np.where(ex, sd, np.nan)))
    if return_members:
        for k, v in per.items():
            df[k] = v
    df.attrs.update(input=report, sse_method=sse_method, n_members=len(per))
    return df
