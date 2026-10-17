"""The released 12-member ensemble (model-ens): flat arithmetic mean over members.

Members = 4 training arms x 3 seeds, all on the same input features. The arms differ only
in what the trunk was pretrained on before fine-tuning on HDX data (none; Best-Vendruscolo
dG_open; MD-derived mean; MD-derived mean and variance). No MD is needed at inference.
The across-member standard deviation is returned as a spread diagnostic; it is NOT a
calibrated uncertainty (members share data and features, so it underestimates error).
"""
from __future__ import annotations

import numpy as np
import torch

from . import weights as W
from .model import ArchConfig, DistGraphTransformer
from .featurize import pair_features, load_feature_norm

FLAT_ATOL = 1e-9


class Ensemble:
    def __init__(self, weights_dir=None, device=None, download=True):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        man = W.manifest()
        feat_cols, _, _, norm = load_feature_norm()
        if norm['spec_hash'] != man['spec_hash'] or feat_cols != man['feat_cols']:
            raise RuntimeError('feature_norm.json and weights_manifest.json disagree on the '
                               'feature spec; the install is inconsistent.')
        a = man['arch']
        self.cfg = ArchConfig(n_in=a['n_in'], d_head=a['d_head'], n_head=a['n_head'],
                              n_layer=a['n_layer'], n_pair=len(man['pair_idx']))
        self.pair_idx = tuple(man['pair_idx'])
        self.members = []
        for mem, path in W.resolve(weights_dir, download=download):
            state = torch.load(path, map_location=self.device, weights_only=True)
            model = DistGraphTransformer(self.cfg).to(self.device).eval()
            model.load_state_dict(state, strict=True)     # never strict=False
            self.members.append((mem['name'], model))

    @torch.no_grad()
    def predict_members(self, X, ca, plm):
        L = X.shape[0]
        if plm.shape != (L, self.cfg.n_in - X.shape[1]):
            raise ValueError(f'embedding shape {plm.shape} does not fit {L} residues')
        Xt = torch.tensor(np.concatenate([X, plm], 1), dtype=torch.float32, device=self.device)
        PF = torch.tensor(pair_features(np.asarray(ca, np.float32), self.pair_idx), device=self.device)
        out = {}
        for name, model in self.members:
            y = model(Xt, PF).cpu().numpy()
            if not np.all(np.isfinite(y)) or np.ptp(y) < FLAT_ATOL:
                raise RuntimeError(f'member {name} returned a degenerate vector; refusing.')
            out[name] = y
        return out

    def predict(self, X, ca, plm):
        per = self.predict_members(X, ca, plm)
        Y = np.stack(list(per.values()))
        return Y.mean(0), Y.std(0, ddof=0), per
