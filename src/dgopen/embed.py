"""ESM-2 650M per-residue embeddings (last hidden layer, fp32, one sequence at a time)."""
from __future__ import annotations

import numpy as np

ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
_CACHE = {}


def _load(device):
    key = (ESM_MODEL, str(device))
    if key not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ESM_MODEL)
        # no pooler: only last_hidden_state is used, and the pooler is not in the checkpoint
        model = AutoModel.from_pretrained(ESM_MODEL, add_pooling_layer=False).eval().to(device)
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def esm2_embed(sequence: str, device='cpu') -> np.ndarray:
    """(L, 1280) float32. Tokens 1..L (drops <cls> and <eos>). fp32 deliberately: this is
    the precision the published evaluation used."""
    import torch
    tok, model = _load(device)
    enc = tok(sequence, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hs = model(**enc).last_hidden_state[0, 1:len(sequence) + 1].float().cpu().numpy()
    if hs.shape[0] != len(sequence):
        raise RuntimeError(f'ESM-2 returned {hs.shape[0]} rows for a {len(sequence)}-residue '
                           'sequence (truncation?); refusing to pad.')
    return hs.astype(np.float32)
