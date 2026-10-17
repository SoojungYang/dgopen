import os
import numpy as np
import pytest
import torch
from conftest import DATA, EXAMPLE
from dgopen import weights as W
from dgopen.model import ArchConfig, DistGraphTransformer


def test_model_forward_shape_and_state_keys():
    m = DistGraphTransformer(ArchConfig()).eval()
    y = m(torch.zeros(7, 1319), torch.zeros(7, 7, 22))
    assert y.shape == (7,)
    keys = set(m.state_dict())
    assert {'inp.weight', 'wb.0', 'wb.5', 'out.bias'} <= keys and len(keys) == 58


def test_manifest_consistent_with_feature_norm():
    from dgopen.featurize import load_feature_norm
    man = W.manifest(); cols, _, _, norm = load_feature_norm()
    assert man['spec_hash'] == norm['spec_hash'] and man['feat_cols'] == cols
    assert len(man['members']) == 12 and len({m['sha256'] for m in man['members']}) == 12


def test_hash_mismatch_refused(tmp_path):
    man = W.manifest()
    for mem in man['members']:
        (tmp_path / mem['file']).write_bytes(b'not a checkpoint')
    with pytest.raises(RuntimeError, match='sha256 mismatch'):
        W.resolve(str(tmp_path), download=False)


def test_missing_weights_message(tmp_path):
    with pytest.raises(FileNotFoundError, match='Download the release weights'):
        W.resolve(str(tmp_path), download=False)


def test_golden_prediction_5pti(weights_dir, tmp_path):
    """End to end with the real weights and a stored ESM-2 embedding (no ESM download)."""
    from dgopen.pdbio import sanitize_pdb
    from dgopen.featurize import build_inputs
    from dgopen.ensemble import Ensemble
    out = str(tmp_path / 'c.pdb'); sanitize_pdb(EXAMPLE, out_path=out)
    I = build_inputs(out)
    ens = Ensemble(weights_dir=weights_dir, device='cpu', download=False)
    mean, sd, per = ens.predict(I['X'], I['ca'], np.load(os.path.join(DATA, '5PTI_esm2_650M.npy')))
    gold = np.load(os.path.join(DATA, '5PTI_golden_pred.npz'))
    np.testing.assert_allclose(mean, gold['mean'], atol=1e-4)
    np.testing.assert_allclose(sd, gold['sd'], atol=1e-4)
    assert np.allclose(mean, np.mean([per[k] for k in per], 0))
