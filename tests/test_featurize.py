import os
import numpy as np
import pandas as pd
import pytest
from conftest import DATA, EXAMPLE
from dgopen.pdbio import sanitize_pdb
from dgopen.featurize import build_inputs, load_feature_norm, pair_features, sse_from_torsion


@pytest.fixture(scope='module')
def bpti(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('p') / 'c.pdb')
    sanitize_pdb(EXAMPLE, out_path=out)
    return build_inputs(out, sse_method='torsion')


def test_golden_features_5pti(bpti):
    gold = pd.read_csv(os.path.join(DATA, '5PTI_features_torsion.csv'))
    got = bpti['features'][gold.columns]
    assert list(got.aa1) == list(gold.aa1)
    num = [c for c in gold.columns if c != 'aa1']
    np.testing.assert_allclose(got[num].to_numpy(float), gold[num].to_numpy(float), atol=1e-6)
    np.testing.assert_allclose(bpti['X'], np.load(os.path.join(DATA, '5PTI_X_torsion.npy')), atol=1e-5)


def test_standardisation_contract(bpti):
    cols, mu, sd, norm = load_feature_norm()
    assert len(cols) == 39 and norm['spec_hash'] == 'f7d4c712ee6d'
    assert np.all(np.abs(bpti['X']) <= 6.0) and np.isfinite(bpti['X']).all()
    assert np.all(bpti['X'][:, cols.index('plddt')] == 0.0)        # dead feature


def test_exchangeable_mask(bpti):
    f = bpti['features']
    assert not bpti['exchangeable'][:2].any()
    assert not bpti['exchangeable'][(f.resn == 'PRO').to_numpy()].any()


def test_pair_features_shape_and_channels():
    ca = np.cumsum(np.full((5, 3), 2.0), 0).astype(np.float32)
    pf = pair_features(ca)
    assert pf.shape == (5, 5, 22)
    assert np.allclose(pf[..., 21], 1.0) and np.allclose(np.diagonal(pf[..., 19]), 1.0)


def test_torsion_sse_run_length():
    phi = np.array([-60.] * 5 + [-120.] * 2 + [-60.] * 3)
    psi = np.array([-45.] * 5 + [130.] * 2 + [-45.] * 3)
    assert ''.join(sse_from_torsion(phi, psi)) == 'HHHHHCCCCC'     # E run of 2 and H run of 3 eroded
