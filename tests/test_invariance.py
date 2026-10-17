"""Rigid-motion invariance and chain-break handling (v0.2.0).

Exact tests use axis-permutation rotations: swapping and negating coordinates is exact under
the PDB 8.3f format, so any feature change is orientation dependence, not rounding.
Fixtures are derived from examples/5PTI.pdb (CC0): a copy with residues 25-27 deleted
(numbering gap) and a copy with residues 40-41 deleted and later residues renumbered down
by 2 (a 'silent' break: consecutive numbers across a real gap)."""
import numpy as np
import pytest
from conftest import EXAMPLE
from dgopen.pdbio import sanitize_pdb
from dgopen.featurize import build_inputs, pair_features, node_features

PERMS = [lambda x, y, z: (y, -x, z), lambda x, y, z: (-x, -y, z), lambda x, y, z: (z, x, y)]


def _rewrite(src, dst, coord=None, drop=(), shift_after=None):
    with open(src) as f, open(dst, 'w') as g:
        for ln in f:
            if ln.startswith(('ATOM', 'HETATM')):
                r = int(ln[22:26])
                if r in drop:
                    continue
                if shift_after is not None and r > shift_after:
                    ln = ln[:22] + f'{r - len(drop):>4}' + ln[26:]
                if coord is not None:
                    x, y, z = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                    ln = ln[:30] + ''.join(f'{v:8.3f}' for v in coord(x, y, z)) + ln[54:]
            g.write(ln)
    return dst


@pytest.fixture(scope='module')
def structures(tmp_path_factory):
    d = tmp_path_factory.mktemp('inv')
    clean = sanitize_pdb(EXAMPLE, out_path=str(d / 'clean.pdb'))[0]
    gap = _rewrite(clean, str(d / 'gap.pdb'), drop=(25, 26, 27))
    silent = _rewrite(clean, str(d / 'silent.pdb'), drop=(40, 41), shift_after=41)
    return dict(clean=clean, gap=gap, silent=silent, dir=d)


@pytest.mark.parametrize('name', ['clean', 'gap', 'silent'])
def test_features_exactly_invariant_to_axis_rotations(structures, name):
    ref = build_inputs(structures[name])
    for k, P in enumerate(PERMS):
        rot = build_inputs(_rewrite(structures[name], str(structures['dir'] / f'{name}_r{k}.pdb'), coord=P))
        diff = np.abs(rot['X'] - ref['X']).max(0)
        assert np.all(diff == 0), f'rotation {k} changed features at columns {np.where(diff > 0)[0].tolist()}'
        # distances only; a rotation reorders the float32 sum of squared components, so allow ulp-level error
        np.testing.assert_allclose(pair_features(rot['ca']), pair_features(ref['ca']), atol=1e-6)


def test_numbering_gap_is_flagged_and_hbond_features_masked(structures):
    df = node_features(structures['gap'])
    row = df[df.resi == 28].iloc[0]
    assert bool(row.after_chain_break)
    assert np.isnan(row.hb_amide_count) and np.isnan(row.hb_is_donor)
    assert not df[df.resi != 28].after_chain_break.any()


def test_silent_break_detected_by_geometry(structures):
    df = node_features(structures['silent'])
    flagged = df[df.after_chain_break].resi.tolist()
    assert flagged == [40], flagged            # old residue 42, now numbered 40, follows a real gap


def test_first_residue_not_flagged(structures):
    df = node_features(structures['clean'])
    assert not df.after_chain_break.any()
