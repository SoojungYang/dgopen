"""The shipped HDX-NMR example data load, align to their structures, and featurize."""
import os
import pandas as pd
from conftest import ROOT
from dgopen.pdbio import sanitize_pdb
from dgopen.featurize import node_features, parse_atoms, by_residue, THREE2ONE

D = os.path.join(ROOT, 'examples', 'hdx_nmr')


def test_measured_values_align_to_structures(tmp_path):
    m = pd.read_csv(os.path.join(D, 'measured_dgopen_nmr.csv'))
    assert len(m) == 335 and m.domain.nunique() == 13
    assert (~m.in_finetune_set).groupby(m.domain).first().sum() == 3
    for dom, g in m.groupby('domain'):
        pdb = os.path.join(D, 'structures', f'{dom}.pdb')
        res = by_residue(parse_atoms(pdb))
        assert all(THREE2ONE[res[r]['resn']] == a for r, a in zip(g.resi, g.aa1)), dom
        clean = sanitize_pdb(pdb, out_path=str(tmp_path / f'{dom}.pdb'))[0]
        assert not node_features(clean).after_chain_break.any(), dom
