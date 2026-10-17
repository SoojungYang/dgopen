import pytest
from dgopen.pdbio import sanitize_pdb, InputError
from conftest import backbone


def _peptide(chain='A', n=4, start=1):
    atoms = []
    for k in range(n):
        atoms += backbone('ALA', chain, start + k, 3.8 * k)
    return atoms


def test_single_chain_passes_and_reports(write_pdb):
    out, rep = sanitize_pdb(write_pdb(_peptide()))
    assert rep['chain'] == 'A' and rep['n_residues'] == 4 and rep['numbering_gaps'] == []


def test_multichain_requires_choice(write_pdb):
    p = write_pdb(_peptide('A') + _peptide('B', start=10))
    with pytest.raises(InputError, match='pass chain='):
        sanitize_pdb(p)
    assert sanitize_pdb(p, chain='B')[1]['first_resi'] == 10


def test_insertion_code_refused(write_pdb):
    atoms = _peptide() + backbone('GLY', 'A', 4, 20.0, ins='A')
    with pytest.raises(InputError, match='insertion code'):
        sanitize_pdb(write_pdb(atoms))


def test_first_altloc_kept_and_flag_cleared(write_pdb, tmp_path):
    atoms = _peptide(n=2) + backbone('SER', 'A', 3, 7.6, alt='A') + backbone('SER', 'A', 3, 9.0, alt='B')
    out, rep = sanitize_pdb(write_pdb(atoms), out_path=str(tmp_path / 'o.pdb'))
    lines = [l for l in open(out) if l.startswith('ATOM') and int(l[22:26]) == 3]
    assert len(lines) == 4 and all(l[16] == ' ' for l in lines)
    assert abs(float(lines[0][30:38]) - 7.6) < 1e-6


def test_hetero_polymer_residue_refused(write_pdb):
    atoms = _peptide(n=3) + [('HETATM', 'N', 'MSE', 'A', 4, (12.0, 0, 0)),
                             ('HETATM', 'CA', 'MSE', 'A', 4, (13.4, 0, 0)),
                             ('HETATM', 'C', 'MSE', 'A', 4, (14.0, 1.4, 0))]
    with pytest.raises(InputError, match='non-standard'):
        sanitize_pdb(write_pdb(atoms))


def test_ligands_and_water_dropped(write_pdb, tmp_path):
    # a calcium ion is named CA too; it must be treated as a ligand, not a residue
    atoms = _peptide() + [('HETATM', 'O', 'HOH', 'A', 101, (0, 9, 9)),
                          ('HETATM', 'CA', ' CA', 'A', 102, (9, 9, 9))]
    out, rep = sanitize_pdb(write_pdb(atoms), out_path=str(tmp_path / 'o.pdb'))
    assert not any(l.startswith('HETATM') for l in open(out))


def test_gaps_reported_not_renumbered(write_pdb):
    atoms = backbone('ALA', 'A', 1, 0.0) + backbone('ALA', 'A', 2, 3.8) + backbone('ALA', 'A', 7, 7.6)
    rep = sanitize_pdb(write_pdb(atoms))[1]
    assert rep['numbering_gaps'] == [(2, 7)] and rep['last_resi'] == 7
