import os
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, 'data')
EXAMPLE = os.path.join(ROOT, 'examples', '5PTI.pdb')


def _line(rec, serial, name, resn, chain, resi, xyz, alt=' ', ins=' ', elem=None):
    elem = elem or name[0]
    return (f'{rec:<6}{serial:>5} {name:<4}{alt}{resn:>3} {chain}{resi:>4}{ins}   '
            f'{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {elem:>2}\n')


@pytest.fixture
def write_pdb(tmp_path):
    """write_pdb([(rec, name, resn, chain, resi, (x,y,z), alt, ins), ...]) -> path"""
    def _w(atoms, fname='t.pdb'):
        p = tmp_path / fname
        with open(p, 'w') as fh:
            for k, a in enumerate(atoms, 1):
                rec, name, resn, chain, resi, xyz = a[:6]
                alt = a[6] if len(a) > 6 else ' '
                ins = a[7] if len(a) > 7 else ' '
                fh.write(_line(rec, k, name, resn, chain, resi, xyz, alt, ins))
            fh.write('END\n')
        return str(p)
    return _w


def backbone(resn, chain, resi, x0, alt=' ', ins=' '):
    return [('ATOM', 'N', resn, chain, resi, (x0, 0.0, 0.0), alt, ins),
            ('ATOM', 'CA', resn, chain, resi, (x0 + 1.46, 0.0, 0.0), alt, ins),
            ('ATOM', 'C', resn, chain, resi, (x0 + 2.0, 1.4, 0.0), alt, ins),
            ('ATOM', 'O', resn, chain, resi, (x0 + 1.4, 2.4, 0.0), alt, ins)]


@pytest.fixture
def weights_dir():
    d = os.environ.get('DGOPEN_WEIGHTS')
    if not d or not os.path.isdir(d):
        pytest.skip('set DGOPEN_WEIGHTS to a directory holding the v0.1.0 weights')
    return d
