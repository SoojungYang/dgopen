"""PDB input handling.

The featurizer reads fixed-column ATOM records and keys residues by residue number alone,
so it is only correct on a single-model, single-chain, one-altloc structure without
insertion codes. `sanitize_pdb` writes such a file, or refuses with a message saying why.
It never renumbers: phi/psi and the |i - j| > 2 contact rule use residue numbers, so
renumbering across a chain break would silently join the residues on either side of it.
"""
from __future__ import annotations

import os
import tempfile

STD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
          'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


class InputError(ValueError):
    """The structure cannot be featurized as given. The message says what to fix."""


def sanitize_pdb(pdb_path: str, chain: str | None = None, out_path: str | None = None):
    """Write a clean single-chain copy of `pdb_path` and return (path, report).

    Keeps: ATOM records of the 20 standard residues, MODEL 1 only, the first altloc of each
    atom (' ' or the first letter seen), the requested chain (or the only chain present).
    Drops: HETATM, waters, hydrogens are left to the featurizer (it ignores them).
    Refuses: several chains with no `chain` given, insertion codes, a residue without CA,
    and non-standard residues inside the chain (e.g. MSE), which would otherwise vanish and
    leave a sequence gap the user did not ask for.
    """
    if not os.path.exists(pdb_path):
        raise InputError(f'no such file: {pdb_path}')
    lines, chains, nonstd, het_atoms, model_seen = [], [], set(), {}, 0
    with open(pdb_path) as fh:
        for ln in fh:
            rec = ln[:6]
            if rec.startswith('MODEL'):
                model_seen += 1
                if model_seen > 1:
                    break
                continue
            if rec.startswith('ENDMDL') and model_seen:
                break
            if rec == 'HETATM':
                resn = ln[17:20].strip()
                if resn not in ('HOH', 'WAT', 'DOD'):
                    key = (ln[21], ln[22:27].strip(), resn)
                    nonstd.add((ln[21], resn, ln[22:27].strip()))
                    het_atoms.setdefault(key, set()).add(ln[12:16].strip())
                continue
            if rec != 'ATOM  ':
                continue
            ln = ln.rstrip('\n').ljust(80)
            ch = ln[21]
            if ch not in chains:
                chains.append(ch)
            lines.append(ln)
    if not lines:
        raise InputError(f'{pdb_path}: no ATOM records')
    if chain is None:
        if len(chains) > 1:
            raise InputError(f'{pdb_path} has chains {chains}; pass chain= (CLI: --chain).')
        chain = chains[0]
    if chain not in chains:
        raise InputError(f'chain {chain!r} not in {pdb_path} (chains: {chains})')
    keep, altloc_of, ca_seen, resn_of, dropped_nonstd = [], {}, set(), {}, set()
    for ln in lines:
        if ln[21] != chain:
            continue
        resn, ins = ln[17:20].strip(), ln[26]
        try:
            resi = int(ln[22:26])
        except ValueError:
            raise InputError(f'unparseable residue number in line: {ln[:27]!r}')
        if ins != ' ':
            raise InputError(
                f'{pdb_path} chain {chain} has insertion code {ins!r} at residue {resi}. '
                'Renumber the structure yourself (keeping chain breaks as numbering gaps) '
                'and rerun; dgopen does not renumber automatically.')
        if resn not in STD_AA:
            dropped_nonstd.add((resi, resn))
            continue
        alt = ln[16]
        if alt != ' ':
            first = altloc_of.setdefault((resi, ln[12:16]), alt)
            if alt != first:
                continue
            ln = ln[:16] + ' ' + ln[17:]
        resn_of.setdefault(resi, resn)
        if ln[12:16].strip() == 'CA':
            ca_seen.add(resi)
        keep.append(ln)
    hets_in_chain = {(r, n) for c, n, r in nonstd if c == chain}
    # A HETATM residue carrying a full N/CA/C backbone is a modified amino acid in the
    # polymer (e.g. MSE), not a ligand; dropping it would leave a silent sequence gap.
    het_polymer = {k for k, names in het_atoms.items() if {'N', 'CA', 'C'} <= names}
    dropped_nonstd |= {(r, n) for c, r, n in het_polymer if c == chain}
    if dropped_nonstd:
        raise InputError(
            f'non-standard residues inside chain {chain}: {sorted(dropped_nonstd)[:8]}. '
            'Convert them (e.g. MSE -> MET) or remove them explicitly before running.')
    missing_ca = sorted(set(resn_of) - ca_seen)
    if missing_ca:
        raise InputError(f'residues without a CA atom in chain {chain}: {missing_ca[:10]}')
    resids = sorted(resn_of)
    gaps = [(a, b) for a, b in zip(resids, resids[1:]) if b - a > 1]
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix='.pdb', prefix='dgopen_')
        os.close(fd)
    with open(out_path, 'w') as fh:
        for ln in keep:
            fh.write(ln.rstrip() + '\n')
        fh.write('END\n')
    report = dict(source=os.path.abspath(pdb_path), chain=chain, n_residues=len(resids),
                  first_resi=resids[0], last_resi=resids[-1], numbering_gaps=gaps,
                  altlocs_resolved=len(altloc_of), hetero_groups_dropped=len(hets_in_chain),
                  models_in_file=max(model_seen, 1))
    return out_path, report
