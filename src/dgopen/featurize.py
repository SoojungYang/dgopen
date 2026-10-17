"""Per-residue node features and CA pair features, as used to train the released models.

Ported from the development repository (src/featurize.py, src/best_vendruscolo.py,
src/track2_data.py, data/heldout/predict_heldout.py). Numerical behaviour is unchanged;
the release test suite checks it against the original code on the evaluation structures.

Two secondary-structure / SASA methods are available:

  sse_method='torsion' (default): the method that built the TRAINING features of every
      released model. SSE from a Ramachandran-region rule on backbone phi/psi with run-length
      smoothing (helix >= 4, strand >= 3 residues), not a published SSE algorithm; SASA by
      Shrake-Rupley (1973), probe 1.4 A, 92-point sphere, VDW radii, normalised by the Tien
      et al. (2013) theoretical maximum.
  sse_method='psea': P-SEA as implemented in biotite (CA trace only) plus biotite
      Shrake-Rupley SASA (ProtOr radii). This is what the published external benchmarks
      used at inference time; it differs from training on 5 of 39 features (rsasa, ss_H,
      ss_E, ss_C, sse_depth). Provided to reproduce those numbers.
"""
from __future__ import annotations

import json
import os
from importlib import resources

import numpy as np
import pandas as pd

R_KCAL = 1.9872041e-3
STD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
          'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
      'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
      'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
VOL = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4,
       'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9,
       'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0}
THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
             'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
             'TYR': 'Y', 'VAL': 'V'}
AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')
CHARGE7 = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
TIEN_MAXASA = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                       [129, 167, 193, 223, 240, 104, 224, 197, 236, 201, 224, 195, 159,
                        225, 274, 155, 172, 174, 285, 263]))
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'H': 1.20}
SSE_METHODS = ('torsion', 'psea')


class FeaturizeError(RuntimeError):
    """The structure could not be featurized; never silently patched over."""


# ----------------------------------------------------------------------------- parsing
def parse_atoms(pdb_path):
    """Heavy-atom ATOM records of standard residues. Expects a sanitized single-chain file."""
    atoms = []
    for ln in open(pdb_path):
        if ln[:4] != 'ATOM':
            continue
        resn = ln[17:20].strip()
        if resn not in STD_AA:
            continue
        try:
            resi = int(ln[22:26])
        except ValueError:
            continue
        name = ln[12:16].strip()
        elem = ln[76:78].strip() or name[:1]
        if elem in ('H', 'D'):
            continue
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        atoms.append(dict(resi=resi, resn=resn, name=name, elem=elem, xyz=xyz))
    return atoms


def by_residue(atoms):
    res = {}
    for a in atoms:
        res.setdefault(a['resi'], {'resn': a['resn'], 'atoms': {}})
        res[a['resi']]['atoms'][a['name']] = a['xyz']
    return res


def reconstruct_amide_H(N, C_prev, CA):
    """Amide H 1.01 A from N along the bisector of (N - C_prev) and (N - CA)."""
    u1 = (N - C_prev); u1 /= (np.linalg.norm(u1) + 1e-9)
    u2 = (N - CA);     u2 /= (np.linalg.norm(u2) + 1e-9)
    d = u1 + u2; d /= (np.linalg.norm(d) + 1e-9)
    return N + 1.01 * d


def parse_ca(pdb_path):
    """resi -> (x, y, z) of CA (first altloc)."""
    out = {}
    for ln in open(pdb_path):
        if ln[:4] != 'ATOM' or ln[12:16].strip() != 'CA' or ln[16] not in (' ', 'A'):
            continue
        try:
            ri = int(ln[22:26])
        except ValueError:
            continue
        if ri not in out:
            out[ri] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return out


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w); y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


# ------------------------------------------------------------------ SSE / SASA methods
def sse_from_torsion(phi_deg, psi_deg):
    """Ramachandran-region SSE with run-length smoothing (helix >= 4, strand >= 3)."""
    n = len(phi_deg)
    ss = np.array(['C'] * n, dtype='<U1')
    helix = (phi_deg > -160) & (phi_deg < -20) & (psi_deg > -100) & (psi_deg < 5)
    strand = (phi_deg > -180) & (phi_deg < -40) & ((psi_deg > 80) | (psi_deg < -150))
    ss[helix] = 'H'
    ss[strand] = 'E'
    for code, minrun in (('H', 4), ('E', 3)):
        i = 0
        while i < n:
            if ss[i] == code:
                j = i
                while j < n and ss[j] == code:
                    j += 1
                if j - i < minrun:
                    ss[i:j] = 'C'
                i = j
            else:
                i += 1
    return ss


def shrake_rupley(pdb_path, resids, probe=1.4, n_pts=92, atoms=None):
    """Per-residue absolute SASA (A^2), Shrake-Rupley, numpy. The test-point sphere is fixed in
    the coordinate frame, so the result depends on orientation unless the caller passes atoms
    already in the canonical frame (see canonical_frame)."""
    atoms = parse_atoms(pdb_path) if atoms is None else atoms
    xyz = np.array([a['xyz'] for a in atoms])
    rad = np.array([VDW.get(a['elem'], 1.70) for a in atoms]) + probe
    ri = np.array([a['resi'] for a in atoms])
    i_ = np.arange(n_pts)
    z = 1 - 2 * (i_ + 0.5) / n_pts
    th = np.pi * (1 + 5 ** 0.5) * i_
    sph = np.c_[np.sqrt(1 - z * z) * np.cos(th), np.sqrt(1 - z * z) * np.sin(th), z]
    sasa = np.zeros(len(atoms))
    rmax = rad.max()
    for k in range(len(atoms)):
        d = np.linalg.norm(xyz - xyz[k], axis=1)
        nb = np.where((d > 1e-6) & (d < rad[k] + rmax))[0]
        surf = xyz[k] + rad[k] * sph
        acc = np.ones(n_pts, bool)
        for j in nb:
            acc &= np.linalg.norm(surf - xyz[j], axis=1) >= rad[j]
        sasa[k] = acc.sum() / n_pts * 4 * np.pi * rad[k] ** 2
    return {int(i): float(sasa[ri == i].sum()) for i in resids}


def _sse_sasa_torsion(df, pdb_path, resids, atoms=None):
    phi = np.degrees(np.arctan2(df['phi_sin'].values, df['phi_cos'].values))
    psi = np.degrees(np.arctan2(df['psi_sin'].values, df['psi_cos'].values))
    ss = sse_from_torsion(phi, psi)
    df['ss_H'] = (ss == 'H').astype(float)
    df['ss_E'] = (ss == 'E').astype(float)
    df['ss_C'] = (ss == 'C').astype(float)
    sa = shrake_rupley(pdb_path, resids, atoms=atoms)
    df['rsasa'] = [sa.get(int(r), np.nan) / TIEN_MAXASA.get(a, 200)
                   for r, a in zip(df['resi'], df['aa1'])]
    return df


def _sse_sasa_psea(df, pdb_path, transform=None):
    try:
        import biotite.structure.io.pdb as pdbio
        import biotite.structure as struc
    except ImportError as e:
        raise FeaturizeError("sse_method='psea' needs biotite (pip install biotite)") from e
    arr = pdbio.PDBFile.read(pdb_path).get_structure(model=1)
    if transform is not None:                     # same rigid motion as the numpy features
        R, c = transform
        arr.coord = ((arr.coord.astype(np.float64) - c) @ R.T).astype(arr.coord.dtype)
    sse = struc.annotate_sse(arr)
    sasa_rw = struc.apply_residue_wise(arr, struc.sasa(arr), np.nansum)
    rids = np.unique(arr.res_id)
    if not (len(sse) == len(sasa_rw) == len(rids)):
        raise FeaturizeError(f'biotite array lengths disagree for {pdb_path}: sse={len(sse)} '
                             f'sasa={len(sasa_rw)} res_id={len(rids)}')
    m = {int(r): (s, a) for r, s, a in zip(rids, sse, sasa_rw)}
    for idx, row in df.iterrows():
        if int(row.resi) in m:
            s, a = m[int(row.resi)]
            df.at[idx, 'ss_H'] = float(s == 'a')
            df.at[idx, 'ss_E'] = float(s == 'b')
            df.at[idx, 'ss_C'] = float(s == 'c')
            df.at[idx, 'rsasa'] = a / TIEN_MAXASA.get(row.aa1, 200)
    return df


def _add_sse_position(df):
    """Distance (residues) to the nearest end of the current SSE segment."""
    ss = np.where(df['ss_H'] > 0, 'H', np.where(df['ss_E'] > 0, 'E', 'C'))
    dist = np.zeros(len(df))
    start = 0
    for end in range(1, len(df) + 1):
        if end == len(df) or ss[end] != ss[start]:
            seg = np.arange(start, end)
            for p, idx in enumerate(seg):
                dist[idx] = min(p, len(seg) - 1 - p)
            start = end
    df['sse_depth'] = dist
    return df


# ------------------------------------------------------------------- canonical frame
FRAMES = ('canonical', 'input')
BREAK_POLICIES = ('mask', 'legacy')
MAX_PEPTIDE_CN = 2.0     # A; a real C(i-1)-N(i) peptide bond is ~1.33 A


def canonical_frame(atoms):
    """Rigid motion (R, c) taking the structure into a deterministic frame: origin at the CA
    centroid, axes along the principal axes of the CA coordinates (largest variance first),
    each of the first two axis signs fixed by the sign of the third moment of the projected
    CA coordinates, third axis = e1 x e2 (a proper rotation, so dihedral signs are preserved).
    Applying x' = R (x - c) makes every feature exactly invariant to rotating or translating
    the input, up to floating-point error."""
    ca = np.array([a['xyz'] for a in atoms if a['name'] == 'CA'], float)
    if len(ca) < 3:
        raise FeaturizeError('need at least 3 CA atoms to define a frame')
    c = ca.mean(0); X = ca - c
    w, V = np.linalg.eigh(X.T @ X / len(X))
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    if w[2] > 0 and (w[0] - w[1] < 0.01 * w[0] or w[1] - w[2] < 0.01 * w[1]):
        import warnings
        warnings.warn('principal axes are nearly degenerate; the canonical frame (and hence the '
                      'Shrake-Rupley SASA) may be sensitive to small coordinate changes')
    e = []
    for k in range(2):
        v = V[:, k]; m3 = np.sum((X @ v) ** 3)
        e.append(v if m3 >= 0 else -v)
    e.append(np.cross(e[0], e[1]))
    return np.array(e), c


def _apply_frame(atoms, R, c):
    return [dict(a, xyz=R @ (a['xyz'] - c)) for a in atoms]


# --------------------------------------------------------------------- node features
def node_features(pdb_path, sse_method='torsion', r_c=6.5, r_h=2.4, contact_cb=8.0,
                  frame='canonical', break_policy='mask'):
    """Per-residue feature table (one row per residue) from a sanitized single-chain PDB.

    frame='canonical' (default) computes every feature in canonical_frame, so outputs do not
    depend on how the input is oriented. frame='input' and break_policy='legacy' reproduce
    v0.1.x exactly (used to check published numbers; not for new predictions).

    A residue is `after_chain_break` when it is not the first residue and the previous residue
    is absent, has no C atom, or sits more than 2.0 A from this N (a real gap with consecutive
    numbering). Under break_policy='mask' its amide-H features (hb_amide_count, hb_is_donor) and
    phi are set to missing, which standardization maps to the training mean / 0, instead of
    being computed from an amide H placed without a preceding carbonyl; psi of the residue before
    a geometric break is masked the same way.
    """
    if sse_method not in SSE_METHODS:
        raise ValueError(f'sse_method must be one of {SSE_METHODS}')
    if frame not in FRAMES or break_policy not in BREAK_POLICIES:
        raise ValueError(f'frame in {FRAMES}, break_policy in {BREAK_POLICIES}')
    atoms = parse_atoms(pdb_path)
    if not atoms:
        raise FeaturizeError(f'no standard-residue heavy atoms in {pdb_path}')
    transform = None
    if frame == 'canonical':
        transform = canonical_frame(atoms)
        atoms = _apply_frame(atoms, *transform)
    res = by_residue(atoms)
    resids = sorted(res)
    first_two = set(resids[:2])
    xyz = np.array([a['xyz'] for a in atoms]); ari = np.array([a['resi'] for a in atoms])
    aelem = np.array([a['elem'] for a in atoms])
    cb = {i: (res[i]['atoms'].get('CB', res[i]['atoms'].get('CA'))) for i in resids}
    rows = []
    for k, i in enumerate(resids):
        ri = res[i]; resn = ri['resn']; aa = THREE2ONE.get(resn, 'X')
        at = ri['atoms']; N, CA, C = at.get('N'), at.get('CA'), at.get('C')
        is_pro = (resn == 'PRO')
        C_before = res.get(i - 1, {}).get('atoms', {}).get('C')
        after_break = k > 0 and (C_before is None or
                                 (N is not None and np.linalg.norm(N - C_before) > MAX_PEPTIDE_CN))
        N_next = res.get(i + 1, {}).get('atoms', {}).get('N')
        before_break = C is not None and N_next is not None and \
            np.linalg.norm(N_next - C) > MAX_PEPTIDE_CN
        N_c = N_h = np.nan
        if N is not None and not is_pro:
            far = np.abs(ari - i) > 2
            N_c = int(np.sum((np.linalg.norm(xyz - N, axis=1) < r_c) & far))
            C_prev = res.get(i - 1, {}).get('atoms', {}).get('C')
            H = reconstruct_amide_H(N, C_prev, CA) if (C_prev is not None and CA is not None) \
                else (N + np.array([0, 0, 1.0]))
            acc = np.isin(aelem, ['O', 'N']) & far
            N_h = int(np.sum((np.linalg.norm(xyz - H, axis=1) < r_h) & acc))
        cnum = 0
        if cb[i] is not None:
            for j in resids:
                if abs(j - i) > 2 and cb[j] is not None and \
                        np.linalg.norm(cb[i] - cb[j]) < contact_cb:
                    cnum += 1
        phi = psi = np.nan
        Cm = res.get(i - 1, {}).get('atoms', {}).get('C')
        Np = res.get(i + 1, {}).get('atoms', {}).get('N')
        if all(a is not None for a in (Cm, N, CA, C)):
            phi = _dihedral(Cm, N, CA, C)
        if all(a is not None for a in (N, CA, C, Np)):
            psi = _dihedral(N, CA, C, Np)
        hb_donor = float((N_h or 0) > 0)
        if break_policy == 'mask':
            if after_break:
                N_h = hb_donor = np.nan
                phi = np.nan
            if before_break:
                psi = np.nan
        rows.append(dict(
            resi=i, resn=resn, aa1=aa,
            seq_relpos=k / max(len(resids) - 1, 1),
            hb_amide_count=N_h, hb_is_donor=hb_donor,
            contacts_amideN=N_c, contacts_cb=cnum,
            kd_hydropathy=KD.get(aa, 0.0), sc_volume=VOL.get(aa, 0.0),
            charge7=CHARGE7.get(aa, 0.0),
            phi_sin=np.sin(phi) if phi == phi else 0.0,
            phi_cos=np.cos(phi) if phi == phi else 0.0,
            psi_sin=np.sin(psi) if psi == psi else 0.0,
            psi_cos=np.cos(psi) if psi == psi else 0.0,
            plddt=np.nan,          # dead feature: its training mean/sd are NaN -> always 0
            is_proline=float(is_pro),
            is_measurable=float((not is_pro) and (i not in first_two)),
            after_chain_break=bool(after_break),
        ))
    df = pd.DataFrame(rows)
    for a in AA_ORDER:
        df[f'aa_{a}'] = (df['aa1'] == a).astype(float)
    df['ss_H'] = df['ss_E'] = df['ss_C'] = 0.0
    df['rsasa'] = np.nan
    df = _sse_sasa_torsion(df, pdb_path, resids, atoms=atoms) if sse_method == 'torsion' \
        else _sse_sasa_psea(df, pdb_path, transform=transform)
    return _add_sse_position(df)


# ------------------------------------------------------------- model input assembly
def load_feature_norm():
    with resources.files('dgopen').joinpath('data/feature_norm.json').open() as fh:
        d = json.load(fh)
    mu = np.array([np.nan if v is None else v for v in d['mu']], float)
    sd = np.array([np.nan if v is None else v for v in d['sd']], float)
    return d['feat_cols'], mu, sd, d


def standardize(df, feat_cols, mu, sd, clip=6.0):
    """(x - mu) / sd, clipped to +-clip, NaN -> 0 (training convention)."""
    F = np.stack([df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)
                  for c in feat_cols], 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        F = (F - mu) / sd
    return np.nan_to_num(np.clip(F, -clip, clip), nan=0.0).astype(np.float32)


RBF_C = np.linspace(2., 22., 16)
RBF_W = 2.5


def pair_features(ca, pair_idx=None):
    """(L, L, 22) pair tensor: 16 Gaussian RBFs of CA-CA distance (centres 2..22 A, width
    2.5 A), 4 sequence-separation indicators (|i-j| <= 1, <= 2, <= 4, == 0), 1/(1+|i-j|),
    and a constant channel. `pair_idx` selects channels by index."""
    L = ca.shape[0]
    diff = ca[:, None, :] - ca[None, :, :]
    Dm = np.sqrt((diff ** 2).sum(-1) + 1e-8)
    rbf = np.exp(-((Dm[..., None] - RBF_C) ** 2) / (2 * RBF_W ** 2))
    idx = np.arange(L); sep = np.abs(idx[:, None] - idx[None, :]).astype(float)
    feats = [rbf,
             (sep <= 1).astype(float)[..., None], (sep <= 2).astype(float)[..., None],
             (sep <= 4).astype(float)[..., None], (sep == 0).astype(float)[..., None],
             (1.0 / (1.0 + sep))[..., None], np.ones((L, L, 1))]
    out = np.concatenate(feats, axis=-1).astype(np.float32)
    if pair_idx is not None and tuple(pair_idx) != tuple(range(out.shape[-1])):
        out = np.ascontiguousarray(out[..., list(pair_idx)])
    return out


def build_inputs(pdb_path, sse_method='torsion', frame='canonical', break_policy='mask'):
    """Sanitized PDB -> dict(X, ca, exchangeable, after_chain_break, resi, aa1, sequence, features).
    frame / break_policy: see node_features; the non-default values exist only to reproduce v0.1.x."""
    feat_cols, mu, sd, _ = load_feature_norm()
    df = node_features(pdb_path, sse_method=sse_method, frame=frame, break_policy=break_policy)
    X = standardize(df, feat_cols, mu, sd)
    cam = parse_ca(pdb_path)
    missing = [r for r in df['resi'] if r not in cam]
    if missing:
        raise FeaturizeError(f'missing CA for residues {missing[:8]}')
    ca = np.array([cam[r] for r in df['resi']], float).astype(np.float32)
    exch = (df['is_proline'].to_numpy(float) < 0.5) & (df['is_measurable'].to_numpy(float) > 0.5)
    return dict(X=X, ca=ca, exchangeable=exch, after_chain_break=df['after_chain_break'].to_numpy(bool),
                resi=df['resi'].to_numpy(int),
                aa1=df['aa1'].tolist(), sequence=''.join(df['aa1']), features=df)
