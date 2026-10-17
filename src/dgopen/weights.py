"""Locate, download and verify the released checkpoints.

Weights are NOT in the git repository. Search order for the weights directory:
  1. the `weights_dir` argument / CLI --weights-dir
  2. $DGOPEN_WEIGHTS
  3. ~/.cache/dgopen/<release tag>
Each file is checked against the sha256 in data/weights_manifest.json before loading.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from importlib import resources


def manifest():
    with resources.files('dgopen').joinpath('data/weights_manifest.json').open() as fh:
        return json.load(fh)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def default_dir():
    m = manifest()
    return os.environ.get('DGOPEN_WEIGHTS') or os.path.join(
        os.path.expanduser('~'), '.cache', 'dgopen', m['release'])


def resolve(weights_dir=None, download=True):
    """-> list of (member_dict, local_path), all hash-verified."""
    m = manifest()
    d = weights_dir or default_dir()
    os.makedirs(d, exist_ok=True)
    out = []
    for mem in m['members']:
        p = os.path.join(d, mem['file'])
        if not os.path.exists(p):
            if not download or not m.get('base_url'):
                raise FileNotFoundError(
                    f'{p} missing. Download the release weights (see README) into {d} '
                    'or pass --weights-dir.')
            url = m['base_url'].rstrip('/') + '/' + mem['file']
            tmp = p + '.part'
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, p)
        got = _sha256(p)
        if got != mem['sha256']:
            raise RuntimeError(f'sha256 mismatch for {p}: {got} != {mem["sha256"]}')
        out.append((mem, p))
    return out
