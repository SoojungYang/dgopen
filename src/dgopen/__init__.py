"""dgopen: per-residue amide opening free energy (dG_open) from a protein structure.

Public entry points:
    dgopen.predict_structure(pdb_path, chain=None, ...) -> pandas.DataFrame
    dgopen.best_vendruscolo(pdb_path, chain=None)        -> pandas.DataFrame
Command line: `dgopen predict --pdb X.pdb --out X.csv`.
"""
__version__ = "0.1.1"

from .predict import predict_structure  # noqa: E402,F401
from .baselines import best_vendruscolo  # noqa: E402,F401
