# Third-party components

| component | how it is used | licence |
|---|---|---|
| ESM-2 650M (`facebook/esm2_t33_650M_UR50D`, Meta AI) | downloaded at run time from Hugging Face; not redistributed | MIT |
| PyTorch, Hugging Face transformers, NumPy, pandas | runtime dependencies | BSD-3 / Apache-2.0 / BSD-3 / BSD-3 |
| biotite (optional, `--sse-method psea`) | runtime dependency | BSD-3 |
| `examples/5PTI.pdb` (RCSB PDB entry 5PTI) | example input, redistributed unmodified | CC0 1.0 (wwPDB) |
| `examples/hdx_nmr/` measured ΔG_open and structures (Ferrari et al. 2026; Zenodo 10.5281/zenodo.14983481) | example data; structures unmodified, measured values averaged over replicates (see its README) | CC BY 4.0 |
