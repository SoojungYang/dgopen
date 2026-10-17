# HDX-NMR examples

Per-residue ΔG_open measured by site-resolved HDX-NMR for 13 small domains (designed
mini-proteins, two point mutants, and three natural domains), compared with dgopen and with
the Best-Vendruscolo baseline.

![HDX-NMR examples](nmr_examples.png)

Each panel shows one domain. Black points are measured values. Open triangles are residues
that exchanged too slowly to measure fully, so their value is a lower bound. The navy line is
the dgopen ensemble mean, with the ±1 SD spread across its 12 members shaded. The dashed red
line is Best-Vendruscolo. The ρ values in each panel are Spearman correlations with the
measured values (navy for dgopen, red for Best-Vendruscolo).

| | dgopen | Best-Vendruscolo |
|---|---|---|
| macro ρ, all 13 domains | 0.631 | 0.387 |
| macro ρ, 3 held-out domains | 0.729 | 0.418 |

**What the model saw.** None of these NMR values were used in training. The domains fall
into two groups:

- **Held out** (3 domains, bold in the figure): no measured data for these proteins was used
  in training at all. Two of them are double mutants (K31L/E36V, R35D/G45L) of domains in the
  other group, so the model has seen data for a protein differing at two positions. One of the
  four pretraining variants also saw their structures, through the structure-only
  Best-Vendruscolo target, which involves no measurement.
- **Others** (10 domains): only their unassigned HDX-MS ΔG_open distributions were used in
  fine-tuning, that is, how many amides have each ΔG_open without saying which residue is
  which. Their per-residue values were never used.

Lower-bound residues are scored as if they were exact values, as in the published evaluation.

## Files

- `measured_dgopen_nmr.csv`: domain, residue number (as in the structure file), amino acid,
  measured ΔG_open (kcal/mol), `lower_bound`, `in_finetune_set`. 335 residues.
- `structures/`: the 13 AlphaFold2 + RosettaRelax models used for prediction.
- `run_nmr_examples.py`: runs dgopen and Best-Vendruscolo and writes `nmr_comparison.csv`
  (per residue) and `nmr_rho.csv` (per domain). Takes about a minute on a CPU.
- `plot_nmr_examples.py`: draws the figure from those two files.

```bash
python examples/hdx_nmr/run_nmr_examples.py
python examples/hdx_nmr/plot_nmr_examples.py
```

## Source and licence

Measured values and structures come from Ferrari et al., *Large-scale discovery, analysis and
design of protein energy landscapes*, Nature (2026), doi:10.1038/s41586-026-10465-z. Data are
from Zenodo record 10.5281/zenodo.14983481 (`Dataset_4_HDXNMR.json`, `Dataset_9_AlphaFoldModels.zip`),
licensed CC BY 4.0. The structure files are unmodified.

We made one processing step. Each domain was measured in several replicates, sometimes on
constructs with different N-terminal tags. We aligned every replicate to the shortest
construct by sequence, averaged the net-charge-corrected ΔG_open (`free_energy_nc_corrected`)
per residue, and mapped the result onto the structure's residue numbering by sequence. A
residue is marked as a lower bound if any replicate flags it as too slow
(`too_slow_array`).
