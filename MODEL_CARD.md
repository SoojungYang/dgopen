# Model card: dgopen model-ens v0.1.0

## Summary

model-ens predicts per-residue amide opening free energy (ΔG_open, kcal/mol) from one protein
structure. On held-out proteins it **ranks residues within a protein** better than the
Best-Vendruscolo structural baseline. Macro Spearman ρ over 12 WT-external folds is 0.39–0.40
for model-ens and 0.35 for the baseline. Its **absolute** values are systematically compressed
and offset, and on one larger metalloprotein it lost to relative solvent accessibility alone.
Use it for within-protein ranking of protected and unprotected sites, on single-domain
proteins comparable in size to the training set (28–64 residues).

## Model

- **Architecture.** Distance-biased graph transformer with 6 layers, 8 heads, head width 32
  and hidden width 256, and no layer normalisation (3.49 M parameters). The attention logits
  for head h get an additive bias B_h[i, j] = Σ_k PF[i, j, k] w[h, k]. The pair tensor PF
  holds 16 Gaussian radial basis functions of the CA–CA distance (centres 2–22 Å, width
  2.5 Å), 5 sequence-separation channels and a constant. There is no other positional
  encoding.
- **Node input** (1319 dimensions): 39 standardised per-residue features (relative SASA,
  distance to the nearest SSE end, heavy-atom contacts to the amide N within 6.5 Å, Cβ
  contacts within 8 Å, amide H-bond count and donor flag, hydropathy, side-chain volume,
  charge at pH 7, relative sequence position, a constant pLDDT column, a proline flag, sin/cos
  of φ and ψ, a 3-state SSE one-hot, a 20-dim amino-acid one-hot) plus the frozen ESM-2 650M
  last-layer embedding (1280 dimensions).
- **Ensemble.** 4 training arms × 3 seeds, averaged with equal weights. The arms differ only
  in what the trunk was pretrained on before HDX fine-tuning: nothing (A1), Best-Vendruscolo
  ΔG_open (A2), an MD-derived mean target (A3), or MD-derived mean and variance targets (A4).
  No molecular dynamics is needed at inference. The across-member SD is reported as a spread
  diagnostic. It is not a calibrated uncertainty, because the members share data and features.

## Training data

- **Fine-tuning:** per-domain distributions of ΔG_open from mass-spectrometry HDX on 3590
  small domains (28–64 residues, median 46; designed mini-proteins and natural domains).
  MS does not resolve which residue carries which ΔG_open, so the loss is permutation
  invariant: a sorted L1 distance between the sorted predictions and the sorted measured
  values. Measurements at the fast-exchange floor get a one-sided hinge (margin
  1.5 kcal/mol) instead. Checkpoints were selected on validation-set L1.
  Source: the mHDX-MS dataset of Ferrari et al., *Nature* 654, 1108–1118 (2026), measured on
  domains from the mega-scale stability study of Tsuboyama et al., *Nature* 620, 434–444
  (2023; data and structures on Zenodo 10.5281/zenodo.7992926, CC BY 4.0).
- **Pretraining** (arms A2–A4): per-residue Best-Vendruscolo and MD-derived protection
  targets. The MD targets come from MegaSim: 271 wild-type domains and 21,458 point mutants
  simulated by Microsoft Research for BioEmu (Lewis et al., bioRxiv
  10.1101/2024.12.05.626885; data on Zenodo 10.5281/zenodo.15641184, CDLA-Permissive-2.0).
- **Structures:** model structures of the training domains. None has alternate locations or
  multiple models. Secondary structure and SASA come from the torsion-rule / Shrake-Rupley
  method that `--sse-method torsion` reproduces.

## Evaluation

Spearman ρ between predicted and measured ΔG_open over measured residues, averaged over
proteins (macro).

| benchmark | model-ens (release default) | model-ens (published, `--sse-method psea`) | Best-Vendruscolo |
|---|---|---|---|
| WT-external, 12 folds (held out) | 0.393 | 0.402 | 0.35 |
| HDX-NMR, 13 domains | 0.630 | 0.631 | 0.39 |

- **WT-external** proteins are not in the training set. Per-fold ρ ranges from about 0.0 to
  0.7, so on any single protein the result can be poor.
- **HDX-NMR:** 10 of the 13 domains are in the fine-tuning set. Their MS distributions, not
  their NMR per-residue labels, were seen in training. Treat this row as partly in-sample.
- The release default differs from the published evaluation in two inference-time details:
  (1) torsion-rule SSE and numpy SASA, which match training, in place of P-SEA and biotite
  SASA, which do not; (2) keeping only the first alternate location of each atom, where the
  published evaluation counted every conformer in contact and H-bond counts. The net change
  is −0.009 macro ρ on WT-external. The median per-residue change is 0.07 kcal/mol and the
  largest is 0.75 kcal/mol.

## Limitations: read before use

1. **Absolute values are not calibrated.** Predictions span a narrower range than
   measurements and can be offset low on larger proteins. On holo SOD1 (153-residue
   subunits, scored as the dimer), point-resolved residues were offset by −4.2 to
   −6.0 kcal/mol across WT, G93A and I113T. The predicted range was up to about 3× smaller
   than measured (I113T: 3.4 vs 10.1 kcal/mol).
2. **Failure case, SOD1.** On metalated SOD1 (WT, G93A, I113T), model-ens lost to plain
   relative SASA in 6 of 6 comparisons (e.g. WT point-resolved residues: ρ 0.365 vs 0.420).
   The leading candidate explanation is that the model has no metal or disulfide channel.
   Domain size is an alternative that this case cannot rule out.
3. **Size.** Training domains are 28–64 residues. Proteins well outside this range are
   extrapolation.
4. **Not represented:** ligands, metals, cofactors, oligomeric partners, disulfide bonds as
   such, pH and temperature. One chain is scored in isolation.
5. **Sensitivity to the structure.** Predictions depend on the input coordinates. On ecDHFR,
   the predicted per-residue differences between alternative crystal structures were
   uncorrelated with how much the structures differed (ρ = 0.03 over 10 structure pairs). Do
   not read small differences between conformers as conformational signal.
6. **Exchangeable sites only.** Prolines and the first two residues get no value.

## Intended use

Within-protein ranking of amide protection for single-domain proteins: prioritising HDX
experiments, designing probes, or generating hypotheses about which segments are most
protected. Not intended for absolute stability (ΔG_unfold) estimation, for ranking
different proteins against each other, or for clinical or diagnostic use.

## Provenance

Each member's seed, training arm, training-code commit and sha256 are recorded in
`src/dgopen/data/weights_manifest.json`. Release reproduction:
the packaged code reproduces the stored predictions of the published evaluation exactly
(max |Δ| = 0 kcal/mol over all members) when given the published inputs.
