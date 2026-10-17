# Changelog

## 0.2.0 (2026-10-17)

Inference-only changes; the model weights are unchanged (Hugging Face revision `v0.1.0`).

- **Predictions no longer depend on how the input structure is oriented.** v0.1.x computed
  relative SASA with a 92-point Shrake-Rupley sphere fixed in the coordinate frame, and placed
  the amide H of the first residue (and of any residue after a chain break) at N + (0, 0, 1)
  Å in that frame. Rotating the input therefore changed predictions by a median of 0.020
  kcal/mol per residue, and by up to 0.38 kcal/mol (99 test structures, 6 random rotations
  each). Features are now computed in a canonical principal-axes frame. The remaining change
  under rotation (median 0.0014 kcal/mol) matches the effect of re-rounding the PDB
  coordinates alone (0.0009).
- **Chain breaks are detected from geometry** (previous residue missing, missing its C, or
  C(i−1)–N(i) > 2.0 Å, which also catches gaps hidden by consecutive numbering). For the
  residue after a break, the amide H-bond features and φ are set to the training mean
  instead of being computed from an amide H placed without a preceding carbonyl. ψ of the
  residue before a geometric break is treated the same way. A new output column,
  `after_chain_break`, flags these residues.
- Effect at the input orientation: median per-residue change 0.008 kcal/mol, at most
  0.14 kcal/mol; at most 0.28 kcal/mol on residues after a chain break. Benchmarks
  (macro Spearman ρ): WT-external 12 folds 0.393 → 0.400; HDX-NMR 13 domains 0.630 → 0.631.

Thanks to the colleague who reported the chain-break amide-H issue.

## 0.1.1 (2026-10-17)

- Weights are hosted at huggingface.co/Soojung/dgopen (0.1.0 pointed at the wrong account).

## 0.1.0 (2026-10-17)

- First release: 12-member ensemble (model-ens), CLI, Best-Vendruscolo baseline.
