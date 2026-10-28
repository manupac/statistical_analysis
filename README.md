# Acoustic vs. Neural Representations of French Vowels (L1 and L2)

Do self-supervised speech models (Whisper, XLS-R) encode vowel structure and L1/L2 differences the way formant measurements do? Statistical comparison on the Russian–French Interference Corpus: French vowels from native French speakers (L1) and native Russian speakers (L2).

## Key findings

- **XLS-R encodes phonological structure far better than Whisper.** Nearest-centroid vowel identification (leave-one-speaker-out): XLS-R L20 **73.5%**, acoustic F1/F2 68.9%, Whisper L20 **33.8%**.
- **XLS-R geometry matches acoustics.** Mantel correlation with the acoustic vowel space: r = 0.58 (XLS-R) vs. 0.16 (Whisper).
- **Clustering recovers vowel height with XLS-R** (ARI 0.71 vs. 0.31 acoustic, 0.24 Whisper). Front/back/rounding is weakly recovered by all.
- **L1/L2 differences** survive FDR correction for 5 vowels in acoustics, 5 in Whisper, 8 in XLS-R. /u/, /y/ and /ɛ/ appear in all three.
- **Significant is not the same as meaningful.** With a ROPE analysis, almost no L1/L2 contrast is practically equivalent to zero in acoustics (mostly indeterminate), while neural contrasts mostly fall inside each model's noise floor.

## Method

DVC pipeline: Praat/Parselmouth formants (Lobanov-normalised), Whisper-medium and XLS-R hidden states (layers 4/12/20), PCA/UMAP, Mantel tests, permutation and nonparametric group tests (BH-FDR), mixed-effects models, bootstrap CIs with ROPE, hierarchical clustering (ARI), and a formant-trajectory sensitivity check.

## Reproduce

```bash
pip install -r requirements.txt
# place the corpus in data/
dvc repro
```

Outputs are written to `results/` (tables and figures per analysis).
