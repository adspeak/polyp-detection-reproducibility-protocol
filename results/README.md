# Recorded per-seed results

These files hold the per-seed outputs reported in the paper, so the summary
statistics can be checked without retraining the sixteen checkpoints.

| File | Contents | Paper item |
| --- | --- | --- |
| `multiseed_8seeds.json` | Size-stratified recall for the baseline and the +P2 candidate, per seed (42-49) | Tables 3 and 4 |
| `crossover_fp_8seeds.json` | Scale-response crossover location and small-scale false-positive change, per seed | Section 5.2 |

Regenerate them with `evaluation/multiseed_eval.py` and
`evaluation/crossover_fp_analysis.py` respectively.
