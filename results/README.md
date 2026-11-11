# Recorded per-seed results of the original (historical) batch

These files hold the per-seed outputs of the original submission. The original
checkpoints were lost, so these records are the historical batch of the revised
paper; `revision/analysis_replication.py` reads them for the historical columns.

| File | Contents | Revised paper |
| --- | --- | --- |
| `multiseed_8seeds.json` | Size-stratified recall for the baseline and the +P2 candidate, per seed (42-49) | Table 4; historical column of Table 6 |
| `crossover_fp_8seeds.json` | Scale-response crossover location and small-scale false-positive change, per seed | Section 5.2 |
| `rebuild_verification.json` | Record of the dataset rebuild and its agreement with the original submission | Section 4.2 |

They were produced by `evaluation/multiseed_eval.py` and
`evaluation/crossover_fp_analysis.py`. The retrained batch and all revision
analyses are attached to release v1.1.0 (see the main README).
