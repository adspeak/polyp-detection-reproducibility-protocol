# Assessing a Single-Run Gain in Polyp Detection at the Small-Lesion Scale: A Multi-Seed, Size-Stratified Replication Study

Code accompanying the paper *Assessing a Single-Run Gain in Polyp Detection at
the Small-Lesion Scale: A Multi-Seed, Size-Stratified Replication Study*
(PeerJ Computer Science, under review; submission CS-2026:07:145772).

The study takes a single-run improvement in small-lesion recall, obtained by
adding a high-resolution detection level (P2) to YOLOv8s, and reassesses it with
multiple training seeds, size-stratified recall and a graded verdict on the 95%
confidence interval of the mean difference. The P2 modification is the case
under assessment, not a proposed detection method.

> **Release for the revised manuscript: `v1.1.0`.** Releases `v1.0.0` and
> `v1.0.1` correspond to the original submission and are kept unchanged.

## Contents

| Folder | Contents |
| --- | --- |
| `data_preparation/` | LDPolypVideo video-level split and size thresholds (original submission) |
| `configs/` | +P2 and control (P2–P4) model definitions |
| `scripts/` | Training scripts of the original submission |
| `evaluation/` | Public evaluation scripts: size-stratified recall and scale-response crossover |
| `figures/` | Figure scripts; `make_fig1_protocol_overview_rev2.py` produces Figure 1 of the revised paper |
| `results/` | Recorded per-seed results of the original (historical) batch |
| `revision/` | Training, caching and analysis scripts of the revision (see below) |
| `ai_disclosure/` | Original code-assistance records (see "Use of AI-assisted tools") |

## Large files: release assets of `v1.1.0`

GitHub stores only code. The model weights, the inference caches and the
analysis outputs are attached to the
[`v1.1.0` release](https://github.com/adspeak/polyp-detection-reproducibility-protocol/releases/tag/v1.1.0)
as the following assets; `SHA256SUMS.txt` lists their checksums.

| Asset | Contents |
| --- | --- |
| `weights_runs.tar` | Retrained batch on LDPolypVideo (input 1280): baseline, +P2 and control (P2–P4), seeds 42–49 (24 runs) |
| `weights_runs_g2.tar` | Same-seed repeats: baseline and +P2, seeds 42–43 (4 runs) |
| `weights_runs_g3.tar` | Positive control: baseline at input 640, seeds 42–49 (8 runs); the 1280 arm is the baseline in `weights_runs.tar` |
| `weights_runs_g6.tar` | Kvasir-SEG group-based split (input 640): baseline and +P2, seeds 42–49 (16 runs) |
| `weights_runs_g7.tar` | Positive-control extension (registration S1): baseline at inputs 640 and 1280, seeds 50–57 (16 runs) |
| `cache_and_analysis.tar.gz` | One-time inference caches (LDPolypVideo; Kvasir-SEG), the outputs of every revision analysis, the exact-match check records of the public scripts, and the Kvasir-SEG split manifest (train/validation/test 700/150/150) |

Each weights archive holds, per run, `weights/best.pt` together with the
training arguments (`args.yaml`) and the per-epoch record (`results.csv`).

The frozen revision analysis addendum (v1.0) is at
https://doi.org/10.6084/m9.figshare.34919340, and the supplementary registration
of the positive-control extension (registration S1, v1.0) at
https://doi.org/10.6084/m9.figshare.34933821.

## Datasets

Both datasets are public and are not redistributed here.

- **LDPolypVideo**: Ma et al., MICCAI 2021, https://doi.org/10.1007/978-3-030-87240-3_37;
  data at https://github.com/dashishi/LDPolypVideo-Benchmark
- **Kvasir-SEG**: Jha et al., MMM 2020, https://doi.org/10.1007/978-3-030-37734-2_37;
  data at https://datasets.simula.no/kvasir-seg/

## Environment

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

All revision models and analyses were run with Python 3.8.10, PyTorch 2.0.0+cu118,
Ultralytics 8.4.104 and cuDNN 8700 on RTX 4090 D GPUs. The package lists
recorded on the hosts are in `revision/env/`. Results are not guaranteed to
reproduce under other Ultralytics versions.

## Revision analyses

Every revision analysis reads the one-time inference cache (confidence 0.001,
NMS IoU 0.7, at most 300 predictions per image) instead of re-running the test
set, and each script checks its inputs against an earlier record before
writing results. Scripts locate their sibling scripts in `revision/` and take
the workspace, dataset and cache locations as arguments (`--root`, `--repo`,
`--cache`, ...); `python <script> --help` lists them. The defaults and the
records in `revision/run_records/` use the paths of the server on which the
study was run (`revision/env/env.sh`).

| Paper item | Script |
| --- | --- |
| Training of all revision runs (Section 4.4) | `revision/train_paper3.py` |
| One-time inference cache (Section 4.5) | `revision/cache_predictions.py` |
| Exact-match check of the cache against the public scripts (Section 4.5; Table S1, no. 5) | `revision/check_public_vs_cache.py` |
| Initialisation-pairing check for the positive control and its extension (Section 4.5) | `revision/check_init_g7.py` |
| Kvasir-SEG group-based split (Section 4.7) | `revision/prepare_kvasir_seg.py`, `revision/prepare_kvasir_v3split.py` |
| Replication of the primary comparison; batch difference D (Section 5.2, Tables 5–6) | `revision/analysis_replication.py` |
| Normal quantile–quantile plots (Fig. S1) | `revision/analysis_qq_small_stratum.py` |
| Average precision and model cost: parameters, GFLOPs, inference time (Sections 4.4 and 5.2) | `revision/analysis_ap_cost.py` |
| Kvasir-SEG multi-seed comparison (Section 5.3) | `revision/analysis_kvasir.py` |
| Positive control: input resolution (Section 5.4) | `revision/analysis_positive_control.py` |
| Positive-control extension, registration S1 (Section 5.4) | `revision/analysis_g7_planning.py`, `revision/analysis_g7_extension.py` |
| Control configuration (Section 5.5, Table 7) | `revision/analysis_control.py` |
| Same-seed repeats (Section 5.6) | `revision/analysis_g2.py`, `revision/analysis_g2_predictions.py` |
| Ensembles at a common operating point (Section 5.7) | `revision/analysis_ensemble_v2.py` |
| Seed subsets (Section 5.8) | `revision/analysis_seed_subsets.py` |
| Field-diameter sensitivity (Section 5.8, Table S2) | `revision/analysis_fov.py` |
| Operating characteristics of the verdict rule (Sections 5.4 and 5.8) | `revision/analysis_oc_simulation.py` |
| Crossover definitions and one-to-one matching (Sections 5.2 and 5.8) | `revision/analysis_a11_v2.py` |
| Test-video cluster bootstrap (Section 5.8) | `revision/analysis_a10_video_bootstrap.py` |
| Recomputation of the recorded historical statistics | `revision/verify_recorded_results.py` |

`revision/evaluate_kvasir.py`, `revision/evaluate_coco_ap.py` and
`revision/ensemble_eval.py` are the evaluation helpers called by
`revision/run_records/autodl_reproduce_v1.0.1.sh`, the end-to-end reproduction
of release v1.0.1. The other files in `revision/run_records/` are the job queues
used to train and cache the runs, kept as run records. The ensemble fusion
(`weighted_boxes_fusion`, ensemble_boxes 1.0.9, MIT licence) is loaded from the
vendored copy in `revision/vendor/`.

## Original submission: public scripts

The scripts in `data_preparation/`, `scripts/`, `evaluation/` and `figures/`
are those of the original submission. Their use is unchanged:

```bash
python data_preparation/prepare_dataset.py --ldpolyp-root /path/to/LDPolypVideo --out ./ldpolyp
python data_preparation/compute_size_thresholds.py --dataset ./ldpolyp --out size_thresholds.npy
bash scripts/train_all_seeds.sh
python evaluation/multiseed_eval.py --dataset ./ldpolyp --thresholds size_thresholds.npy \
    --weights-dir ./weights --baseline baseline_seed42.pt ... --candidate p2_seed42.pt ... \
    --seeds 42 43 44 45 46 47 48 49 --imgsz 1280 --conf 0.25 --iou-match 0.5
python evaluation/crossover_fp_analysis.py --dataset ./ldpolyp --thresholds size_thresholds.npy \
    --weights-dir ./weights --baseline ... --candidate ... --seeds 42 43 44 45 46 47 48 49
```

The video-level split yields 2,118 training, 402 validation and 1,570 test
frames; the training-set tercile thresholds are 1.87% and 6.31% relative box
area. `evaluation/multiseed_eval.py` reports the three-condition rule of the
original submission; the revised paper uses the graded verdict implemented in
`revision/analysis_replication.py` and reports the earlier rule alongside
(Table 7). The paper's crossover is the first crossing from candidate above
to at or below baseline, as computed by `evaluation/crossover_fp_analysis.py`.

## Use of AI-assisted tools

ChatGPT (OpenAI) and Claude (Anthropic) assisted with translation, English
editing, manuscript drafting and organisation, and code preparation and
checking. Their code-related use included repackaging some of the authors'
plotting and analysis scripts and correcting a pooled-standard-deviation
calculation. The authors make the final decisions on the study design, analyses
and interpretation, run the experiments themselves, review all assisted code and
text, and take responsibility for the reported findings. The original
code-assistance records are retained in `ai_disclosure/`.

## Citation

Citation details will be added upon acceptance.
