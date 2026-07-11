# A Multi-Seed, Clinically-Anchored Protocol for Reproducible Evaluation of Small Polyp Detection

This repository contains the code accompanying the paper *A Multi-Seed,
Clinically-Anchored Protocol for Reproducible Evaluation of Small Polyp
Detection*. It provides everything needed to reproduce the case study that
instantiates the proposed evaluation protocol, using the addition of a
high-resolution detection head (P2) to a YOLOv8s detector as a **representative
case study** — not as a proposed detection method.

The protocol itself is the contribution of the paper. The P2 modification is
only a test case used to demonstrate that a single-seed evaluation can report a
clinically framed "improvement" that the protocol reveals to be an artefact of
training variability.

> **Paper DOI:** _to be added upon acceptance._

## What this code reproduces

| Paper item | Script |
| --- | --- |
| Video-level train/val/test split with leakage control (Sec. 4.2) | `data_preparation/prepare_dataset.py` |
| Clinically anchored size thresholds, 1.87% / 6.31% (Sec. 4.3) | `data_preparation/compute_size_thresholds.py` |
| Model configurations: baseline, +P2, control (Table 2) | `configs/` + `scripts/train.py` |
| Per-seed and across-seed size-stratified recall (Tables 3, 4) | `evaluation/multiseed_eval.py` |
| Operational assessment rule on the small stratum (Sec. 5.2) | `evaluation/multiseed_eval.py` |

## Datasets

Both datasets are publicly available and are **not** redistributed here.

- **LDPolypVideo** (primary dataset): Ma et al., *LDPolypVideo Benchmark: A
  Large-Scale Colonoscopy Video Dataset of Diverse Polyps*, MICCAI 2021.
  https://doi.org/10.1007/978-3-030-87240-3_37
- **Kvasir-SEG** (contrasting dataset, dataset-scale dependency analysis):
  Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020.
  https://doi.org/10.1007/978-3-030-37734-2_37

Download them from their official sources and point the scripts at the local
paths as shown below.

## Reproducibility settings

All stochastic elements are controlled by fixed seeds and settings so that a
reported result can be closely replicated:

- **Training seeds:** 42, 43, 44 (the multi-seed assessment uses all three)
- **Frame sampling stride:** 10 (one frame in every ten per video)
- **Split:** video-level, 85:15 train/val on the development videos, with the 60
  official test videos kept intact; seed 42 controls the train/val video split
- **Training:** 100 epochs, image size 1280, batch 16, early-stopping patience
  30, `deterministic=True`
- **Size thresholds:** computed on the training set only, then frozen; stored as
  decimal fractions (0.0187 / 0.0631), not percentages

## Installation

```bash
git clone <this-repository-url>
cd <this-repository>
pip install -r requirements.txt
```

The code was developed with Ultralytics YOLOv8 on Python 3.10+ (Google Colab,
Tesla T4 GPU).

## Step-by-step reproduction

Run all commands from the repository root.

**1. Prepare the dataset** (video-level split, leakage control):

```bash
python data_preparation/prepare_dataset.py \
    --ldpolyp-root /path/to/LDPolypVideo \
    --out ./ldpolyp
```

This produces `./ldpolyp/` with `images/{train,val,test}`,
`labels/{train,val,test}`, and `data.yaml`. On the paper's build this yields
2,118 training, 402 validation, and 1,570 test images.

**2. Compute the frozen size thresholds** (training set only):

```bash
python data_preparation/compute_size_thresholds.py \
    --dataset ./ldpolyp \
    --out size_thresholds.npy
```

This reproduces the tercile boundaries 1.87% and 6.31% and the balanced strata
of 626 / 611 / 640 lesions (small / medium / large), and reports the clinical
anchoring check (the 5 mm boundary maps to ~2.04% relative area).

**3. Train the configurations under all three seeds:**

```bash
bash scripts/train_all_seeds.sh
```

Or train a single configuration/seed directly, e.g.:

```bash
python scripts/train.py --config baseline --seed 42 --data ./ldpolyp/data.yaml
python scripts/train.py --config p2       --seed 42 --data ./ldpolyp/data.yaml
```

Best checkpoints are written to `runs/<config>_seed<seed>/weights/best.pt`.

**4. Run the multi-seed, size-stratified evaluation:**

Collect the six checkpoints (baseline and +P2 for seeds 42/43/44) into a
directory, then:

```bash
python evaluation/multiseed_eval.py \
    --dataset ./ldpolyp \
    --thresholds size_thresholds.npy \
    --weights-dir ./weights \
    --baseline  baseline_seed42.pt baseline_seed43.pt baseline_seed44.pt \
    --candidate p2_seed42.pt       p2_seed43.pt       p2_seed44.pt \
    --seeds 42 43 44 \
    --imgsz 1280 --conf 0.25 --iou-match 0.5 \
    --out multiseed_results.json
```

This prints the per-seed small-lesion recall, the across-seed mean +/- SD for
each stratum (Tables 3 and 4), and the operational assessment rule outcome for
the small stratum. On the paper's checkpoints the small-stratum mean difference
is +2.4 +/- 2.8 pp with a paired t-test p = 0.30, which does not satisfy the
operational assessment rule.

## Terminology (kept precise, as in the paper)

- The across-seed difference `C_bar_k - B_bar_k` is the **mean difference**
  (Delta_k), the primary quantity of interest.
- **Cohen's d** (the standardized effect size) is reported only descriptively;
  with three seeds it is highly uncertain and is not the basis for the
  reliability decision.
- The scale-response crossover is reported as an **empirical observation**, not
  a theoretical threshold.
- P2 is a **case study / representative modification** throughout, never a
  claimed contribution.

## Repository layout

```
configs/
    yolov8s-P2P3P4P5.yaml     # +P2 candidate modification
    yolov8s-P2P3P4.yaml       # control (P5 detection head removed)
data_preparation/
    prepare_dataset.py         # video-level split + YOLO-format conversion
    compute_size_thresholds.py # training-set tercile + clinical anchoring
scripts/
    train.py                   # train one config under one seed
    train_all_seeds.sh         # train baseline + P2 across seeds 42/43/44
evaluation/
    multiseed_eval.py          # size-stratified recall + operational rule
requirements.txt
LICENSE
```

Note: the baseline uses the stock YOLOv8s architecture loaded directly from
`yolov8s.pt`, so no `baseline.yaml` is needed.

## Citation

_Citation details (BibTeX) will be added upon acceptance._
