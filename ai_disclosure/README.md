# AI Disclosure Materials

This directory documents the use of AI-assisted tools in the development of
code for this study, as described in Section 4.9 ("Use of AI-assisted
tools") of the manuscript, for the following five scripts:

```
figures/make_fig2_dataset_distribution.py
figures/make_fig3_singleseed_crossover.py
figures/make_fig4_multiseed_stability.py
evaluation/multiseed_eval.py
evaluation/crossover_fp_analysis.py
```

## What AI assistance actually consisted of

For all five scripts, the core algorithmic logic — IoU-based ground-truth
matching, sliding-window recall computation, crossover location by linear
interpolation, false-positive binning, the per-seed statistics (mean
difference, paired t-test, Cohen's d, pooled seed-to-seed SD), the
operational assessment rule, and the plotting and styling of each figure — was
originally authored by the researchers in a series of Google Colab notebook
cells, prior to any AI involvement. The pre-AI-assisted originals for all
five scripts are archived alongside the final versions in this directory
(see below).

AI assistance (Claude, Anthropic) was limited to:

1. **Repackaging** the notebook cells into standalone, reusable command-line
   scripts: adding `argparse` interfaces, replacing hardcoded Colab/Drive
   paths with configurable arguments, adding English-language docstrings and
   print output, and adding an optional `--results` flag to
   `make_fig4_multiseed_stability.py` to recompute its bar heights from a
   recorded results file instead of the hardcoded summary values.
2. **One bug fix** in `evaluation/multiseed_eval.py`: a separately maintained
   copy of this script (already in the GitHub repository, distinct from the
   notebook version archived here) computed the seed-to-seed standard
   deviation by pooling all 16 per-seed values together, rather than the
   pooled standard deviation of the two per-configuration SDs defined in
   Section 3.1 of the manuscript, sqrt((SD_B^2 + SD_C^2) / 2). This was
   corrected to match the notebook version's (already correct) formula and
   the numbers reported in Table 3. The default seed list and docstring
   example numbers in that same repository copy were also outdated (three
   seeds instead of eight) and were updated to match.
3. **Verification, not authorship, of the core logic.** Before adopting the
   repackaged versions, the `sliding_recall` and `find_crossover` functions
   were checked against the original notebook code with 200 rounds of
   randomised-input equivalence testing (200/200 identical output), and the
   output PDF dimensions of the repackaged figure scripts were checked
   against the published figures byte-for-byte (Fig. 2: 422.386 x 292.969
   pt; Fig. 4: 422.386 x 292.573 pt; both exact matches).

The statistical definitions, the operational assessment rule, and the
evaluation methodology itself (Section 3 of the manuscript) were authored by
the researchers independently of any AI tool; AI assistance did not extend
to defining or choosing the statistical method.

## Pre-AI-assisted versions

Genuine pre-AI-assisted originals are available for all five scripts and are
included here, extracted verbatim (including original Chinese comments and
notebook-specific hardcoded paths) from the authors' Google Colab notebook
history:

```
evaluation/multiseed_eval_pre_ai.py
evaluation/crossover_fp_analysis_pre_ai.py
figures/make_fig4_multiseed_stability_pre_ai.py
figures/make_fig2_dataset_distribution_pre_ai.py
figures/make_fig3_singleseed_crossover_pre_ai.py
```

Each archived original is a notebook cell that depended on variables defined
in earlier cells of the same notebook session (dataset paths, checkpoint
paths, frozen size thresholds); this context is documented in a header
comment in each file rather than reconstructed, to keep the archived
fragment faithful to what was actually run. Where a script went through
several iterative draft versions in the notebook (this applies to the Figure
2 and Figure 3 scripts), only the final version — the one the authors marked
as the corrected/adopted iteration and that was used to produce the
published figures — is archived here; intermediate drafts are not included.

For `evaluation/crossover_fp_analysis.py`, an AI-authored draft was written
independently before the notebook original was located; that draft was
discarded in favour of the notebook original once found, owing to substantive
algorithmic differences (evaluation grid range and point count, crossover
direction, handling of multiple crossings, and interpolation space), and is
not part of this repository or this disclosure.

## Scope of this disclosure

This directory documents AI-assisted development for the five scripts listed
above, for which the authors can specifically confirm the extent of AI
involvement and have verified the output against the results reported in the
manuscript. Other code in the repository (including any additional analysis
or training scripts not listed above, such as the Kvasir-SEG comparison in
Section 5.3) may also have involved comparable AI assistance during
development; detailed records distinguishing the exact extent of AI
involvement were not retained for every script, so these are not itemised
individually here. This disclosure is limited to what the authors can
specifically and accurately confirm, consistent with the transparency
principle stated in Section 4.9 of the manuscript.

## Verification and responsibility

All AI-assisted repackaging was reviewed, tested, and verified by the
authors against the original notebook code and against the results reported
in the manuscript prior to submission. The design of the evaluation
protocol, the choice of statistical procedures, the plotting logic, and the
interpretation of all results were determined and authored by the
researchers, who take full responsibility for the accuracy and integrity of
the code and the results it produces.

## Prompts

A representative record of the prompts and instructions used during the
AI-assisted repackaging of the scripts in this directory is provided in
`prompts.md`. Full records are retained by the authors and are available
upon request.
