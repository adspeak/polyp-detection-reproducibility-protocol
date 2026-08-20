# AI Prompt Records

Representative record of the prompts and instructions used during the
AI-assisted repackaging described in `README.md`. This reflects the actual
sequence of requests made during development, in the authors' own words
(paraphrased); it is a representative record, not necessarily an exhaustive
transcript. Full records are retained by the authors and available upon
request.

## Context

The GitHub repository initially shipped with a three-seed version of the
evaluation code and README, inconsistent with the eight-seed results
reported in the manuscript, and was missing the crossover/false-positive
analysis code, the figure-generation scripts, and the per-seed results
files. The AI-assisted work below was requested to bring the repository into
alignment with the manuscript before making it public, and to identify and
package the authors' own original notebook code wherever it could be
located, rather than have the AI author replacement code from scratch.

## evaluation/multiseed_eval.py

- Tool: Claude (Anthropic)
- Instruction (paraphrased): review the existing repository copy of this
  script against the numbers reported in the manuscript (Table 3: mean
  difference +0.82pp, pooled SD 3.33pp, p=0.41, Cohen's d 0.25) and correct
  any discrepancies; update the default seed list from three seeds to the
  eight seeds actually used (42-49) and update the docstring accordingly.
- Outcome: identified that the repository copy computed the seed-to-seed
  standard deviation incorrectly (pooling all 16 per-seed values instead of
  using the pooled-SD formula defined in Section 3.1); corrected the formula
  and the default seed list to match the authors' original notebook logic
  (archived as `multiseed_eval_pre_ai.py`), which already used the correct
  formula.

## evaluation/crossover_fp_analysis.py

- Tool: Claude (Anthropic)
- Instruction (paraphrased, first pass): write a script implementing the
  scale-response stability analysis exactly as specified in Section 3.3 of
  the manuscript (logarithmic sliding window, w=0.5, minimum 10 lesions per
  window, crossover by linear interpolation, false-positive binning by the
  frozen size thresholds), since no such script existed yet in the
  repository.
- Instruction (paraphrased, second pass, after the author located the
  original notebook code in a backup file): check whether this backup file
  contains the actual original crossover/false-positive code, and if so,
  discard the independently AI-authored version and repackage the original
  instead, verifying that the two are numerically equivalent.
- Outcome: the original notebook code was found and shown to differ from the
  AI-authored draft in the evaluation grid, crossover direction, handling of
  multiple crossings, and interpolation space; the AI-authored draft was
  discarded, and the notebook original (archived as
  `crossover_fp_analysis_pre_ai.py`) was repackaged with argparse and
  verified against the notebook's own `sliding_recall`/`find_crossover`
  behaviour using 200 rounds of randomised-input equivalence testing.

## figures/make_fig2_dataset_distribution.py

- Tool: Claude (Anthropic)
- Instruction (paraphrased): search the same backup files for the original
  plotting code for Figure 2 (LDPolypVideo vs. Kvasir-SEG relative-area
  distribution histogram) rather than reimplementing it from the figure as
  published; if found, repackage the original into a standalone script.
- Outcome: the original was located (multiple iterative draft versions; the
  author's own final "corrected" version was identified and adopted,
  archived as `make_fig2_dataset_distribution_pre_ai.py`) and repackaged
  with argparse and generalised paths. Output PDF dimensions were checked
  against the published figure and matched exactly.

## figures/make_fig3_singleseed_crossover.py

- Tool: Claude (Anthropic)
- Instruction (paraphrased): same as Figure 2 — locate the original plotting
  code for Figure 3 (single-seed recall-vs-scale curve with crossover
  marker) in the backup files rather than reimplementing it.
- Outcome: the original was located (three iterative draft versions; the
  author's final "legend font size unified" version was identified and
  adopted, archived as `make_fig3_singleseed_crossover_pre_ai.py`) and
  repackaged with argparse. Its `sliding_recall`/crossover logic was checked
  for equivalence with the adopted `crossover_fp_analysis.py` logic.

## figures/make_fig4_multiseed_stability.py

- Tool: Claude (Anthropic)
- Instruction (paraphrased): locate the original plotting code for Figure 4
  (recall by lesion-size stratum, grouped bar chart with error bars) in the
  same backup file as the crossover/false-positive analysis code, and
  repackage it; optionally add a way to recompute the bar heights from the
  recorded per-seed results file instead of the hardcoded values.
- Outcome: the original was located in the same backup file (immediately
  following the crossover/false-positive code) and repackaged with argparse
  and an optional `--results` flag. Output PDF dimensions were checked
  against the published figure and matched exactly.

---

Note: the paraphrasing above reflects the authors' recollection of what was
requested, reconstructed from the retained chat records of this work. Where
the retained records include enough detail to state a request precisely
(e.g. specific numeric targets, specific file locations), that detail is
included above; where they do not, the instruction is described at the level
of what can be confirmed.
