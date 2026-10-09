# Third-party code vendored into this repository

Both files in this directory are **copied verbatim, unmodified**, from
[`google-research/mt-metrics-eval`](https://github.com/google-research/mt-metrics-eval)
and are used as the reference implementations of the two meta-evaluation
measures reported in `sec:metaeval` of the paper. They were vendored rather
than installed because the compute cluster used for this work has no network
access; neither file needs anything beyond NumPy.

| file | upstream path | copyright | licence |
|---|---|---|---|
| `tau_optimization.py` | `mt_metrics_eval/tau_optimization.py` | Copyright 2021 Google LLC | Apache License 2.0 |
| `pce.py` | `mt_metrics_eval/pce.py` | Copyright 2024 Brian Thompson. All rights reserved. | Apache License 2.0 |

Each file retains its original Apache 2.0 header; nothing has been stripped
or altered, and no modifications were made, so no "changed files" notice is
required under section 4(b) of the licence. A full copy of the licence text
is at <http://www.apache.org/licenses/LICENSE-2.0>.

**What they implement.**

* `tau_optimization.py` is tie-calibrated pairwise accuracy,
  $\mathrm{acc}^{\ast}_{eq}$, from Deutsch, Foster and Freitag (2023),
  "Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and
  Tie Calibration", EMNLP 2023.
* `pce.py` is Soft Pairwise Accuracy, from Thompson, Mathur, Deutsch and
  Khayrallah (2024), "Improving Statistical Significance in Human Evaluation
  of Automatic Metrics via Soft Pairwise Accuracy", WMT 2024. Note that the
  class is named `PairwiseConfidenceError` upstream, which is why grepping
  for "SPA" in this directory finds nothing.

Our own pooled `acc*_eq` sweep was cross-checked against
`tau_optimization.py` on two worked cases and agrees to floating-point
identity, including the optimal $\varepsilon$
(`results/tables/metaeval_acc_validation.csv`).

This notice must ship with any release of the reproducibility package.
