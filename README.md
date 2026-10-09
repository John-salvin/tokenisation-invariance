# Tokenisation Invariance in Learned Machine Translation Metrics

### Diagnosis and Repair

> **Status:** anonymous submission. This repository contains all the code and data behind the paper, every number it reports, and **SI-Check**, the auditing tool the paper releases.

---

## What we did

A learned metric such as COMET reads a translation through a tokeniser, which cuts the text into pieces. The same text can be cut in many valid ways that all decode to the identical string. A metric that scores meaning should give the same score whichever way the text is cut. We call this **Tokenisation Invariance**.

We test it by re-cutting each translation into more and more fragmented token sequences, keeping the text, the reference and the human score fixed, and scoring it again. We then test the related **Script Invariance**: does the metric still agree with humans when the same translation is written in Latin script? Finally we ask which part of the failure rescaling can fix, and repair the rest by training.

---

## Key findings

| Finding | Value | Notebook |
|---|---|---|
| Agreement falls as fragmentation rises (Table 1) | 18 of 20 settings; Gujarati 0.452 -> 0.038; Arabic falls below zero (-0.339) | `02_tokenisation` |
| The same with a different vocabulary (MetricX-24, mT5) | same direction in all five Indic languages | `02_tokenisation` |
| Romanisation, IndicMT Eval, 11 metrics (Table 3) | every learned metric loses agreement | `03_script_indic` |
| Romanisation, 7 WMT settings, 11 metrics (Table 4) | drop in 42 of 49 compared cases, rise in none | `04_script_wmt` |
| Serbian: Latin script without extra tokens | no change, -0.009 [-0.028, 0.009] | `04_script_wmt` |
| System rankings | romanisation reorders 18 of 75 system pairs for COMET-22 | `05_rankings` |
| Lost meaning or distortion? | a romanisation that loses nothing causes most of the drop | `06_meaning` |
| Rescaling (COMET-QN) | fixes the scale between groups, never the ranking within one | `07_rescaling` |
| Repair by training (A2) | 107% of the plain adapter's repair at a cost of 0.017; positive in 5 of 6 unseen-script settings | `08_repair` |

---

## Quickstart

```bash
# 1. Environment (Python 3.10-3.12, CPU only)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run notebooks 01 -> 10 in order: rebuild every table, then check every number in the paper
make notebooks            # about 30 minutes

# 3. Tests
make test
```

To read the analysis step by step instead, start `jupyter lab` and open `notebooks/01_data.ipynb`. No GPU is needed and no model is downloaded: the metric scores were computed on GPUs and are committed in `data/`. The tokenisers' vocabulary files (a few MB) are fetched from the Hugging Face Hub on first use; if the Hub is blocked, set `HF_ENDPOINT=https://hf-mirror.com`.

---

## The notebooks

Each notebook covers one part of the paper. It explains in plain words what was done, rebuilds the tables of that part, and shows the numbers the paper prints.

| Notebook | Paper | What it shows |
|---|---|---|
| `01_data` | Section 3 | the corpora, the twenty settings, the off-target systems, an example of re-segmentation |
| `02_tokenisation` | Section 4 | Table 1, Table 2, the second vocabulary, constant length, the encoder probe |
| `03_script_indic` | Section 5.1 | Table 3: eleven metrics, native against romanised |
| `04_script_wmt` | Section 5.2 | Table 4: seven WMT settings, and the Serbian control |
| `05_rankings` | Section 5.3 | system pairs that change order, pairwise accuracy |
| `06_meaning` | Section 6 | how much of the drop is lost meaning and how much distortion |
| `07_rescaling` | Section 7 | what quantile normalisation fixes and what it cannot |
| `08_repair` | Section 8 | the adapters, the choice of lambda, transfer to new scripts |
| `09_appendix` | Limitations, appendices | what predicts the effect size, vocabulary, sampler checks, data quality |
| `10_verify_numbers` | all | recomputes every number in the paper from the tables |

The tables are written to `results/tables/` (they are also committed, so you can read them without running anything). `paper_numbers.yaml` lists every number the paper prints and the table it comes from; notebook 10 (or `make verify`) checks all of them.

---

## Repository layout

```
tokenisation-invariance/
├── README.md  LICENSE  Makefile  CITATIONS.bib
├── requirements.txt          for the notebooks (CPU)
├── requirements-gpu.txt      for experiments/ (GPU)
├── notebooks/                01-10, one per part of the paper
├── src/tokinv/               the analysis code the notebooks call
├── src/sicheck/              SI-Check
├── experiments/              all the GPU code that produced data/, grouped by paper section
├── data/                     the data (see data/README.md)
├── results/tables/           every table of the paper
├── paper_numbers.yaml        every number in the paper and where it comes from
├── scripts/                  run_all.py, verify_paper_numbers.py, download_adapters.py
└── tests/
```

`experiments/README.md` lists, for every file in `data/`, the script that produced it. The trained adapters of Section 8 (74 files, about 770 MB) are too large for git; `python scripts/download_adapters.py` downloads them.

---

## SI-Check

The tool the paper releases. It tests any metric for both invariances on any data with a segment-level human score.

```bash
pip install -e ".[score]"      # adds torch and unbabel-comet

# Tokenisation Invariance: re-cut, score and report in one command
si-check run --input my_data.tsv --metric comet22 --out audit/

# Script Invariance: two scored versions of the same segments
si-check script --a native_scores.tsv --b romanised_scores.tsv
```

The input needs the columns `mt` (the translation) and `human` (a human score, higher is better), plus `src` and `ref` if the metric uses them. The report gives agreement in every fragmentation bucket, the trend across buckets with its 95 percent interval, and a verdict. `--metric` accepts `comet22`, `cometkiwi22`, `cometkiwi23` or another COMET checkpoint on the Hugging Face Hub (xCOMET is not supported: part of its score cannot be computed from token ids). For any other metric, run `si-check sample`, score the re-cut samples yourself, add a `score` column, and run `si-check report`.

Example output on the paper's Gujarati data:

```
Tokenisation Invariance  (293 segments)
  canonical agreement      +0.597
  [1.00,1.25)   ratio 1.22  agreement +0.452  (n=1465)
  ...
  [2.75,3.00)   ratio 2.78  agreement +0.038  (n=980)
  monotonicity rho         -1.000   exact p 5e-05
  95% segment bootstrap    [-1.000, -0.833]
  => VIOLATED: agreement falls with fragmentation
```

---

## Settings that fix every number

| | |
|---|---|
| **Python** | 3.10-3.12 (run on 3.12) |
| **Dependencies** | pinned in `requirements.txt`; `transformers==4.40.2`, because token counts can change between releases |
| **Tokenisers** | pinned to exact Hugging Face revisions in `src/tokinv/tokenisers.py` |
| **Re-segmentation** | five seeds (0-4) per segment and bucket |
| **Intervals** | 300 bootstrap resamples of whole segments, seed 0, percentile interval |
| **Language order** | Gujarati, Tamil, Malayalam, Marathi, Hindi, everywhere |

---

## Data and licences

* **IndicMT Eval** (Sai et al., ACL 2023), created by [AI4Bharat](https://github.com/AI4Bharat): five Indic languages, 1,400 segments each, six MT systems, MQM scores. The dataset is governed by its own licence from [AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval). The romanised version and all metric scores are ours.
* **WMT24 and WMT25** human scores and system outputs, through [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval), governed by its terms.

`data/README.md` describes every file and where to download the originals. `CITATIONS.bib` lists every dataset, tool, model and method used.

Code: MIT (`LICENSE`). `src/tokinv/vendor/` holds two files from mt-metrics-eval under Apache 2.0, with their notice.
