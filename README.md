This repository contains code to reproduce the results from the paper [<em>Fixation Sequences as Time Series: A Topological Approach to Dyslexia Detection</em>]().
In that paper, fixation sequences are interpreted as time series, based on which a topological pipeline for dyslexia detection is introduced.
The results that this repository creates are obtained by training and evaluating this pipeline on the corpus introduced in  [<em>The Copenhagen Corpus of Eye Tracking Recordings from Natural Reading of Danish Texts</em>](https://aclanthology.org/2022.lrec-1.182/) and expanded in [<em>Reading Does Not Equal Reading: Comparing, Simulating and Exploiting Reading Behavior across Populations</em>](https://aclanthology.org/2024.lrec-main.1187/).

---

__Requirements__

Python 3.11 or higher is required. Other dependencies are specified in `pyproject.toml`.

---

__Data__

The CopCo corpus is available for download [here](https://osf.io/ud8s5/).
For the scripts of this repository to run, the data must be placed in `./data_copco/` with the following structure:

```
data_copco/
├── ExtractedFeatures/
│   └── P{subject_id}.csv
└── FixationReports/
    └── FIX_report_P{subject_id}.txt
```

---

__Reproducing results__

To reproduce all results from the paper, run `python run_experiments.py`. This will run all model configurations.

To run individual experiments, use `python -m scripts.experiment` with the following arguments:

__Required arguments (all models):__
- `--model-name`: one of
    - `tsh`, for "time series homology", the method introduced in the paper;
    - `baseline_bjornsdottir` for the baseline method from [<em>Dyslexia Prediction from Natural Reading of Danish Texts</em>](https://aclanthology.org/2023.nodalida-1.7/);
    - `baseline_raatikainen` for the baseline method from [<em>Detection of developmental dyslexia with machine learning using eye movement data</em>](https://www.sciencedirect.com/science/article/pii/S2590005621000345);
    - `baseline_bjornsdottir_with_tsh` for the hybrid model combining the first baseline above with the features from time series homology; and
    - `baseline_raatikainen_with_tsh` for the hybrid model combining the second baseline above with the features from time series homology.
- `--level`: one of `trial` (for trial-level aggregation) and `reader` (for reader-level aggregation).
- `--classifier`: classifier to use (admissible values depend on the model; see below).

__Model-specific arguments:__

| Model | Admissible classifiers | Requires `--filtration-type` | Supports `--use-extended-persistence` |
|-------|------------------------|------------------------------|---------------------------------------|
| `tsh` | `svc` | Yes | Yes |
| `baseline_bjornsdottir` | `rf` | No | No |
| `baseline_raatikainen` | `svc`, `rf` | No | No |
| `baseline_bjornsdottir_with_tsh` | `rf` | Yes | Yes |
| `baseline_raatikainen_with_tsh` | `svc`, `rf` | Yes | Yes |

- `--filtration-type`: one of `horizontal`, `sloped`, `sigmoid`, `arctan`.
- `--use-extended-persistence`: flag to compute extended persistence instead of ordinary persistence (optional for models using TSH).

__Optional arguments (all models):__
- `--exclude-l2`: exclude CopCo-L2-readers from the dataset.
- `--min-n-fixations`: minimum number of fixations for a trial not to be discarded; only relevant for models using TSH (default: 5).
- `--n-splits`: number of splits for nested CV (default: 10 for trial-level, 5 for reader-level).
- `--n-iter`: number of iterations for randomized hyperparameter search (default: 200).
- `--n-jobs`: number of parallel jobs (default: 1).
- `--outdir`: output directory (default: `outfiles`).
- `--verbose`, `-v`: verbosity level (default: 0).
- `--overwrite`, `-o`: overwrite existing output files.
- `--seed`, `-s`: random seed (default: 42).

__Example:__

```bash
python -m scripts.experiment --model-name tsh --level trial --classifier svc --filtration-type sloped --n-jobs 8 --verbose 1
```

Executing the above will create a directory named `outfiles` that contains the CV results as a JSON file.

---

__For users of `uv`__

If `uv` is installed, required dependencies can be installed by running `uv pip install -r pyproject.toml`.
The environment specified in `uv.lock` can be recreated by running `uv sync`.

To reproduce all results from the paper, run `uv run run_experiments.py`. To run individual experiments, use `uv run -m scripts.experiment` with arguments as specified above.

---

__License__

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
