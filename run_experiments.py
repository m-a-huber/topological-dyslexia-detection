# ruff: noqa: E501
import subprocess

n_splits = 10
n_iter = 100
n_jobs = 8
verbose = 1
seed = 42

common_flag = f"--n-splits {n_splits} --n-iter {n_iter} --n-jobs {n_jobs} --verbose {verbose} --seed {seed}"

flags_baselines = (
    f"--model-name baseline_bjornsdottir --classifier svc {common_flag}",
    f"--model-name baseline_raatikainen --classifier svc {common_flag}",
    f"--model-name baseline_raatikainen --classifier rf {common_flag}",
)

flags_tda_experiment_ordinary = tuple(
    f"--model-name tda_experiment --filtration-type {filtration_type} --classifier {classifier} {common_flag}"
    for classifier in ("svc", "rf")
    for filtration_type in ("horizontal", "sloped", "sigmoid", "arctan")
)

flags_tda_experiment_extended = tuple(
    f"--model-name tda_experiment --filtration-type {filtration_type} --classifier {classifier} --use-extended-persistence {common_flag}"
    for classifier in ("svc", "rf")
    for filtration_type in ("horizontal", "sloped", "sigmoid", "arctan")
)

for flag_set in flags_baselines + flags_tda_experiment_ordinary + flags_tda_experiment_extended:
    subprocess.run(f"uv run -m scripts.experiment {flag_set}", shell=True)
