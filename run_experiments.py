# ruff: noqa: E501
import subprocess

n_iter = 100
n_jobs = 8
verbose = 1
seed = 42

common_flag = (
    f"--n-iter {n_iter} --n-jobs {n_jobs} --verbose {verbose} --seed {seed}"
)

flag_sets_baselines = (
    f"--model-name baseline_bjornsdottir --classifier rf {common_flag}",
    f"--model-name baseline_raatikainen --classifier svc {common_flag}",
    f"--model-name baseline_raatikainen --classifier rf {common_flag}",
)

flag_sets_tsh_ordinary = tuple(
    f"--model-name tsh --filtration-type {filtration_type} --classifier {classifier} {common_flag}"
    for classifier in ("svc", "rf")
    for filtration_type in ("horizontal", "sloped", "sigmoid", "arctan")
)

flag_sets_tsh_extended = tuple(
    f"--model-name tsh --filtration-type {filtration_type} --classifier {classifier} --use-extended-persistence {common_flag}"
    for classifier in ("svc", "rf")
    for filtration_type in ("horizontal", "sloped", "sigmoid", "arctan")
)

flag_sets_tsh_aggregated_ordinary = tuple(
    f"--model-name tsh_aggregated --filtration-type {filtration_type} --classifier {classifier} {common_flag}"
    for classifier in ("svc", "rf")
    for filtration_type in ("horizontal", "sloped", "sigmoid", "arctan")
)

flag_sets_tsh_aggregated_extended = tuple(
    f"--model-name tsh_aggregated --filtration-type {filtration_type} --classifier {classifier} --use-extended-persistence {common_flag}"
    for classifier in ("svc", "rf")
    for filtration_type in ("horizontal", "sloped", "sigmoid", "arctan")
)

for flag_set in (
    flag_sets_baselines
    + flag_sets_tsh_ordinary
    + flag_sets_tsh_extended
    + flag_sets_tsh_aggregated_ordinary
    + flag_sets_tsh_aggregated_extended
):
    subprocess.run(
        f"uv run -m scripts.experiment {flag_set}",
        shell=True,
    )
    subprocess.run(
        f"uv run -m scripts.experiment {flag_set} --exclude-l2",
        shell=True,
    )
