# ruff: noqa: E501
import subprocess

n_splits = 10
n_iter = 100
n_jobs = 8
verbose = 1
seed = 42

common_flag = (
    f"--n-splits {n_splits} --n-iter {n_iter} --n-jobs {n_jobs} --verbose {verbose} --seed {seed}"
)

flag_sets = (
    f"--model-name baseline_bjornsdottir --classifier svc {common_flag}",
    f"--model-name baseline_raatikainen --classifier svc {common_flag}",
    f"--model-name baseline_raatikainen --classifier rf {common_flag}",
    f"--model-name tda_experiment --filtration-type horizontal --classifier svc {common_flag}",
    f"--model-name tda_experiment --filtration-type sloped --classifier svc {common_flag}",
    f"--model-name tda_experiment --filtration-type sigmoid --classifier svc {common_flag}",
    f"--model-name tda_experiment --filtration-type horizontal --classifier svc --use-extended-persistence {common_flag}",
    f"--model-name tda_experiment --filtration-type sloped --classifier svc --use-extended-persistence {common_flag}",
    f"--model-name tda_experiment --filtration-type sigmoid --classifier svc --use-extended-persistence {common_flag}",
    # f"--model-name tda_experiment --filtration-type horizontal --classifier rf {common_flag}",
    # f"--model-name tda_experiment --filtration-type sloped --classifier rf {common_flag}",
    # f"--model-name tda_experiment --filtration-type sigmoid --classifier rf {common_flag}",
    # f"--model-name tda_experiment --filtration-type horizontal --classifier rf --use-extended-persistence {common_flag}",
    # f"--model-name tda_experiment --filtration-type sloped --classifier rf --use-extended-persistence {common_flag}",
    # f"--model-name tda_experiment --filtration-type sigmoid --classifier rf --use-extended-persistence {common_flag}",
)

for flag_set in flag_sets:
    subprocess.run(f"uv run -m scripts.experiment {flag_set}", shell=True)
