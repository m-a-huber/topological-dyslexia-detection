# ruff: noqa: E501
import subprocess

flag_sets = (
    "--model-name baseline_bjornsdottir --n-repeats 10 --n-jobs 8 --verbose",
    "--model-name baseline_raatikainen_svc --n-repeats 10 --n-jobs 8 --verbose",
    "--model-name baseline_raatikainen_rf --n-repeats 10 --n-jobs 8 --verbose",
    "--model-name tda_experiment_horizontal --no-extended-persistence --n-repeats 10 --n-jobs 8 --n-iter 10 --verbose",
    "--model-name tda_experiment_sloped --no-extended-persistence --n-repeats 10 --n-jobs 8 --n-iter 10 --verbose",
    "--model-name tda_experiment_sigmoid --no-extended-persistence --n-repeats 10 --n-jobs 8 --n-iter 10 --verbose",
    "--model-name tda_experiment_horizontal --extended-persistence --n-repeats 10 --n-jobs 8 --n-iter 10 --verbose",
    "--model-name tda_experiment_sloped --extended-persistence --n-repeats 10 --n-jobs 8 --n-iter 10 --verbose",
    "--model-name tda_experiment_sigmoid --extended-persistence --n-repeats 10 --n-jobs 8 --n-iter 10 --verbose",
)

for flag_set in flag_sets:
    cmd = " ".join(["uv run -m scripts.experiment", flag_set])
    subprocess.run(cmd, shell=True)
