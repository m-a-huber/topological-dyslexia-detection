# ruff: noqa: E501
import subprocess

n_repeats = 10
n_iter = 75
n_jobs = 8

flag_sets = (
    f"--model-name baseline_bjornsdottir --n-repeats {n_repeats} --n-jobs {n_jobs} --verbose",
    f"--model-name baseline_raatikainen_svc --n-repeats {n_repeats} --n-jobs {n_jobs} --verbose",
    f"--model-name baseline_raatikainen_rf --n-repeats {n_repeats} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_horizontal_ordinary --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sloped_ordinary --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sigmoid_ordinary --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_horizontal_extended --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sloped_extended --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sigmoid_extended --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
)

for flag_set in flag_sets:
    cmd = " ".join(["uv run -m scripts.experiment", flag_set])
    subprocess.run(cmd, shell=True)
