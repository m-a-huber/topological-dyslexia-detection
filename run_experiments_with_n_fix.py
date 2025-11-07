# ruff: noqa: E501
import subprocess

n_repeats = 10
n_iter = 75
n_jobs = 8

flag_sets = (
    f"--model-name tda_experiment_horizontal_ordinary --with-n-fix --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sloped_ordinary --with-n-fix --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sigmoid_ordinary --with-n-fix --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_horizontal_extended --with-n-fix --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sloped_extended --with-n-fix --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
    f"--model-name tda_experiment_sigmoid_extended --with-n-fix --n-repeats {n_repeats} --n-iter {n_iter} --n-jobs {n_jobs} --verbose",
)

for flag_set in flag_sets:
    cmd = " ".join(["uv run -m scripts.experiment", flag_set])
    subprocess.run(cmd, shell=True)
