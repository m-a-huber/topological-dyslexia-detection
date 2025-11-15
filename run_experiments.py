import subprocess

n_splits = 10
n_iter = 100
n_jobs = 8

common_flag = (
    f"--n-splits {n_splits} --n-iter {n_iter} --n-jobs {n_jobs} --verbose"
)

flag_sets = (
    f"--model-name baseline_bjornsdottir {common_flag}",
    f"--model-name baseline_raatikainen_svc {common_flag}",
    f"--model-name baseline_raatikainen_rf {common_flag}",
    f"--model-name tda_experiment_horizontal_ordinary {common_flag}",
    f"--model-name tda_experiment_sloped_ordinary {common_flag}",
    f"--model-name tda_experiment_sigmoid_ordinary {common_flag}",
    f"--model-name tda_experiment_horizontal_extended {common_flag}",
    f"--model-name tda_experiment_sloped_extended {common_flag}",
    f"--model-name tda_experiment_sigmoid_extended {common_flag}",
)

for flag_set in flag_sets:
    cmd = " ".join(["uv run -m scripts.experiment", flag_set])
    subprocess.run(cmd, shell=True)
