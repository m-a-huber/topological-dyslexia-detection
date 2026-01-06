# ruff: noqa: E501
import subprocess

from scripts import constants

n_iter = 200
n_jobs = 8
verbose = 1
seed = 42

common_flag = (
    f"--n-iter {n_iter} --n-jobs {n_jobs} --verbose {verbose} --seed {seed}"
)

flag_sets_baseline_bjornsdottir = tuple(  # 2 flag sets
    f"--model-name baseline_bjornsdottir --level {level} --classifier {classifier} {common_flag}"
    for level in constants.admissible_levels
    for classifier in constants.admissible_classifiers_bjornsdottir
)

flag_sets_baseline_raatikainen = tuple(  # 4 flag sets
    f"--model-name baseline_raatikainen --level {level} --classifier {classifier} {common_flag}"
    for level in constants.admissible_levels
    for classifier in constants.admissible_classifiers_raatikainen
)

flag_sets_tsh_ordinary = tuple(  # 8 flag sets
    f"--model-name tsh --level {level} --filtration-type {filtration_type} --classifier {classifier} {common_flag}"
    for level in constants.admissible_levels
    for filtration_type in constants.admissible_filtration_types_tsh
    for classifier in constants.admissible_classifiers_tsh
)

flag_sets_tsh_extended = tuple(  # 8 flag sets
    f"--model-name tsh --level {level} --filtration-type {filtration_type} --classifier {classifier} {common_flag} --use-extended-persistence"
    for level in constants.admissible_levels
    for filtration_type in constants.admissible_filtration_types_tsh
    for classifier in constants.admissible_classifiers_tsh
)

flag_sets_baseline_bjornsdottir_with_tsh_ordinary = tuple(  # 8 flag sets
    f"--model-name baseline_bjornsdottir_with_tsh --level {level} --filtration-type {filtration_type} --classifier {classifier} {common_flag}"
    for level in constants.admissible_levels
    for filtration_type in constants.admissible_filtration_types_tsh
    for classifier in constants.admissible_classifiers_bjornsdottir
)

flag_sets_baseline_raatikainen_with_tsh_ordinary = tuple(  # 16 flag sets
    f"--model-name baseline_raatikainen_with_tsh --level {level} --filtration-type {filtration_type} --classifier {classifier} {common_flag}"
    for level in constants.admissible_levels
    for filtration_type in constants.admissible_filtration_types_tsh
    for classifier in constants.admissible_classifiers_raatikainen
)

flag_sets_baseline_bjornsdottir_with_tsh_extended = tuple(  # 8 flag sets
    f"--model-name baseline_bjornsdottir_with_tsh --level {level} --filtration-type {filtration_type} --classifier {classifier} {common_flag} --use-extended-persistence"
    for level in constants.admissible_levels
    for filtration_type in constants.admissible_filtration_types_tsh
    for classifier in constants.admissible_classifiers_bjornsdottir
)

flag_sets_baseline_raatikainen_with_tsh_extended = tuple(  # 16 flag sets
    f"--model-name baseline_raatikainen_with_tsh --level {level} --filtration-type {filtration_type} --classifier {classifier} {common_flag} --use-extended-persistence"
    for level in constants.admissible_levels
    for filtration_type in constants.admissible_filtration_types_tsh
    for classifier in constants.admissible_classifiers_raatikainen
)

# Run experiments including L2-readers
for flag_set in (  # 86 flag sets
    flag_sets_baseline_bjornsdottir
    + flag_sets_baseline_raatikainen
    + flag_sets_tsh_ordinary
    + flag_sets_tsh_extended
    + flag_sets_baseline_bjornsdottir_with_tsh_ordinary
    + flag_sets_baseline_raatikainen_with_tsh_ordinary
    + flag_sets_baseline_bjornsdottir_with_tsh_extended
    + flag_sets_baseline_raatikainen_with_tsh_extended
):
    subprocess.run(
        f"uv run -m scripts.experiment {flag_set}",
        shell=True,
    )

# Run experiments excluding L2-readers
for flag_set in (  # 86 flag sets
    flag_sets_baseline_bjornsdottir
    + flag_sets_baseline_raatikainen
    + flag_sets_tsh_ordinary
    + flag_sets_tsh_extended
    + flag_sets_baseline_bjornsdottir_with_tsh_ordinary
    + flag_sets_baseline_raatikainen_with_tsh_ordinary
    + flag_sets_baseline_bjornsdottir_with_tsh_extended
    + flag_sets_baseline_raatikainen_with_tsh_extended
):
    subprocess.run(
        f"uv run -m scripts.experiment {flag_set} --exclude-l2",
        shell=True,
    )
