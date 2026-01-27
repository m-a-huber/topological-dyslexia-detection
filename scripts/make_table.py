# ruff: noqa: E501
import json
from pathlib import Path

import numpy as np

# Metric to compute ("roc_auc" or "pr_auc")
METRIC = "roc_auc"
# Outdirs containing CV-result files
OUTDIRS = [
    Path("outfiles"),
    Path("outfiles_without_l2"),
]


def mean_std(cv_file: Path) -> str:
    try:
        with open(cv_file, "r") as f:
            data = json.load(f)
        mean = np.mean(data[METRIC])
        std = np.std(data[METRIC])
        return rf"{mean:.2f}\pm {std:.2f}"
    except FileNotFoundError:
        return r"\text{---}"


def get_tsh_line(
    model_name: str,  # must be "tsh_<filtration_type>"
    level: str,
    outdirs: list[Path],
) -> str:
    filtration_type = model_name.split("_")[1]
    cv_files = (
        outdir
        / f"cv_results_tsh_{level}_level_{filtration_type}_{persistence_type}_svc_seed_42.json"
        for outdir in outdirs
        for persistence_type in ["ordinary", "extended"]
    )
    model_name_pretty = rf"TSH\textsubscript{{{filtration_type}}}"
    numbers = [mean_std(cv_file) for cv_file in cv_files]
    return (
        rf"& {model_name_pretty} "
        + " ".join([rf"& {s}" for s in numbers])
        + r" \\"
    )


def get_baseline_line(
    model_name: str,  # must be "baseline_<name>_<classifier>"
    level: str,
    outdirs: list[Path],
) -> str:
    name, classifier = model_name.split("_")[1:]
    cv_files = (
        outdir
        / f"cv_results_baseline_{name}_{level}_level_{classifier}_seed_42.json"
        for outdir in outdirs
    )
    if name == "bjornsdottir":
        model_name_pretty = r"BL\textsubscript{Bjö}"
    elif name == "raatikainen":
        if classifier == "rf":
            model_name_pretty = r"BL\textsubscript{Raa-RF}"
        elif classifier == "svc":
            model_name_pretty = r"BL\textsubscript{Raa-SVC}"
    numbers = [mean_std(cv_file) for cv_file in cv_files]
    return (
        rf"& {model_name_pretty} "
        + " ".join([rf"& \multicolumn{{2}}{{C}}{{{s}}}" for s in numbers])
        + r" \\"
    )


def get_baseline_with_tsh_line(
    model_name: str,  # must be "baseline_<name>_<classifier>_with_tsh_<filtration_type>"
    level: str,
    outdirs: list[Path],
) -> str:
    baseline_name, classifier, _, _, filtration_type = model_name.split("_")[
        1:
    ]
    cv_files = (
        outdir
        / f"cv_results_baseline_{baseline_name}_with_tsh_{level}_level_{filtration_type}_{persistence_type}_{classifier}_seed_42.json"
        for outdir in outdirs
        for persistence_type in ["ordinary", "extended"]
    )
    if baseline_name == "bjornsdottir":
        model_name_pretty = (
            rf"BL\textsubscript{{Bjö}}+TSH\textsubscript{{{filtration_type}}}"
        )
    elif baseline_name == "raatikainen":
        if classifier == "rf":
            model_name_pretty = rf"BL\textsubscript{{Raa-RF}}+TSH\textsubscript{{{filtration_type}}}"
        elif classifier == "svc":
            model_name_pretty = rf"BL\textsubscript{{Raa-SVC}}+TSH\textsubscript{{{filtration_type}}}"
    numbers = [mean_std(cv_file) for cv_file in cv_files]
    return (
        rf"& {model_name_pretty} "
        + " ".join([rf"& {s}" for s in numbers])
        + r" \\"
    )


header = rf"""\begin{{table}}[ht]
\caption{{Mean {METRIC.upper().replace("_", " ")} scores by model and aggreagation level}}
\centering
\label{{table:results_{METRIC}}}
\begin{{tabular}}{{clCCCC}}
\toprule
&  & \multicolumn{{4}}{{c}}{{Mean {METRIC.upper().replace("_", " ")} score}}\\
\cmidrule(lr){{3-6}}
& Model name
& \multicolumn{{2}}{{c}}{{Including L2}}
& \multicolumn{{2}}{{c}}{{Excluding L2}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}
&  & \text{{Ord. persistence}} & \text{{Ext. persistence}} & \text{{Ord. persistence}} & \text{{Ext. persistence}}\\
\midrule
\multirow{{19}}{{*}}{{\rotatebox{{90}}{{TRIAL-LEVEL}}}}"""

footer = r"""\bottomrule
\end{tabular}
\end{table}"""


def main(
    outdirs: list[Path],
) -> None:
    model_names_tsh = [
        "tsh_horizontal",
        "tsh_sloped",
        "tsh_sigmoid",
        "tsh_arctan",
    ]
    model_names_baseline = [
        "baseline_bjornsdottir_rf",
        "baseline_raatikainen_rf",
        "baseline_raatikainen_svc",
    ]
    model_names_baseline_with_tsh = [
        "baseline_bjornsdottir_rf_with_tsh_horizontal",
        "baseline_bjornsdottir_rf_with_tsh_sloped",
        "baseline_bjornsdottir_rf_with_tsh_sigmoid",
        "baseline_bjornsdottir_rf_with_tsh_arctan",
        "baseline_raatikainen_rf_with_tsh_horizontal",
        "baseline_raatikainen_rf_with_tsh_sloped",
        "baseline_raatikainen_rf_with_tsh_sigmoid",
        "baseline_raatikainen_rf_with_tsh_arctan",
        "baseline_raatikainen_svc_with_tsh_horizontal",
        "baseline_raatikainen_svc_with_tsh_sloped",
        "baseline_raatikainen_svc_with_tsh_sigmoid",
        "baseline_raatikainen_svc_with_tsh_arctan",
    ]
    tex_table_list = [header]
    tex_table_list.extend(
        [
            get_baseline_with_tsh_line(model_name, "trial", outdirs)
            for model_name in model_names_baseline_with_tsh
        ]
    )
    tex_table_list.extend(
        [
            get_baseline_line(model_name, "trial", outdirs)
            for model_name in model_names_baseline
        ]
    )
    tex_table_list.extend(
        [
            get_tsh_line(model_name, "trial", outdirs)
            for model_name in model_names_tsh
        ]
    )
    tex_table_list.extend(
        [r"\midrule", r"\multirow{19}{*}{\rotatebox{90}{READER-LEVEL}}"]
    )
    tex_table_list.extend(
        [
            get_baseline_with_tsh_line(model_name, "reader", outdirs)
            for model_name in model_names_baseline_with_tsh
        ]
    )
    tex_table_list.extend(
        [
            get_baseline_line(model_name, "reader", outdirs)
            for model_name in model_names_baseline
        ]
    )
    tex_table_list.extend(
        [
            get_tsh_line(model_name, "reader", outdirs)
            for model_name in model_names_tsh
        ]
    )
    tex_table_list.append(footer)
    tex_table = "\n".join(tex_table_list)
    print(tex_table)
    return


if __name__ == "__main__":
    main(OUTDIRS)
