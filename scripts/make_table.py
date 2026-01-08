import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    def int_or_none(value):
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid value for seed; seed must be either an integer or "
                f"None, but got '{value}' instead."
            )

    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment results"
    )
    parser.add_argument(
        "--exclude-l2",
        action="store_true",
        help="Exclude CopCo-L2-readers",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int_or_none,
        default=42,
        help="Seed for reproducibility (int or None)",
    )
    return parser.parse_args()


def load_roc_auc(filepath: Path) -> tuple[float, float] | None:
    """Load ROC AUC results from JSON file. Returns (mean, std) or None if file
    doesn't exist.
    """
    if not filepath.exists():
        return None
    with open(filepath, "r") as f:
        data = json.load(f)
    roc_auc = data["roc_auc"]
    return np.mean(roc_auc), np.std(roc_auc)


def format_result(result: tuple[float, float] | None) -> str:
    """Format result as 'mean±std' or 'N/A' if None."""
    if result is None:
        return "N/A"
    mean, std = result
    return f"{mean:.2f}\\pm {std:.2f}"


def make_tda_row(
    tda_results: dict,
    clf: str,
    filtration: str,
) -> str:
    """Generate a LaTeX table row for a TDA model."""
    clf_label = "SVC" if clf == "svc" else "RF"
    ord_result = format_result(tda_results[filtration, "ordinary", clf])
    ext_result = format_result(tda_results[filtration, "extended", clf])
    return (
        f"    TDA-{clf_label}\\textsubscript{{{filtration}}} & {ord_result} & "
        f"{ext_result} \\\\"
    )


def make_baseline_row(label: str, result: tuple[float, float] | None) -> str:
    """Generate a LaTeX table row for a baseline model."""
    formatted = format_result(result)
    return (
        f"    Baseline\\textsubscript{{{label}}} & "
        f"\\multicolumn{{2}}{{C}}{{{formatted}}} \\\\"
    )


def main(
    args: argparse.Namespace,
) -> None:
    # Determine output directory
    outdir = Path("outfiles_without_l2" if args.exclude_l2 else "outfiles")
    seed = args.seed
    # Load results for TDA models
    filtration_types = ["horizontal", "sloped", "sigmoid", "arctan"]
    persistence_types = ["ordinary", "extended"]
    classifiers = ["svc", "rf"]
    tda_results = {}
    for filtration_type in filtration_types:
        for persistence_type in persistence_types:
            for clf in classifiers:
                filepath = outdir / (
                    f"cv_results_tsh_{filtration_type}_{persistence_type}_{clf}_seed_{seed}.json"
                )
                tda_results[filtration_type, persistence_type, clf] = (
                    load_roc_auc(filepath)
                )
    # Load Björnsdottir baseline result
    bjornsdottir_result = load_roc_auc(
        outdir / f"cv_results_baseline_bjornsdottir_rf_seed_{seed}.json"
    )
    # Load Raatikainen baseline results
    raatikainen_svc_result = load_roc_auc(
        outdir / f"cv_results_baseline_raatikainen_svc_seed_{seed}.json"
    )
    raatikainen_rf_result = load_roc_auc(
        outdir / f"cv_results_baseline_raatikainen_rf_seed_{seed}.json"
    )
    # Build TDA rows
    tda_rows = [
        make_tda_row(tda_results, clf, filtration_type)
        for clf in classifiers
        for filtration_type in filtration_types
    ]
    # Generate table
    l2_status = "excluding" if args.exclude_l2 else "including"
    label_suffix = "without_l2" if args.exclude_l2 else "with_l2"
    tda_rows_str = "\n".join(tda_rows)
    baseline_bjo = make_baseline_row("Bjö", bjornsdottir_result)
    baseline_raa_svc = make_baseline_row("Raa-SVC", raatikainen_svc_result)
    baseline_raa_rf = make_baseline_row("Raa-RF", raatikainen_rf_result)
    table_content = f"""\\begin{{table}}
  \\caption{{Evaluation results on CopCo Dataset {l2_status} L2-readers}}
  \\label{{table:results_copco_{label_suffix}}}
  \\centering
  \\begin{{tabular}}{{lCC}}
    \\toprule
    Filtration type & \\multicolumn{{2}}{{c}}{{Mean ROC AUC score}} \\\\
    \\midrule
    & \\text{{Ordinary persistence}} & \\text{{Extended persistence}} \\\\
    \\cmidrule(r){{2-3}}
{tda_rows_str}
    \\cmidrule(r){{2-3}}
{baseline_bjo}
{baseline_raa_svc}
{baseline_raa_rf}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""
    print(table_content)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
