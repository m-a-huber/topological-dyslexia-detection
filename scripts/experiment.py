import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from scripts import constants, make_dataframes


def parse_args():
    def int_or_none(value):
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for seed: {value}"
            )

    parser = argparse.ArgumentParser(
        description="Run TDA-experiment or baselines on CopCo"
    )
    parser.add_argument("--model_name", help="Name of model to run")
    parser.add_argument(
        "--min_n_fixations",
        type=int,
        default=5,
        help=(
            "Minimum number of fixation for a trial not to be discarded "
            "(ignored unless `model_name` is 'tda_experiment')"
        ),
    )
    parser.add_argument(
        "--include_l2",
        dest="include_l2",
        action="store_true",
        help="Whether or not to include CopCo-L2-readers",
    )
    parser.add_argument(
        "--exclude_l2",
        dest="include_l2",
        action="store_false",
        help="Whether or not to exclude CopCo-L2-readers",
    )
    parser.add_argument(
        "--percentage_val_split",
        type=float,
        default=0.1,
        help=("Percentage of reader IDs to be used for validation split"),
    )
    parser.add_argument(
        "--n_splits_train_test",
        type=int,
        default=5,
        help=("Number of splits to break up non-validation data into"),
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=("Number of jobs to run in parallel in `GridSearchCV`."),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display information during execution of program",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int_or_none,
        default=42,
        help="Seed for reproducibility (int or None)",
    )
    parser.set_defaults(include_l2=True)
    return parser.parse_args()


def get_best_params(
    X_train: npt.NDArray,
    X_val: npt.NDArray,
    y_train: npt.NDArray,
    y_val: npt.NDArray,
    pipeline: Pipeline,
    hyperparams: dict[str, list[float]],
    scoring: str,
    n_jobs: int,
) -> dict[str, list[float]]:
    """Function performing grid search to tune model parameters by training on
    `X_train` and evaluating on `X_val` for all possible combinations of
    hyperparameters. Returns a dictionary containing the best combination of
    hyperparameters found."""
    ps = PredefinedSplit(
        np.concatenate(
            [
                -np.ones(len(X_train), dtype=int),
                np.zeros(len(X_val), dtype=int),
            ]
        )
    )
    grid = GridSearchCV(
        pipeline,
        hyperparams,
        cv=ps,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    X_train_val, y_train_val = (
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    grid.fit(X_train_val, y_train_val)
    return grid.best_params_


def main(
    args: argparse.Namespace,
) -> None:
    result_file_path = Path(
        f"./outfiles/results_{args.model_name}_seed_{args.seed}.json"
    )
    if not result_file_path.exists() or args.overwrite:
        if args.model_name == "tda_experiment":
            raise NotImplementedError()
        elif args.model_name == "baseline_bjornsdottir":
            pipeline = Pipeline(
                [
                    (
                        "scaler",
                        MinMaxScaler(
                            feature_range=(-1, 1),
                        ),
                    ),
                    (
                        "rf",
                        RandomForestClassifier(
                            random_state=args.seed,
                        ),
                    ),
                ]
            )
            hyperparams = constants.hyperparams["baseline_bjornsdottir"]
        elif args.model_name == "baseline_raatikainen":
            raise NotImplementedError()
        elif args.model_name == "baseline_benfatto":
            raise NotImplementedError()
        elif args.model_name == "baseline_haller":
            raise NotImplementedError()
        else:
            raise ValueError(
                "Invalid choice of `model_name`; must be one of "
                "`'tda_experiment'`, `'baseline_bjornsdottir'`, "
                "`'baseline_raatikainen'`, `'baseline_benfatto'` and "
                "`'baseline_haller'`."
            )
        df = make_dataframes.make_df(
            model_name=args.model_name,
            min_n_fixations=args.min_n_fixations,
            include_l2=args.include_l2,
            verbose=args.verbose,
            overwrite=args.overwrite,
            seed=args.seed,
        )
        df_val, dfs_train_test = make_dataframes.get_dfs_splits(
            df=df,
            percentage_val_split=args.percentage_val_split,
            n_splits_train_test=args.n_splits_train_test,
            random_state=args.seed,
        )
        roc_aucs, pr_aucs = [], []
        result_dict = dict()
        for i, _ in enumerate(dfs_train_test):
            df_train = pl.concat(
                [
                    dfs_train_test[j]
                    for j, _ in enumerate(dfs_train_test)
                    if j != i
                ]
            )
            df_test = dfs_train_test[i]
            X_train = df_train.drop(
                "READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"
            ).to_numpy()
            X_test = df_test.drop(
                "READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"
            ).to_numpy()
            y_train = df_train["LABEL"].to_numpy()
            y_test = df_test["LABEL"].to_numpy()
            if i == 0:
                X_val = df_val.drop(
                    "READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"
                ).to_numpy()
                y_val = df_val["LABEL"].to_numpy()
                best_params = get_best_params(
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    pipeline=pipeline,
                    hyperparams=hyperparams,
                    scoring="roc_auc",
                    n_jobs=args.n_jobs,
                )
            pipeline_best = pipeline.set_params(**best_params)
            pipeline_best.fit(X_train, y_train)
            y_proba = pipeline_best.predict_proba(X_test)
            roc_aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
            pr_aucs.append(average_precision_score(y_test, y_proba[:, 1]))
        result_dict["roc_aucs"] = roc_aucs
        result_dict["pr_aucs"] = pr_aucs
        result_dict["best_params"] = best_params
        result_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file_path, "w") as f_out:
            json.dump(result_dict, f_out, indent=2)
        if args.verbose:
            print(
                "Saved dictionary containing results and best hyperparameters "
                f"for model `'{args.model_name}'` to `{result_file_path}`."
            )
    else:
        with open(result_file_path, "r") as f_in:
            result_dict = json.load(f_in)
        if args.verbose:
            print(
                "Found dictionary containing results and best hyperparameters "
                f"for model `'{args.model_name}'` at `{result_file_path}`; "
                "not overwriting."
            )
        roc_aucs = result_dict["roc_aucs"]
        pr_aucs = result_dict["pr_aucs"]
    print(f"ROC AUC: {np.mean(roc_aucs):.2f}\u00b1{np.mean(roc_aucs):.2f}")
    print(f"PR AUC : {np.mean(pr_aucs):.2f}\u00b1{np.mean(pr_aucs):.2f}")
    return


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
