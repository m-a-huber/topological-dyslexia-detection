import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import gudhi.representations as gdrep
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    RandomizedSearchCV,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from scripts import constants, make_dataframes
from scripts.time_series_homology import TimeSeriesHomology
from scripts.utils import (
    ListTransformer,
    PersistenceImageProcessor,
    PersistenceProcessor,
)


def parse_args():
    def int_or_none(value):
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid value for seed; seed must be either an integer or "
                f"None, but got {value} instead."
            )

    parser = argparse.ArgumentParser(
        description="Run TDA-experiment or baselines on CopCo"
    )
    parser.add_argument("--model-name", help="Name of model to run")
    parser.add_argument(
        "--min-n-fixations",
        type=int,
        default=5,
        help=(
            "Minimum number of fixation for a trial not to be discarded "
            "(ignored when running baseline models)"
        ),
    )
    parser.add_argument(
        "--include-l2",
        dest="include_l2",
        action="store_true",
        help="Include CopCo-L2-readers",
    )
    parser.add_argument(
        "--exclude-l2",
        dest="include_l2",
        action="store_false",
        help="Exclude CopCo-L2-readers",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help=("Number of times to repeat evaluation"),
    )
    parser.add_argument(
        "--percentage-val-split",
        type=float,
        default=0.1,
        help=("Percentage of reader IDs to be used for validation split"),
    )
    parser.add_argument(
        "--n-splits-train-test",
        type=int,
        default=5,
        help=("Number of splits to break up non-validation data into"),
    )
    parser.add_argument(
        "--with-n-fix",
        action="store_true",
        help="Append the length of a scanpath as an extra feature",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help=(
            "Number of iterations for randomized hyperparameter search "
            "(ignored when running baseline models)"
        ),
    )
    parser.add_argument(
        "--n-jobs",
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


def validate_model_name(model_name: str) -> None:
    """Validates the model name provided."""
    if not model_name:
        raise ValueError("No model name was provided.")
    if model_name.startswith("tda_experiment"):
        parts = args.model_name.split("_")
        filtration_type, persistence_type = parts[-2:]
        if not (
            len(parts) == 4
            and filtration_type
            in constants.admissible_filtration_types_tda_experiment
            and persistence_type
            in constants.admissible_persistence_types_tda_experiment
        ):
            raise ValueError(
                "Invalid model name for TDA-experiment; model name must be of "
                "the form 'tda_experiment_<horizontal|sloped|sigmoid|arctan>_"
                f"<ordinary|extended>', but got {model_name} instead."
            )
    elif model_name.startswith("baseline_bjornsdottir"):
        if model_name != "baseline_bjornsdottir":
            raise ValueError(
                "Invalid model name for Björnsdottir-baseline; model name "
                "must be 'baseline_bjornsdottir'."
            )
    elif model_name.startswith("baseline_raatikainen"):
        parts = model_name.split("_")
        model_kind = parts[-1]
        if not (
            len(parts) == 3
            and model_kind in constants.admissible_model_kinds_raatikainen
        ):
            raise ValueError(
                "Invalid model name for Raatikainen-baseline; model name must "
                "be of the form 'baseline_raatikainen_<rf|svc>', but got "
                f"{model_name} instead."
            )
    else:
        raise ValueError(
            "Invalid choice of `model_name`; must be one of "
            "`'tda_experiment_<horizontal|sloped|sigmoid|arctan>_"
            "<ordinary|extended>'`, `'baseline_bjornsdottir'`, and "
            "`'baseline_raatikainen_<rf|svc>'`."
        )


def get_best_params(
    X_train: npt.NDArray,
    X_val: npt.NDArray,
    y_train: npt.NDArray,
    y_val: npt.NDArray,
    pipeline: Pipeline,
    hyperparams: dict[str, list[float]],
    scoring: str,
    search_kind: str,  # either 'grid' or 'random'
    n_iter: int,  # no. of iterations for randomized search
    n_jobs: int,
    random_state: Optional[int],
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
    if search_kind == "grid":
        search = GridSearchCV(
            pipeline,
            hyperparams,
            cv=ps,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=False,
        )
        try:
            X_train_val = np.concatenate([X_train, X_val])
        except ValueError:
            X_train_val = X_train + X_val
        y_train_val = np.concatenate([y_train, y_val])
        with threadpool_limits(limits=1):
            search.fit(X_train_val, y_train_val)
    elif search_kind == "random":
        search = RandomizedSearchCV(
            pipeline,
            hyperparams,
            n_iter=n_iter,
            cv=ps,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=False,
            random_state=random_state,
        )
        try:
            X_train_val = np.concatenate([X_train, X_val])
        except ValueError:
            X_train_val = X_train + X_val
        y_train_val = np.concatenate([y_train, y_val])
        with threadpool_limits(limits=1):
            search.fit(X_train_val, y_train_val)
    else:
        raise ValueError(
            "Invalid value for `search_kind`; `search_kind` must be either "
            f"`'grid'` or `'random'`, but got {search_kind} instead."
        )
    return search.best_params_


def get_pipeline(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Pipeline:
    if args.model_name.startswith("tda_experiment"):
        parts = args.model_name.split("_")
        filtration_type, persistence_type = parts[-2:]
        use_extended_persistence = persistence_type == "extended"
        pipeline = Pipeline(
            [
                (
                    "time_series_scaler",
                    ListTransformer(
                        base_estimator=MinMaxScaler(feature_range=(0, 1))
                    ),
                ),
                (
                    "time_series_homology",
                    TimeSeriesHomology(
                        filtration_type=filtration_type,
                        use_extended_persistence=use_extended_persistence,
                        drop_inf_persistence=not use_extended_persistence,
                    ),
                ),
                ("persistence_processor", PersistenceProcessor()),
                (
                    "persistence_imager",
                    ListTransformer(
                        gdrep.PersistenceImage(
                            resolution=(50, 50),
                        )
                    ),
                ),
                (
                    "persistence_image_scaler",
                    PersistenceImageProcessor(feature_range=(0, 1)),
                ),
                (
                    "pca",
                    PCA(
                        svd_solver="randomized",
                        whiten=True,
                        n_components=250,
                        random_state=rng.integers(low=0, high=2**32),
                    ),
                ),
                (
                    "svc",
                    SVC(
                        probability=True,
                        random_state=rng.integers(low=0, high=2**32),
                    ),
                ),
            ]
        )
        pipeline_no_svc, svc = pipeline[:-1], pipeline[-1]
        if args.with_n_fix:
            extra_feature_transformer = Pipeline(
                [
                    (
                        "get_lengths",
                        FunctionTransformer(
                            lambda X: np.array(
                                [len(x) for x in X], dtype=float
                            ).reshape(-1, 1)
                        ),
                    ),
                    ("scale_lengths", MinMaxScaler(feature_range=(0, 1))),
                ]
            )
        else:
            extra_feature_transformer = FunctionTransformer(
                lambda X: np.empty(shape=(len(X), 0), dtype=float)
            )
        pipeline_union = FeatureUnion(
            [
                ("time_series_features", pipeline_no_svc),
                ("extra_feature", extra_feature_transformer),
            ]
        )
        pipeline = Pipeline(
            [
                (
                    "feature_union",
                    pipeline_union,
                ),
                ("svc", svc),
            ]
        )
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
                        random_state=rng.integers(low=0, high=2**32),
                    ),
                ),
            ]
        )
    elif args.model_name.startswith("baseline_raatikainen"):
        model_kind = args.model_name.split("_")[-1]
        if model_kind == "rf":
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
                            random_state=rng.integers(low=0, high=2**32),
                        ),
                    ),
                ]
            )
        elif model_kind == "svc":
            pipeline = Pipeline(
                [
                    (
                        "scaler",
                        MinMaxScaler(
                            feature_range=(-1, 1),
                        ),
                    ),
                    (
                        "svc",
                        SVC(
                            probability=True,
                            random_state=rng.integers(low=0, high=2**32),
                        ),
                    ),
                ]
            )
    return pipeline


def main(
    args: argparse.Namespace,
) -> None:
    validate_model_name(args.model_name)
    tqdm.write(f" RUNNING MODEL '{args.model_name}' ".center(120, "*"))
    rng = np.random.default_rng(seed=args.seed)
    result_file_path = Path(
        f"./outfiles/results_{args.model_name}_{args.n_repeats}_repeats_seed_"
        f"{args.seed}.json"
    )
    if not result_file_path.exists() or args.overwrite:
        if args.model_name.startswith("tda_experiment"):
            model_class = "tda_experiment"
            hyperparams = constants.hyperparams[
                "_".join(args.model_name.split("_")[:3])
            ]
            search_kind = "random"
            pipeline = get_pipeline(args, rng)
        elif args.model_name == "baseline_bjornsdottir":
            model_class = "baseline_bjornsdottir"
            hyperparams = constants.hyperparams[args.model_name]
            search_kind = "grid"
            pipeline = get_pipeline(args, rng)
        elif args.model_name.startswith("baseline_raatikainen"):
            model_class = "baseline_raatikainen"
            hyperparams = constants.hyperparams[args.model_name]
            search_kind = "grid"
            pipeline = get_pipeline(args, rng)
        df = make_dataframes.make_df(
            model_class=model_class,
            min_n_fixations=args.min_n_fixations,
            include_l2=args.include_l2,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
        result_dict = defaultdict(dict)
        for idx_repeat in tqdm(
            range(args.n_repeats),
            desc=f"Repeating evaluation of model {args.model_name}",
        ):
            df_val, dfs_train_test = make_dataframes.get_dfs_splits(
                df=df,
                percentage_val_split=args.percentage_val_split,
                n_splits_train_test=args.n_splits_train_test,
                random_state=rng.integers(low=0, high=2**32),
            )
            roc_aucs, pr_aucs = [], []
            for i, _ in tqdm(
                enumerate(dfs_train_test),
                desc="Iterating over non-validation folds",
                total=args.n_splits_train_test,
                leave=False,
            ):
                df_train = pl.concat(
                    [
                        dfs_train_test[j]
                        for j, _ in enumerate(dfs_train_test)
                        if j != i
                    ]
                )
                df_test = dfs_train_test[i]
                if args.model_name.startswith("tda_experiment"):
                    X_train = [
                        np.array(x, dtype=float)
                        for x in df_train["time_series"].to_list()
                    ]
                    X_test = [
                        np.array(x, dtype=float)
                        for x in df_test["time_series"].to_list()
                    ]
                else:
                    X_train = df_train.drop(
                        "READER_ID",
                        "LABEL",
                        "TRIAL_ID",
                        "SAMPLE_ID",
                        strict=False,
                    ).to_numpy()
                    X_test = df_test.drop(
                        "READER_ID",
                        "LABEL",
                        "TRIAL_ID",
                        "SAMPLE_ID",
                        strict=False,
                    ).to_numpy()
                y_train = df_train["LABEL"].to_numpy()
                y_test = df_test["LABEL"].to_numpy()
                if i == 0:
                    if args.model_name.startswith("tda_experiment"):
                        X_val = [
                            np.array(x, dtype=float)
                            for x in df_val["time_series"].to_list()
                        ]
                    else:
                        X_val = df_val.drop(
                            "READER_ID",
                            "LABEL",
                            "TRIAL_ID",
                            "SAMPLE_ID",
                            strict=False,
                        ).to_numpy()
                    y_val = df_val["LABEL"].to_numpy()
                    if args.verbose:
                        tqdm.write(
                            "Optimizing hyperparameters on validation split..."
                        )
                    best_params = get_best_params(
                        X_train=X_train,
                        X_val=X_val,
                        y_train=y_train,
                        y_val=y_val,
                        pipeline=pipeline,
                        hyperparams=hyperparams,
                        scoring="roc_auc",
                        search_kind=search_kind,
                        n_iter=args.n_iter,
                        n_jobs=args.n_jobs,
                        random_state=rng.integers(low=0, high=2**32),
                    )
                    if args.verbose:
                        tqdm.write(
                            "Finished optimizing hyperparameters on "
                            "validation split."
                        )
                pipeline_best = pipeline.set_params(**best_params)
                pipeline_best.fit(X_train, y_train)
                y_proba = pipeline_best.predict_proba(X_test)
                roc_aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
                pr_aucs.append(average_precision_score(y_test, y_proba[:, 1]))
            result_dict[f"repeat {idx_repeat}"]["best_params"] = best_params
            result_dict[f"repeat {idx_repeat}"]["roc_aucs"] = roc_aucs
            result_dict[f"repeat {idx_repeat}"]["pr_aucs"] = pr_aucs
            result_dict[f"repeat {idx_repeat}"]["roc_auc_mean"] = np.mean(
                roc_aucs
            )
            result_dict[f"repeat {idx_repeat}"]["roc_auc_std"] = np.std(
                roc_aucs
            )
            result_dict[f"repeat {idx_repeat}"]["pr_auc_mean"] = np.mean(
                pr_aucs
            )
            result_dict[f"repeat {idx_repeat}"]["pr_auc_std"] = np.std(pr_aucs)
        result_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file_path, "w") as f_out:
            json.dump(result_dict, f_out, indent=2)
        if args.verbose:
            tqdm.write(
                "Saved dictionary containing results and best hyperparameters "
                f"for model `'{args.model_name}'` across {args.n_repeats} "
                f"repeats to `{result_file_path}`."
            )
    else:
        with open(result_file_path, "r") as f_in:
            result_dict = json.load(f_in)
        if args.verbose:
            tqdm.write(
                "Found dictionary containing results and best hyperparameters "
                f"for model `'{args.model_name}'` across {args.n_repeats} "
                f"repeats at `{result_file_path}`; not overwriting."
            )
    roc_auc_means = [
        result_dict[f"repeat {idx_repeat}"]["roc_auc_mean"]
        for idx_repeat in range(args.n_repeats)
    ]
    roc_auc_stds = [
        result_dict[f"repeat {idx_repeat}"]["roc_auc_std"]
        for idx_repeat in range(args.n_repeats)
    ]
    roc_auc_mean_overall = np.mean(roc_auc_means)
    roc_auc_std_overall = np.sqrt(np.mean(np.square(roc_auc_stds)))
    pr_auc_means = [
        result_dict[f"repeat {idx_repeat}"]["pr_auc_mean"]
        for idx_repeat in range(args.n_repeats)
    ]
    pr_auc_stds = [
        result_dict[f"repeat {idx_repeat}"]["pr_auc_std"]
        for idx_repeat in range(args.n_repeats)
    ]
    pr_auc_mean_overall = np.mean(pr_auc_means)
    pr_auc_std_overall = np.sqrt(np.mean(np.square(pr_auc_stds)))
    tqdm.write(
        f"ROC AUC: {roc_auc_mean_overall:.2f}\u00b1{roc_auc_std_overall:.2f} "
        f"(mean & SD aggregated over {args.n_repeats} repetition means)"
    )
    tqdm.write(
        f" PR AUC: {pr_auc_mean_overall:.2f}\u00b1{pr_auc_std_overall:.2f} "
        f"(mean & SD aggregated over {args.n_repeats} repetition means)"
    )
    return


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
