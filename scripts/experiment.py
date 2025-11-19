import argparse
import json
from pathlib import Path

import gudhi.representations as gdrep
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

from scripts import constants, get_data, get_dataframes
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
        "--exclude-l2",
        action="store_true",
        help="Exclude CopCo-L2-readers",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help=(
            "Number of splits to be used in nested CV (same number will be "
            "used in inner and outer loop)"
        ),
    )
    parser.add_argument(
        "--with-n-fix",
        action="store_true",
        help="Append the length of a scanpath as an extra feature",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced class weights in final SVC",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        default=1.0,
        help=(
            "Representative percentage of trials to consider (in terms of "
            "length; applied after trials with fewer than --min-n-fixations "
            "fixations)"
        ),
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
        help=(
            "Number of jobs to run in parallel in `GridSearchCV` or "
            "`RandomizedSearchCV`."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
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
                f"must be 'baseline_bjornsdottir', but got {model_name} "
                "instead."
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
            "Invalid model name; must be one of `'tda_experiment_"
            "<horizontal|sloped|sigmoid|arctan>_<ordinary|extended>'`, "
            "`'baseline_bjornsdottir'`, and `'baseline_raatikainen_"
            f"<rf|svc>'`, but got {model_name} instead."
        )


def get_pipeline(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Pipeline:
    if args.model_name.startswith("tda_experiment"):
        parts = args.model_name.split("_")
        filtration_type, persistence_type = parts[-2:]
        use_extended_persistence = persistence_type == "extended"
        pipeline_topological_features = Pipeline(
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
                        n_components=750 if use_extended_persistence else 250,
                        random_state=rng.integers(low=0, high=2**32),
                    ),
                ),
            ]
        )
        if args.with_n_fix:
            # transformer that extracts length of each time series
            pipeline_extra_features = Pipeline(
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
            # dummy transformer that does nothing
            pipeline_extra_features = FunctionTransformer(
                lambda X: np.empty(shape=(len(X), 0), dtype=float)
            )
        pipeline_union = FeatureUnion(
            [
                ("topological_features", pipeline_topological_features),
                ("extra_features", pipeline_extra_features),
            ]
        )
        pipeline = Pipeline(
            [
                (
                    "feature_union",
                    pipeline_union,
                ),
                (
                    "svc",
                    SVC(
                        class_weight="balanced" if args.balanced else None,
                        probability=True,
                        random_state=rng.integers(low=0, high=2**32),
                    ),
                ),
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


def get_split_idxs(
    X: list[npt.NDArray] | npt.NDArray,
    y: npt.NDArray,
    groups: npt.NDArray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[npt.NDArray]:
    split_idxs_ok = False
    while not split_idxs_ok:
        if args.verbose:
            tqdm.write(
                f"Finding splitting of data into {args.n_splits} splits..."
            )
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng.integers(low=0, high=2**32),
        )
        split_idxs = [test_idx for _, test_idx in splitter.split(X, y, groups)]
        n_classes_per_split = np.array(
            [len(np.unique(y[split_idx])) for split_idx in split_idxs]
        )
        if (n_classes_per_split > 1).all():
            split_idxs_ok = True
    if args.verbose:
        tqdm.write(f"Found splitting of data into {args.n_splits} splits.")
    return split_idxs


def main(
    args: argparse.Namespace,
) -> None:
    validate_model_name(args.model_name)
    tqdm.write(f" RUNNING MODEL '{args.model_name}' ".center(120, "*"))
    rng = np.random.default_rng(seed=args.seed)
    outdir = "outfiles"
    if args.exclude_l2:
        outdir += "_without_l2"
    if args.balanced:
        outdir += "_balanced"
    if args.truncate < 1.0:
        outdir += f"_truncate_{args.truncate}"
    if args.with_n_fix:
        outdir += "_with_n_fix"
    cv_results_file_path = Path(
        f"./{outdir}/cv_results_{args.model_name}_seed_{args.seed}.json"
    )
    if not cv_results_file_path.exists() or args.overwrite:
        # Get pipeline and corresponding hyperparameter distributions
        if args.model_name.startswith("tda_experiment"):
            model_class = "tda_experiment"
            filtration_type = args.model_name.split("_")[-2]
            hyperparams = constants.hyperparams[
                "_".join([model_class, filtration_type])
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
        # Get dataframe with data
        df = get_dataframes.get_df(
            model_class=model_class,
            min_n_fixations=args.min_n_fixations,
            include_l2=not args.exclude_l2,
            verbose=bool(args.verbose),
            overwrite=args.overwrite,
        )
        # Truncate data if required
        if (
            args.model_name.startswith("tda_experiment")
            and args.truncate < 1.0
        ):
            df_with_len = df.with_columns(
                pl.col("time_series").list.len().alias("length")
            )
            coverage = 100 * args.truncate
            min_len, max_len = np.percentile(
                df_with_len["length"],
                [
                    0.5 * (100 - coverage),
                    coverage + 0.5 * (100 - coverage),
                ],
            ).astype(int)
            df = df_with_len.filter(
                pl.col("length").is_between(min_len, max_len)
            ).drop("length")
        # Get array of samples, labels and reader IDs
        X, y, groups = get_data.get_X_y_groups(
            df=df,
            model_class=model_class,
        )
        # Get indices of splits
        split_idxs = get_split_idxs(
            X=X,
            y=y,
            groups=groups,
            n_splits=args.n_splits,
            rng=rng,
        )
        # Verify that no reader ID appears in more than one split
        reader_ids_per_split = [
            set(groups[split_idx]) for split_idx in split_idxs
        ]
        assert len(set().union(*reader_ids_per_split)) == sum(
            map(len, reader_ids_per_split)
        )
        # Prepare dict for storing of results
        cv_results = {
            "roc_curve": [],
            "roc_auc": [],
            "pr_curve": [],
            "pr_auc": [],
            "accuracy": [],
            "best_params_list": [],
        }
        # Perform nested CV
        for test_fold_idx, test_idxs in tqdm(
            enumerate(split_idxs), total=len(split_idxs)
        ):
            # Get test split
            try:  # in case X is a NumPy-array
                X_test = X[test_idxs]
            except TypeError:  # fallback in case X is a list
                X_test = [X[i] for i in test_idxs]
            y_test = y[test_idxs]
            # Set up splits for hyperparameter optimization
            cv = (
                (
                    np.concatenate(  # train indices
                        [
                            train_idxs
                            for train_fold_idx, train_idxs in enumerate(
                                split_idxs
                            )
                            if train_fold_idx
                            not in [test_fold_idx, val_fold_idx]
                        ]
                    ),
                    val_idxs,  # val indices
                )
                for val_fold_idx, val_idxs in enumerate(split_idxs)
                if val_fold_idx != test_fold_idx
            )
            # Set up hyperparameter search
            if search_kind == "random":
                inner_search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=hyperparams,
                    n_iter=args.n_iter,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=args.n_jobs,
                    refit=True,
                    verbose=args.verbose,
                    random_state=rng.integers(low=0, high=2**32),
                )
            else:
                inner_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=hyperparams,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=args.n_jobs,
                    refit=True,
                    verbose=args.verbose,
                )
            # Optimize hyperparameters
            inner_search.fit(X, y)
            # Evaluate best model on outer test fold
            y_pred = inner_search.predict(X_test)
            y_pred_proba = inner_search.predict_proba(X_test)[:, 1]
            # Get ROC AUC metrics
            fp_rate, tp_rate, thresholds_roc = roc_curve(y_test, y_pred_proba)
            cv_results["roc_curve"].append(
                (fp_rate.tolist(), tp_rate.tolist(), thresholds_roc.tolist())
            )
            test_roc_auc = roc_auc_score(y_test, y_pred_proba)
            cv_results["roc_auc"].append(test_roc_auc)
            # Get precision-recall metrics
            precision, recall, thresholds_pr = precision_recall_curve(
                y_test, y_pred_proba
            )
            cv_results["pr_curve"].append(
                (precision.tolist(), recall.tolist(), thresholds_pr.tolist())
            )
            test_pr_auc = average_precision_score(y_test, y_pred_proba)
            cv_results["pr_auc"].append(test_pr_auc)
            # Get accuracy
            test_accuracy = accuracy_score(y_test, y_pred)
            cv_results["accuracy"].append(test_accuracy)
            # Retrieve best hyperparams
            cv_results["best_params_list"].append(inner_search.best_params_)
        # Save CV results to disk
        cv_results_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cv_results_file_path, "w") as f_out:
            json.dump(cv_results, f_out, indent=2)
        if args.verbose:
            tqdm.write(
                f"Saved CV results for model `'{args.model_name}'`to "
                f"`{cv_results_file_path}`."
            )
    else:
        with open(cv_results_file_path, "r") as f_in:
            cv_results = json.load(f_in)
        if args.verbose:
            tqdm.write(
                f"Found CV results for model `'{args.model_name}'`at "
                f"`{cv_results_file_path}`; not overwriting."
            )
    # Print results
    roc_auc_mean = np.mean(cv_results["roc_auc"])
    roc_auc_std = np.std(cv_results["roc_auc"])
    pr_auc_mean = np.mean(cv_results["pr_auc"])
    pr_auc_std = np.std(cv_results["pr_auc"])
    accuracy_mean = np.mean(cv_results["accuracy"])
    accuracy_std = np.std(cv_results["accuracy"])
    tqdm.write(f"ROC AUC  | {roc_auc_mean:.2f}\u00b1{roc_auc_std:.2f}")
    tqdm.write(f"PR AUC   | {pr_auc_mean:.2f}\u00b1{pr_auc_std:.2f}")
    tqdm.write(f"Accuracy | {accuracy_mean:.2f}\u00b1{accuracy_std:.2f}")
    return


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
