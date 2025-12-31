import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import gudhi.representations as gdrep
import numpy as np
import numpy.typing as npt
import polars as pl
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

from scripts import constants, get_data, get_dataframes
from scripts.time_series_homology import TimeSeriesHomology
from scripts.utils import (
    ListTransformer,
    PersistenceImageProcessor,
    PersistenceProcessor,
    group_by_mean,
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
                f"None, but got '{value}' instead."
            )

    parser = argparse.ArgumentParser(
        description="Run TDA-experiment or baselines on CopCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=constants.admissible_model_names,
        help=("Name of model to run"),
    )
    parser.add_argument(
        "--classifier",
        type=str,
        help=(
            "Classifier to use (must be one of 'svc' and 'rf', depending on "
            "the model name)"
        ),
    )
    parser.add_argument(
        "--filtration-type",
        type=str,
        help=(
            "Filtration type to use (must be one of 'horizontal', 'sloped', "
            "'sigmoid' and 'arctan'; ignored unless model name is "
            "'tsh' or 'tsh_aggregated')"
        ),
    )
    parser.add_argument(
        "--use-extended-persistence",
        action="store_true",
        help=(
            "Compute extended persistence of time series (as opposed to "
            "ordinary persistence; ignored unless model name is "
            "'tsh' or 'tsh_aggregated')"
        ),
    )
    parser.add_argument(
        "--exclude-l2",
        action="store_true",
        help="Exclude CopCo-L2-readers",
    )
    parser.add_argument(
        "--min-n-fixations",
        type=int,
        default=5,
        help=(
            "Minimum number of fixation for a trial not to be discarded "
            "(ignored unless model name is 'tsh' or 'tsh_aggregated')"
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help=("Number of splits to create for nested CV"),
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


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validates the arguments provided."""
    if args.model_name in ["tsh", "tsh_aggregated"]:
        if (
            args.filtration_type
            not in constants.admissible_filtration_types_tsh
        ):
            raise ValueError(
                "Invalid filtration type for TDA-experiment; must be in "
                f"{constants.admissible_filtration_types_tsh}, but "
                f"got '{args.filtration_type}' instead."
            )
        if args.classifier not in constants.admissible_classifiers_tsh:
            raise ValueError(
                "Invalid classifier for TDA-experiment; must be in "
                f"{constants.admissible_classifiers_tsh}, but got "
                f"'{args.classifier}' instead."
            )
    elif args.model_name == "baseline_bjornsdottir":
        if (
            args.classifier
            not in constants.admissible_classifiers_bjornsdottir
        ):
            raise ValueError(
                "Invalid classifier for Björnsdottir-baseline; must be in "
                f"{constants.admissible_classifiers_bjornsdottir}, but got "
                f"'{args.classifier}' instead."
            )
    elif args.model_name == "baseline_raatikainen":
        if args.classifier not in constants.admissible_classifiers_raatikainen:
            raise ValueError(
                "Invalid classifier for Raatikainen-baseline; must be in "
                f"{constants.admissible_classifiers_raatikainen}, but got "
                f"'{args.classifier}' instead."
            )
    return


def get_cv_results_file_path(
    outdir: Path,
    args: argparse.Namespace,
) -> Path:
    """Creates name of file storing results."""
    if args.model_name in ["tsh", "tsh_aggregated"]:
        persistence_type = (
            "extended" if args.use_extended_persistence else "ordinary"
        )
        cv_results_file_path = outdir / (
            f"cv_results_{args.model_name}_{args.filtration_type}"
            f"_{persistence_type}_{args.classifier}_seed_{args.seed}.json"
        )
    elif args.model_name in ["baseline_bjornsdottir", "baseline_raatikainen"]:
        cv_results_file_path = outdir / (
            f"cv_results_{args.model_name}_{args.classifier}"
            f"_seed_{args.seed}.json"
        )
    return cv_results_file_path


def get_pipeline(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Pipeline:
    if args.classifier == "svc":
        clf = (
            "svc",
            SVC(
                probability=True,
                random_state=rng.integers(low=0, high=2**32),
            ),
        )
    elif args.classifier == "rf":
        clf = (
            "rf",
            RandomForestClassifier(
                random_state=rng.integers(low=0, high=2**32),
            ),
        )
    if args.model_name in ["tsh", "tsh_aggregated"]:
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
                        filtration_type=args.filtration_type,
                        use_extended_persistence=args.use_extended_persistence,
                        drop_inf_persistence=not args.use_extended_persistence,
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
                        n_components=750
                        if args.use_extended_persistence
                        else 250,
                        random_state=rng.integers(low=0, high=2**32),
                    ),
                ),
                clf,
            ]
        )
    elif args.model_name == "baseline_bjornsdottir":
        pipeline = ImbalancedPipeline(
            [
                (
                    "scaler",
                    MinMaxScaler(
                        feature_range=(-1, 1),
                    ),
                ),
                (
                    "downsampler",  # to balance classes to equal sizes
                    RandomUnderSampler(
                        random_state=rng.integers(low=0, high=2**32)
                    ),
                ),
                clf,
            ]
        )
    elif args.model_name == "baseline_raatikainen":
        pipeline = Pipeline(
            [
                (
                    "scaler",
                    MinMaxScaler(
                        feature_range=(-1, 1),
                    ),
                ),
                clf,
            ]
        )
    return pipeline


def get_split_idxs(
    df: pl.DataFrame,
    n_splits: int,
    verbose: bool,
    rng: np.random.Generator,
) -> list[npt.NDArray]:
    split_idxs_ok = False
    while not split_idxs_ok:
        if verbose:
            tqdm.write(f"Finding splitting of data into {n_splits} splits...")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng.integers(low=0, high=2**32),
        )
        y, groups = df["LABEL"], df["READER_ID"]
        split_idxs = [
            test_idx for _, test_idx in splitter.split(df, y, groups)
        ]
        n_classes_per_split = np.array(
            [len(np.unique(y[split_idx])) for split_idx in split_idxs]
        )
        if (n_classes_per_split > 1).all():
            split_idxs_ok = True
    if verbose:
        tqdm.write(f"Found splitting of data into {n_splits} splits.")
    return split_idxs


def sample_random_hyperparams(
    param_distributions: list[dict],
    n_iter: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample hyperparameter combinations from distributions."""
    all_params = []
    for _ in range(n_iter):
        # Randomly select one dict from the list
        dist_dict = param_distributions[
            rng.integers(0, len(param_distributions))
        ]
        params = {}
        for key, value in dist_dict.items():
            if hasattr(value, "rvs"):
                params[key] = value.rvs(random_state=rng)
            elif isinstance(value, list):
                params[key] = value[rng.integers(0, len(value))]
        all_params.append(params)
    return all_params


def generate_grid_hyperparams(param_grid: dict) -> list[dict]:
    """Generate all hyperparameter combinations from grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_params = []
    for combination in itertools.product(*values):
        all_params.append(dict(zip(keys, combination)))
    return all_params


def verify_no_mixed_labels(y: npt.NDArray, groups: npt.NDArray) -> None:
    """Verify that no group has mixed labels."""
    for group in np.unique(groups):
        group_labels = y[groups == group]
        assert len(np.unique(group_labels)) == 1, (
            f"Group {group} has mixed labels: {np.unique(group_labels)}"
        )


def evaluate_single_fold(
    model_name: str,
    params: dict,
    split_idxs: list[npt.NDArray],
    test_fold_idx: int,
    val_fold_idx: int,
    pipeline: Pipeline,
    X: npt.NDArray | list[npt.NDArray] | pl.DataFrame,
    y: npt.NDArray,
    groups: npt.NDArray,
) -> tuple[dict, int, float]:
    """Evaluate a single hyperparameter combination on a single fold.

    Returns:
        Tuple of (params, val_fold_idx, val_score) for aggregation.
    """
    # Get train and validation data
    train_idxs = np.concatenate(
        [
            idxs
            for fold_idx, idxs in enumerate(split_idxs)
            if fold_idx not in [test_fold_idx, val_fold_idx]
        ]
    )
    val_idxs = split_idxs[val_fold_idx]
    try:  # in case X is a NumPy-array
        X_train = X[train_idxs]
        X_val = X[val_idxs]
    except TypeError:  # fallback in case X is a list
        X_train = [X[i] for i in train_idxs]
        X_val = [X[i] for i in val_idxs]
    y_train = y[train_idxs]
    y_val = y[val_idxs]
    # Fit model with given params on train data
    estimator = clone(pipeline).set_params(**params)
    if model_name == "tsh_aggregated":
        # Separate feature extractor and classifier for aggregation
        feature_extractor, clf = estimator[:-1], estimator[-1]
        train_features = feature_extractor.fit_transform(X_train, y_train)
        train_groups = groups[train_idxs]
        verify_no_mixed_labels(y_train, train_groups)
        train_features = group_by_mean(train_features, train_groups)
        y_train = group_by_mean(y_train, train_groups)
        clf.fit(train_features, y_train)
        # Get predictions on validation data
        val_features = feature_extractor.transform(X_val)
        val_groups = groups[val_idxs]
        verify_no_mixed_labels(y_val, val_groups)
        val_features = group_by_mean(val_features, val_groups)
        y_val = group_by_mean(y_val, val_groups)
        y_val_pred_proba = clf.predict_proba(val_features)[:, 1]
    else:
        # Get predictions on validation data
        estimator.fit(X_train, y_train)
        y_val_pred_proba = estimator.predict_proba(X_val)[:, 1]
    # Get score on validation data
    val_score = roc_auc_score(y_val, y_val_pred_proba)
    return (params, val_fold_idx, val_score)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(seed=args.seed)
    outdir = "outfiles"
    if args.exclude_l2:
        outdir += "_without_l2"
    cv_results_file_path = get_cv_results_file_path(Path(outdir), args)
    experiment_name = cv_results_file_path.stem[11:]
    tqdm.write(f" RUNNING MODEL '{experiment_name}' ".center(120, "*"))
    if not cv_results_file_path.exists() or args.overwrite:
        # Get pipeline
        pipeline = get_pipeline(args, rng)
        # Get hyperparameter distributions
        if args.model_name in ["tsh", "tsh_aggregated"]:
            hyperparams = constants.hyperparams[
                "_".join(
                    [args.model_name, args.filtration_type, args.classifier]
                )
            ]
            search_kind = "random"
        elif args.model_name in [
            "baseline_bjornsdottir",
            "baseline_raatikainen",
        ]:
            hyperparams = constants.hyperparams[
                "_".join([args.model_name, args.classifier])
            ]
            search_kind = "grid"
        # Get dataframe with data
        df = get_dataframes.get_df(
            model_name=args.model_name,
            min_n_fixations=args.min_n_fixations,
            include_l2=not args.exclude_l2,
            verbose=bool(args.verbose),
            overwrite=args.overwrite,
        )
        # Get indices of splits
        split_idxs = get_split_idxs(
            df=df,
            n_splits=args.n_splits,
            verbose=args.verbose,
            rng=rng,
        )
        # Verify that no reader ID appears in more than one split
        reader_ids_per_split = [
            set(df["READER_ID"][split_idx]) for split_idx in split_idxs
        ]
        assert len(set().union(*reader_ids_per_split)) == sum(
            map(len, reader_ids_per_split)
        )
        # Get array of samples, labels and reader IDs
        X, y, groups = get_data.get_X_y_groups(
            df=df,
            model_name=args.model_name,
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
            # Generate hyperparameter combinations
            if search_kind == "random":
                assert isinstance(hyperparams, list)
                param_combinations = sample_random_hyperparams(
                    hyperparams, args.n_iter, rng
                )
            elif search_kind == "grid":
                assert isinstance(hyperparams, dict)
                param_combinations = generate_grid_hyperparams(hyperparams)
            # Generate all (param_combination, val_fold) pairs
            val_fold_idxs = [
                idx for idx in range(len(split_idxs)) if idx != test_fold_idx
            ]
            # Evaluate all (param_combination, val_fold) pairs in parallel
            if args.verbose:
                tqdm.write(
                    f"Evaluating {len(param_combinations)} parameter "
                    f"combinations on {len(val_fold_idxs)} validation folds "
                    "in parallel, for a total of "
                    f"{len(param_combinations) * len(val_fold_idxs)} "
                    "evaluations."
                )
            fold_results = Parallel(n_jobs=args.n_jobs)(
                delayed(evaluate_single_fold)(
                    model_name=args.model_name,
                    params=params,
                    split_idxs=split_idxs,
                    test_fold_idx=test_fold_idx,
                    val_fold_idx=val_fold_idx,
                    pipeline=pipeline,
                    X=X,
                    y=y,
                    groups=groups,
                )
                for val_fold_idx in val_fold_idxs
                for params in param_combinations
            )
            # Aggregate results by parameter combination
            params_to_val_scores = defaultdict(list)
            for params, val_fold_idx, val_score in fold_results:
                # Use tuple to make dict keys hashable
                params_key = tuple(sorted(params.items()))
                params_to_val_scores[params_key].append(val_score)
            # Compute average scores per parameter combination
            params_to_mean_val_scores = {
                params: np.mean(val_scores)
                for params, val_scores in params_to_val_scores.items()
            }
            # Find best hyperparameters
            best_params = dict(
                max(
                    params_to_mean_val_scores,
                    key=lambda k: params_to_mean_val_scores[k],
                )
            )
            assert best_params is not None
            # Fit best model on non-test data
            non_test_idxs = np.concatenate(
                [
                    idxs
                    for fold_idx, idxs in enumerate(split_idxs)
                    if fold_idx != test_fold_idx
                ]
            )
            try:  # in case X is a NumPy-array
                X_non_test = X[non_test_idxs]
            except TypeError:  # fallback in case X is a list
                X_non_test = [X[i] for i in non_test_idxs]
            y_non_test = y[non_test_idxs]
            best_estimator = clone(pipeline).set_params(**best_params)
            if args.model_name == "tsh_aggregated":
                # Separate feature extractor and classifier for aggregation
                best_feature_extractor, best_clf = (
                    best_estimator[:-1],
                    best_estimator[-1],
                )
                non_test_features = best_feature_extractor.fit_transform(
                    X_non_test, y_non_test
                )
                non_test_groups = groups[non_test_idxs]
                verify_no_mixed_labels(y_non_test, non_test_groups)
                non_test_features = group_by_mean(
                    non_test_features, non_test_groups
                )
                y_non_test = group_by_mean(y_non_test, non_test_groups)
                best_clf.fit(non_test_features, y_non_test)
                # Get predictions on test data
                test_features = best_feature_extractor.transform(X_test)
                test_groups = groups[test_idxs]
                verify_no_mixed_labels(y_test, test_groups)
                test_features = group_by_mean(test_features, test_groups)
                y_test = group_by_mean(y_test, test_groups)
                y_test_pred_proba = best_clf.predict_proba(test_features)[:, 1]
                y_test_pred = best_clf.predict(test_features)
            else:
                # Get predictions on test data
                best_estimator.fit(X_non_test, y_non_test)
                y_test_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
                y_test_pred = best_estimator.predict(X_test)
            # Get ROC AUC metrics
            fp_rate, tp_rate, thresholds_roc = roc_curve(
                y_test, y_test_pred_proba
            )
            cv_results["roc_curve"].append(
                (fp_rate.tolist(), tp_rate.tolist(), thresholds_roc.tolist())
            )
            cv_results["roc_auc"].append(
                roc_auc_score(y_test, y_test_pred_proba)
            )
            # Get PR AUC metrics
            precision, recall, thresholds_pr = precision_recall_curve(
                y_test, y_test_pred_proba
            )
            cv_results["pr_curve"].append(
                (precision.tolist(), recall.tolist(), thresholds_pr.tolist())
            )
            cv_results["pr_auc"].append(
                average_precision_score(y_test, y_test_pred_proba)
            )
            # Get accuracy
            cv_results["accuracy"].append(accuracy_score(y_test, y_test_pred))
            # Store best hyperparams
            cv_results["best_params_list"].append(best_params)
        # Save CV results to disk
        cv_results_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cv_results_file_path, "w") as f_out:
            json.dump(cv_results, f_out, indent=2)
        if args.verbose:
            tqdm.write(
                f"Saved CV results for model '{args.model_name}' to "
                f"`{cv_results_file_path}`."
            )
    else:
        with open(cv_results_file_path, "r") as f_in:
            cv_results = json.load(f_in)
        if args.verbose:
            tqdm.write(
                f"Found CV results for model '{args.model_name}' at "
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
    raise SystemExit(main())
