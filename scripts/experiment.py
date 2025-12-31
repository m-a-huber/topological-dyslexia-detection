import argparse
import json
from pathlib import Path

import gudhi.representations as gdrep
import numpy as np
import numpy.typing as npt
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
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
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
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
            "'tsh')"
        ),
    )
    parser.add_argument(
        "--use-extended-persistence",
        action="store_true",
        help=(
            "Compute extended persistence of time series (as opposed to "
            "ordinary persistence; ignored unless model name is "
            "'tsh')"
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
            "(ignored unless model name is 'tsh')"
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
    if args.model_name == "tsh":
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
    if args.model_name == "tsh":
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
    if args.model_name == "tsh":
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
    X: list[npt.NDArray] | npt.NDArray,
    y: npt.NDArray,
    groups: npt.NDArray,
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
        split_idxs = [test_idx for _, test_idx in splitter.split(X, y, groups)]
        n_classes_per_split = np.array(
            [len(np.unique(y[split_idx])) for split_idx in split_idxs]
        )
        if (n_classes_per_split > 1).all():
            split_idxs_ok = True
    if verbose:
        tqdm.write(f"Found splitting of data into {n_splits} splits.")
    return split_idxs


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
        # Get pipeline and corresponding hyperparameter distributions
        if args.model_name == "tsh":
            hyperparams = constants.hyperparams[
                "_".join(
                    [args.model_name, args.filtration_type, args.classifier]
                )
            ]
            search_kind = "random"
            pipeline = get_pipeline(args, rng)
        elif args.model_name in [
            "baseline_bjornsdottir",
            "baseline_raatikainen",
        ]:
            hyperparams = constants.hyperparams[
                "_".join([args.model_name, args.classifier])
            ]
            search_kind = "grid"
            pipeline = get_pipeline(args, rng)
        # Get dataframe with data
        df = get_dataframes.get_df(
            model_name=args.model_name,
            min_n_fixations=args.min_n_fixations,
            include_l2=not args.exclude_l2,
            verbose=bool(args.verbose),
            overwrite=args.overwrite,
        )
        # Get array of samples, labels and reader IDs
        X, y, groups = get_data.get_X_y_groups(
            df=df,
            model_name=args.model_name,
        )
        # Get indices of splits
        split_idxs = get_split_idxs(
            X=X,
            y=y,
            groups=groups,
            n_splits=args.n_splits,
            verbose=args.verbose,
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
                    refit=False,
                    verbose=args.verbose,
                    random_state=rng.integers(low=0, high=2**32),
                )
            elif search_kind == "grid":
                inner_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=hyperparams,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=args.n_jobs,
                    refit=False,
                    verbose=args.verbose,
                )
            # Optimize hyperparameters
            inner_search.fit(X, y)
            # Fit best model on non-test data
            non_test_idxs = np.concatenate(
                [
                    non_test_idxs
                    for non_test_fold_idx, non_test_idxs in enumerate(
                        split_idxs
                    )
                    if non_test_fold_idx != test_fold_idx
                ]
            )
            try:  # in case X is a NumPy-array
                X_non_test = X[non_test_idxs]
            except TypeError:  # fallback in case X is a list
                X_non_test = [X[i] for i in non_test_idxs]
            y_non_test = y[non_test_idxs]
            best_params = inner_search.best_params_
            best_estimator = (
                clone(pipeline)
                .set_params(**best_params)
                .fit(X_non_test, y_non_test)
            )
            # Get predictions from best model on test data
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
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
