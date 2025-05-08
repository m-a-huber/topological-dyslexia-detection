import sys
from pathlib import Path  # type: ignore

import gudhi.representations as gdrep  # type: ignore
import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import (  # type: ignore
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
from skopt import BayesSearchCV  # type: ignore
from skopt.space import Categorical, Real  # type: ignore

from scripts.process_data import (  # type: ignore
    get_labels,
    process_fixation_reports,
)
from scripts.time_series_homology import TimeSeriesHomology  # type: ignore
from scripts.utils import (  # type: ignore
    ListTransformer,
    PersistenceImageProcessor,
    PersistenceProcessor,
)


def get_data(
    time_series_dir: Path,
) -> tuple[list[npt.NDArray], npt.NDArray]:
    X = [
        np.load(time_series_path)
        for time_series_path in sorted(time_series_dir.glob("*.npy"))
    ]
    y = np.load(time_series_dir / "labels" / "is_dyslexic.npy")
    assert len(X) == len(y)
    return X, y


def weight_abs1p(pt):
    """Custom weight function for persistence images that weighs points in a
    persistence diagram by lifetime plus 1.
    """
    return np.abs(pt[1]) + 1


def train_eval_svm(
    X: list[npt.NDArray],
    y: npt.NDArray,
    out_dir: Path,
    filtration_type: str,
    use_extended_persistence: bool,
    n_splits: int,
    search_space: dict,
    n_iter: int,
    n_jobs: int | None,
    verbose: int,
    overwrite: bool,
    random_state: int | None,
) -> tuple[dict, dict, npt.NDArray]:
    if not out_dir.is_dir() or overwrite:
        TimeSeriesScaler = ListTransformer(base_estimator=StandardScaler())
        PersistenceImager = ListTransformer(gdrep.PersistenceImage(
            resolution=(75, 75),
            weight=weight_abs1p
        ))
        pipeline = Pipeline([
            ("time_series_scaler", TimeSeriesScaler),
            ("time_series_homology", TimeSeriesHomology(
                filtration_type=filtration_type,
                use_extended_persistence=use_extended_persistence,
                drop_infinite_persistence=not use_extended_persistence,
            )),
            ("persistence_processor", PersistenceProcessor()),
            ("persistence_imager", PersistenceImager),
            ("persistence_image_scaler", PersistenceImageProcessor(
                scaler=MinMaxScaler()
            )),
            ("svc", SVC(class_weight="balanced"))
        ])
        rng = np.random.default_rng(random_state)
        inner_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng.integers(0, 10_000),
        )
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=n_iter,
            scoring="roc_auc",
            n_jobs=n_jobs,
            cv=inner_cv,
            verbose=verbose,
            random_state=rng.integers(0, 10_000),
        )
        bayes_search.fit(X, y)
        outer_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng.integers(0, 10_000),
        )
        cv_results, best_params, cv_roc_scores = (
            bayes_search.cv_results_,
            bayes_search.best_params_,
            cross_val_score(
                bayes_search.best_estimator_,
                X,
                y,
                cv=outer_cv,
                scoring="roc_auc",
                n_jobs=n_jobs,
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(cv_results, out_dir / "cv_results.pkl")
        joblib.dump(best_params, out_dir / "best_params.pkl")
        joblib.dump(cv_roc_scores, out_dir / "cv_roc_scores.pkl")
        if verbose:
            print(
                "Saved `cv_results`, `best_params` and `cv_roc_scores` "
                f"in {out_dir}."
            )
    else:
        cv_results = joblib.load(out_dir / "cv_results.pkl")
        best_params = joblib.load(out_dir / "best_params.pkl")
        cv_roc_scores = joblib.load(out_dir / "cv_roc_scores.pkl")
        if verbose:
            print(
                "Found `cv_results`, `best_params` and `cv_roc_scores` "
                f"in {out_dir}; not overwriting."
            )
    return cv_results, best_params, cv_roc_scores


if __name__ == "__main__":
    overwrite = sys.argv[1] == "True"

    n_splits = 5  # number of splits in StratifiedKFold
    n_iter = 50  # number of iterations for BayesSearchCV
    n_jobs = -1  # parallelism for BayesSearchCV
    verbose = 2
    random_state = 42

    # Process corpus files and create time series data
    fixation_reports_dir = Path("data/FixationReports")
    dataset_statistics_dir = Path("data/DatasetStatistics")
    participants_stats_path = dataset_statistics_dir / Path(
        "participant_stats.csv"
    )
    time_series_dir = Path("data/TimeSeriesData")
    process_fixation_reports(
        fixation_reports_dir=fixation_reports_dir,
        out_dir=time_series_dir,
        verbose=bool(verbose),
        overwrite=overwrite,
    )
    get_labels(
        participants_stats_path=participants_stats_path,
        out_dir=time_series_dir / "labels",
        verbose=bool(verbose),
        overwrite=overwrite,
    )

    # Load data
    X, y = get_data(
        time_series_dir=time_series_dir,
    )

    for filtration_type in ["horizontal", "sloped", "sigmoid", "arctan"]:
        if filtration_type == "horizontal":
            search_space = {
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1, 10000, prior="log-uniform"),
                "svc__gamma": Real(0.0001, 100, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        if filtration_type == "sloped":
            search_space = {
                "time_series_homology__linear_slope": Real(
                    -4, 4, prior="uniform"
                ),
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1, 10000, prior="log-uniform"),
                "svc__gamma": Real(0.0001, 100, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        if filtration_type == "sigmoid":
            search_space = {
                "time_series_homology__sigmoid_slope": Real(
                    -4, 4, prior="uniform"
                ),
                "time_series_homology__padding_factor": Real(
                    0.01, 0.05, prior="uniform"
                ),
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1, 10000, prior="log-uniform"),
                "svc__gamma": Real(0.0001, 100, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        if filtration_type == "arctan":
            search_space = {
                "time_series_homology__arctan_slope": Real(
                    -4, 4, prior="uniform"
                ),
                "time_series_homology__padding_factor": Real(
                    0.01, 0.05, prior="uniform"
                ),
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1, 10000, prior="log-uniform"),
                "svc__gamma": Real(0.0001, 100, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        for use_extended_persistence in [True, False]:
            suffix = (
                "with_extended_persistence"
                if use_extended_persistence
                else "without_extended_persistence"
            )
            out_dir = Path(
                f"out_files/{filtration_type}_{suffix}"
            )
            print(f"Running {filtration_type}_{suffix}...")
            cv_results, best_params, cv_roc_scores = (
                train_eval_svm(
                    X=X,
                    y=y,
                    out_dir=out_dir,
                    filtration_type=filtration_type,
                    use_extended_persistence=use_extended_persistence,
                    n_splits=n_splits,
                    search_space=search_space,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    overwrite=overwrite,
                    verbose=verbose,
                    random_state=random_state,
                )
            )
