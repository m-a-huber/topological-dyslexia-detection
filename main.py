import sys
from pathlib import Path  # type: ignore

import gudhi.representations as gdrep  # type: ignore
import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
from sklearn.metrics import (  # type: ignore
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (  # type: ignore
    StratifiedKFold,
    train_test_split,
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
    test_size: float,
    random_state: int | None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    X = [
        np.load(time_series_path)
        for time_series_path in sorted(time_series_dir.glob("*.npy"))
    ]
    y = np.load(time_series_dir / "labels" / "is_dyslexic.npy")
    assert len(X) == len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def weight_abs1p(pt):
    """Custom weight function for persistence images that weighs points in a
    persistence diagram by lifetime plus 1.
    """
    return np.abs(pt[1]) + 1


def train_eval_svm(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    out_dir: Path,
    filtration_type: str,
    use_extended_persistence: bool,
    n_splits: int,
    param_grid: dict,
    n_iter: int,
    n_jobs: int | None,
    verbose: int,
    overwrite: bool,
    random_state: int | None,
) -> tuple[dict, dict, dict, npt.NDArray, float]:
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
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
        )
        bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        y_pred = best_model.predict(X_test)
        cv_results, best_params, clf_report, conf_matrix, roc_score = (
            bayes_search.cv_results_,
            bayes_search.best_params_,
            classification_report(y_test, y_pred, output_dict=True),
            confusion_matrix(y_test, y_pred),
            roc_auc_score(y_test, best_model.decision_function(X_test)),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(cv_results, out_dir / "cv_results.pkl")
        joblib.dump(best_params, out_dir / "best_params.pkl")
        joblib.dump(clf_report, out_dir / "clf_report.pkl")
        joblib.dump(conf_matrix, out_dir / "conf_matrix.pkl")
        joblib.dump(roc_score, out_dir / "roc_score.pkl")
        if verbose:
            print(
                "Saved `cv_results`, `best_params`, `clf_report`, "
                f"`conf_matrix` and `roc_score` in {out_dir}."
            )
    else:
        cv_results = joblib.load(out_dir / "cv_results.pkl")
        best_params = joblib.load(out_dir / "best_params.pkl")
        clf_report = joblib.load(out_dir / "clf_report.pkl")
        conf_matrix = joblib.load(out_dir / "conf_matrix.pkl")
        roc_score = joblib.load(out_dir / "roc_score.pkl")
        if verbose:
            print(
                "Found `cv_results`, `best_params`, `clf_report`, "
                f"`conf_matrix` and `roc_score` in {out_dir}; not overwriting."
            )
    return cv_results, best_params, clf_report, conf_matrix, roc_score


if __name__ == "__main__":
    filtration_type = sys.argv[1]
    use_extended_persistence = sys.argv[2] == "True"
    overwrite = sys.argv[3] == "True"

    n_splits = 5  # number of splits in StratifiedKFold
    n_iter = 2  # number of iterations for BayesSearchCV
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
    X_train, X_test, y_train, y_test = get_data(
        time_series_dir=time_series_dir,
        test_size=0.2,
        random_state=random_state,
    )

    # Set parameter space for BayesSearchCV; best params found by hand are [
    #     -0.25, 0.1, 1, "auto", "linear"
    # ] (in the order below)
    param_grid = {
        "time_series_homology__sigmoid_slope": Real(-1, 1, prior="uniform"),
        "persistence_imager__base_estimator__bandwidth": Real(
            0.01, 0.1,
            prior="log-uniform"
        ),
        "svc__C": Real(0.1, 10, prior="log-uniform"),
        "svc__gamma": Real(0.01, 10, prior='log-uniform'),
        "svc__kernel": Categorical(["linear", "rbf"]),
    }
    suffix = (
        "with_extended_persistence"
        if use_extended_persistence
        else "without_extended_persistence"
    )
    out_dir = Path(
        f"out_files_{filtration_type}_{suffix}"
    )
    cv_results, best_params, clf_report, conf_matrix, roc_score = (
        train_eval_svm(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            out_dir=out_dir,
            filtration_type=filtration_type,
            use_extended_persistence=use_extended_persistence,
            n_splits=n_splits,
            param_grid=param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            overwrite=overwrite,
            verbose=verbose,
            random_state=random_state,
        )
    )
    print("Best Parameters:")
    print(best_params)
    print("Classification Report:")
    print(clf_report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("ROC AUC Score:")
    print(roc_score)
