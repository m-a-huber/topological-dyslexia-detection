from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (  # type: ignore
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (  # type: ignore
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
from tqdm import tqdm  # type: ignore


def get_data(
    time_series_dir: Path,
    persistence_images_dir: Path,
    test_size: float,
    random_state: int | None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    X = np.array(
        [
            np.concatenate(np.load(persistence_images_path))
            for persistence_images_path in persistence_images_dir.glob("*.npy")
        ]
    )
    y = np.load(time_series_dir / "labels" / "is_dyslexic.npy")
    assert len(X) == len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def get_hyperparameters(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    n_jobs: int | None,
    verbose: int,
    random_state: int | None,
) -> tuple[float, float, str]:
    std_scaler = StandardScaler()
    svm = SVC()
    svm_pipeline = Pipeline([("scaler", std_scaler), ("svc", svm)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {
        "svc__C": [0.1, 1, 10, 100, 1000],
        "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100],
        "svc__kernel": ["linear", "rbf"]
    }
    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="f1",
        n_jobs=n_jobs,
        cv=cv,
        verbose=verbose,
    )
    grid_search.fit(X_train, y_train)
    C = grid_search.best_params_.get("svc__C", "Error")
    gamma = grid_search.best_params_.get("svc__gamma", "Error")
    kernel = grid_search.best_params_.get("svc__kernel", "Error")
    return C, gamma, kernel


def train_eval_svm(
    time_series_dir: Path,
    persistence_images_dir: Path,
    test_size: float = 0.2,
    n_runs: int = 10,
    n_jobs: int | None = None,
    verbose: int = 0,
    random_state: int | None = None,
) -> tuple[list[dict], list[float], float, float, str]:
    reports = []
    roc_auc_scores = []
    rng = np.random.default_rng(seed=random_state)
    random_states_svm = rng.integers(low=0, high=10_000, size=n_runs)
    X_train, X_test, y_train, y_test = get_data(
        time_series_dir=time_series_dir,
        persistence_images_dir=persistence_images_dir,
        test_size=test_size,
        random_state=random_state
    )
    C, gamma, kernel = get_hyperparameters(
        X_train,
        y_train,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
    )
    for random_state_svm in tqdm(
        random_states_svm, desc="Fitting and evaluating SVMs"
    ):
        std_scaler = StandardScaler()
        pipeline = Pipeline(
            [
                ("scaler", std_scaler),
                (
                    "SVM",
                    SVC(
                        C=C,
                        gamma=gamma,
                        kernel=kernel,
                        random_state=random_state_svm,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        reports.append(
            classification_report(
                y_test,
                pipeline.predict(X_test),
                target_names=["non-dyslexic", "dyslexic"],
                output_dict=True,
            )
        )
        roc_auc_scores.append(
            roc_auc_score(y_test, pipeline.decision_function(X_test))
        )
    return reports, roc_auc_scores, C, gamma, kernel
