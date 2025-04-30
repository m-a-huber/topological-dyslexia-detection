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
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
from time_series_homology import TimeSeriesHomology  # type: ignore
from utils import (  # type: ignore
    ListTransformer,
    PersistenceImageProcessor,
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
    return pt[1] + 1


def train_eval_svm(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    out_dir: Path,
    n_jobs: int | None,
    overwrite: bool,
    verbose: int,
    random_state: int | None,
) -> tuple[dict, dict, dict, npt.NDArray, float]:
    if not out_dir.is_dir() or overwrite:
        TimeSeriesScaler = ListTransformer(base_estimator=StandardScaler())
        PersistenceImager = ListTransformer(gdrep.PersistenceImage(
            weight=weight_abs1p
        ))
        pipeline = Pipeline([
            ("time_series_scaler", TimeSeriesScaler),
            ("time_series_homology", TimeSeriesHomology(
                drop_infinite_persistence=True
            )),
            ("persistence_imager", PersistenceImager),
            ("persistence_image_scaler", PersistenceImageProcessor(
                scaler=MinMaxScaler()
            )),
            ("svc", SVC(class_weight="balanced"))
        ])
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
        param_grid = {
            "time_series_homology__type": ["sigmoidal"],
            "time_series_homology__sigmoid_slope": [-0.5, -1, -1.5, -2],
            "persistence_imager__base_estimator__resolution": [(75, 75)],
            "persistence_imager__base_estimator__bandwidth": [0.01, 0.05, 0.1],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": [0.001, 0.01, 0.1, "scale", "auto"],
            "svc__kernel": ["linear", "rbf"],
        }
        # cv = StratifiedKFold(
        #     n_splits=2,
        #     shuffle=True,
        #     random_state=random_state,
        # )
        # param_grid = {
        #     "time_series_homology__type": ["sigmoidal"],
        #     "time_series_homology__sigmoid_slope": [-1],
        #     "persistence_imager__base_estimator__resolution": [(75, 75)],
        #     "persistence_imager__base_estimator__bandwidth": [0.1],
        #     "svc__C": [1],
        #     "svc__gamma": ["auto"],
        #     "svc__kernel": ["linear"],
        # }
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        cv_results, best_params, clf_report, conf_matrix, roc_score = (
            grid_search.cv_results_,
            grid_search.best_params_,
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
    time_series_dir = Path("data/TimeSeriesData")
    out_dir = Path("out_files")
    n_jobs = -1
    overwrite = False
    verbose = 2
    random_state = 42

    X_train, X_test, y_train, y_test = get_data(
        time_series_dir=time_series_dir,
        test_size=0.2,
        random_state=random_state,
    )

    cv_results, best_params, clf_report, conf_matrix, roc_score = (
        train_eval_svm(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            out_dir=out_dir,
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
