import sys
from collections import defaultdict
from pathlib import Path  # type: ignore

import gudhi.representations as gdrep  # type: ignore
import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.base import clone  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import (  # type: ignore
    PredefinedSplit,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
from skopt import BayesSearchCV  # type: ignore
from skopt.space import Categorical, Real  # type: ignore
from tqdm import tqdm  # type: ignore

from scripts import (  # type: ignore
    process_data_copco,
    process_data_reading_trials,
)
from scripts.time_series_homology import TimeSeriesHomology  # type: ignore
from scripts.utils import (  # type: ignore
    ListTransformer,
    PersistenceImageProcessor,
    PersistenceProcessor,
    weight_abs1p,
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


def train_eval_svm(
    X: list[npt.NDArray],
    y: npt.NDArray,
    out_dir: Path,
    filtration_type: str,
    use_extended_persistence: bool,
    n_splits: int,
    search_space: dict,
    n_iter: int,
    n_points: int,
    n_jobs: int | None,
    verbose: int,
    overwrite: bool,
    random_state: int | None,
) -> tuple[npt.NDArray, list[dict]]:
    best_params_path = out_dir / "best_params.pkl"
    roc_scores_path = out_dir / "roc_scores.npy"
    if not out_dir.is_dir() or overwrite:
        TimeSeriesScaler = ListTransformer(base_estimator=StandardScaler())
        PersistenceImager = ListTransformer(gdrep.PersistenceImage(
            resolution=(75, 75),
            weight=weight_abs1p
        ))
        memory = joblib.Memory(location=out_dir / "pipeline_cache", verbose=0)
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
        ], memory=memory)
        rng = np.random.default_rng(random_state)
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=rng.integers(0, 10_000),
        )
        fold_ixss = [
            test_ixs
            for train_ixs, test_ixs in skf.split(X, y)
        ]
        best_params_list = []
        roc_scores_list = []
        for test_fold in tqdm(
            range(n_splits),
            desc=f"Training and evaluating on {n_splits} splits with {n_iter} "
            "iterations each."
        ):
            # Pick val fold randomly from non-test folds
            val_fold = rng.choice(
                [fold for fold in range(n_splits) if fold != test_fold]
            )
            # Combine non-test and non-val fold into train fold
            train_folds = [
                fold
                for fold in range(n_splits)
                if fold not in (test_fold, val_fold)
            ]
            X_test, y_test = (
                [X[fold_ix] for fold_ix in fold_ixss[test_fold]],
                [y[fold_ix] for fold_ix in fold_ixss[test_fold]],
            )
            X_val, y_val = (
                [X[fold_ix] for fold_ix in fold_ixss[val_fold]],
                [y[fold_ix] for fold_ix in fold_ixss[val_fold]],
            )
            X_train, y_train = (
                [
                    X[fold_ix]
                    for fold_ix in np.concatenate(
                        [fold_ixss[i] for i in train_folds]
                    )
                ],
                [
                    y[fold_ix]
                    for fold_ix in np.concatenate(
                        [fold_ixss[i] for i in train_folds]
                    )
                ],
            )
            X_combined = X_train + X_val
            y_combined = y_train + y_val
            ps = PredefinedSplit([-1] * len(X_train) + [0] * len(X_val))
            bayes_search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=search_space,
                n_iter=n_iter,
                n_points=n_points,
                scoring="roc_auc",
                n_jobs=n_jobs,
                cv=ps,
                verbose=verbose,
                random_state=rng.integers(0, 10_000),
                refit=False,
            )
            bayes_search.fit(X_combined, y_combined)
            best_params_list.append(
                bayes_search.best_params_
            )
            best_model = clone(pipeline).set_params(
                **bayes_search.best_params_
            )
            best_model.fit(
                X_combined,
                y_combined
            )
            roc_scores_list.append(
                roc_auc_score(y_test, best_model.decision_function(X_test))
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_params_list, best_params_path)
        roc_scores = np.array(roc_scores_list)
        np.save(roc_scores_path, roc_scores)
        if verbose:
            print(
                f"Saved `best_params_list` and `roc_scores` in {out_dir}."
            )
    else:
        best_params_list = joblib.load(best_params_path)
        roc_scores = np.load(roc_scores_path)
        if verbose:
            print(
                f"Found `best_params_list` and `roc_scores` in {out_dir}; "
                "not overwriting."
            )
    return roc_scores, best_params_list


def make_df(
    df_dict: defaultdict[str, dict],
    df_path: Path,
    verbose: bool,
    overwrite: bool,
) -> pl.DataFrame:
    if not df_path.is_file() or overwrite:
        rows = sorted(
            set(key for col in df_dict.values() for key in col.keys())
        )
        table = []
        for row in rows:
            row_dict = {"index": row}
            for col, values in df_dict.items():
                row_dict[col] = values.get(row, None)
            table.append(row_dict)
        df_results = pl.DataFrame(table).sort(
            ["with_extended_persistence", "without_extended_persistence"],
            descending=True,
        )
        df_results.write_csv(df_path)
        if verbose:
            print(
                f"Saved `df_results` to {df_path}."
            )
    else:
        df_results = pl.read_csv(df_path)
        if verbose:
            print(
                f"Found `df_results` at {df_path}; not overwriting."
            )
    return df_results


if __name__ == "__main__":
    corpus_name = sys.argv[1]  # one of "copco" and "reading_trials"
    overwrite = sys.argv[2] == "True"

    n_splits = 10  # number of splits in StratifiedKFold
    n_iter = 40  # number of iterations for BayesSearchCV
    n_points = 8  # number of parallel points for BayesSearchCV
    n_jobs = -1  # parallelism for BayesSearchCV
    verbose = 2
    random_state = 42

    # Process corpus files and create time series data
    if corpus_name == "copco":
        fixation_reports_dir = Path("data_copco/FixationReports")
        dataset_statistics_dir = Path("data_copco/DatasetStatistics")
        participants_stats_path = dataset_statistics_dir / Path(
            "participant_stats.csv"
        )
        time_series_dir = Path("data_copco/TimeSeriesData")
        process_data_copco.process_fixation_reports(
            fixation_reports_dir=fixation_reports_dir,
            out_dir=time_series_dir,
            verbose=bool(verbose),
            overwrite=overwrite,
        )
        process_data_copco.get_labels(
            participants_stats_path=participants_stats_path,
            out_dir=time_series_dir / "labels",
            verbose=bool(verbose),
            overwrite=overwrite,
        )
    elif corpus_name == "reading_trials":
        data_dir = Path("data_reading_trials")
        fixation_reports_dir = Path("data_reading_trials/FixationReports")
        time_series_dir = Path("data_reading_trials/TimeSeriesData")
        process_data_reading_trials.unzip_and_clean(
            data_dir=data_dir,
            fixation_reports_dir=fixation_reports_dir,
            min_n_fixations=5,
        )
        process_data_reading_trials.process_fixation_reports(
            fixation_reports_dir=fixation_reports_dir,
            out_dir=time_series_dir,
            verbose=bool(verbose),
            overwrite=overwrite,
        )
        process_data_reading_trials.get_labels(
            data_dir=data_dir,
            time_series_dir=time_series_dir,
            verbose=bool(verbose),
            overwrite=overwrite,
        )
    else:
        raise ValueError(
            "Got invalid value for `corpus_name`, must be one of `'copco'` "
            "and `'reading_trials'`."
        )

    # Load data
    X, y = get_data(
        time_series_dir=time_series_dir,
    )

    df_dict: defaultdict[str, dict] = defaultdict(dict)
    for filtration_type in ["horizontal", "sloped", "sigmoid", "arctan"]:
        if filtration_type == "horizontal":
            search_space = {
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1e-2, 1e2, prior="log-uniform"),
                "svc__gamma": Real(1e-3, 10, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        if filtration_type == "sloped":
            search_space = {
                "time_series_homology__slope": Real(
                    -4, 4, prior="uniform"
                ),
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1e-2, 1e2, prior="log-uniform"),
                "svc__gamma": Real(1e-3, 10, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        if filtration_type == "sigmoid":
            search_space = {
                "time_series_homology__slope": Real(
                    -4, 4, prior="uniform"
                ),
                "time_series_homology__padding_factor": Real(
                    0.01, 0.05, prior="uniform"
                ),
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1e-2, 1e2, prior="log-uniform"),
                "svc__gamma": Real(1e-3, 10, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        if filtration_type == "arctan":
            search_space = {
                "time_series_homology__slope": Real(
                    -4, 4, prior="uniform"
                ),
                "time_series_homology__padding_factor": Real(
                    0.01, 0.05, prior="uniform"
                ),
                "persistence_imager__base_estimator__bandwidth": Real(
                    0.01, 1, prior="log-uniform"
                ),
                "svc__C": Real(1e-2, 1e2, prior="log-uniform"),
                "svc__gamma": Real(1e-3, 10, prior="log-uniform"),
                "svc__kernel": Categorical(["linear", "rbf"]),
            }
        for use_extended_persistence in [True, False]:
            suffix = (
                "with_extended_persistence"
                if use_extended_persistence
                else "without_extended_persistence"
            )
            print(f"Started {filtration_type}_{suffix}.")
            out_dir = Path(
                f"out_files_{corpus_name}/{filtration_type}_{suffix}"
            )
            roc_scores, best_params_list = (
                train_eval_svm(
                    X=X,
                    y=y,
                    out_dir=out_dir,
                    filtration_type=filtration_type,
                    use_extended_persistence=use_extended_persistence,
                    n_splits=n_splits,
                    n_points=n_points,
                    search_space=search_space,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    overwrite=overwrite,
                    verbose=verbose,
                    random_state=random_state,
                )
            )
            print(f"Finished {filtration_type}_{suffix}.")
            df_dict[suffix][filtration_type] = np.around(
                roc_scores.mean(), 4
            )
    df = make_df(
        df_dict=df_dict,
        df_path=out_dir.parent / f"eval_scores_{corpus_name}.csv",
        verbose=bool(verbose),
        overwrite=overwrite,
    )
    if verbose:
        print("\nAverage ROC AUC scores:")
        print(df)
