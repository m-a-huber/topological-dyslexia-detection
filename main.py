from pathlib import Path

import numpy as np

from scripts.get_persistence_data import (  # type: ignore
    compute_persistences,
    make_persistence_images,
)
from scripts.process_data import (  # type: ignore
    get_labels,
    process_fixation_reports,
)
from scripts.train_eval_svm import train_eval_svm  # type: ignore


def aggregate_classification_reports(
    reports,
    labels=("0", "1")
):
    metrics = ["precision", "recall"]
    agg = {}
    keys_to_aggregate = list(labels)
    for label in keys_to_aggregate:
        for metric in metrics:
            values = [r[label][metric] for r in reports]
            agg[f"{label}_{metric}_mean"] = np.mean(values)
            agg[f"{label}_{metric}_std"] = np.std(values)
    return agg


if __name__ == "__main__":
    fixation_reports_dir = Path("data/FixationReports")
    dataset_statistics_dir = Path("data/DatasetStatistics")
    participants_stats_path = dataset_statistics_dir / Path(
        "participant_stats.csv"
    )
    time_series_dir = Path("data/TimeSeriesData")
    persistences_dir = Path("data/Persistences")
    persistence_images_dir = Path("data/PersistenceImages")
    process_fixation_reports(
        fixation_reports_dir=fixation_reports_dir,
        out_dir=time_series_dir,
        overwrite=False,
    )
    get_labels(
        participants_stats_path=participants_stats_path,
        out_dir=time_series_dir / "labels",
        overwrite=True,
    )
    compute_persistences(
        time_series_dir=time_series_dir,
        out_dir=persistences_dir,
        overwrite=False,
    )
    make_persistence_images(
        persistences_dir=persistences_dir,
        out_dir=persistence_images_dir,
        bandwidth=10.0,
        weight=lambda pt: 1,
        resolution=[50, 50],
        overwrite=False,
    )
    reports, roc_auc_scores, C_best, gamma_best, kernel_best = train_eval_svm(
        time_series_dir=time_series_dir,
        persistence_images_dir=persistence_images_dir,
        test_size=0.2,
        n_runs=2,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    print(f"{C_best = }")
    print(f"{gamma_best = }")
    print(f"{kernel_best = }")
    agg_report = aggregate_classification_reports(
        reports,
        labels=["non-dyslexic", "dyslexic"],
    )
    for key, value in agg_report.items():
        print(f"{key}:\t\t{value}")
    print(f"{roc_auc_scores = }")
