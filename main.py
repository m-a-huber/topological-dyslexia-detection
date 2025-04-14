from pathlib import Path

import numpy as np

from scripts.process_data import (  # type: ignore
    get_labels,
    process_fixation_reports,
)

if __name__ == "__main__":
    fixation_reports_dir = Path("data/FixationReports")
    dataset_statistics_dir = Path("data/DatasetStatistics")
    participants_stats_path = dataset_statistics_dir / Path(
        "participant_stats.csv"
    )
    out_dir = Path("data/TimeSeriesData")
    process_fixation_reports(
        fixation_reports_dir,
        out_dir
    )
    get_labels(
        participants_stats_path,
        out_dir / "labels",
    )
    n_time_series = len(sorted(out_dir.glob("*.npy")))
    n_labels_native = len(np.load(out_dir / "labels" / "is_native.npy"))
    n_labels_dyslexic = len(np.load(out_dir / "labels" / "is_dyslexic.npy"))
    assert n_time_series == n_labels_native
    assert n_time_series == n_labels_dyslexic
