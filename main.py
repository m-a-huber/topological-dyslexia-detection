from pathlib import Path

from scripts.process_fixation_reports import process_fixation_reports

if __name__ == "__main__":
    fixation_reports_dir = Path("data/FixationReports")
    dataset_statistics_dir = Path("data/DatasetStatistics")
    participants_stats_path = dataset_statistics_dir / Path(
        "participant_stats.csv"
    )
    time_series_dir = Path("data/TimeSeriesData")
    process_fixation_reports(
        fixation_reports_dir,
        time_series_dir
    )
