import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from tqdm import tqdm  # type: ignore


def unzip_and_clean(
    data_dir: Path,
    fixation_reports_dir: Path,
    min_n_fixations: int,
) -> None:
    """Extracts those csv-files from data_dir/event_data_csv.zip that contain a
    fixation report that is at least `min_n_fixations` long and whose label
    appears in `data_dir/slrt_results_new.csv`.
    """
    zip_path = data_dir / "event_data_csv.zip"
    if not fixation_reports_dir.is_dir():
        fixation_reports_dir.mkdir(parents=True, exist_ok=True)
    df_participants = pl.read_csv(data_dir / "slrt_results_new.csv")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            filestem = Path(file).stem
            filename = Path(file).name
            if (
                filestem in df_participants["Trial_ID"]
            ):
                file_data = zip_ref.read(file)
                csv_reader = csv.reader(io.StringIO(file_data.decode("utf-8")))
                rows = list(csv_reader)
                if len(rows) >= min_n_fixations:  # exclude header
                    out_path = fixation_reports_dir / filename
                    out_path.write_bytes(file_data)
    return


def get_time_series_data(
    sp_path: Path,
) -> npt.NDArray:
    """Creates NumPy-array containing x- and y-coordinates of a fixation and
    its start time.

    Args:
        sp_path (Path): Path pointing to a scanpath-file in
            "data_reading_trials/event_data_trial_1_csv".

    Returns:
        numpy.ndarray: NumPy-array of shape (n_fixations, 3) where each row
            contains (start time, x-coordinate, y-coordinate) of a fixation.
    """
    df_sp = pl.read_csv(sp_path).select([
            "onset",
            "x",
            "y",
        ])
    return df_sp.to_numpy()


def process_fixation_reports(
    fixation_reports_dir: Path,
    out_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    for sp_path in tqdm(
        sorted(fixation_reports_dir.glob("*.csv")),
        desc="Processing fixation reports"
    ):
        id = sp_path.stem
        out_file = out_dir / f"time_series_data_reading_trials_{id}.npy"
        if not out_file.is_file() or overwrite:
            out_file.parent.mkdir(exist_ok=True, parents=True)
            time_series_data = get_time_series_data(sp_path)
            np.save(out_file, time_series_data)
            if verbose:
                tqdm.write(
                    f"Saved processed fixation report to {out_dir}."
                )
        else:
            if verbose:
                tqdm.write(
                    f"Found processed fixation report at {out_dir}; not "
                    "overwriting."
                )
    return


def get_labels(
    data_dir: Path,
    time_series_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    df_participants = pl.read_csv(data_dir / "slrt_results_new.csv")
    out_file_dyslexic = time_series_dir / "labels/is_dyslexic.npy"
    if not out_file_dyslexic.is_file() or overwrite:
        is_dyslexic = []
        for time_series_file in sorted(time_series_dir.glob("*.npy")):
            id = "_".join(time_series_file.stem.split("_")[-2:])
            label = (
                df_participants.filter(pl.col("Trial_ID") == id)
                .select("SLRT<=M-SD")
                .item()
            )
            is_dyslexic.append(label)
        out_file_dyslexic.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_file_dyslexic, np.array(is_dyslexic, dtype=int))
        if verbose:
            tqdm.write(
                f"Saved dyslexia labels to {out_file_dyslexic}."
            )
    else:
        if verbose:
            tqdm.write(
                f"Found dyslexia labels at {out_file_dyslexic}; "
                "not overwriting."
            )
    return
