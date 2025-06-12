import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm  # type: ignore


def unzip_and_clean(
    data_dir: Path,
    fixation_reports_dir: Path,
    min_n_fixations: int,
) -> None:
    """Extracts those csv-files from data_dir/event_data_csv.zip that contain a
    fixation report that is at least `min_n_fixations` long and whose label
    appears in `data_dir/slrt_results.csv`.
    """
    zip_path = data_dir / "event_data_csv.zip"
    if not fixation_reports_dir.is_dir():
        fixation_reports_dir.mkdir(parents=True, exist_ok=True)
    df_participants = pl.read_csv(data_dir / "slrt_results.csv")
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
                if len(rows) >= min_n_fixations + 1:  # exclude header
                    out_path = fixation_reports_dir / filename
                    out_path.write_bytes(file_data)
    return


def process_fixation_reports(
    fixation_reports_dir: Path,
    out_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    """Processes fixation reports and creates time series data from them.
    """
    for sp_path in tqdm(
        sorted(fixation_reports_dir.glob("*.csv")),
        desc="Processing fixation reports"
    ):
        id = sp_path.stem
        out_file = out_dir / f"time_series_data_beginning_readers_{id}.npy"
        if not out_file.is_file() or overwrite:
            out_file.parent.mkdir(exist_ok=True, parents=True)
            time_series_data = pl.read_csv(sp_path).select([
                "onset",
                "x",
                "y",
            ]).to_numpy()
            np.save(out_file, time_series_data)
            if verbose:
                tqdm.write(
                    f"Saved processed fixation report to {out_file}."
                )
        else:
            if verbose:
                tqdm.write(
                    f"Found processed fixation report at {out_file}; not "
                    "overwriting."
                )
    return


def get_labels(
    data_dir: Path,
    time_series_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    df_participants = pl.read_csv(data_dir / "slrt_results.csv")
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
