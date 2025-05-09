from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl  # type: ignore
from tqdm import tqdm  # type: ignore


def get_time_series_data(
    sp_path: Path,
) -> npt.NDArray:
    """Creates NumPy-array containing x- and y-coordinates of a fixation and
    its start time. The latter is computed as a cumulative sum of fixation
    duration and duration of following saccade.

    Args:
        sp_path (Path): Path pointing to a scanpath-file in
            "data_copco/FixationReports".

    Returns:
        numpy.ndarray: NumPy-array of shape (n_fixations, 3) where each row
            contains (start time, x-coordinate, y-coordinate) of a fixation.
    """
    with open(sp_path, "r", encoding="utf-8-sig") as f_in:
        header_line = f_in.readline()
    header = [
        col_name.strip('"') for col_name in header_line.strip().split("\t")
    ]
    df_sp = pl.read_csv(
        sp_path,
        separator="\t",
        skip_lines=1,
        quote_char=None,
        infer_schema=False,
        has_header=False,
        new_columns=header,
    ).select([
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
            "CURRENT_FIX_DURATION",
            "NEXT_SAC_DURATION",
        ])
    df_sp = df_sp.with_columns(
        [
            pl.col("CURRENT_FIX_X").str.replace(",", ".").cast(pl.Float64),
            pl.col("CURRENT_FIX_Y").str.replace(",", ".").cast(pl.Float64),
            pl.col("CURRENT_FIX_DURATION").cast(pl.Float64),
            pl.col("NEXT_SAC_DURATION").str.replace(".", "0").cast(pl.Float64),
        ]
    )
    df_sp = df_sp.with_columns(
        [
            pl.concat(
                [
                    pl.Series([0.0]),
                    (
                        df_sp["CURRENT_FIX_DURATION"]
                        + df_sp["NEXT_SAC_DURATION"]
                    ).cum_sum()[:-1],
                ]
            ).alias("CURRENT_FIX_START")
        ]
    )
    return df_sp.select(
        [
            "CURRENT_FIX_START",
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
        ]
    ).to_numpy()


def process_fixation_reports(
    fixation_reports_dir: Path,
    out_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    for sp_path in tqdm(
        sorted(fixation_reports_dir.glob("*.txt")),
        desc="Processing fixation reports"
    ):
        id = sp_path.stem.split("_")[-1]
        out_file = out_dir / f"time_series_data_copco_{id}.npy"
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
    participants_stats_path: Path,
    out_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    df_participants = pl.read_csv(participants_stats_path).filter(
        pl.col("subj") != "P14"  # because fixation data from P14 is missing
    ).sort("subj")
    is_native = (
        df_participants["native_language"].str.contains("Danish").to_numpy()
    )
    is_dyslexic = (df_participants["dyslexia"] == "yes").to_numpy()
    out_file_native = out_dir / "is_native.npy"
    if not out_file_native.is_file() or overwrite:
        out_file_native.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_file_native, is_native.astype(int))
        if verbose:
            tqdm.write(
                f"Saved native speaker labels to {out_file_native}."
            )
    else:
        if verbose:
            tqdm.write(
                f"Found native speaker labels at {out_file_native}; not "
                "overwriting."
            )
    out_file_dyslexic = out_dir / "is_dyslexic.npy"
    if not out_file_dyslexic.is_file() or overwrite:
        out_file_dyslexic.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_file_dyslexic, is_dyslexic.astype(int))
        if verbose:
            tqdm.write(
                f"Saved dyslexia labels to {out_file_native}."
            )
    else:
        if verbose:
            tqdm.write(
                f"Found dyslexia labels at {out_file_native}; not overwriting."
            )
    return
