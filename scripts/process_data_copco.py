from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm  # type: ignore


def process_fixation_reports(
    fixation_reports_dir: Path,
    n_trials: int,
    min_n_fixations: int,
    out_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    """Processes fixation reports and creates time series data from them.
    """
    for sp_path in tqdm(
        sorted(fixation_reports_dir.glob("*.txt")),
        desc="Processing fixation reports"
    ):
        with open(sp_path, "r", encoding="utf-8-sig") as f_in:
            header_line = f_in.readline()
        header = [
            col_name.strip('"') for col_name in header_line.strip().split("\t")
        ]
        df_fixrep = pl.read_csv(
            sp_path,
            separator="\t",
            skip_lines=1,
            quote_char=None,
            infer_schema=False,
            has_header=False,
            new_columns=header,
        ).select([
            "TRIAL_INDEX",
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
            "CURRENT_FIX_DURATION",
            "NEXT_SAC_DURATION",
        ])
        df_fixrep = df_fixrep.with_columns(
            [
                pl.col("TRIAL_INDEX").cast(pl.Int64),
                pl.col("CURRENT_FIX_X").str.replace(",", ".").cast(pl.Float64),
                pl.col("CURRENT_FIX_Y").str.replace(",", ".").cast(pl.Float64),
                pl.col("CURRENT_FIX_DURATION").cast(pl.Float64),
                pl.col("NEXT_SAC_DURATION").str.replace(".", "0").cast(
                    pl.Float64
                ),
            ]
        )
        id = sp_path.stem.split("_")[-1]
        # drop first 10 trials because those are practice trials
        for trial_ix in list(range(11, 11 + n_trials)):
            df_trial = df_fixrep.filter(pl.col("TRIAL_INDEX") == trial_ix)
            if len(df_trial) < min_n_fixations:  # drop short trials
                continue
            df_trial = df_trial.with_columns(
                [
                    pl.concat(
                        [
                            pl.Series([0.0]),
                            (
                                df_trial["CURRENT_FIX_DURATION"]
                                + df_trial["NEXT_SAC_DURATION"]
                            ).cum_sum()[:-1],
                        ]
                    ).alias("CURRENT_FIX_START")
                ]
            ).select(
                [
                    "CURRENT_FIX_START",
                    "CURRENT_FIX_X",
                    "CURRENT_FIX_Y",
                ]
            )
            out_file = out_dir / f"time_series_data_copco_{id}_{trial_ix}.npy"
            if not out_file.is_file() or overwrite:
                out_file.parent.mkdir(exist_ok=True, parents=True)
                time_series_data = df_trial.to_numpy()
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
    participants_stats_path: Path,
    label: str,
    time_series_dir: Path,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    """Creates array of labels corresponding to the processed fixation reports.
    Parameter `label` must be one of `'dyslexic'` and `'native'`.
    """
    df_participants = pl.read_csv(participants_stats_path)
    if label == "dyslexic":
        def _aux_fcn_dyslexic(s):
            return int(s == "yes")
        out_file_labels = time_series_dir / "labels/is_dyslexic.npy"
        if not out_file_labels.is_file() or overwrite:
            is_dyslexic = []
            for time_series_file in sorted(time_series_dir.glob("*.npy")):
                id = time_series_file.stem.split("_")[-2]
                label = _aux_fcn_dyslexic(
                    df_participants.filter(pl.col("subj") == id)
                    .select("dyslexia")
                    .item()
                )
                is_dyslexic.append(label)
            out_file_labels.parent.mkdir(exist_ok=True, parents=True)
            np.save(out_file_labels, np.array(is_dyslexic, dtype=int))
            if verbose:
                tqdm.write(
                    f"Saved dyslexia labels to {out_file_labels}."
                )
        else:
            if verbose:
                tqdm.write(
                    f"Found dyslexia labels at {out_file_labels}; "
                    "not overwriting."
                )
    elif label == "native":
        def _aux_fcn_native(s):
            return int("danish" in s.lower())
        out_file_labels = time_series_dir / "labels/is_native.npy"
        if not out_file_labels.is_file() or overwrite:
            is_native = []
            for time_series_file in sorted(time_series_dir.glob("*.npy")):
                id = time_series_file.stem.split("_")[-2]
                label = _aux_fcn_native(
                    df_participants.filter(pl.col("subj") == id)
                    .select("native_language")
                    .item()
                )
                is_native.append(label)
            out_file_labels.parent.mkdir(exist_ok=True, parents=True)
            np.save(out_file_labels, np.array(is_native, dtype=int))
            if verbose:
                tqdm.write(
                    f"Saved nativeness labels to {out_file_labels}."
                )
        else:
            if verbose:
                tqdm.write(
                    f"Found nativeness labels at {out_file_labels}; "
                    "not overwriting."
                )
    else:
        raise ValueError(
            "Invalid choice of `label`; must be one of `'dyslexic'` and "
            "`'native'`."
        )
    return
