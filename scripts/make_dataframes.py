from pathlib import Path
from typing import Optional

import polars as pl

subjects_non_dys = [
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "15",
    "16",
    "18",
    "19",
    "20",
    "21",
    "22",
    "42",  # this and following are L2
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
]  # 01, 13, 14, 17 excluded because of POOR calibration or attention disorder
subjects_dys = [
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
]  # excluding P32 because no dyslexia screening result


def get_df(
    baseline_name: str,
    seed: Optional[int] = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Given a `baseline_name`, returns a `pl.DataFrame` whose first four
    columns are `READER_ID`, `LABEL` and , possibly, `TRIAL_ID` and `SAMPLE_ID`
    (depending on the aggregation level of the corresponding model). The
    remaining columns contain the input data required for the corresponding
    model.
    """
    df_out_path = Path(f"./dataframes/dataframe_baseline_{baseline_name}.csv")
    if not df_out_path.exists() or overwrite:
        if baseline_name == "tda":
            # Create df for all subjects
            subject_dfs = []
            for subject in subjects_non_dys + subjects_dys:
                # Parse fixation report
                fixation_report_path = Path(
                    f"./data_copco/FixationReports/FIX_report_P{subject}.txt"
                )
                df_subject = parse_fixation_report(fixation_report_path)
                # Create column containing reader ID
                df_subject = df_subject.with_columns(
                    pl.lit(subject).alias("READER_ID")
                )
                # Create column containing label
                label = 1 if subject in subjects_dys else 0
                df_subject = df_subject.with_columns(
                    pl.lit(label).alias("LABEL")
                )
                # Create column with sample ID
                df_subject = df_subject.rename(
                    {
                        "TRIAL_INDEX": "TRIAL_ID",
                    }
                )
                df_subject = df_subject.with_columns(
                    pl.concat_str(
                        [
                            "READER_ID",
                            "TRIAL_ID",
                        ],
                        separator="-",
                    ).alias("SAMPLE_ID"),
                )
                # Select relevant columns
                df_subject = df_subject.select(
                    [
                        "READER_ID",
                        "LABEL",
                        "TRIAL_ID",
                        "SAMPLE_ID",
                        "CURRENT_FIX_DURATION",
                        "NEXT_SAC_DURATION",
                        "CURRENT_FIX_X",
                        "CURRENT_FIX_Y",
                    ]
                )
                # Cast columns to correct types
                df_subject = df_subject.with_columns(
                    pl.col("READER_ID").cast(pl.Int64),
                    pl.col("LABEL").cast(pl.Int64),
                    pl.col("TRIAL_ID").cast(pl.Int64),
                    pl.col("CURRENT_FIX_DURATION").cast(pl.Float64),
                    pl.col("NEXT_SAC_DURATION")
                    .str.replace(".", "0")
                    .cast(pl.Float64),
                    pl.col("CURRENT_FIX_X")
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                    pl.col("CURRENT_FIX_Y")
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                )
                # Drop practice trials
                df_subject = df_subject.filter(pl.col("TRIAL_ID") > 10)
                # Create column containing start time of fixation
                df_subject = df_subject.with_columns(
                    pl.concat(
                        [
                            pl.Series([0.0]),
                            (
                                df_subject["CURRENT_FIX_DURATION"]
                                + df_subject["NEXT_SAC_DURATION"]
                            ).cum_sum()[:-1],
                        ]
                    ).alias("CURRENT_FIX_START")
                )
                df_subject = df_subject.rename(
                    {
                        "CURRENT_FIX_START": "current_fix_start",
                        "CURRENT_FIX_X": "current_fix_x",
                        "CURRENT_FIX_Y": "current_fix_y",
                    }
                )
                df_subject = df_subject.drop(
                    "CURRENT_FIX_DURATION",
                    "NEXT_SAC_DURATION",
                )
                subject_dfs.append(df_subject)
            df_out = pl.concat(subject_dfs)
        elif baseline_name == "bjornsdottir":
            # Create df for all subjects
            subject_dfs = []
            for subject in subjects_non_dys + subjects_dys:
                extracted_features_path = Path(
                    f"./data_copco/ExtractedFeatures/P{subject}.csv"
                )
                df_subject = pl.read_csv(extracted_features_path)
                df_subject = df_subject.fill_null(0)
                df_subject = df_subject.with_columns(
                    pl.lit(subject).alias("READER_ID")
                )
                # Set dyslexia label
                label = 1 if subject in subjects_dys else 0
                df_subject = df_subject.with_columns(
                    pl.lit(label).cast(pl.Int64).alias("LABEL")
                )
                subject_dfs.append(df_subject)
            df_all = pl.concat(subject_dfs)
            df_all = df_all.rename(
                {
                    "trialId": "TRIAL_ID",
                }
            )
            df_all = df_all.with_columns(
                pl.concat_str(
                    [
                        "READER_ID",
                        "TRIAL_ID",
                    ],
                    separator="-",
                ).alias("SAMPLE_ID")
            )
            # Downsample dyslexic subjects
            df_non_dys = df_all.filter(pl.col("LABEL") == 0)
            df_dys = df_all.filter(pl.col("LABEL") == 1)
            max_number_of_samples = len(df_dys["SAMPLE_ID"].unique())
            df_non_dys_grouped = df_non_dys.group_by(
                "SAMPLE_ID", maintain_order=True
            )
            df_non_dys_downsampled = pl.concat(
                [
                    df
                    for _, df in list(df_non_dys_grouped)[
                        :max_number_of_samples
                    ]
                ]
            )
            # Combine dataframes
            df_all = pl.concat([df_non_dys_downsampled, df_dys])
            df_all = df_all.drop(
                "word",
                "char_IA_ids",
                "part",
                "wordId",
                "sentenceId",
                "speechId",
                "paragraphId",
                "landing_position",
            )
            df_all = df_all.with_columns(pl.col("READER_ID").cast(pl.Int64))
            # Create aggregated reading measures
            aggs = []
            for col in df_all.columns:
                if col in ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]:
                    continue
                aggs.extend(
                    [
                        pl.col(col).mean().alias(f"mean_{col}"),
                        pl.col(col).std().alias(f"std_{col}"),
                        pl.col(col).max().alias(f"max_{col}"),
                    ]
                )
            df_out = df_all.group_by(
                ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
            ).agg(aggs)
            assert df_out.shape == df_out.drop_nans().shape
            # Sort dataframe for reproducibility of final result
            df_out = df_out.sort(
                ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
            )
            df_out = df_out.sample(fraction=1.0, shuffle=True, seed=seed)
        elif baseline_name == "raatikainen":
            data_dict = {
                "READER_ID": [],
                "LABEL": [],
                "mean_fixation_duration": [],
                "mean_saccade_duration": [],
                "mean_saccade_amplitude": [],
                "total_fixation_count": [],
            }
            for subject in subjects_non_dys + subjects_dys:
                # Parse fixation report
                fixation_report_path = Path(
                    f"./data_copco/FixationReports/FIX_report_P{subject}.txt"
                )
                df_subject = parse_fixation_report(fixation_report_path)
                # Select relevant columns
                df_subject = df_subject.select(
                    [
                        "TRIAL_INDEX",
                        "CURRENT_FIX_DURATION",
                        "NEXT_SAC_DURATION",
                        "NEXT_SAC_AMPLITUDE",
                    ]
                )
                # Cast columns to correct types
                df_subject = df_subject.with_columns(
                    pl.col("TRIAL_INDEX").cast(pl.Int64),
                    pl.col("CURRENT_FIX_DURATION").cast(pl.Float64),
                    pl.col("NEXT_SAC_DURATION")
                    .str.replace(".", "0")
                    .cast(pl.Float64),
                    pl.col("NEXT_SAC_AMPLITUDE")
                    .str.replace(".", "0")
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                )
                # Drop practice trials
                df_subject = df_subject.filter(pl.col("TRIAL_INDEX") > 10)
                # Set dyslexia label
                label = 1 if subject in subjects_dys else 0
                # Set values of data-dictionary
                data_dict["READER_ID"].append(int(subject))
                data_dict["LABEL"].append(label)
                data_dict["mean_fixation_duration"].append(
                    df_subject["CURRENT_FIX_DURATION"].mean()
                )
                data_dict["mean_saccade_duration"].append(
                    df_subject["NEXT_SAC_DURATION"].mean()
                )
                data_dict["mean_saccade_amplitude"].append(
                    df_subject["NEXT_SAC_AMPLITUDE"].mean()
                )
                data_dict["total_fixation_count"].append(len(df_subject))
            df_out = pl.from_dict(data_dict)
        elif baseline_name == "benfatto":
            raise NotImplementedError()
        elif baseline_name == "haller":
            raise NotImplementedError()
        else:
            raise ValueError(
                "Invalid choice of `baseline_name`; must be one of `'tda'`, "
                "`'bjornsdottir'`, `'raatikainen'`, `'benfatto'` and "
                "`'haller'`."
            )
        df_out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.write_csv(df_out_path)
        if verbose:
            print(
                f"Saved dataframe for baseline '{baseline_name}' to "
                f"`{df_out_path}`."
            )
    else:
        df_out = pl.read_csv(df_out_path)
        if verbose:
            print(
                f"Found dataframe for baseline '{baseline_name}' at "
                f"`{df_out_path}`; not overwriting."
            )
    return df_out


def parse_fixation_report(
    fixation_report_path: Path,
) -> pl.DataFrame:
    """Creates a dataframe from a fixation report as in the directory
    FixationReports of the CopCo.
    """
    with open(fixation_report_path, "r", encoding="utf-8-sig") as f_in:
        header_line = f_in.readline()
    header = [
        col_name.strip('"') for col_name in header_line.strip().split("\t")
    ]
    df = pl.read_csv(
        fixation_report_path,
        separator="\t",
        skip_lines=1,
        quote_char=None,
        infer_schema=False,
        has_header=False,
        new_columns=header,
    )
    return df
