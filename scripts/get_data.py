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
    "42",
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


def get_data(
    model_name: str,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """Given a `baseline_name`, returns a `pl.DataFrame` whose first four
    columns are `READER_ID`, `LABEL` and , possibly, `TRIAL_ID` and `SAMPLE_ID`
    (depending on the aggregation level of the corresponding model). The
    remaining columns contain the input data required for the corresponding
    model.
    """
    if model_name == "tda":
        raise NotImplementedError()
    elif model_name == "bjornsdottir":
        # Create df for non-dyslexic subjects
        subject_dfs_non_dys = []
        for subject in subjects_non_dys:
            extracted_features_path = Path(
                f"./data_copco/ExtractedFeatures/P{subject}.csv"
            )
            df = pl.read_csv(extracted_features_path)
            df = df.fill_null(0)
            df = df.with_columns(pl.lit(subject).alias("participantId"))
            df = df.with_columns(pl.lit(0).alias("LABEL"))
            subject_dfs_non_dys.append(df)
        df_non_dys = pl.concat(subject_dfs_non_dys)
        df_non_dys = df_non_dys.rename(
            {
                "participantId": "READER_ID",
                "trialId": "TRIAL_ID",
            }
        )
        df_non_dys = df_non_dys.with_columns(
            pl.concat_str(
                [
                    "READER_ID",
                    "TRIAL_ID",
                ],
                separator="-",
            ).alias("SAMPLE_ID")
        )
        # Create df for dyslexic subjects
        subject_dfs_dys = []
        for subject in subjects_dys:
            extracted_features_path = Path(
                f"./data_copco/ExtractedFeatures/P{subject}.csv"
            )
            df = pl.read_csv(extracted_features_path)
            df = df.fill_null(0)
            df = df.with_columns(pl.lit(subject).alias("participantId"))
            df = df.with_columns(pl.lit(1).alias("LABEL"))
            subject_dfs_dys.append(df)
        df_dys = pl.concat(subject_dfs_dys)
        df_dys = df_dys.rename(
            {
                "participantId": "READER_ID",
                "trialId": "TRIAL_ID",
            }
        )
        df_dys = df_dys.with_columns(
            pl.concat_str(
                [
                    "READER_ID",
                    "TRIAL_ID",
                ],
                separator="-",
            ).alias("SAMPLE_ID")
        )
        # Downsample dyslexic subjects
        max_number_of_samples = len(df_dys["SAMPLE_ID"].unique())
        df_non_dys_grouped = df_non_dys.group_by(
            "SAMPLE_ID", maintain_order=True
        )
        df_non_dys_downsampled = pl.concat(
            [df for _, df in list(df_non_dys_grouped)[:max_number_of_samples]]
        )
        # Combine dataframes
        df_all = pl.concat([df_non_dys_downsampled, df_dys])
        df_all = df_all.drop(
            [
                "word",
                "char_IA_ids",
                "part",
                "wordId",
                "sentenceId",
                "speechId",
                "paragraphId",
                "landing_position",
            ]
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
        df_aggregated = df_all.group_by(
            ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
        ).agg(aggs)
        assert df_aggregated.shape == df_aggregated.drop_nans().shape
        df_aggregated = df_aggregated.sample(
            fraction=1.0, shuffle=True, seed=seed
        )
        return df_aggregated
    elif model_name == "raatikainen":
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
            with open(fixation_report_path, "r", encoding="utf-8-sig") as f_in:
                header_line = f_in.readline()
            header = [
                col_name.strip('"')
                for col_name in header_line.strip().split("\t")
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
            # Pick relevant columns
            df = df.select(
                [
                    "TRIAL_INDEX",
                    "CURRENT_FIX_DURATION",
                    "NEXT_SAC_DURATION",
                    "NEXT_SAC_AMPLITUDE",
                ]
            )
            # Cast columns to correct types
            df = df.with_columns(
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
            # Set dyslexia label
            label = 1 if subject in subjects_dys else 0
            # Set values of data-dictionary
            data_dict["READER_ID"].append(int(subject))
            data_dict["LABEL"].append(label)
            data_dict["mean_fixation_duration"].append(
                df["CURRENT_FIX_DURATION"].mean()
            )
            data_dict["mean_saccade_duration"].append(
                df["NEXT_SAC_DURATION"].mean()
            )
            data_dict["mean_saccade_amplitude"].append(
                df["NEXT_SAC_AMPLITUDE"].mean()
            )
            data_dict["total_fixation_count"].append(len(df))
        df_aggregated = pl.from_dict(data_dict)
        return df_aggregated
    elif model_name == "haller":
        raise NotImplementedError()
    else:
        raise ValueError(
            "Invalid choice of `baseline_name`; must be one of `'tda'`, "
            "`'bjornsdottir'`, `'raatikainen'` and `'haller'`."
        )
