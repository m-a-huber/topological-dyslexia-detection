# ruff: noqa: E501
"""Constants and hyperparameter definitions for CopCo experiments."""

from scipy.stats import loguniform, randint, uniform

from scripts.utils import UniformSlopeSym

# ============================================================================
# Subject IDs
# ============================================================================

subjects_non_dys_l1 = [
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
]  # P01, P13, P14, P17 excluded because of poor calibration or attention disorder

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
]  # P32 excluded because no dyslexia screening result

subjects_non_dys_l2 = [
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
]

# ============================================================================
# Validation Constants
# ============================================================================

admissible_model_names = [
    "tsh",  # time series homology
    "tsh_aggregated",
    "baseline_bjornsdottir",
    "baseline_raatikainen",
    "baseline_bjornsdottir_with_tsh",
    "baseline_raatikainen_with_tsh_aggregated",
]

admissible_filtration_types_tsh = [
    "horizontal",
    "sloped",
    "sigmoid",
    "arctan",
]

admissible_classifiers_tsh = [
    "svc",
    "rf",
]

admissible_classifiers_bjornsdottir = [
    "rf",
]

admissible_classifiers_raatikainen = [
    "svc",
    "rf",
]

# ============================================================================
# Hyperparameter Helper Functions for TSH Models
# ============================================================================


def _get_common_svc_hyperparams(bandwidth_param: str) -> list[dict]:
    """Generate common SVC hyperparameters for TSH models."""
    return [
        {
            bandwidth_param: loguniform(1e-3, 1e-2),
            "svc__kernel": ["rbf"],
            "svc__C": loguniform(1e-1, 1e2),
            "svc__gamma": loguniform(1e-4, 1e-2),
        },
        {
            bandwidth_param: loguniform(1e-3, 1e-2),
            "svc__kernel": ["linear"],
            "svc__C": loguniform(1e-2, 1e1),
        },
    ]


def _get_common_rf_hyperparams(bandwidth_param: str) -> list[dict]:
    """Generate common RF hyperparameters for TSH models."""
    return [
        {
            bandwidth_param: loguniform(1e-3, 1e-2),
            "rf__n_estimators": randint(100, 2000),
            "rf__max_depth": [None],
            "rf__min_samples_split": randint(2, 50),
            "rf__min_samples_leaf": randint(1, 50),
            "rf__max_features": ["sqrt", "log2", None],
        },
        {
            bandwidth_param: loguniform(1e-3, 1e-2),
            "rf__n_estimators": randint(100, 2000),
            "rf__max_depth": randint(3, 50),
            "rf__min_samples_split": randint(2, 50),
            "rf__min_samples_leaf": randint(1, 50),
            "rf__max_features": uniform(0.05, 0.95),
        },
    ]


def _get_slope_hyperparams(slope_param: str) -> dict:
    """Generate slope hyperparameters for TSH models."""
    return {
        slope_param: UniformSlopeSym(min_slope=0.5, max_slope=4),
    }


# ============================================================================
# TSH Model Hyperparameter Definitions
# ============================================================================

hyperparams_tsh_common_svc = _get_common_svc_hyperparams(
    "persistence_imager__base_estimator__bandwidth"
)
hyperparams_tsh_common_rf = _get_common_rf_hyperparams(
    "persistence_imager__base_estimator__bandwidth"
)
hyperparams_tsh_slope = _get_slope_hyperparams("time_series_homology__slope")

hyperparams_tsh_aggregated_common_svc = _get_common_svc_hyperparams(
    "persistence_imager__base_estimator__base_estimator__bandwidth"
)
hyperparams_tsh_aggregated_common_rf = _get_common_rf_hyperparams(
    "persistence_imager__base_estimator__base_estimator__bandwidth"
)
hyperparams_tsh_aggregated_slope = _get_slope_hyperparams(
    "time_series_homology__base_estimator__slope"
)

# ============================================================================
# Hyperparameter Building Functions
# ============================================================================


def _build_tsh_hyperparams(
    model_prefix: str,
    common_svc: list[dict],
    common_rf: list[dict],
    slope: dict,
) -> dict:
    """Build hyperparameter dictionary for TSH models."""
    filtration_types_with_slope = ["sloped", "sigmoid", "arctan"]
    result = {}

    for filtration_type in ["horizontal", *filtration_types_with_slope]:
        for classifier in ["svc", "rf"]:
            key = f"{model_prefix}_{filtration_type}_{classifier}"
            common = common_svc if classifier == "svc" else common_rf

            if filtration_type == "horizontal":
                result[key] = common
            else:
                result[key] = [
                    slope | hyperparam_dict for hyperparam_dict in common
                ]

    return result


def _build_combined_hyperparams(
    tsh_model_prefix: str,
    baseline_model_name: str,
    classifier: str,
    tsh_common_rf: list[dict],
    tsh_slope: dict,
    baseline_hyperparams: dict,
) -> dict:
    """Build hyperparameter dictionary for combined models."""
    filtration_types_with_slope = ["sloped", "sigmoid", "arctan"]
    result = {}
    # Determine parameter name prefixes based on TSH model type
    if tsh_model_prefix == "tsh":
        bandwidth_param = (
            "feature_union__tsh_features__persistence_imager__"
            "base_estimator__bandwidth"
        )
        slope_param = (
            "feature_union__tsh_features__time_series_homology__slope"
        )
    elif tsh_model_prefix == "tsh_aggregated":
        bandwidth_param = (
            "feature_union__tsh_features__persistence_imager__"
            "base_estimator__base_estimator__bandwidth"
        )
        slope_param = (
            "feature_union__tsh_features__time_series_homology__"
            "base_estimator__slope"
        )
    # Extract and prefix TSH hyperparameters (only bandwidth, not classifier)
    # Since both dicts in tsh_common_rf have the same bandwidth distribution,
    # we only need one (they differ only in classifier params we don't use)
    first_tsh_dict = tsh_common_rf[0]
    base_tsh_dict = {}
    for param_name, param_value in first_tsh_dict.items():
        if "bandwidth" in param_name:
            base_tsh_dict[bandwidth_param] = param_value

    # Build combined hyperparameters for each filtration type
    for filtration_type in ["horizontal", *filtration_types_with_slope]:
        key = (
            f"baseline_{baseline_model_name}_with_{tsh_model_prefix}_"
            f"{filtration_type}_{classifier}"
        )
        # Add slope parameter for non-horizontal filtrations
        if filtration_type != "horizontal":
            tsh_dict_with_slope = base_tsh_dict.copy()
            tsh_dict_with_slope[slope_param] = tsh_slope[
                next(iter(tsh_slope.keys()))
            ]
        else:
            tsh_dict_with_slope = base_tsh_dict.copy()
        # Merge baseline hyperparameters
        final_dict = tsh_dict_with_slope.copy()
        for param_name, param_value in baseline_hyperparams.items():
            final_dict[param_name] = param_value
        # Return as single dict (RandomizedSearchCV can accept dict or list of dicts)
        result[key] = final_dict
    return result


# ============================================================================
# Main Hyperparameter Dictionary
# ============================================================================

hyperparams = {
    **_build_tsh_hyperparams(
        "tsh",
        hyperparams_tsh_common_svc,
        hyperparams_tsh_common_rf,
        hyperparams_tsh_slope,
    ),
    **_build_tsh_hyperparams(
        "tsh_aggregated",
        hyperparams_tsh_aggregated_common_svc,
        hyperparams_tsh_aggregated_common_rf,
        hyperparams_tsh_aggregated_slope,
    ),
    "baseline_bjornsdottir_rf": {
        "rf__n_estimators": [1, 10, 100, 200],
        "rf__max_depth": [1, 3, 5, 7, 9],
        "rf__max_features": [  # "auto" is excluded because deprecated
            "sqrt",
            "log2",
        ],
    },
    # Grids below were found in the code at
    # https://gitlab.jyu.fi/nieminen/dyslexia-detection-public/
    "baseline_raatikainen_svc": {
        "svc__C": [
            1000,
            2000,
            3000,
            5000,
            7000,
            10000,
            20000,
            50000,
            100000,
            500000,
            1000000,
        ],
        "svc__gamma": [
            0.00004,
            0.00006,
            0.00008,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
        ],
    },
    "baseline_raatikainen_rf": {
        "rf__n_estimators": [
            5,
            20,
            30,
            50,
            80,
            100,
            300,
            500,
            1000,
            2000,
        ],
        "rf__max_features": [
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            15,
            20,
        ],
    },
}

# ============================================================================
# Combined Model Hyperparameters
# ============================================================================

# baseline_bjornsdottir_with_tsh (trial-level TSH + baseline features)
hyperparams.update(
    _build_combined_hyperparams(
        "tsh",
        "bjornsdottir",
        "rf",
        hyperparams_tsh_common_rf,
        hyperparams_tsh_slope,
        hyperparams["baseline_bjornsdottir_rf"],
    )
)

# baseline_raatikainen_with_tsh_aggregated (reader-level TSH + baseline features)
hyperparams.update(
    _build_combined_hyperparams(
        "tsh_aggregated",
        "raatikainen",
        "svc",
        hyperparams_tsh_aggregated_common_rf,
        hyperparams_tsh_aggregated_slope,
        hyperparams["baseline_raatikainen_svc"],
    )
)
hyperparams.update(
    _build_combined_hyperparams(
        "tsh_aggregated",
        "raatikainen",
        "rf",
        hyperparams_tsh_aggregated_common_rf,
        hyperparams_tsh_aggregated_slope,
        hyperparams["baseline_raatikainen_rf"],
    )
)
