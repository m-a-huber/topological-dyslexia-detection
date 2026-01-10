# ruff: noqa: E501
"""Constants and hyperparameter definitions for CopCo experiments."""

from scipy.stats import loguniform

from scripts.utils import (
    NestedDict,
    UniformSlopeSym,
    weight_by_lifetime,
    weight_constant,
)

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
    "baseline_bjornsdottir",
    "baseline_raatikainen",
    "baseline_bjornsdottir_with_tsh",
    "baseline_raatikainen_with_tsh",
]

admissible_levels = [
    "trial",
    "reader",
]

admissible_filtration_types_tsh = [
    "horizontal",
    "sloped",
    "sigmoid",
    "arctan",
]

admissible_classifiers_tsh = [
    "svc",
]

admissible_classifiers_bjornsdottir = [
    "rf",
]

admissible_classifiers_raatikainen = [
    "svc",
    "rf",
]

# ============================================================================
# Hyperparameter grid
# ============================================================================

# Initialize hyperparameter dict


hyperparams = NestedDict()

# Hyperparameters for Bjornsdottir-baseline

hyperparams_baseline_bjornsdottir_rf = {
    "rf__n_estimators": [1, 10, 100, 200],
    "rf__max_depth": [1, 3, 5, 7, 9],
    "rf__max_features": [  # "auto" is excluded because deprecated
        "sqrt",
        "log2",
    ],
}

hyperparams["baseline_bjornsdottir"]["trial"]["rf"] = (
    hyperparams_baseline_bjornsdottir_rf
)

hyperparams["baseline_bjornsdottir"]["reader"]["rf"] = (
    hyperparams_baseline_bjornsdottir_rf
)

# Hyperparameters for Raatikainen-baseline

hyperparams_baseline_raatikainen_rf = {
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
}

# Grids below were found in the code at
# https://gitlab.jyu.fi/nieminen/dyslexia-detection-public/
hyperparams_baseline_raatikainen_svc = {
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
}

hyperparams["baseline_raatikainen"]["trial"]["svc"] = (
    hyperparams_baseline_raatikainen_svc
)

hyperparams["baseline_raatikainen"]["trial"]["rf"] = (
    hyperparams_baseline_raatikainen_rf
)

hyperparams["baseline_raatikainen"]["reader"]["svc"] = (
    hyperparams_baseline_raatikainen_svc
)

hyperparams["baseline_raatikainen"]["reader"]["rf"] = (
    hyperparams_baseline_raatikainen_rf
)

# Hyperparameters for trial-level TSH-models

MIN_SLOPE = 0.5
MAX_SLOPE = 4
MIN_BANDWIDTH = 1e-3
MAX_BANDWIDTH = 1e-1

hyperparams_tsh_trial_features = {
    "time_series_homology__slope": UniformSlopeSym(
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE
    ),
    "persistence_imager__base_estimator__bandwidth": loguniform(
        MIN_BANDWIDTH, MAX_BANDWIDTH
    ),
    "persistence_imager__base_estimator__weight": [
        weight_by_lifetime,
        weight_constant,
    ],
}

hyperparams_tsh_trial_svc = [
    {
        **hyperparams_tsh_trial_features,
        "svc__kernel": ["rbf"],
        "svc__C": loguniform(1e-1, 1e2),
        "svc__gamma": loguniform(1e-4, 1e-2),
    },
    {
        **hyperparams_tsh_trial_features,
        "svc__kernel": ["linear"],
        "svc__C": loguniform(1e-2, 1e1),
    },
]

hyperparams["tsh"]["trial"]["svc"] = hyperparams_tsh_trial_svc

# Hyperparameters for reader-level TSH-models

hyperparams_tsh_reader_features = {
    "time_series_homology__base_estimator__slope": UniformSlopeSym(
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE
    ),
    "persistence_imager__base_estimator__base_estimator__bandwidth": loguniform(
        MIN_BANDWIDTH, MAX_BANDWIDTH
    ),
    "persistence_imager__base_estimator__base_estimator__weight": [
        weight_by_lifetime,
        weight_constant,
    ],
}

hyperparams_tsh_reader_svc = [
    {
        **hyperparams_tsh_reader_features,
        "svc__kernel": ["rbf"],
        "svc__C": loguniform(1e-1, 1e2),
        "svc__gamma": loguniform(1e-4, 1e-2),
    },
    {
        **hyperparams_tsh_reader_features,
        "svc__kernel": ["linear"],
        "svc__C": loguniform(1e-2, 1e1),
    },
]

hyperparams["tsh"]["reader"]["svc"] = hyperparams_tsh_reader_svc

# Hyperparameters for trial-level Bjornsdottir-baseline+TSH-models

hyperparams_baseline_bjornsdottir_with_tsh_trial_features = {
    "feature_union__tsh_features__time_series_homology__slope": UniformSlopeSym(
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__bandwidth": loguniform(
        MIN_BANDWIDTH, MAX_BANDWIDTH
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__weight": [
        weight_by_lifetime,
        weight_constant,
    ],
}

hyperparams["baseline_bjornsdottir_with_tsh"]["trial"]["rf"] = {
    **hyperparams_baseline_bjornsdottir_with_tsh_trial_features,
    **hyperparams_baseline_bjornsdottir_rf,
}

# Hyperparameters for reader-level Bjornsdottir-baseline+TSH-models

hyperparams_baseline_bjornsdottir_with_tsh_reader_features = {
    "feature_union__tsh_features__time_series_homology__base_estimator__slope": UniformSlopeSym(
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__base_estimator__bandwidth": loguniform(
        MIN_BANDWIDTH, MAX_BANDWIDTH
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__base_estimator__weight": [
        weight_by_lifetime,
        weight_constant,
    ],
}

hyperparams["baseline_bjornsdottir_with_tsh"]["reader"]["rf"] = {
    **hyperparams_baseline_bjornsdottir_with_tsh_reader_features,
    **hyperparams_baseline_bjornsdottir_rf,
}

# Hyperparameters for trial-level Raatikainen-baseline+TSH-models

hyperparams_baseline_raatikainen_with_tsh_trial_features = {
    "feature_union__tsh_features__time_series_homology__slope": UniformSlopeSym(
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__bandwidth": loguniform(
        MIN_BANDWIDTH, MAX_BANDWIDTH
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__weight": [
        weight_by_lifetime,
        weight_constant,
    ],
}

hyperparams["baseline_raatikainen_with_tsh"]["trial"]["rf"] = {
    **hyperparams_baseline_raatikainen_with_tsh_trial_features,
    **hyperparams_baseline_raatikainen_rf,
}

hyperparams["baseline_raatikainen_with_tsh"]["trial"]["svc"] = {
    **hyperparams_baseline_raatikainen_with_tsh_trial_features,
    **hyperparams_baseline_raatikainen_svc,
}

# Hyperparameters for reader-level Raatikainen-baseline+TSH-models

hyperparams_baseline_raatikainen_with_tsh_reader_features = {
    "feature_union__tsh_features__time_series_homology__base_estimator__slope": UniformSlopeSym(
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__base_estimator__bandwidth": loguniform(
        MIN_BANDWIDTH, MAX_BANDWIDTH
    ),
    "feature_union__tsh_features__persistence_imager__base_estimator__base_estimator__weight": [
        weight_by_lifetime,
        weight_constant,
    ],
}

hyperparams["baseline_raatikainen_with_tsh"]["reader"]["rf"] = {
    **hyperparams_baseline_raatikainen_with_tsh_reader_features,
    **hyperparams_baseline_raatikainen_rf,
}

hyperparams["baseline_raatikainen_with_tsh"]["reader"]["svc"] = {
    **hyperparams_baseline_raatikainen_with_tsh_reader_features,
    **hyperparams_baseline_raatikainen_svc,
}

# Cast hyperparams back to dict

hyperparams = hyperparams.to_dict()
