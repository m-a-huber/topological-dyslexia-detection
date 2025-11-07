# ruff: noqa: E501
from scipy.stats import loguniform

from scripts.utils import UniformSlope

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
]  # 01, 13, 14, 17 excluded because of poor calibration or attention disorder

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

admissible_filtration_types_tda_experiment = [
    "horizontal",
    "sloped",
    "sigmoid",
    "arctan",
]

admissible_persistence_types_tda_experiment = [
    "ordinary",
    "extended",
]

admissible_model_kinds_raatikainen = [
    "rf",
    "svc",
]

hyperparams = {
    "tda_experiment_horizontal": [
        {
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["rbf"],
            "svc__C": loguniform(1e-1, 1e2),
            "svc__gamma": loguniform(1e-4, 1e-2),
        },
        {
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["linear"],
            "svc__C": loguniform(1e-2, 1e1),
        },
    ],
    "tda_experiment_sloped": [
        {
            "feature_union__time_series_features__time_series_homology__slope": UniformSlope(
                min_slope=-2, max_slope=2
            ),
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["rbf"],
            "svc__C": loguniform(1e-1, 1e2),
            "svc__gamma": loguniform(1e-4, 1e-2),
        },
        {
            "feature_union__time_series_features__time_series_homology__slope": UniformSlope(
                min_slope=-2, max_slope=2
            ),
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["linear"],
            "svc__C": loguniform(1e-2, 1e1),
        },
    ],
    "tda_experiment_sigmoid": [
        {
            "feature_union__time_series_features__time_series_homology__slope": UniformSlope(
                min_slope=-2, max_slope=2
            ),
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["rbf"],
            "svc__C": loguniform(1e-1, 1e2),
            "svc__gamma": loguniform(1e-4, 1e-2),
        },
        {
            "feature_union__time_series_features__time_series_homology__slope": UniformSlope(
                min_slope=-2, max_slope=2
            ),
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["linear"],
            "svc__C": loguniform(1e-2, 1e1),
        },
    ],
    "tda_experiment_arctan": [
        {
            "feature_union__time_series_features__time_series_homology__slope": UniformSlope(
                min_slope=-2, max_slope=2
            ),
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["rbf"],
            "svc__C": loguniform(1e-1, 1e2),
            "svc__gamma": loguniform(1e-4, 1e-2),
        },
        {
            "feature_union__time_series_features__time_series_homology__slope": UniformSlope(
                min_slope=-2, max_slope=2
            ),
            "feature_union__time_series_features__persistence_imager__base_estimator__bandwidth": loguniform(
                1e-3, 1e-1
            ),
            "svc__kernel": ["linear"],
            "svc__C": loguniform(1e-2, 1e1),
        },
    ],
    "baseline_bjornsdottir": {
        "rf__n_estimators": [
            1,
            10,
            100,
            200,
        ],
        "rf__max_depth": [
            1,
            3,
            5,
            7,
            9,
        ],
        "rf__max_features": [  # "auto" is excluded because deprecated
            "sqrt",
            "log2",
        ],
    },
    # Grids below were found in the code at
    # https://gitlab.jyu.fi/nieminen/dyslexia-detection-public/
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
}
