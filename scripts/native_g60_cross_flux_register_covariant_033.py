from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SCAN_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_register_invariant_scan_032.json"
)

SCAN_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_register_invariant_scan_032.npz"
)

IDENTIFICATION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_maximizer_axis_identification_029.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_register_covariant_033.json"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_register_covariant_axes_033.csv"
)

SAMPLE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_register_covariant_samples_033.csv"
)

REGRESSION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_register_covariant_regressions_033.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_register_covariant_033.npz"
)

TRACE_TARGET = 5.0 / 2.0

TRACE_TOLERANCE = 2e-10
AXIS_SPECTRUM_TOLERANCE = 2e-9

SAVED_SAMPLE_COUNT = 5000
BIN_COUNT = 200


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(array)
        )
    )


def rational_label(
    value: float,
    denominator_limit: int = 100000,
) -> str:
    return str(
        Fraction(float(value))
        .limit_denominator(
            denominator_limit
        )
    )


def register_covariants(
    directions: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    overlaps = (
        directions @ axis_lines.T
    )

    weights = overlaps**2

    return np.einsum(
        "ni,ia,ib->nab",
        weights,
        axis_lines,
        axis_lines,
    )


def covariant_invariants(
    covariants: np.ndarray,
) -> dict[str, np.ndarray]:
    eigenvalues = np.linalg.eigvalsh(
        covariants
    )

    trace = np.trace(
        covariants,
        axis1=1,
        axis2=2,
    )

    square = np.einsum(
        "nij,njk->nik",
        covariants,
        covariants,
    )

    cube = np.einsum(
        "nij,njk->nik",
        square,
        covariants,
    )

    trace_square = np.trace(
        square,
        axis1=1,
        axis2=2,
    )

    trace_cube = np.trace(
        cube,
        axis1=1,
        axis2=2,
    )

    determinant = np.linalg.det(
        covariants
    )

    return {
        "eigenvalues": eigenvalues,
        "trace": trace,
        "trace_square": trace_square,
        "trace_cube": trace_cube,
        "determinant": determinant,
        "minimum_eigenvalue": (
            eigenvalues[:, 0]
        ),
        "second_eigenvalue": (
            eigenvalues[:, 1]
        ),
        "third_eigenvalue": (
            eigenvalues[:, 2]
        ),
        "maximum_eigenvalue": (
            eigenvalues[:, 3]
        ),
        "spectral_gap_top": (
            eigenvalues[:, 3]
            - eigenvalues[:, 2]
        ),
        "spectral_spread": (
            eigenvalues[:, 3]
            - eigenvalues[:, 0]
        ),
    }


def regression_summary(
    name: str,
    design: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, np.ndarray]:
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    prediction = (
        design @ coefficients
    )

    residual = (
        target - prediction
    )

    centered = (
        target - np.mean(target)
    )

    denominator = float(
        np.dot(
            centered,
            centered,
        )
    )

    r_squared = (
        1.0
        - float(
            np.dot(
                residual,
                residual,
            )
        )
        / denominator
        if denominator > 0.0
        else 1.0
    )

    row = {
        "model": name,
        "feature_names": json.dumps(
            feature_names
        ),
        "coefficient_count": len(
            coefficients
        ),
        "coefficients": json.dumps(
            coefficients.tolist()
        ),
        "r_squared": r_squared,
        "root_mean_square_residual": float(
            np.sqrt(
                np.mean(
                    residual**2
                )
            )
        ),
        "maximum_absolute_residual": max_abs(
            residual
        ),
        "minimum_residual": float(
            np.min(residual)
        ),
        "maximum_residual": float(
            np.max(residual)
        ),
    }

    return row, prediction


def binned_target_spread(
    scalar: np.ndarray,
    target: np.ndarray,
    bin_count: int = BIN_COUNT,
) -> dict:
    minimum = float(
        np.min(scalar)
    )

    maximum = float(
        np.max(scalar)
    )

    edges = np.linspace(
        minimum,
        maximum,
        bin_count + 1,
    )

    indices = np.clip(
        np.digitize(
            scalar,
            edges,
        )
        - 1,
        0,
        bin_count - 1,
    )

    maximum_spread = 0.0
    populated_count = 0
    widest = None

    for bin_index in range(bin_count):
        values = target[
            indices == bin_index
        ]

        if len(values) < 2:
            continue

        populated_count += 1

        spread = float(
            np.max(values)
            - np.min(values)
        )

        if spread > maximum_spread:
            maximum_spread = spread

            widest = {
                "bin_index": bin_index,
                "scalar_minimum": float(
                    edges[bin_index]
                ),
                "scalar_maximum": float(
                    edges[
                        bin_index + 1
                    ]
                ),
                "sample_count": len(
                    values
                ),
                "target_spread": spread,
            }

    return {
        "bin_count": bin_count,
        "populated_bin_count": (
            populated_count
        ),
        "maximum_target_spread": (
            maximum_spread
        ),
        "widest_bin": widest,
    }


def nearest_neighbor_invariant_ambiguity(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    maximum_points: int = 8000,
) -> dict:
    from scipy.spatial import cKDTree

    point_count = min(
        len(feature_matrix),
        maximum_points,
    )

    indices = np.linspace(
        0,
        len(feature_matrix) - 1,
        point_count,
        dtype=np.int64,
    )

    features = np.array(
        feature_matrix[indices],
        dtype=np.float64,
    )

    targets = np.array(
        target[indices],
        dtype=np.float64,
    )

    means = np.mean(
        features,
        axis=0,
    )

    scales = np.std(
        features,
        axis=0,
    )

    scales[
        scales < 1e-14
    ] = 1.0

    normalized_features = (
        features - means
    ) / scales

    tree = cKDTree(
        normalized_features
    )

    distances, neighbors = tree.query(
        normalized_features,
        k=2,
    )

    nearest_distances = (
        distances[:, 1]
    )

    nearest_indices = (
        neighbors[:, 1]
    )

    target_differences = np.abs(
        targets
        - targets[
            nearest_indices
        ]
    )

    return {
        "sample_count": point_count,
        "median_feature_distance": float(
            np.median(
                nearest_distances
            )
        ),
        "maximum_feature_distance": float(
            np.max(
                nearest_distances
            )
        ),
        "median_target_difference": float(
            np.median(
                target_differences
            )
        ),
        "maximum_target_difference": float(
            np.max(
                target_differences
            )
        ),
        "q95_target_difference": float(
            np.quantile(
                target_differences,
                0.95,
            )
        ),
        "q99_target_difference": float(
            np.quantile(
                target_differences,
                0.99,
            )
        ),
    }


def axis_covariant_profile(
    axis_lines: np.ndarray,
) -> tuple[
    list[dict],
    np.ndarray,
    np.ndarray,
]:
    covariants = register_covariants(
        axis_lines,
        axis_lines,
    )

    invariants = covariant_invariants(
        covariants
    )

    eigenvalues = invariants[
        "eigenvalues"
    ]

    mean_eigenvalues = np.mean(
        eigenvalues,
        axis=0,
    )

    maximum_spectrum_residual = max_abs(
        eigenvalues
        - mean_eigenvalues
    )

    rows = []

    for axis_id in range(
        len(axis_lines)
    ):
        values = eigenvalues[
            axis_id
        ]

        rows.append(
            {
                "axis_id": axis_id,
                "trace": float(
                    invariants[
                        "trace"
                    ][axis_id]
                ),
                "trace_square": float(
                    invariants[
                        "trace_square"
                    ][axis_id]
                ),
                "trace_cube": float(
                    invariants[
                        "trace_cube"
                    ][axis_id]
                ),
                "determinant": float(
                    invariants[
                        "determinant"
                    ][axis_id]
                ),
                "eigenvalue_0": float(
                    values[0]
                ),
                "eigenvalue_1": float(
                    values[1]
                ),
                "eigenvalue_2": float(
                    values[2]
                ),
                "eigenvalue_3": float(
                    values[3]
                ),
                "rational_eigenvalue_0": (
                    rational_label(
                        values[0]
                    )
                ),
                "rational_eigenvalue_1": (
                    rational_label(
                        values[1]
                    )
                ),
                "rational_eigenvalue_2": (
                    rational_label(
                        values[2]
                    )
                ),
                "rational_eigenvalue_3": (
                    rational_label(
                        values[3]
                    )
                ),
            }
        )

    return (
        rows,
        covariants,
        np.array(
            mean_eigenvalues,
            dtype=np.float64,
        ),
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AXIS_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGRESSION_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scan_receipt = json.loads(
        SCAN_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    scan_data = np.load(
        SCAN_NPZ_PATH
    )

    identification_data = np.load(
        IDENTIFICATION_NPZ_PATH
    )

    axis_lines = np.array(
        identification_data[
            "native_line_matrix"
        ],
        dtype=np.float64,
    )

    directions = np.array(
        scan_data[
            "random_directions"
        ],
        dtype=np.float64,
    )

    top_eigenvalues = np.array(
        scan_data[
            "top_eigenvalues"
        ],
        dtype=np.float64,
    )

    s4 = np.array(
        scan_data["s4"],
        dtype=np.float64,
    )

    s6 = np.array(
        scan_data["s6"],
        dtype=np.float64,
    )

    s8 = np.array(
        scan_data["s8"],
        dtype=np.float64,
    )

    axis_distances = np.array(
        scan_data[
            "axis_distances"
        ],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis line shape: {axis_lines.shape}"
        )

    if directions.shape[1] != 4:
        raise RuntimeError(
            f"unexpected direction shape: {directions.shape}"
        )

    (
        axis_rows,
        axis_covariants,
        axis_mean_spectrum,
    ) = axis_covariant_profile(
        axis_lines
    )

    axis_spectra = np.array(
        [
            [
                row["eigenvalue_0"],
                row["eigenvalue_1"],
                row["eigenvalue_2"],
                row["eigenvalue_3"],
            ]
            for row in axis_rows
        ],
        dtype=np.float64,
    )

    axis_spectrum_residual = max_abs(
        axis_spectra
        - axis_mean_spectrum
    )

    covariants = register_covariants(
        directions,
        axis_lines,
    )

    invariants = covariant_invariants(
        covariants
    )

    trace_residual = max_abs(
        invariants["trace"]
        - TRACE_TARGET
    )

    regression_rows = []
    regression_predictions = {}

    sample_count = len(
        directions
    )

    ones = np.ones(
        sample_count,
        dtype=np.float64,
    )

    models = {
        "s4_affine": (
            np.column_stack(
                [
                    ones,
                    s4,
                ]
            ),
            [
                "1",
                "S4",
            ],
        ),
        "covariant_trace2_affine": (
            np.column_stack(
                [
                    ones,
                    invariants[
                        "trace_square"
                    ],
                ]
            ),
            [
                "1",
                "tr(C^2)",
            ],
        ),
        "covariant_elementary": (
            np.column_stack(
                [
                    ones,
                    invariants[
                        "trace_square"
                    ],
                    invariants[
                        "trace_cube"
                    ],
                    invariants[
                        "determinant"
                    ],
                ]
            ),
            [
                "1",
                "tr(C^2)",
                "tr(C^3)",
                "det(C)",
            ],
        ),
        "covariant_spectrum_affine": (
            np.column_stack(
                [
                    ones,
                    invariants[
                        "minimum_eigenvalue"
                    ],
                    invariants[
                        "second_eigenvalue"
                    ],
                    invariants[
                        "third_eigenvalue"
                    ],
                    invariants[
                        "maximum_eigenvalue"
                    ],
                ]
            ),
            [
                "1",
                "c0",
                "c1",
                "c2",
                "c3",
            ],
        ),
        "covariant_spectrum_quadratic": (
            np.column_stack(
                [
                    ones,
                    invariants[
                        "minimum_eigenvalue"
                    ],
                    invariants[
                        "second_eigenvalue"
                    ],
                    invariants[
                        "third_eigenvalue"
                    ],
                    invariants[
                        "maximum_eigenvalue"
                    ],
                    invariants[
                        "minimum_eigenvalue"
                    ] ** 2,
                    invariants[
                        "second_eigenvalue"
                    ] ** 2,
                    invariants[
                        "third_eigenvalue"
                    ] ** 2,
                    invariants[
                        "maximum_eigenvalue"
                    ] ** 2,
                    invariants[
                        "minimum_eigenvalue"
                    ]
                    * invariants[
                        "maximum_eigenvalue"
                    ],
                    invariants[
                        "second_eigenvalue"
                    ]
                    * invariants[
                        "third_eigenvalue"
                    ],
                ]
            ),
            [
                "1",
                "c0",
                "c1",
                "c2",
                "c3",
                "c0^2",
                "c1^2",
                "c2^2",
                "c3^2",
                "c0*c3",
                "c1*c2",
            ],
        ),
        "register_joint_scalar_covariant": (
            np.column_stack(
                [
                    ones,
                    s4,
                    s6,
                    s8,
                    invariants[
                        "trace_square"
                    ],
                    invariants[
                        "trace_cube"
                    ],
                    invariants[
                        "determinant"
                    ],
                    invariants[
                        "spectral_gap_top"
                    ],
                    invariants[
                        "spectral_spread"
                    ],
                ]
            ),
            [
                "1",
                "S4",
                "S6",
                "S8",
                "tr(C^2)",
                "tr(C^3)",
                "det(C)",
                "gap_top(C)",
                "spread(C)",
            ],
        ),
    }

    for name, (
        design,
        feature_names,
    ) in models.items():
        row, prediction = (
            regression_summary(
                name,
                design,
                top_eigenvalues,
                feature_names,
            )
        )

        regression_rows.append(row)

        regression_predictions[
            name
        ] = prediction

    s4_spread = binned_target_spread(
        s4,
        top_eigenvalues,
    )

    covariant_max_eigenvalue_spread = (
        binned_target_spread(
            invariants[
                "maximum_eigenvalue"
            ],
            top_eigenvalues,
        )
    )

    covariant_trace_square_spread = (
        binned_target_spread(
            invariants[
                "trace_square"
            ],
            top_eigenvalues,
        )
    )

    spectrum_feature_matrix = (
        invariants["eigenvalues"]
    )

    elementary_feature_matrix = np.column_stack(
        [
            invariants[
                "trace_square"
            ],
            invariants[
                "trace_cube"
            ],
            invariants[
                "determinant"
            ],
        ]
    )

    spectrum_ambiguity = (
        nearest_neighbor_invariant_ambiguity(
            spectrum_feature_matrix,
            top_eigenvalues,
        )
    )

    elementary_ambiguity = (
        nearest_neighbor_invariant_ambiguity(
            elementary_feature_matrix,
            top_eigenvalues,
        )
    )

    scalar_moment_ambiguity = (
        nearest_neighbor_invariant_ambiguity(
            np.column_stack(
                [
                    s4,
                    s6,
                    s8,
                ]
            ),
            top_eigenvalues,
        )
    )

    saved_indices = np.linspace(
        0,
        sample_count - 1,
        SAVED_SAMPLE_COUNT,
        dtype=np.int64,
    )

    sample_rows = []

    for sample_id, source_index in enumerate(
        saved_indices
    ):
        direction = directions[
            source_index
        ]

        eigenvalues = invariants[
            "eigenvalues"
        ][source_index]

        sample_rows.append(
            {
                "sample_id": sample_id,
                "source_index": int(
                    source_index
                ),
                "f0": float(
                    direction[0]
                ),
                "f1": float(
                    direction[1]
                ),
                "f2": float(
                    direction[2]
                ),
                "f3": float(
                    direction[3]
                ),
                "top_pencil_eigenvalue": float(
                    top_eigenvalues[
                        source_index
                    ]
                ),
                "s4": float(
                    s4[source_index]
                ),
                "s6": float(
                    s6[source_index]
                ),
                "s8": float(
                    s8[source_index]
                ),
                "covariant_trace": float(
                    invariants[
                        "trace"
                    ][source_index]
                ),
                "covariant_trace_square": float(
                    invariants[
                        "trace_square"
                    ][source_index]
                ),
                "covariant_trace_cube": float(
                    invariants[
                        "trace_cube"
                    ][source_index]
                ),
                "covariant_determinant": float(
                    invariants[
                        "determinant"
                    ][source_index]
                ),
                "covariant_eigenvalue_0": float(
                    eigenvalues[0]
                ),
                "covariant_eigenvalue_1": float(
                    eigenvalues[1]
                ),
                "covariant_eigenvalue_2": float(
                    eigenvalues[2]
                ),
                "covariant_eigenvalue_3": float(
                    eigenvalues[3]
                ),
                "covariant_top_gap": float(
                    invariants[
                        "spectral_gap_top"
                    ][source_index]
                ),
                "covariant_spread": float(
                    invariants[
                        "spectral_spread"
                    ][source_index]
                ),
                "projective_axis_distance": float(
                    axis_distances[
                        source_index
                    ]
                ),
            }
        )


    regression_by_name = {
        row["model"]: row
        for row in regression_rows
    }

    s4_r_squared = regression_by_name[
        "s4_affine"
    ]["r_squared"]

    best_covariant_r_squared = max(
        row["r_squared"]
        for row in regression_rows
        if row["model"]
        != "s4_affine"
    )

    best_covariant_model = max(
        (
            row
            for row in regression_rows
            if row["model"]
            != "s4_affine"
        ),
        key=lambda row: row[
            "r_squared"
        ],
    )

    covariant_improves_over_s4 = (
        best_covariant_r_squared
        > s4_r_squared
        + 1e-4
    )

    checks = {
        "input_032_audit_pass": (
            scan_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "axis_line_count_is_10": (
            len(axis_lines) == 10
        ),
        "covariant_shape_is_N_by_4_by_4": (
            covariants.shape
            == (
                sample_count,
                4,
                4,
            )
        ),
        "covariant_trace_is_five_halves": (
            trace_residual
            < TRACE_TOLERANCE
        ),
        "axis_covariant_spectrum_is_uniform": (
            axis_spectrum_residual
            < AXIS_SPECTRUM_TOLERANCE
        ),
        "all_regression_models_completed": (
            len(regression_rows)
            == len(models)
        ),
        "covariant_model_improves_over_s4_affine": (
            covariant_improves_over_s4
        ),
        "nearest_neighbor_ambiguity_audits_completed": (
            spectrum_ambiguity[
                "sample_count"
            ]
            > 0
            and elementary_ambiguity[
                "sample_count"
            ]
            > 0
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    verdict = (
        "native_g60_cross_flux_register_covariant_retains_additional_norm_information"
        if audit_pass
        else "native_g60_cross_flux_register_covariant_audit_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_register_covariant_033"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "definition": {
            "covariant": (
                "C(f)=sum_i <f,q_i>^2 q_i q_i^T"
            ),
            "trace_law": (
                "tr C(f)=5/2 for unit f"
            ),
            "line_count": 10,
            "ambient_dimension": 4,
        },
        "axis_covariant": {
            "axis_count": 10,
            "mean_spectrum": (
                axis_mean_spectrum
            ),
            "mean_spectrum_rational": [
                rational_label(value)
                for value in (
                    axis_mean_spectrum
                )
            ],
            "maximum_axis_spectrum_residual": (
                axis_spectrum_residual
            ),
            "trace": float(
                np.sum(
                    axis_mean_spectrum
                )
            ),
            "trace_rational": (
                rational_label(
                    np.sum(
                        axis_mean_spectrum
                    )
                )
            ),
        },
        "random_scan": {
            "sample_count": (
                sample_count
            ),
            "trace_maximum_residual": (
                trace_residual
            ),
            "covariant_eigenvalue_minimum": (
                np.min(
                    invariants[
                        "eigenvalues"
                    ],
                    axis=0,
                )
            ),
            "covariant_eigenvalue_maximum": (
                np.max(
                    invariants[
                        "eigenvalues"
                    ],
                    axis=0,
                )
            ),
        },
        "regression_comparison": {
            "s4_affine_r_squared": (
                s4_r_squared
            ),
            "best_covariant_model": (
                best_covariant_model[
                    "model"
                ]
            ),
            "best_covariant_r_squared": (
                best_covariant_r_squared
            ),
            "r_squared_improvement": (
                best_covariant_r_squared
                - s4_r_squared
            ),
            "covariant_improves_over_s4": (
                covariant_improves_over_s4
            ),
            "models": regression_rows,
        },
        "one_dimensional_binned_spreads": {
            "s4": s4_spread,
            "covariant_maximum_eigenvalue": (
                covariant_max_eigenvalue_spread
            ),
            "covariant_trace_square": (
                covariant_trace_square_spread
            ),
        },
        "nearest_neighbor_ambiguity": {
            "scalar_moments_s4_s6_s8": (
                scalar_moment_ambiguity
            ),
            "covariant_elementary_invariants": (
                elementary_ambiguity
            ),
            "full_covariant_spectrum": (
                spectrum_ambiguity
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "matrix_covariant_constructed": (
                audit_pass
            ),
            "covariant_trace_law_verified": (
                audit_pass
            ),
            "axis_covariant_spectrum_uniform": (
                audit_pass
            ),
            "covariant_retains_more_norm_information_than_s4_affine": (
                covariant_improves_over_s4
            ),
            "covariant_spectrum_fully_determines_operator_norm": (
                spectrum_ambiguity[
                    "maximum_target_difference"
                ]
                < 1e-8
            ),
            "global_one_third_bound_proved": (
                False
            ),
        },
        "boundary": {
            "matrix_valued_register_scan_completed": (
                audit_pass
            ),
            "exact_axis_covariant_spectrum_derived": (
                False
            ),
            "operator_norm_exactly_reconstructed_from_covariant": (
                False
            ),
            "global_operator_norm_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "axis_csv": str(
                AXIS_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "sample_csv": str(
                SAMPLE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "regression_csv": str(
                REGRESSION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "covariant_npz": str(
                NPZ_OUT.relative_to(
                    ROOT
                )
            ),
        },
    }

    JSON_OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    with AXIS_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                axis_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            axis_rows
        )

    with SAMPLE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                sample_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            sample_rows
        )

    with REGRESSION_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                regression_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            regression_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        axis_covariants=(
            axis_covariants
        ),
        axis_mean_spectrum=(
            axis_mean_spectrum
        ),
        random_directions=(
            directions
        ),
        covariants=covariants,
        covariant_eigenvalues=(
            invariants[
                "eigenvalues"
            ]
        ),
        covariant_trace=(
            invariants["trace"]
        ),
        covariant_trace_square=(
            invariants[
                "trace_square"
            ]
        ),
        covariant_trace_cube=(
            invariants[
                "trace_cube"
            ]
        ),
        covariant_determinant=(
            invariants[
                "determinant"
            ]
        ),
        top_pencil_eigenvalues=(
            top_eigenvalues
        ),
        s4=s4,
        s6=s6,
        s8=s8,
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "covariant_trace_residual:",
        trace_residual,
    )
    print(
        "axis_covariant_mean_spectrum:",
        axis_mean_spectrum.tolist(),
    )
    print(
        "axis_covariant_rational_spectrum:",
        [
            rational_label(value)
            for value in (
                axis_mean_spectrum
            )
        ],
    )
    print(
        "axis_covariant_spectrum_residual:",
        axis_spectrum_residual,
    )
    print(
        "s4_affine_r_squared:",
        s4_r_squared,
    )
    print(
        "best_covariant_model:",
        best_covariant_model[
            "model"
        ],
    )
    print(
        "best_covariant_r_squared:",
        best_covariant_r_squared,
    )
    print(
        "r_squared_improvement:",
        best_covariant_r_squared
        - s4_r_squared,
    )
    print(
        "binned_spreads:",
        {
            "s4": s4_spread,
            "covariant_max_eigenvalue": (
                covariant_max_eigenvalue_spread
            ),
            "covariant_trace_square": (
                covariant_trace_square_spread
            ),
        },
    )
    print(
        "nearest_neighbor_ambiguity:",
        {
            "scalar_moments": (
                scalar_moment_ambiguity
            ),
            "elementary_covariant": (
                elementary_ambiguity
            ),
            "covariant_spectrum": (
                spectrum_ambiguity
            ),
        },
    )
    print("wrote:", JSON_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", SAMPLE_CSV_OUT)
    print("wrote:", REGRESSION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
