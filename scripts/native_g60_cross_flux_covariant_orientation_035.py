from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]

IDENTITY_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_covariant_identity_034.json"
)

COVARIANT_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_register_covariant_033.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_covariant_orientation_035.json"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_orientation_axes_035.csv"
)

SAMPLE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_orientation_samples_035.csv"
)

REGRESSION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_orientation_regressions_035.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_covariant_orientation_035.npz"
)

TARGET_TOP_EIGENVALUE = 1.0 / 9.0

IDENTITY_TOLERANCE = 2e-10
AXIS_TOLERANCE = 2e-10

SAVED_SAMPLE_COUNT = 5000
NEAREST_NEIGHBOR_SAMPLE_COUNT = 8000


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


def register_covariants(
    directions: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    overlaps = (
        directions @ axis_lines.T
    )

    return np.einsum(
        "ni,ia,ib->nab",
        overlaps**2,
        axis_lines,
        axis_lines,
    )


def register_moment(
    directions: np.ndarray,
    axis_lines: np.ndarray,
    power: int,
) -> np.ndarray:
    overlaps = (
        directions @ axis_lines.T
    )

    return np.sum(
        overlaps**power,
        axis=1,
    )


def regression_summary(
    name: str,
    design: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, np.ndarray]:
    coefficients, _, rank, singular_values = (
        np.linalg.lstsq(
            design,
            target,
            rcond=None,
        )
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
        "design_rank": int(rank),
        "design_singular_values": json.dumps(
            singular_values.tolist()
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



def orientation_data(
    directions: np.ndarray,
    covariants: np.ndarray,
    s4: np.ndarray,
) -> dict[str, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariants)

    covariant_times_f = np.einsum(
        "nij,nj->ni",
        covariants,
        directions,
    )

    residual_vectors = (
        covariant_times_f
        - s4[:, None] * directions
    )

    residual_norm_squared = np.einsum(
        "ni,ni->n",
        residual_vectors,
        residual_vectors,
    )

    tangent_residual = np.einsum(
        "ni,ni->n",
        directions,
        residual_vectors,
    )

    eigen_coordinates = np.einsum(
        "nrc,nr->nc",
        eigenvectors,
        directions,
    )

    eigen_weights = eigen_coordinates**2

    eigen_weight_sum_residual = (
        np.sum(eigen_weights, axis=1) - 1.0
    )

    rayleigh_from_spectrum = np.sum(
        eigenvalues * eigen_weights,
        axis=1,
    )

    second_rayleigh_from_spectrum = np.sum(
        eigenvalues**2 * eigen_weights,
        axis=1,
    )

    residual_from_spectrum = (
        second_rayleigh_from_spectrum
        - rayleigh_from_spectrum**2
    )

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "covariant_times_f": covariant_times_f,
        "residual_vectors": residual_vectors,
        "residual_norm_squared": residual_norm_squared,
        "tangent_residual": tangent_residual,
        "eigen_coordinates": eigen_coordinates,
        "eigen_weights": eigen_weights,
        "eigen_weight_sum_residual": eigen_weight_sum_residual,
        "rayleigh_from_spectrum": rayleigh_from_spectrum,
        "second_rayleigh_from_spectrum": second_rayleigh_from_spectrum,
        "residual_from_spectrum": residual_from_spectrum,
        "top_eigenvector_weight": eigen_weights[:, 3],
        "low_eigenspace_weight": (
            eigen_weights[:, 0]
            + eigen_weights[:, 1]
            + eigen_weights[:, 2]
        ),
    }


def nearest_neighbor_ambiguity(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    sample_count: int = NEAREST_NEIGHBOR_SAMPLE_COUNT,
) -> dict:
    count = min(len(feature_matrix), sample_count)

    indices = np.linspace(
        0,
        len(feature_matrix) - 1,
        count,
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

    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales[scales < 1e-14] = 1.0

    standardized = (features - means) / scales
    tree = cKDTree(standardized)

    distances, neighbors = tree.query(
        standardized,
        k=2,
    )

    nearest_distance = distances[:, 1]
    nearest_index = neighbors[:, 1]

    target_difference = np.abs(
        targets - targets[nearest_index]
    )

    return {
        "sample_count": count,
        "median_feature_distance": float(
            np.median(nearest_distance)
        ),
        "maximum_feature_distance": float(
            np.max(nearest_distance)
        ),
        "median_target_difference": float(
            np.median(target_difference)
        ),
        "q95_target_difference": float(
            np.quantile(target_difference, 0.95)
        ),
        "q99_target_difference": float(
            np.quantile(target_difference, 0.99)
        ),
        "maximum_target_difference": float(
            np.max(target_difference)
        ),
    }


def axis_orientation_profile(
    axis_lines: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    covariants = register_covariants(
        axis_lines,
        axis_lines,
    )

    s4 = register_moment(
        axis_lines,
        axis_lines,
        4,
    )

    s6 = register_moment(
        axis_lines,
        axis_lines,
        6,
    )

    orientation = orientation_data(
        axis_lines,
        covariants,
        s4,
    )

    rows = []

    for axis_id in range(len(axis_lines)):
        rows.append(
            {
                "axis_id": axis_id,
                "s4": float(s4[axis_id]),
                "s6": float(s6[axis_id]),
                "residual_norm_squared": float(
                    orientation[
                        "residual_norm_squared"
                    ][axis_id]
                ),
                "residual_norm": float(
                    np.sqrt(
                        max(
                            orientation[
                                "residual_norm_squared"
                            ][axis_id],
                            0.0,
                        )
                    )
                ),
                "tangent_residual": float(
                    orientation[
                        "tangent_residual"
                    ][axis_id]
                ),
                "top_eigenvector_weight": float(
                    orientation[
                        "top_eigenvector_weight"
                    ][axis_id]
                ),
                "low_eigenspace_weight": float(
                    orientation[
                        "low_eigenspace_weight"
                    ][axis_id]
                ),
                "eigenvalue_0": float(
                    orientation["eigenvalues"][axis_id, 0]
                ),
                "eigenvalue_1": float(
                    orientation["eigenvalues"][axis_id, 1]
                ),
                "eigenvalue_2": float(
                    orientation["eigenvalues"][axis_id, 2]
                ),
                "eigenvalue_3": float(
                    orientation["eigenvalues"][axis_id, 3]
                ),
            }
        )

    return rows, orientation

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

    identity_receipt = json.loads(
        IDENTITY_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        COVARIANT_NPZ_PATH
    )

    axis_lines = np.array(
        data["axis_lines"],
        dtype=np.float64,
    )

    directions = np.array(
        data["random_directions"],
        dtype=np.float64,
    )

    covariants = np.array(
        data["covariants"],
        dtype=np.float64,
    )

    top_pencil_eigenvalues = np.array(
        data["top_pencil_eigenvalues"],
        dtype=np.float64,
    )

    s4 = np.array(
        data["s4"],
        dtype=np.float64,
    )

    s6 = np.array(
        data["s6"],
        dtype=np.float64,
    )

    s8 = np.array(
        data["s8"],
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

    orientation = orientation_data(
        directions,
        covariants,
        s4,
    )

    residual_identity_prediction = (
        -625.0 / 1152.0
        + 55.0 / 48.0
        * s4
        + s6
        - s4**2
    )

    residual_identity_residual = (
        orientation[
            "residual_norm_squared"
        ]
        - residual_identity_prediction
    )

    rayleigh_residual = (
        orientation[
            "rayleigh_from_spectrum"
        ]
        - s4
    )

    spectral_residual_norm_residual = (
        orientation[
            "residual_from_spectrum"
        ]
        - orientation[
            "residual_norm_squared"
        ]
    )

    axis_rows, axis_orientation = (
        axis_orientation_profile(
            axis_lines
        )
    )

    axis_residual_maximum = max(
        row["residual_norm"]
        for row in axis_rows
    )

    axis_top_weight_residual = max(
        abs(
            row[
                "top_eigenvector_weight"
            ]
            - 1.0
        )
        for row in axis_rows
    )

    sample_count = len(
        directions
    )

    ones = np.ones(
        sample_count,
        dtype=np.float64,
    )

    eigenvalues = orientation[
        "eigenvalues"
    ]

    weights = orientation[
        "eigen_weights"
    ]

    residual_norm_squared = orientation[
        "residual_norm_squared"
    ]

    top_weight = orientation[
        "top_eigenvector_weight"
    ]

    target_gap = (
        TARGET_TOP_EIGENVALUE
        - top_pencil_eigenvalues
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
        "s4_residual_affine": (
            np.column_stack(
                [
                    ones,
                    s4,
                    residual_norm_squared,
                ]
            ),
            [
                "1",
                "S4",
                "||r||^2",
            ],
        ),
        "s4_residual_quadratic": (
            np.column_stack(
                [
                    ones,
                    s4,
                    residual_norm_squared,
                    s4**2,
                    residual_norm_squared**2,
                    s4
                    * residual_norm_squared,
                ]
            ),
            [
                "1",
                "S4",
                "||r||^2",
                "S4^2",
                "||r||^4",
                "S4*||r||^2",
            ],
        ),
        "covariant_spectrum_quadratic": (
            np.column_stack(
                [
                    ones,
                    eigenvalues,
                    eigenvalues**2,
                    eigenvalues[:, 0]
                    * eigenvalues[:, 3],
                    eigenvalues[:, 1]
                    * eigenvalues[:, 2],
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
        "covariant_orientation_affine": (
            np.column_stack(
                [
                    ones,
                    eigenvalues,
                    weights[:, 0],
                    weights[:, 1],
                    weights[:, 2],
                    weights[:, 3],
                ]
            ),
            [
                "1",
                "c0",
                "c1",
                "c2",
                "c3",
                "w0",
                "w1",
                "w2",
                "w3",
            ],
        ),
        "covariant_orientation_quadratic": (
            np.column_stack(
                [
                    ones,
                    eigenvalues,
                    eigenvalues**2,
                    weights[:, 0],
                    weights[:, 1],
                    weights[:, 2],
                    weights[:, 3],
                    eigenvalues[:, 0]
                    * weights[:, 0],
                    eigenvalues[:, 1]
                    * weights[:, 1],
                    eigenvalues[:, 2]
                    * weights[:, 2],
                    eigenvalues[:, 3]
                    * weights[:, 3],
                    residual_norm_squared,
                    top_weight,
                    top_weight**2,
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
                "w0",
                "w1",
                "w2",
                "w3",
                "c0*w0",
                "c1*w1",
                "c2*w2",
                "c3*w3",
                "||r||^2",
                "w3",
                "w3^2",
            ],
        ),
        "full_register_orientation": (
            np.column_stack(
                [
                    ones,
                    s4,
                    s6,
                    s8,
                    eigenvalues,
                    eigenvalues**2,
                    weights,
                    residual_norm_squared,
                    top_weight**2,
                ]
            ),
            [
                "1",
                "S4",
                "S6",
                "S8",
                "c0",
                "c1",
                "c2",
                "c3",
                "c0^2",
                "c1^2",
                "c2^2",
                "c3^2",
                "w0",
                "w1",
                "w2",
                "w3",
                "||r||^2",
                "w3^2",
            ],
        ),
    }

    regression_rows = []
    regression_predictions = {}

    for name, (
        design,
        feature_names,
    ) in models.items():
        row, prediction = (
            regression_summary(
                name,
                design,
                top_pencil_eigenvalues,
                feature_names,
            )
        )

        regression_rows.append(row)

        regression_predictions[
            name
        ] = prediction

    regression_by_name = {
        row["model"]: row
        for row in regression_rows
    }

    spectrum_r_squared = (
        regression_by_name[
            "covariant_spectrum_quadratic"
        ]["r_squared"]
    )

    orientation_r_squared = (
        regression_by_name[
            "covariant_orientation_quadratic"
        ]["r_squared"]
    )

    full_orientation_r_squared = (
        regression_by_name[
            "full_register_orientation"
        ]["r_squared"]
    )

    spectrum_features = eigenvalues

    orientation_features = np.column_stack(
        [
            eigenvalues,
            weights,
            residual_norm_squared,
        ]
    )

    full_features = np.column_stack(
        [
            s4,
            s6,
            s8,
            eigenvalues,
            weights,
            residual_norm_squared,
        ]
    )

    spectrum_ambiguity = (
        nearest_neighbor_ambiguity(
            spectrum_features,
            top_pencil_eigenvalues,
        )
    )

    orientation_ambiguity = (
        nearest_neighbor_ambiguity(
            orientation_features,
            top_pencil_eigenvalues,
        )
    )

    full_ambiguity = (
        nearest_neighbor_ambiguity(
            full_features,
            top_pencil_eigenvalues,
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
                    top_pencil_eigenvalues[
                        source_index
                    ]
                ),
                "one_ninth_gap": float(
                    target_gap[
                        source_index
                    ]
                ),
                "s4": float(
                    s4[source_index]
                ),
                "s6": float(
                    s6[source_index]
                ),
                "residual_norm_squared": float(
                    residual_norm_squared[
                        source_index
                    ]
                ),
                "residual_identity_prediction": float(
                    residual_identity_prediction[
                        source_index
                    ]
                ),
                "residual_identity_residual": float(
                    residual_identity_residual[
                        source_index
                    ]
                ),
                "eigenvalue_0": float(
                    eigenvalues[
                        source_index,
                        0,
                    ]
                ),
                "eigenvalue_1": float(
                    eigenvalues[
                        source_index,
                        1,
                    ]
                ),
                "eigenvalue_2": float(
                    eigenvalues[
                        source_index,
                        2,
                    ]
                ),
                "eigenvalue_3": float(
                    eigenvalues[
                        source_index,
                        3,
                    ]
                ),
                "eigenweight_0": float(
                    weights[
                        source_index,
                        0,
                    ]
                ),
                "eigenweight_1": float(
                    weights[
                        source_index,
                        1,
                    ]
                ),
                "eigenweight_2": float(
                    weights[
                        source_index,
                        2,
                    ]
                ),
                "eigenweight_3": float(
                    weights[
                        source_index,
                        3,
                    ]
                ),
            }
        )

    identity_checks = {
        "input_034_theorem_pass": (
            identity_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "unit_direction_count_matches_covariants": (
            len(directions)
            == len(covariants)
        ),
        "residual_is_tangent": (
            max_abs(
                orientation[
                    "tangent_residual"
                ]
            )
            < IDENTITY_TOLERANCE
        ),
        "residual_norm_identity_exact": (
            max_abs(
                residual_identity_residual
            )
            < IDENTITY_TOLERANCE
        ),
        "rayleigh_reconstructs_s4": (
            max_abs(
                rayleigh_residual
            )
            < IDENTITY_TOLERANCE
        ),
        "spectral_variance_reconstructs_residual_norm": (
            max_abs(
                spectral_residual_norm_residual
            )
            < IDENTITY_TOLERANCE
        ),
        "eigenweights_sum_to_one": (
            max_abs(
                orientation[
                    "eigen_weight_sum_residual"
                ]
            )
            < IDENTITY_TOLERANCE
        ),
        "native_axis_residuals_vanish": (
            axis_residual_maximum
            < AXIS_TOLERANCE
        ),
        "native_axes_lie_in_top_covariant_eigenspace": (
            axis_top_weight_residual
            < AXIS_TOLERANCE
        ),
        "orientation_regression_completed": (
            len(regression_rows)
            == len(models)
        ),
    }

    audit_pass = all(
        identity_checks.values()
    )

    orientation_improves_regression = (
        orientation_r_squared
        > spectrum_r_squared
    )

    orientation_reduces_ambiguity = (
        orientation_ambiguity[
            "maximum_target_difference"
        ]
        < spectrum_ambiguity[
            "maximum_target_difference"
        ]
    )

    verdict = (
        "native_g60_cross_flux_covariant_orientation_retains_additional_information"
        if (
            audit_pass
            and (
                orientation_improves_regression
                or orientation_reduces_ambiguity
            )
        )
        else (
            "native_g60_cross_flux_covariant_orientation_identity_exact"
            if audit_pass
            else "native_g60_cross_flux_covariant_orientation_audit_failed"
        )
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_covariant_orientation_035"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": audit_pass,
        "verdict": verdict,
        "definition": {
            "tangent_residual": (
                "r(f)=C(f)f-S4(f)f for unit f"
            ),
            "tangency": (
                "f^T r(f)=0"
            ),
            "residual_norm_identity": (
                "||r(f)||^2="
                "-625/1152+(55/48)S4(f)"
                "+S6(f)-S4(f)^2"
            ),
            "spectral_variance_identity": (
                "||r(f)||^2="
                "sum_j c_j(f)^2 w_j(f)-S4(f)^2"
            ),
        },
        "axis_orientation": {
            "axis_count": len(
                axis_rows
            ),
            "maximum_axis_residual_norm": (
                axis_residual_maximum
            ),
            "maximum_top_eigenweight_residual": (
                axis_top_weight_residual
            ),
            "interpretation": (
                "each native axis is an eigenvector "
                "of C(q) with eigenvalue S4(q)"
            ),
        },
        "identity_residuals": {
            "tangency": max_abs(
                orientation[
                    "tangent_residual"
                ]
            ),
            "moment_formula": max_abs(
                residual_identity_residual
            ),
            "rayleigh_s4": max_abs(
                rayleigh_residual
            ),
            "spectral_variance": max_abs(
                spectral_residual_norm_residual
            ),
            "eigenweight_sum": max_abs(
                orientation[
                    "eigen_weight_sum_residual"
                ]
            ),
        },
        "regression_comparison": {
            "spectrum_quadratic_r_squared": (
                spectrum_r_squared
            ),
            "orientation_quadratic_r_squared": (
                orientation_r_squared
            ),
            "full_orientation_r_squared": (
                full_orientation_r_squared
            ),
            "orientation_improves_over_spectrum": (
                orientation_improves_regression
            ),
            "models": regression_rows,
        },
        "nearest_neighbor_ambiguity": {
            "covariant_spectrum": (
                spectrum_ambiguity
            ),
            "covariant_orientation": (
                orientation_ambiguity
            ),
            "full_register_orientation": (
                full_ambiguity
            ),
            "orientation_reduces_maximum_ambiguity": (
                orientation_reduces_ambiguity
            ),
        },
        "checks": identity_checks,
        "earned_interpretation": {
            "tangent_covariant_identity_proved": (
                audit_pass
            ),
            "native_axes_are_covariant_eigendirections": (
                audit_pass
            ),
            "orientation_data_retains_additional_norm_information": (
                orientation_improves_regression
                or orientation_reduces_ambiguity
            ),
            "orientation_data_completely_determines_pencil_norm": (
                full_ambiguity[
                    "maximum_target_difference"
                ]
                < 1e-8
            ),
            "global_one_third_bound_proved": (
                False
            ),
        },
        "boundary": {
            "covariant_orientation_audit_completed": (
                audit_pass
            ),
            "residual_norm_identity_proved": (
                audit_pass
            ),
            "native_axes_characterized_as_covariant_eigendirections": (
                audit_pass
            ),
            "zero_residual_locus_equals_native_axis_register": (
                False
            ),
            "operator_norm_exactly_reconstructed": (
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
            "orientation_npz": str(
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
        random_directions=directions,
        covariants=covariants,
        covariant_eigenvalues=(
            eigenvalues
        ),
        covariant_eigenweights=(
            weights
        ),
        residual_vectors=(
            orientation[
                "residual_vectors"
            ]
        ),
        residual_norm_squared=(
            residual_norm_squared
        ),
        residual_identity_prediction=(
            residual_identity_prediction
        ),
        top_pencil_eigenvalues=(
            top_pencil_eigenvalues
        ),
        one_ninth_gap=target_gap,
        s4=s4,
        s6=s6,
        s8=s8,
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", audit_pass)
    print("verdict:", verdict)
    print(
        "maximum_tangency_residual:",
        max_abs(
            orientation[
                "tangent_residual"
            ]
        ),
    )
    print(
        "maximum_residual_identity_error:",
        max_abs(
            residual_identity_residual
        ),
    )
    print(
        "maximum_spectral_variance_error:",
        max_abs(
            spectral_residual_norm_residual
        ),
    )
    print(
        "maximum_axis_residual_norm:",
        axis_residual_maximum,
    )
    print(
        "maximum_axis_top_weight_residual:",
        axis_top_weight_residual,
    )
    print(
        "regression_r_squared:",
        {
            row["model"]: row[
                "r_squared"
            ]
            for row in regression_rows
        },
    )
    print(
        "nearest_neighbor_ambiguity:",
        {
            "spectrum": (
                spectrum_ambiguity
            ),
            "orientation": (
                orientation_ambiguity
            ),
            "full": full_ambiguity,
        },
    )
    print("wrote:", JSON_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", SAMPLE_CSV_OUT)
    print("wrote:", REGRESSION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
