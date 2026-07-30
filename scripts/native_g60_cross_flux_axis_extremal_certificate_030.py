from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

IDENTIFICATION_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_maximizer_axis_identification_029.json"
)

IDENTIFICATION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_maximizer_axis_identification_029.npz"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_axis_extremal_certificate_030.json"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_axis_spectral_certificate_030.csv"
)

LOCAL_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_axis_local_extremal_probes_030.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_axis_extremal_certificate_030.npz"
)

HIGH_EIGENVALUE = 1.0 / 9.0
LOW_EIGENVALUE = 1.0 / 144.0

HIGH_SINGULAR_VALUE = 1.0 / 3.0
LOW_SINGULAR_VALUE = 1.0 / 12.0

SPECTRAL_TOLERANCE = 2e-9
POLYNOMIAL_TOLERANCE = 2e-9
PROJECTOR_TOLERANCE = 2e-9

RANDOM_SEED = 46030
TANGENT_DIRECTION_COUNT = 512

GEODESIC_ANGLES = (
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
)

LOCAL_DROP_TOLERANCE = 1e-12


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


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0.0:
        raise RuntimeError(
            "cannot normalize zero vector"
        )

    return vector / norm


def pencil_matrix(
    slices: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "r,rab->ab",
        direction,
        slices,
    )


def pencil_singular_values(
    slices: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return np.linalg.svd(
        pencil_matrix(
            slices,
            direction,
        ),
        compute_uv=False,
    )


def operator_norm(
    slices: np.ndarray,
    direction: np.ndarray,
) -> float:
    return float(
        pencil_singular_values(
            slices,
            direction,
        )[0]
    )


def tangent_basis(
    axis: np.ndarray,
) -> np.ndarray:
    axis = normalized(axis)

    projector = (
        np.eye(4)
        - np.outer(axis, axis)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        projector
    )

    basis = eigenvectors[
        :,
        eigenvalues > 0.5
    ]

    if basis.shape != (4, 3):
        raise RuntimeError(
            f"unexpected tangent basis shape: {basis.shape}"
        )

    return basis


def geodesic_point(
    axis: np.ndarray,
    tangent: np.ndarray,
    angle: float,
) -> np.ndarray:
    return (
        math.cos(angle) * axis
        + math.sin(angle) * tangent
    )



def certify_axis_spectrum(
    slices: np.ndarray,
    axis_id: int,
    axis: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    axis = normalized(axis)

    matrix = pencil_matrix(
        slices,
        axis,
    )

    gram = matrix.T @ matrix

    eigenvalues = np.linalg.eigvalsh(
        gram
    )

    singular_values = np.sqrt(
        np.maximum(
            eigenvalues,
            0.0,
        )
    )

    expected_eigenvalues = np.array(
        [
            LOW_EIGENVALUE,
            LOW_EIGENVALUE,
            LOW_EIGENVALUE,
            LOW_EIGENVALUE,
            HIGH_EIGENVALUE,
            HIGH_EIGENVALUE,
        ],
        dtype=np.float64,
    )

    expected_singular_values = np.array(
        [
            LOW_SINGULAR_VALUE,
            LOW_SINGULAR_VALUE,
            LOW_SINGULAR_VALUE,
            LOW_SINGULAR_VALUE,
            HIGH_SINGULAR_VALUE,
            HIGH_SINGULAR_VALUE,
        ],
        dtype=np.float64,
    )

    identity = np.eye(
        6,
        dtype=np.float64,
    )

    polynomial_residual = (
        (
            gram
            - HIGH_EIGENVALUE * identity
        )
        @ (
            gram
            - LOW_EIGENVALUE * identity
        )
    )

    high_projector = (
        gram
        - LOW_EIGENVALUE * identity
    ) / (
        HIGH_EIGENVALUE
        - LOW_EIGENVALUE
    )

    low_projector = (
        identity
        - high_projector
    )

    high_projector_eigenvalues = (
        np.linalg.eigvalsh(
            high_projector
        )
    )

    high_rank = int(
        np.count_nonzero(
            high_projector_eigenvalues
            > 0.5
        )
    )

    low_rank = 6 - high_rank

    row = {
        "axis_id": axis_id,
        "operator_norm": float(
            singular_values[-1]
        ),
        "minimum_singular_value": float(
            singular_values[0]
        ),
        "singular_values": (
            singular_values.tolist()
        ),
        "maximum_eigenvalue_residual": max_abs(
            eigenvalues
            - expected_eigenvalues
        ),
        "maximum_singular_value_residual": max_abs(
            singular_values
            - expected_singular_values
        ),
        "polynomial_identity_max_abs": max_abs(
            polynomial_residual
        ),
        "high_projector_idempotence_max_abs": max_abs(
            high_projector
            @ high_projector
            - high_projector
        ),
        "low_projector_idempotence_max_abs": max_abs(
            low_projector
            @ low_projector
            - low_projector
        ),
        "projector_orthogonality_max_abs": max_abs(
            high_projector
            @ low_projector
        ),
        "projector_completeness_max_abs": max_abs(
            high_projector
            + low_projector
            - identity
        ),
        "high_projector_trace_residual": abs(
            float(
                np.trace(
                    high_projector
                )
            )
            - 2.0
        ),
        "low_projector_trace_residual": abs(
            float(
                np.trace(
                    low_projector
                )
            )
            - 4.0
        ),
        "high_projector_rank": high_rank,
        "low_projector_rank": low_rank,
    }

    row["axis_certificate_pass"] = (
        row[
            "maximum_eigenvalue_residual"
        ]
        < SPECTRAL_TOLERANCE
        and row[
            "maximum_singular_value_residual"
        ]
        < SPECTRAL_TOLERANCE
        and row[
            "polynomial_identity_max_abs"
        ]
        < POLYNOMIAL_TOLERANCE
        and row[
            "high_projector_idempotence_max_abs"
        ]
        < PROJECTOR_TOLERANCE
        and row[
            "low_projector_idempotence_max_abs"
        ]
        < PROJECTOR_TOLERANCE
        and row[
            "projector_orthogonality_max_abs"
        ]
        < PROJECTOR_TOLERANCE
        and row[
            "projector_completeness_max_abs"
        ]
        < PROJECTOR_TOLERANCE
        and row[
            "high_projector_trace_residual"
        ]
        < PROJECTOR_TOLERANCE
        and row[
            "low_projector_trace_residual"
        ]
        < PROJECTOR_TOLERANCE
        and high_rank == 2
        and low_rank == 4
    )

    return (
        row,
        high_projector,
        low_projector,
    )


def probe_axis_neighborhood(
    slices: np.ndarray,
    axis_id: int,
    axis: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    axis = normalized(axis)
    basis = tangent_basis(axis)

    rows = []

    maximum_off_axis_sigma = -float("inf")
    minimum_positive_drop = float("inf")
    minimum_drop_over_angle_squared = float("inf")
    nonpositive_drop_count = 0

    for direction_id in range(
        TANGENT_DIRECTION_COUNT
    ):
        tangent_coordinates = normalized(
            rng.normal(size=3)
        )

        tangent = normalized(
            basis @ tangent_coordinates
        )

        tangent_dot_axis = abs(
            float(
                np.dot(
                    tangent,
                    axis,
                )
            )
        )

        for angle in GEODESIC_ANGLES:
            point = geodesic_point(
                axis,
                tangent,
                angle,
            )

            sigma = operator_norm(
                slices,
                point,
            )

            drop = (
                HIGH_SINGULAR_VALUE
                - sigma
            )

            drop_over_angle_squared = (
                drop / angle**2
            )

            maximum_off_axis_sigma = max(
                maximum_off_axis_sigma,
                sigma,
            )

            if drop > 0.0:
                minimum_positive_drop = min(
                    minimum_positive_drop,
                    drop,
                )
            else:
                nonpositive_drop_count += 1

            minimum_drop_over_angle_squared = min(
                minimum_drop_over_angle_squared,
                drop_over_angle_squared,
            )

            rows.append(
                {
                    "axis_id": axis_id,
                    "direction_id": direction_id,
                    "angle": angle,
                    "sigma": sigma,
                    "drop_from_one_third": drop,
                    "drop_over_angle_squared": (
                        drop_over_angle_squared
                    ),
                    "tangent_dot_axis_abs": (
                        tangent_dot_axis
                    ),
                    "local_drop_positive": (
                        drop
                        > LOCAL_DROP_TOLERANCE
                    ),
                }
            )

    summary = {
        "axis_id": axis_id,
        "probe_count": len(rows),
        "maximum_off_axis_sigma": (
            maximum_off_axis_sigma
        ),
        "minimum_positive_drop": (
            minimum_positive_drop
        ),
        "minimum_drop_over_angle_squared": (
            minimum_drop_over_angle_squared
        ),
        "nonpositive_drop_count": (
            nonpositive_drop_count
        ),
        "all_sampled_off_axis_points_drop": (
            nonpositive_drop_count == 0
        ),
    }

    return rows, summary

def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AXIS_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCAL_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    identification_receipt = json.loads(
        IDENTIFICATION_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    identification_data = np.load(
        IDENTIFICATION_NPZ_PATH,
        allow_pickle=True,
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    axis_lines = np.array(
        identification_data[
            "native_line_matrix"
        ],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            "unexpected axis line shape: "
            f"{axis_lines.shape}"
        )

    if slices.shape != (4, 6, 6):
        raise RuntimeError(
            "unexpected slice shape: "
            f"{slices.shape}"
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    axis_rows = []
    local_rows = []
    local_summaries = []

    high_projectors = []
    low_projectors = []

    for axis_id, axis in enumerate(
        axis_lines
    ):
        (
            axis_row,
            high_projector,
            low_projector,
        ) = certify_axis_spectrum(
            slices,
            axis_id,
            axis,
        )

        axis_rows.append(
            axis_row
        )

        high_projectors.append(
            high_projector
        )

        low_projectors.append(
            low_projector
        )

        (
            current_local_rows,
            local_summary,
        ) = probe_axis_neighborhood(
            slices,
            axis_id,
            axis,
            rng,
        )

        local_rows.extend(
            current_local_rows
        )

        local_summaries.append(
            local_summary
        )

        print(
            "axis:",
            axis_id + 1,
            "/10",
            "certificate:",
            axis_row[
                "axis_certificate_pass"
            ],
            "max_off_axis_sigma:",
            local_summary[
                "maximum_off_axis_sigma"
            ],
            "minimum_drop:",
            local_summary[
                "minimum_positive_drop"
            ],
            flush=True,
        )

    all_axis_certificates_pass = all(
        row["axis_certificate_pass"]
        for row in axis_rows
    )

    all_local_probes_drop = all(
        summary[
            "all_sampled_off_axis_points_drop"
        ]
        for summary in local_summaries
    )

    maximum_axis_spectral_residual = max(
        row[
            "maximum_singular_value_residual"
        ]
        for row in axis_rows
    )

    maximum_axis_polynomial_residual = max(
        row[
            "polynomial_identity_max_abs"
        ]
        for row in axis_rows
    )

    maximum_sampled_off_axis_sigma = max(
        summary[
            "maximum_off_axis_sigma"
        ]
        for summary in local_summaries
    )

    minimum_sampled_drop = min(
        summary[
            "minimum_positive_drop"
        ]
        for summary in local_summaries
    )

    minimum_sampled_drop_over_angle_squared = min(
        summary[
            "minimum_drop_over_angle_squared"
        ]
        for summary in local_summaries
    )

    total_nonpositive_drop_count = sum(
        summary[
            "nonpositive_drop_count"
        ]
        for summary in local_summaries
    )

    checks = {
        "input_029_theorem_pass": (
            identification_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "axis_line_count_is_10": (
            len(axis_lines) == 10
        ),
        "slice_shape_is_4_by_6_by_6": (
            slices.shape == (4, 6, 6)
        ),
        "all_axis_spectral_certificates_pass": (
            all_axis_certificates_pass
        ),
        "every_axis_has_high_multiplicity_two": all(
            row[
                "high_projector_rank"
            ]
            == 2
            for row in axis_rows
        ),
        "every_axis_has_low_multiplicity_four": all(
            row[
                "low_projector_rank"
            ]
            == 4
            for row in axis_rows
        ),
        "all_sampled_neighborhood_points_drop_below_one_third": (
            all_local_probes_drop
        ),
        "all_local_probe_rows_completed": (
            len(local_rows)
            == (
                10
                * TANGENT_DIRECTION_COUNT
                * len(GEODESIC_ANGLES)
            )
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = (
        all_axis_certificates_pass
        and checks[
            "input_029_theorem_pass"
        ]
    )

    verdict = (
        "native_g60_cross_flux_ten_axes_have_exact_extremal_spectrum"
        if theorem_pass
        else "native_g60_cross_flux_axis_extremal_certificate_failed"
    )

    theorem_statement = (
        "For each of the ten native unoriented four-flux axis lines q, "
        "the pencil Gram matrix B_q=A(q)^T A(q) satisfies "
        "(B_q-I/9)(B_q-I/144)=0. Its eigenvalue 1/9 has "
        "multiplicity two and its eigenvalue 1/144 has multiplicity "
        "four. Therefore A(q) has singular values 1/3 twice and "
        "1/12 four times."
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_axis_extremal_certificate_030"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "theorem": {
            "statement": theorem_statement,
            "axis_line_count": 10,
            "axis_polynomial_identity": (
                "(B_q - I6/9)(B_q - I6/144) = 0"
            ),
            "high_eigenvalue": "1/9",
            "high_multiplicity": 2,
            "low_eigenvalue": "1/144",
            "low_multiplicity": 4,
            "high_singular_value": "1/3",
            "low_singular_value": "1/12",
        },
        "finite_axis_certificate": {
            "all_axes_pass": (
                all_axis_certificates_pass
            ),
            "maximum_spectral_residual": (
                maximum_axis_spectral_residual
            ),
            "maximum_polynomial_residual": (
                maximum_axis_polynomial_residual
            ),
            "axis_rows": axis_rows,
        },
        "local_neighborhood_probe": {
            "status": (
                "sampled strict-local-max evidence"
            ),
            "tangent_direction_count_per_axis": (
                TANGENT_DIRECTION_COUNT
            ),
            "geodesic_angles": list(
                GEODESIC_ANGLES
            ),
            "total_probe_count": len(
                local_rows
            ),
            "all_sampled_points_drop": (
                all_local_probes_drop
            ),
            "total_nonpositive_drop_count": (
                total_nonpositive_drop_count
            ),
            "maximum_sampled_off_axis_sigma": (
                maximum_sampled_off_axis_sigma
            ),
            "minimum_sampled_drop": (
                minimum_sampled_drop
            ),
            "minimum_sampled_drop_over_angle_squared": (
                minimum_sampled_drop_over_angle_squared
            ),
            "axis_summaries": (
                local_summaries
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "native_axes_have_exact_one_third_extremal_spectrum": (
                theorem_pass
            ),
            "native_axes_are_sampled_strict_local_maxima": (
                all_local_probes_drop
            ),
            "native_axes_are_proved_strict_local_maxima": (
                False
            ),
            "global_one_third_upper_bound_proved": (
                False
            ),
            "complete_equality_locus_proved": (
                False
            ),
        },
        "boundary": {
            "finite_axis_spectral_certificate_completed": (
                theorem_pass
            ),
            "local_neighborhood_sampling_completed": (
                audit_pass
            ),
            "local_maximality_analytically_proved": (
                False
            ),
            "global_operator_norm_bound_proved": (
                False
            ),
            "all_global_maximizers_classified": (
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
            "local_probe_csv": str(
                LOCAL_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "certificate_npz": str(
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

    with LOCAL_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                local_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            local_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        slices=slices,
        high_projectors=np.array(
            high_projectors,
            dtype=np.float64,
        ),
        low_projectors=np.array(
            low_projectors,
            dtype=np.float64,
        ),
        expected_singular_values=np.array(
            [
                LOW_SINGULAR_VALUE,
                LOW_SINGULAR_VALUE,
                LOW_SINGULAR_VALUE,
                LOW_SINGULAR_VALUE,
                HIGH_SINGULAR_VALUE,
                HIGH_SINGULAR_VALUE,
            ],
            dtype=np.float64,
        ),
        geodesic_angles=np.array(
            GEODESIC_ANGLES,
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "all_axis_spectral_certificates_pass:",
        all_axis_certificates_pass,
    )
    print(
        "maximum_axis_spectral_residual:",
        maximum_axis_spectral_residual,
    )
    print(
        "maximum_axis_polynomial_residual:",
        maximum_axis_polynomial_residual,
    )
    print(
        "all_sampled_neighborhood_points_drop:",
        all_local_probes_drop,
    )
    print(
        "maximum_sampled_off_axis_sigma:",
        maximum_sampled_off_axis_sigma,
    )
    print(
        "minimum_sampled_drop:",
        minimum_sampled_drop,
    )
    print(
        "minimum_drop_over_angle_squared:",
        minimum_sampled_drop_over_angle_squared,
    )
    print(
        "nonpositive_drop_count:",
        total_nonpositive_drop_count,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", LOCAL_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
