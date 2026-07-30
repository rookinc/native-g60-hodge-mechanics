from __future__ import annotations

import csv
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import qr


sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]

FACE_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_e5_gram_face_probe_050.py"
)

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_e5_gram_face_probe_050.json"
)

SOURCE_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_e5_gram_face_probe_050.npz"
)

SDP_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_higher_sdp_probe_048.npz"
)

ORIENTATION_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_covariant_orientation_035.npz"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_e5_rational_kernel_chart_051.json"
)

CANDIDATE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_kernel_candidates_051.csv"
)

CHART_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_kernel_chart_051.csv"
)

NULLSPACE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_face_basis_051.csv"
)

COMPARISON_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_kernel_comparisons_051.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_e5_rational_kernel_chart_051.npz"
)

KERNEL_DIMENSION = 10
FACE_DIMENSION = 46

DENOMINATOR_LIMITS = (
    12,
    24,
    60,
    120,
    360,
    1080,
    10_000,
    100_000,
    1_000_000,
)

CHART_CONDITION_LIMIT = 1e6
ANGLE_AUDIT_TOLERANCE_DEGREES = 0.01
PROJECTOR_AUDIT_TOLERANCE = 1e-4
ANNIHILATION_AUDIT_TOLERANCE = 1e-5


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(values)
        )
    )


def load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def rationalize_matrix(
    matrix: np.ndarray,
    denominator_limit: int,
) -> list[list[Fraction]]:
    return [
        [
            Fraction(
                float(value)
            ).limit_denominator(
                denominator_limit
            )
            for value in row
        ]
        for row in matrix
    ]


def fraction_matrix_float(
    matrix: list[list[Fraction]],
) -> np.ndarray:
    return np.array(
        [
            [
                float(value)
                for value in row
            ]
            for row in matrix
        ],
        dtype=np.float64,
    )


def orthogonal_projector(
    basis: np.ndarray,
) -> np.ndarray:
    q_basis, _ = np.linalg.qr(
        basis,
        mode="reduced",
    )

    return (
        q_basis
        @ q_basis.T
    )


def principal_angles_degrees(
    first_basis: np.ndarray,
    second_basis: np.ndarray,
) -> np.ndarray:
    first_q, _ = np.linalg.qr(
        first_basis,
        mode="reduced",
    )

    second_q, _ = np.linalg.qr(
        second_basis,
        mode="reduced",
    )

    singular_values = np.linalg.svd(
        first_q.T @ second_q,
        compute_uv=False,
    )

    singular_values = np.clip(
        singular_values,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(
            singular_values
        )
    )


def exact_identity(
    size: int,
) -> list[list[Fraction]]:
    return [
        [
            Fraction(
                1 if row == column else 0,
                1,
            )
            for column in range(size)
        ]
        for row in range(size)
    ]


def choose_pivot_rows(
    kernel_basis: np.ndarray,
) -> dict:
    _, _, pivot_indices = qr(
        kernel_basis.T,
        pivoting=True,
        mode="economic",
    )

    pivot_rows = np.array(
        pivot_indices[
            :KERNEL_DIMENSION
        ],
        dtype=np.int64,
    )

    pivot_block = kernel_basis[
        pivot_rows,
        :,
    ]

    condition_number = float(
        np.linalg.cond(
            pivot_block
        )
    )

    chart = (
        kernel_basis
        @ np.linalg.inv(
            pivot_block
        )
    )

    pivot_identity_residual = max_abs(
        chart[
            pivot_rows,
            :,
        ]
        - np.eye(
            KERNEL_DIMENSION
        )
    )

    nonpivot_rows = np.array(
        [
            index
            for index in range(
                kernel_basis.shape[0]
            )
            if index not in set(
                int(value)
                for value in pivot_rows
            )
        ],
        dtype=np.int64,
    )

    permutation = np.concatenate(
        [
            pivot_rows,
            nonpivot_rows,
        ]
    )

    inverse_permutation = np.empty_like(
        permutation
    )

    inverse_permutation[
        permutation
    ] = np.arange(
        len(permutation),
        dtype=np.int64,
    )

    permuted_chart = chart[
        permutation,
        :,
    ]

    return {
        "pivot_rows": pivot_rows,
        "nonpivot_rows": (
            nonpivot_rows
        ),
        "permutation": permutation,
        "inverse_permutation": (
            inverse_permutation
        ),
        "pivot_block": (
            pivot_block
        ),
        "condition_number": (
            condition_number
        ),
        "chart": chart,
        "permuted_chart": (
            permuted_chart
        ),
        "pivot_identity_residual": (
            pivot_identity_residual
        ),
    }


def exact_kernel_from_chart(
    rational_nonpivot: list[
        list[Fraction]
    ],
    permutation: np.ndarray,
) -> list[list[Fraction]]:
    identity = exact_identity(
        KERNEL_DIMENSION
    )

    permuted_kernel = (
        identity
        + rational_nonpivot
    )

    kernel = [
        [
            Fraction(0, 1)
            for _ in range(
                KERNEL_DIMENSION
            )
        ]
        for _ in range(
            KERNEL_DIMENSION
            + FACE_DIMENSION
        )
    ]

    for permuted_row, original_row in enumerate(
        permutation
    ):
        kernel[
            int(original_row)
        ] = list(
            permuted_kernel[
                permuted_row
            ]
        )

    return kernel


def exact_face_from_chart(
    rational_nonpivot: list[
        list[Fraction]
    ],
    permutation: np.ndarray,
) -> list[list[Fraction]]:
    identity = exact_identity(
        FACE_DIMENSION
    )

    permuted_face = [
        [
            -rational_nonpivot[
                column
            ][row]
            for column in range(
                FACE_DIMENSION
            )
        ]
        for row in range(
            KERNEL_DIMENSION
        )
    ] + identity

    face = [
        [
            Fraction(0, 1)
            for _ in range(
                FACE_DIMENSION
            )
        ]
        for _ in range(
            KERNEL_DIMENSION
            + FACE_DIMENSION
        )
    ]

    for permuted_row, original_row in enumerate(
        permutation
    ):
        face[
            int(original_row)
        ] = list(
            permuted_face[
                permuted_row
            ]
        )

    return face


def exact_transpose_product_zero(
    kernel: list[list[Fraction]],
    face: list[list[Fraction]],
) -> bool:
    for kernel_column in range(
        KERNEL_DIMENSION
    ):
        for face_column in range(
            FACE_DIMENSION
        ):
            value = sum(
                (
                    kernel[row][
                        kernel_column
                    ]
                    * face[row][
                        face_column
                    ]
                    for row in range(
                        KERNEL_DIMENSION
                        + FACE_DIMENSION
                    )
                ),
                Fraction(0, 1),
            )

            if value != 0:
                return False

    return True


def recover_kernel_solves(
    face_module,
    target: np.ndarray,
    buckets,
    monomial_count: int,
) -> list[dict]:
    results = []

    for configuration in (
        face_module.SOLVE_CONFIGURATIONS
    ):
        results.append(
            face_module.solve_full_gram(
                (
                    "chart_051_"
                    + configuration[
                        "name"
                    ]
                ),
                target,
                buckets,
                monomial_count,
                configuration,
            )
        )

    return results


def score_candidate(
    denominator_limit: int,
    rational_nonpivot: list[
        list[Fraction]
    ],
    chart_record: dict,
    comparison_bases: list[
        tuple[str, np.ndarray]
    ],
    gram_records: list[
        tuple[str, np.ndarray]
    ],
) -> dict:
    rational_kernel = (
        exact_kernel_from_chart(
            rational_nonpivot,
            chart_record[
                "permutation"
            ],
        )
    )

    rational_face = (
        exact_face_from_chart(
            rational_nonpivot,
            chart_record[
                "permutation"
            ],
        )
    )

    kernel_float = fraction_matrix_float(
        rational_kernel
    )

    face_float = fraction_matrix_float(
        rational_face
    )

    projector = orthogonal_projector(
        kernel_float
    )

    comparison_rows = []

    maximum_angle = 0.0
    maximum_projector_distance = 0.0

    for name, basis in comparison_bases:
        angles = principal_angles_degrees(
            kernel_float,
            basis,
        )

        source_projector = (
            orthogonal_projector(
                basis
            )
        )

        projector_distance = float(
            np.linalg.norm(
                projector
                - source_projector,
                ord=2,
            )
        )

        maximum_angle = max(
            maximum_angle,
            float(
                np.max(angles)
            ),
        )

        maximum_projector_distance = max(
            maximum_projector_distance,
            projector_distance,
        )

        comparison_rows.append(
            {
                "source": name,
                "minimum_angle_degrees": float(
                    np.min(angles)
                ),
                "maximum_angle_degrees": float(
                    np.max(angles)
                ),
                "mean_angle_degrees": float(
                    np.mean(angles)
                ),
                "projector_distance": (
                    projector_distance
                ),
            }
        )

    maximum_annihilation_residual = 0.0

    annihilation_rows = []

    for name, gram in gram_records:
        residual = max_abs(
            gram @ kernel_float
        )

        maximum_annihilation_residual = max(
            maximum_annihilation_residual,
            residual,
        )

        annihilation_rows.append(
            {
                "source": name,
                "annihilation_residual": (
                    residual
                ),
            }
        )

    exact_nullspace_pass = (
        exact_transpose_product_zero(
            rational_kernel,
            rational_face,
        )
    )

    pivot_rows = chart_record[
        "pivot_rows"
    ]

    pivot_identity_residual = max_abs(
        kernel_float[
            pivot_rows,
            :,
        ]
        - np.eye(
            KERNEL_DIMENSION
        )
    )

    chart_difference = max_abs(
        kernel_float
        - chart_record["chart"]
    )

    audit_candidate = (
        exact_nullspace_pass
        and maximum_angle
        < ANGLE_AUDIT_TOLERANCE_DEGREES
        and maximum_projector_distance
        < PROJECTOR_AUDIT_TOLERANCE
        and maximum_annihilation_residual
        < ANNIHILATION_AUDIT_TOLERANCE
    )

    return {
        "denominator_limit": (
            denominator_limit
        ),
        "rational_nonpivot": (
            rational_nonpivot
        ),
        "rational_kernel": (
            rational_kernel
        ),
        "rational_face": (
            rational_face
        ),
        "kernel_float": kernel_float,
        "face_float": face_float,
        "projector": projector,
        "maximum_angle_degrees": (
            maximum_angle
        ),
        "maximum_projector_distance": (
            maximum_projector_distance
        ),
        "maximum_annihilation_residual": (
            maximum_annihilation_residual
        ),
        "chart_difference": (
            chart_difference
        ),
        "pivot_identity_residual": (
            pivot_identity_residual
        ),
        "exact_nullspace_pass": (
            exact_nullspace_pass
        ),
        "audit_candidate": (
            audit_candidate
        ),
        "comparison_rows": (
            comparison_rows
        ),
        "annihilation_rows": (
            annihilation_rows
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANDIDATE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NULLSPACE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_receipt = json.loads(
        SOURCE_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    source_data = np.load(
        SOURCE_NPZ_PATH
    )

    sdp_data = np.load(
        SDP_NPZ_PATH
    )

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    face_module = load_module(
        "e5_face_probe_050",
        FACE_SCRIPT_PATH,
    )

    elementary = face_module.load_module(
        "gap_elementary_044",
        face_module.ELEMENTARY_SCRIPT_PATH,
    )

    axis_lines = np.array(
        orientation_data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    entries = (
        elementary.construct_gap_entries(
            slices
        )
    )

    target_polynomial = (
        elementary.elementary_polynomial(
            entries,
            5,
        )
    )

    target_exponents = (
        elementary.degree_exponents(
            10
        )
    )

    target = (
        elementary.polynomial_vector(
            target_polynomial,
            target_exponents,
        )
    )

    monomial_exponents = (
        face_module.homogeneous_exponents(
            5
        )
    )

    buckets = face_module.coefficient_buckets(
        monomial_exponents,
        target_exponents,
    )

    recovered_solves = recover_kernel_solves(
        face_module,
        target,
        buckets,
        len(monomial_exponents),
    )

    stabilized_kernel = np.array(
        source_data[
            "stabilized_kernel_basis"
        ],
        dtype=np.float64,
    )

    saved_048_gram = np.array(
        sdp_data["e5_gram"],
        dtype=np.float64,
    )

    saved_048_gram = 0.5 * (
        saved_048_gram
        + saved_048_gram.T
    )

    saved_eigenvalues, saved_eigenvectors = (
        np.linalg.eigh(
            saved_048_gram
        )
    )

    saved_048_kernel = (
        saved_eigenvectors[
            :,
            :KERNEL_DIMENSION,
        ]
    )

    comparison_bases = [
        (
            "stabilized_050",
            stabilized_kernel,
        ),
        (
            "saved_048",
            saved_048_kernel,
        ),
    ] + [
        (
            result["name"],
            result["kernel_basis"],
        )
        for result in recovered_solves
    ]

    gram_records = [
        (
            "saved_048",
            saved_048_gram,
        )
    ] + [
        (
            result["name"],
            result["gram"],
        )
        for result in recovered_solves
    ]

    chart_record = choose_pivot_rows(
        stabilized_kernel
    )

    if (
        chart_record[
            "condition_number"
        ]
        > CHART_CONDITION_LIMIT
    ):
        raise RuntimeError(
            "selected pivot chart is poorly conditioned"
        )

    floating_nonpivot = (
        chart_record[
            "permuted_chart"
        ][
            KERNEL_DIMENSION:,
            :,
        ]
    )

    candidates = []

    for denominator_limit in (
        DENOMINATOR_LIMITS
    ):
        rational_nonpivot = (
            rationalize_matrix(
                floating_nonpivot,
                denominator_limit,
            )
        )

        candidate = score_candidate(
            denominator_limit,
            rational_nonpivot,
            chart_record,
            comparison_bases,
            gram_records,
        )

        candidates.append(candidate)

        print(
            "chart_candidate:",
            "denominator_limit:",
            denominator_limit,
            "chart_difference:",
            candidate[
                "chart_difference"
            ],
            "maximum_angle_degrees:",
            candidate[
                "maximum_angle_degrees"
            ],
            "maximum_projector_distance:",
            candidate[
                "maximum_projector_distance"
            ],
            "maximum_annihilation_residual:",
            candidate[
                "maximum_annihilation_residual"
            ],
            "exact_nullspace:",
            candidate[
                "exact_nullspace_pass"
            ],
            "audit_candidate:",
            candidate[
                "audit_candidate"
            ],
            flush=True,
        )

    passing_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "audit_candidate"
        ]
    ]

    pool = (
        passing_candidates
        if passing_candidates
        else candidates
    )

    best = min(
        pool,
        key=lambda candidate: (
            candidate[
                "maximum_annihilation_residual"
            ],
            candidate[
                "maximum_projector_distance"
            ],
            candidate[
                "denominator_limit"
            ],
        ),
    )


    candidate_rows = []

    for candidate in candidates:
        candidate_rows.append(
            {
                "denominator_limit": (
                    candidate[
                        "denominator_limit"
                    ]
                ),
                "chart_difference": (
                    candidate[
                        "chart_difference"
                    ]
                ),
                "pivot_identity_residual": (
                    candidate[
                        "pivot_identity_residual"
                    ]
                ),
                "maximum_angle_degrees": (
                    candidate[
                        "maximum_angle_degrees"
                    ]
                ),
                "maximum_projector_distance": (
                    candidate[
                        "maximum_projector_distance"
                    ]
                ),
                "maximum_annihilation_residual": (
                    candidate[
                        "maximum_annihilation_residual"
                    ]
                ),
                "exact_nullspace_pass": (
                    candidate[
                        "exact_nullspace_pass"
                    ]
                ),
                "audit_candidate": (
                    candidate[
                        "audit_candidate"
                    ]
                ),
            }
        )

    chart_rows = []

    best_kernel = best[
        "rational_kernel"
    ]

    for row in range(
        len(best_kernel)
    ):
        for column in range(
            KERNEL_DIMENSION
        ):
            value = best_kernel[
                row
            ][column]

            chart_rows.append(
                {
                    "row": row,
                    "kernel_column": (
                        column
                    ),
                    "is_pivot_row": (
                        row
                        in set(
                            int(value)
                            for value in chart_record[
                                "pivot_rows"
                            ]
                        )
                    ),
                    "numerator": (
                        value.numerator
                    ),
                    "denominator": (
                        value.denominator
                    ),
                    "exact_value": str(
                        value
                    ),
                    "floating_value": float(
                        value
                    ),
                }
            )

    nullspace_rows = []

    best_face = best[
        "rational_face"
    ]

    for row in range(
        len(best_face)
    ):
        for column in range(
            FACE_DIMENSION
        ):
            value = best_face[
                row
            ][column]

            nullspace_rows.append(
                {
                    "row": row,
                    "face_column": (
                        column
                    ),
                    "numerator": (
                        value.numerator
                    ),
                    "denominator": (
                        value.denominator
                    ),
                    "exact_value": str(
                        value
                    ),
                    "floating_value": float(
                        value
                    ),
                }
            )

    comparison_rows = []

    for row in best[
        "comparison_rows"
    ]:
        comparison_rows.append(
            {
                "comparison_type": (
                    "subspace"
                ),
                "source": (
                    row["source"]
                ),
                "minimum_angle_degrees": (
                    row[
                        "minimum_angle_degrees"
                    ]
                ),
                "maximum_angle_degrees": (
                    row[
                        "maximum_angle_degrees"
                    ]
                ),
                "mean_angle_degrees": (
                    row[
                        "mean_angle_degrees"
                    ]
                ),
                "projector_distance": (
                    row[
                        "projector_distance"
                    ]
                ),
                "annihilation_residual": (
                    ""
                ),
            }
        )

    for row in best[
        "annihilation_rows"
    ]:
        comparison_rows.append(
            {
                "comparison_type": (
                    "gram_annihilation"
                ),
                "source": (
                    row["source"]
                ),
                "minimum_angle_degrees": (
                    ""
                ),
                "maximum_angle_degrees": (
                    ""
                ),
                "mean_angle_degrees": (
                    ""
                ),
                "projector_distance": (
                    ""
                ),
                "annihilation_residual": (
                    row[
                        "annihilation_residual"
                    ]
                ),
            }
        )

    checks = {
        "input_050_audit_pass": (
            source_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "stable_kernel_dimension_is_10": (
            stabilized_kernel.shape
            == (56, 10)
        ),
        "pivot_chart_condition_bounded": (
            chart_record[
                "condition_number"
            ]
            < CHART_CONDITION_LIMIT
        ),
        "floating_pivot_rows_normalize_to_identity": (
            chart_record[
                "pivot_identity_residual"
            ]
            < 1e-10
        ),
        "rational_kernel_has_expected_shape": (
            best["kernel_float"].shape
            == (56, 10)
        ),
        "rational_face_has_expected_shape": (
            best["face_float"].shape
            == (56, 46)
        ),
        "exact_kernel_face_orthogonality": (
            best[
                "exact_nullspace_pass"
            ]
        ),
        "rational_chart_tracks_all_numerical_kernels": (
            best[
                "maximum_angle_degrees"
            ]
            < ANGLE_AUDIT_TOLERANCE_DEGREES
        ),
        "rational_chart_nearly_annihilates_all_gram_solves": (
            best[
                "maximum_annihilation_residual"
            ]
            < ANNIHILATION_AUDIT_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    verdict = (
        "rational_e5_kernel_chart_candidate_recovered_with_exact_nullspace"
        if audit_pass
        else "rational_e5_kernel_chart_not_resolved"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_e5_rational_kernel_chart_051"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "kernel_dimension": (
            KERNEL_DIMENSION
        ),
        "face_dimension": (
            FACE_DIMENSION
        ),
        "pivot_rows": (
            chart_record[
                "pivot_rows"
            ]
        ),
        "nonpivot_rows": (
            chart_record[
                "nonpivot_rows"
            ]
        ),
        "chart_condition_number": (
            chart_record[
                "condition_number"
            ]
        ),
        "floating_pivot_identity_residual": (
            chart_record[
                "pivot_identity_residual"
            ]
        ),
        "best_candidate": {
            "denominator_limit": (
                best[
                    "denominator_limit"
                ]
            ),
            "chart_difference": (
                best[
                    "chart_difference"
                ]
            ),
            "maximum_angle_degrees": (
                best[
                    "maximum_angle_degrees"
                ]
            ),
            "maximum_projector_distance": (
                best[
                    "maximum_projector_distance"
                ]
            ),
            "maximum_annihilation_residual": (
                best[
                    "maximum_annihilation_residual"
                ]
            ),
            "exact_kernel_face_orthogonality": (
                best[
                    "exact_nullspace_pass"
                ]
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "stable_kernel_has_well_conditioned_coordinate_chart": (
                audit_pass
            ),
            "rational_kernel_chart_candidate_found": (
                audit_pass
            ),
            "exact_rational_nullspace_of_candidate_chart_constructed": (
                audit_pass
            ),
            "candidate_chart_proved_to_be_true_e5_gram_kernel": (
                False
            ),
            "exact_rational_e5_sos_certificate_found": (
                False
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "rational_subspace_candidate_only": (
                True
            ),
            "exact_kernel_identity_not_yet_derived_from_coefficient_map": (
                True
            ),
            "reduced_sdp_not_run_in_this_artifact": (
                True
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "candidate_csv": str(
                CANDIDATE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "chart_csv": str(
                CHART_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "nullspace_csv": str(
                NULLSPACE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "comparison_csv": str(
                COMPARISON_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "chart_npz": str(
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

    for path, rows in (
        (
            CANDIDATE_CSV_OUT,
            candidate_rows,
        ),
        (
            CHART_CSV_OUT,
            chart_rows,
        ),
        (
            NULLSPACE_CSV_OUT,
            nullspace_rows,
        ),
        (
            COMPARISON_CSV_OUT,
            comparison_rows,
        ),
    ):
        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[0]
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    np.savez_compressed(
        NPZ_OUT,
        pivot_rows=(
            chart_record[
                "pivot_rows"
            ]
        ),
        nonpivot_rows=(
            chart_record[
                "nonpivot_rows"
            ]
        ),
        permutation=(
            chart_record[
                "permutation"
            ]
        ),
        floating_chart=(
            chart_record["chart"]
        ),
        rational_kernel_float=(
            best["kernel_float"]
        ),
        rational_face_float=(
            best["face_float"]
        ),
        rational_kernel_projector=(
            best["projector"]
        ),
        target_coefficients=(
            target
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "pivot_rows:",
        chart_record[
            "pivot_rows"
        ].tolist(),
    )
    print(
        "chart_condition_number:",
        chart_record[
            "condition_number"
        ],
    )
    print(
        "best_denominator_limit:",
        best[
            "denominator_limit"
        ],
    )
    print(
        "best_chart_difference:",
        best[
            "chart_difference"
        ],
    )
    print(
        "best_maximum_angle_degrees:",
        best[
            "maximum_angle_degrees"
        ],
    )
    print(
        "best_maximum_projector_distance:",
        best[
            "maximum_projector_distance"
        ],
    )
    print(
        "best_maximum_annihilation_residual:",
        best[
            "maximum_annihilation_residual"
        ],
    )
    print(
        "exact_kernel_face_orthogonality:",
        best[
            "exact_nullspace_pass"
        ],
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CANDIDATE_CSV_OUT)
    print("wrote:", CHART_CSV_OUT)
    print("wrote:", NULLSPACE_CSV_OUT)
    print("wrote:", COMPARISON_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
