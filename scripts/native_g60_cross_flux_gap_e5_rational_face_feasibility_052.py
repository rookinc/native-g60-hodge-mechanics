from __future__ import annotations

import csv
import importlib.util
import json
import time
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import lstsq


ROOT = Path(__file__).resolve().parents[1]

ELEMENTARY_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_elementary_invariants_044.py"
)

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_e5_rational_kernel_chart_051.json"
)

FACE_CSV_PATH = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_face_basis_051.csv"
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
    / "native_g60_cross_flux_gap_e5_rational_face_feasibility_052.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_face_feasibility_summary_052.csv"
)

RESIDUAL_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_face_residuals_052.csv"
)

SPECTRUM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_face_spectrum_052.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_rational_face_gram_entries_052.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_e5_rational_face_feasibility_052.npz"
)

KERNEL_DIMENSION = 10
FACE_DIMENSION = 46
FULL_DIMENSION = 56

LINEAR_FEASIBILITY_TOLERANCE = 5e-10
SDP_COEFFICIENT_TOLERANCE = 5e-8
POSITIVE_MARGIN_TOLERANCE = 1e-9

SVD_RANK_TOLERANCE = 1e-11

RUN_SDP_RESIDUAL_LIMIT = 1e-7

SCS_OPTIONS = {
    "eps": 1e-9,
    "max_iters": 1_000_000,
    "scale": 10.0,
    "normalize": True,
    "acceleration_lookback": 20,
    "verbose": False,
}


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


def load_exact_face_basis() -> list[list[Fraction]]:
    matrix = [
        [
            Fraction(0, 1)
            for _ in range(
                FACE_DIMENSION
            )
        ]
        for _ in range(
            FULL_DIMENSION
        )
    ]

    with FACE_CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        row_count = 0

        for record in reader:
            row = int(record["row"])
            column = int(
                record["face_column"]
            )

            numerator = int(
                record["numerator"]
            )

            denominator = int(
                record["denominator"]
            )

            matrix[row][column] = Fraction(
                numerator,
                denominator,
            )

            row_count += 1

    expected_count = (
        FULL_DIMENSION
        * FACE_DIMENSION
    )

    if row_count != expected_count:
        raise RuntimeError(
            "unexpected rational face CSV row count: "
            f"{row_count} != {expected_count}"
        )

    return matrix


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


def homogeneous_exponents(
    degree: int,
) -> list[tuple[int, int, int, int]]:
    values = []

    for a in range(degree + 1):
        for b in range(
            degree + 1 - a
        ):
            for c in range(
                degree + 1 - a - b
            ):
                d = degree - a - b - c

                values.append(
                    (a, b, c, d)
                )

    values.sort(reverse=True)

    return values


def exponent_add(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return tuple(
        first[index] + second[index]
        for index in range(4)
    )


def coefficient_buckets(
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
) -> list[list[tuple[int, int, float]]]:
    target_index = {
        exponent: index
        for index, exponent in enumerate(
            target_exponents
        )
    }

    buckets = [
        []
        for _ in target_exponents
    ]

    for row in range(
        len(monomial_exponents)
    ):
        for column in range(
            row + 1
        ):
            exponent = exponent_add(
                monomial_exponents[row],
                monomial_exponents[column],
            )

            weight = (
                1.0
                if row == column
                else 2.0
            )

            buckets[
                target_index[exponent]
            ].append(
                (
                    row,
                    column,
                    weight,
                )
            )

    return buckets


def symmetric_pairs(
    size: int,
) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(size)
        for column in range(row + 1)
    ]


def full_coefficient_matrix(
    bucket: list[
        tuple[int, int, float]
    ],
    size: int,
) -> np.ndarray:
    matrix = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    for row, column, weight in bucket:
        if row == column:
            matrix[row, column] += weight
        else:
            half_weight = (
                weight / 2.0
            )

            matrix[row, column] += (
                half_weight
            )

            matrix[column, row] += (
                half_weight
            )

    return matrix


def symmetric_matrix_vector(
    matrix: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    values = []

    for row, column in pairs:
        if row == column:
            values.append(
                matrix[row, column]
            )
        else:
            values.append(
                2.0
                * matrix[row, column]
            )

    return np.array(
        values,
        dtype=np.float64,
    )


def vector_symmetric_matrix(
    vector: np.ndarray,
    pairs: list[tuple[int, int]],
    size: int,
) -> np.ndarray:
    matrix = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    for value, (
        row,
        column,
    ) in zip(vector, pairs):
        matrix[row, column] = value
        matrix[column, row] = value

    return matrix


def build_reduced_operator(
    face_basis: np.ndarray,
    buckets: list[
        list[tuple[int, int, float]]
    ],
) -> dict:
    pairs = symmetric_pairs(
        FACE_DIMENSION
    )

    design = np.zeros(
        (
            len(buckets),
            len(pairs),
        ),
        dtype=np.float64,
    )

    reduced_matrices = []

    print(
        "operator_precompute_start:",
        "constraint_count:",
        len(buckets),
        "variable_count:",
        len(pairs),
        flush=True,
    )

    for coefficient_index, bucket in enumerate(
        buckets
    ):
        full_matrix = (
            full_coefficient_matrix(
                bucket,
                FULL_DIMENSION,
            )
        )

        reduced_matrix = (
            face_basis.T
            @ full_matrix
            @ face_basis
        )

        reduced_matrix = 0.5 * (
            reduced_matrix
            + reduced_matrix.T
        )

        reduced_matrices.append(
            reduced_matrix
        )

        design[
            coefficient_index,
            :,
        ] = symmetric_matrix_vector(
            reduced_matrix,
            pairs,
        )

        if (
            coefficient_index == 0
            or (
                coefficient_index + 1
            ) % 50 == 0
            or (
                coefficient_index + 1
                == len(buckets)
            )
        ):
            print(
                "operator_precompute_progress:",
                f"{coefficient_index + 1}/{len(buckets)}",
                flush=True,
            )

    return {
        "pairs": pairs,
        "design": design,
        "reduced_matrices": (
            reduced_matrices
        ),
    }


def evaluate_coefficients(
    gram: np.ndarray,
    reduced_matrices: list[
        np.ndarray
    ],
) -> np.ndarray:
    return np.array(
        [
            float(
                np.sum(
                    matrix * gram
                )
            )
            for matrix in (
                reduced_matrices
            )
        ],
        dtype=np.float64,
    )


def linear_feasibility_probe(
    design: np.ndarray,
    target: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict:
    print(
        "linear_feasibility_start:",
        "design_shape:",
        design.shape,
        flush=True,
    )

    solution, residuals, rank, singular_values = lstsq(
        design,
        target,
        cond=SVD_RANK_TOLERANCE,
        lapack_driver="gelsd",
    )

    predicted = design @ solution

    residual = (
        predicted - target
    )

    gram = vector_symmetric_matrix(
        solution,
        pairs,
        FACE_DIMENSION,
    )

    gram = 0.5 * (
        gram + gram.T
    )

    eigenvalues = np.linalg.eigvalsh(
        gram
    )

    condition_number = (
        float(
            singular_values[0]
            / singular_values[
                int(rank) - 1
            ]
        )
        if rank > 0
        else float("inf")
    )

    result = {
        "solution": solution,
        "gram": gram,
        "eigenvalues": eigenvalues,
        "rank": int(rank),
        "nullity": int(
            design.shape[1] - rank
        ),
        "singular_values": (
            singular_values
        ),
        "condition_number": (
            condition_number
        ),
        "maximum_residual": (
            max_abs(residual)
        ),
        "residual_l2": float(
            np.linalg.norm(
                residual
            )
        ),
        "minimum_gram_eigenvalue": float(
            eigenvalues[0]
        ),
        "maximum_gram_eigenvalue": float(
            eigenvalues[-1]
        ),
    }

    print(
        "linear_feasibility_result:",
        "rank:",
        result["rank"],
        "nullity:",
        result["nullity"],
        "condition_number:",
        result[
            "condition_number"
        ],
        "maximum_residual:",
        result[
            "maximum_residual"
        ],
        "minimum_gram_eigenvalue:",
        result[
            "minimum_gram_eigenvalue"
        ],
        flush=True,
    )

    return result


def reduced_sdp_probe(
    reduced_matrices: list[np.ndarray],
    target: np.ndarray,
) -> dict:
    gram = cp.Variable(
        (
            FACE_DIMENSION,
            FACE_DIMENSION,
        ),
        symmetric=True,
        name="Y_e5_rational_face",
    )

    margin = cp.Variable(
        name="margin_e5_rational_face"
    )

    constraints = [
        gram
        - margin
        * np.eye(
            FACE_DIMENSION
        )
        >> 0
    ]

    for target_index, matrix in enumerate(
        reduced_matrices
    ):
        constraints.append(
            cp.sum(
                cp.multiply(
                    matrix,
                    gram,
                )
            )
            == target[target_index]
        )

    problem = cp.Problem(
        cp.Maximize(margin),
        constraints,
    )

    print(
        "rational_face_sdp_start:",
        "constraint_count:",
        len(reduced_matrices),
        flush=True,
    )

    started = time.perf_counter()

    objective = problem.solve(
        solver="SCS",
        **SCS_OPTIONS,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if gram.value is None:
        return {
            "attempted": True,
            "status": problem.status,
            "objective": (
                float(objective)
                if objective is not None
                else float("nan")
            ),
            "margin": float("nan"),
            "elapsed_seconds": elapsed,
            "gram": np.full(
                (
                    FACE_DIMENSION,
                    FACE_DIMENSION,
                ),
                np.nan,
            ),
            "eigenvalues": np.full(
                FACE_DIMENSION,
                np.nan,
            ),
            "maximum_coefficient_residual": (
                float("inf")
            ),
        }

    gram_value = np.array(
        gram.value,
        dtype=np.float64,
    )

    gram_value = 0.5 * (
        gram_value
        + gram_value.T
    )

    eigenvalues = np.linalg.eigvalsh(
        gram_value
    )

    coefficients = evaluate_coefficients(
        gram_value,
        reduced_matrices,
    )

    residual = (
        coefficients - target
    )

    result = {
        "attempted": True,
        "status": problem.status,
        "objective": (
            float(objective)
            if objective is not None
            else float("nan")
        ),
        "margin": (
            float(margin.value)
            if margin.value is not None
            else float("nan")
        ),
        "elapsed_seconds": elapsed,
        "gram": gram_value,
        "eigenvalues": eigenvalues,
        "maximum_coefficient_residual": (
            max_abs(residual)
        ),
        "coefficient_residual_l2": float(
            np.linalg.norm(
                residual
            )
        ),
        "minimum_gram_eigenvalue": float(
            eigenvalues[0]
        ),
        "maximum_gram_eigenvalue": float(
            eigenvalues[-1]
        ),
        "numerical_rank": int(
            np.count_nonzero(
                eigenvalues > 1e-9
            )
        ),
    }

    print(
        "rational_face_sdp_result:",
        "status:",
        result["status"],
        "margin:",
        result["margin"],
        "coefficient_residual:",
        result[
            "maximum_coefficient_residual"
        ],
        "minimum_gram_eigenvalue:",
        result[
            "minimum_gram_eigenvalue"
        ],
        "rank:",
        result["numerical_rank"],
        "elapsed_seconds:",
        elapsed,
        flush=True,
    )

    return result


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESIDUAL_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPECTRUM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAM_CSV_OUT.parent.mkdir(
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

    elementary = load_module(
        "gap_elementary_044",
        ELEMENTARY_SCRIPT_PATH,
    )

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    axis_lines = np.array(
        orientation_data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    exact_face = load_exact_face_basis()

    face_basis = fraction_matrix_float(
        exact_face
    )

    face_rank = int(
        np.linalg.matrix_rank(
            face_basis,
            tol=1e-12,
        )
    )

    face_condition_number = float(
        np.linalg.cond(
            face_basis
        )
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
        homogeneous_exponents(
            5
        )
    )

    buckets = coefficient_buckets(
        monomial_exponents,
        target_exponents,
    )

    operator = build_reduced_operator(
        face_basis,
        buckets,
    )

    linear = linear_feasibility_probe(
        operator["design"],
        target,
        operator["pairs"],
    )

    run_sdp = (
        linear["maximum_residual"]
        < RUN_SDP_RESIDUAL_LIMIT
    )

    if run_sdp:
        sdp = reduced_sdp_probe(
            operator[
                "reduced_matrices"
            ],
            target,
        )
    else:
        print(
            "rational_face_sdp_skipped:",
            "linear residual exceeds limit:",
            linear[
                "maximum_residual"
            ],
            flush=True,
        )

        sdp = {
            "attempted": False,
            "status": "skipped",
            "objective": float("nan"),
            "margin": float("nan"),
            "elapsed_seconds": 0.0,
            "gram": np.full(
                (
                    FACE_DIMENSION,
                    FACE_DIMENSION,
                ),
                np.nan,
            ),
            "eigenvalues": np.full(
                FACE_DIMENSION,
                np.nan,
            ),
            "maximum_coefficient_residual": (
                float("inf")
            ),
            "coefficient_residual_l2": (
                float("inf")
            ),
            "minimum_gram_eigenvalue": (
                float("nan")
            ),
            "maximum_gram_eigenvalue": (
                float("nan")
            ),
            "numerical_rank": 0,
        }

    linear_consistent = (
        linear["maximum_residual"]
        < LINEAR_FEASIBILITY_TOLERANCE
    )

    sdp_feasible = (
        sdp["attempted"]
        and sdp["status"]
        in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        )
        and sdp[
            "maximum_coefficient_residual"
        ]
        < SDP_COEFFICIENT_TOLERANCE
    )

    sdp_interior = (
        sdp_feasible
        and sdp["margin"]
        > POSITIVE_MARGIN_TOLERANCE
        and sdp[
            "minimum_gram_eigenvalue"
        ]
        > POSITIVE_MARGIN_TOLERANCE
    )

    checks = {
        "input_051_audit_pass": (
            source_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "rational_face_shape_is_56_by_46": (
            face_basis.shape
            == (
                FULL_DIMENSION,
                FACE_DIMENSION,
            )
        ),
        "rational_face_has_full_column_rank": (
            face_rank
            == FACE_DIMENSION
        ),
        "coefficient_count_is_286": (
            len(target) == 286
        ),
        "reduced_symmetric_variable_count_is_1081": (
            operator["design"].shape[1]
            == (
                FACE_DIMENSION
                * (
                    FACE_DIMENSION + 1
                )
                // 2
            )
        ),
        "linear_feasibility_probe_completed": (
            np.isfinite(
                linear[
                    "maximum_residual"
                ]
            )
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    if (
        audit_pass
        and linear_consistent
        and sdp_interior
    ):
        verdict = (
            "rational_e5_face_numerically_feasible_and_interior"
        )
    elif (
        audit_pass
        and linear_consistent
        and sdp_feasible
    ):
        verdict = (
            "rational_e5_face_numerically_feasible_boundary_status_open"
        )
    elif audit_pass and linear_consistent:
        verdict = (
            "rational_e5_face_linear_slice_consistent_sdp_open"
        )
    elif audit_pass:
        verdict = (
            "rational_e5_face_candidate_not_exactly_aligned_with_coefficient_slice"
        )
    else:
        verdict = (
            "rational_e5_face_feasibility_probe_failed"
        )

    summary_rows = [
        {
            "face_rank": face_rank,
            "face_condition_number": (
                face_condition_number
            ),
            "design_rows": (
                operator[
                    "design"
                ].shape[0]
            ),
            "design_columns": (
                operator[
                    "design"
                ].shape[1]
            ),
            "design_rank": (
                linear["rank"]
            ),
            "design_nullity": (
                linear["nullity"]
            ),
            "design_condition_number": (
                linear[
                    "condition_number"
                ]
            ),
            "linear_maximum_residual": (
                linear[
                    "maximum_residual"
                ]
            ),
            "linear_residual_l2": (
                linear[
                    "residual_l2"
                ]
            ),
            "linear_slice_consistent": (
                linear_consistent
            ),
            "sdp_attempted": (
                sdp["attempted"]
            ),
            "sdp_status": (
                sdp["status"]
            ),
            "sdp_margin": (
                sdp["margin"]
            ),
            "sdp_maximum_coefficient_residual": (
                sdp[
                    "maximum_coefficient_residual"
                ]
            ),
            "sdp_minimum_gram_eigenvalue": (
                sdp[
                    "minimum_gram_eigenvalue"
                ]
            ),
            "sdp_numerical_rank": (
                sdp[
                    "numerical_rank"
                ]
            ),
            "sdp_feasible": (
                sdp_feasible
            ),
            "sdp_interior": (
                sdp_interior
            ),
        }
    ]

    linear_prediction = (
        operator["design"]
        @ linear["solution"]
    )

    linear_residual = (
        linear_prediction - target
    )

    residual_rows = []

    for coefficient_id, (
        exponent,
        target_value,
        predicted_value,
        residual_value,
    ) in enumerate(
        zip(
            target_exponents,
            target,
            linear_prediction,
            linear_residual,
        )
    ):
        residual_rows.append(
            {
                "coefficient_id": (
                    coefficient_id
                ),
                "f0_power": exponent[0],
                "f1_power": exponent[1],
                "f2_power": exponent[2],
                "f3_power": exponent[3],
                "target_coefficient": (
                    target_value
                ),
                "linear_prediction": (
                    predicted_value
                ),
                "linear_residual": (
                    residual_value
                ),
            }
        )

    spectrum_rows = []

    for index, value in enumerate(
        linear["eigenvalues"]
    ):
        spectrum_rows.append(
            {
                "source": (
                    "linear_least_squares"
                ),
                "eigenvalue_index": index,
                "eigenvalue": value,
            }
        )

    if sdp["attempted"]:
        for index, value in enumerate(
            sdp["eigenvalues"]
        ):
            spectrum_rows.append(
                {
                    "source": (
                        "reduced_sdp"
                    ),
                    "eigenvalue_index": (
                        index
                    ),
                    "eigenvalue": value,
                }
            )

    gram_rows = []

    gram_to_save = (
        sdp["gram"]
        if sdp["attempted"]
        and np.all(
            np.isfinite(
                sdp["gram"]
            )
        )
        else linear["gram"]
    )

    for row in range(
        FACE_DIMENSION
    ):
        for column in range(
            row + 1
        ):
            gram_rows.append(
                {
                    "row": row,
                    "column": column,
                    "gram_entry": float(
                        gram_to_save[
                            row,
                            column,
                        ]
                    ),
                }
            )


    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_e5_rational_face_feasibility_052"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "rational_face": {
            "shape": list(
                face_basis.shape
            ),
            "rank": face_rank,
            "condition_number": (
                face_condition_number
            ),
        },
        "linear_feasibility": {
            "design_shape": list(
                operator[
                    "design"
                ].shape
            ),
            "rank": (
                linear["rank"]
            ),
            "nullity": (
                linear["nullity"]
            ),
            "condition_number": (
                linear[
                    "condition_number"
                ]
            ),
            "maximum_residual": (
                linear[
                    "maximum_residual"
                ]
            ),
            "residual_l2": (
                linear[
                    "residual_l2"
                ]
            ),
            "slice_consistent": (
                linear_consistent
            ),
        },
        "reduced_sdp": {
            key: value
            for key, value in sdp.items()
            if key not in (
                "gram",
                "eigenvalues",
            )
        },
        "sdp_feasible": sdp_feasible,
        "sdp_interior": sdp_interior,
        "checks": checks,
        "earned_interpretation": {
            "rational_face_candidate_is_numerically_consistent_with_e5_slice": (
                linear_consistent
            ),
            "rational_face_candidate_supports_feasible_psd_gram": (
                sdp_feasible
            ),
            "e5_is_interior_on_rational_candidate_face": (
                sdp_interior
            ),
            "exact_rational_face_feasibility_proved": (
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
            "floating_linear_and_sdp_feasibility_only": (
                True
            ),
            "exact_rational_linear_consistency_not_proved": (
                True
            ),
            "exact_rational_ldl_not_attempted": (
                True
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "summary_csv": str(
                SUMMARY_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "residual_csv": str(
                RESIDUAL_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "spectrum_csv": str(
                SPECTRUM_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "gram_csv": str(
                GRAM_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "feasibility_npz": str(
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
            SUMMARY_CSV_OUT,
            summary_rows,
        ),
        (
            RESIDUAL_CSV_OUT,
            residual_rows,
        ),
        (
            SPECTRUM_CSV_OUT,
            spectrum_rows,
        ),
        (
            GRAM_CSV_OUT,
            gram_rows,
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
        rational_face_basis=(
            face_basis
        ),
        reduced_design=(
            operator["design"]
        ),
        target_coefficients=target,
        linear_solution=(
            linear["solution"]
        ),
        linear_gram=(
            linear["gram"]
        ),
        linear_eigenvalues=(
            linear["eigenvalues"]
        ),
        sdp_gram=(
            sdp["gram"]
        ),
        sdp_eigenvalues=(
            sdp["eigenvalues"]
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "face_rank:",
        face_rank,
    )
    print(
        "face_condition_number:",
        face_condition_number,
    )
    print(
        "design_shape:",
        operator[
            "design"
        ].shape,
    )
    print(
        "design_rank:",
        linear["rank"],
    )
    print(
        "design_nullity:",
        linear["nullity"],
    )
    print(
        "linear_maximum_residual:",
        linear[
            "maximum_residual"
        ],
    )
    print(
        "linear_slice_consistent:",
        linear_consistent,
    )
    print(
        "sdp_attempted:",
        sdp["attempted"],
    )
    print(
        "sdp_status:",
        sdp["status"],
    )
    print(
        "sdp_margin:",
        sdp["margin"],
    )
    print(
        "sdp_maximum_coefficient_residual:",
        sdp[
            "maximum_coefficient_residual"
        ],
    )
    print(
        "sdp_minimum_gram_eigenvalue:",
        sdp[
            "minimum_gram_eigenvalue"
        ],
    )
    print(
        "sdp_feasible:",
        sdp_feasible,
    )
    print(
        "sdp_interior:",
        sdp_interior,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)
    print("wrote:", RESIDUAL_CSV_OUT)
    print("wrote:", SPECTRUM_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
