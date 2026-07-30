from __future__ import annotations

import csv
import importlib.util
import json
import time
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ELEMENTARY_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_elementary_invariants_044.py"
)

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_higher_sdp_probe_048.json"
)

SOURCE_NPZ_PATH = (
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
    / "native_g60_cross_flux_gap_e5_gram_face_probe_050.json"
)

SOLVE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_face_solves_050.csv"
)

ANGLE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_kernel_angles_050.csv"
)

PROJECTOR_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_kernel_projector_050.csv"
)

RATIONAL_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_projector_rationalization_050.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_e5_gram_face_probe_050.npz"
)

KERNEL_DIMENSION = 10
FACE_DIMENSION = 46

SOLVE_CONFIGURATIONS = (
    {
        "name": "scs_eps_1e8",
        "eps": 1e-8,
        "max_iters": 250_000,
        "scale": 1.0,
    },
    {
        "name": "scs_eps_3e9",
        "eps": 3e-9,
        "max_iters": 500_000,
        "scale": 1.0,
    },
    {
        "name": "scs_eps_1e9",
        "eps": 1e-9,
        "max_iters": 750_000,
        "scale": 10.0,
    },
    {
        "name": "scs_eps_3e10",
        "eps": 3e-10,
        "max_iters": 1_000_000,
        "scale": 10.0,
    },
)

PROJECTOR_DENOMINATOR_LIMITS = (
    12,
    24,
    60,
    120,
    360,
    1080,
    10_000,
    100_000,
)

COEFFICIENT_TOLERANCE = 5e-8
KERNEL_EIGENVALUE_TOLERANCE = 5e-9
PROJECTOR_STABILITY_TOLERANCE = 5e-5
PRINCIPAL_ANGLE_TOLERANCE_DEGREES = 0.1
REDUCED_MARGIN_TOLERANCE = 1e-9


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

    return float(np.max(np.abs(values)))


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


def exponent_label(
    exponent: tuple[int, int, int, int],
) -> str:
    factors = []

    for index, power in enumerate(exponent):
        if power == 0:
            continue

        if power == 1:
            factors.append(
                f"f{index}"
            )
        else:
            factors.append(
                f"f{index}^{power}"
            )

    return "*".join(factors) or "1"


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


def gram_coefficients(
    gram: np.ndarray,
    buckets: list[
        list[tuple[int, int, float]]
    ],
) -> np.ndarray:
    return np.array(
        [
            sum(
                weight
                * gram[row, column]
                for row, column, weight
                in bucket
            )
            for bucket in buckets
        ],
        dtype=np.float64,
    )


def coefficient_constraints(
    gram_expression,
    target: np.ndarray,
    buckets: list[
        list[tuple[int, int, float]]
    ],
) -> list:
    constraints = []

    for target_index, bucket in enumerate(
        buckets
    ):
        expression = 0

        for row, column, weight in bucket:
            expression += (
                weight
                * gram_expression[
                    row,
                    column,
                ]
            )

        constraints.append(
            expression
            == target[target_index]
        )

    return constraints


def solve_full_gram(
    name: str,
    target: np.ndarray,
    buckets: list[
        list[tuple[int, int, float]]
    ],
    size: int,
    configuration: dict,
) -> dict:
    gram = cp.Variable(
        (size, size),
        symmetric=True,
        name=f"Q_{name}",
    )

    margin = cp.Variable(
        name=f"margin_{name}"
    )

    constraints = [
        gram
        - margin * np.eye(size)
        >> 0
    ]

    constraints.extend(
        coefficient_constraints(
            gram,
            target,
            buckets,
        )
    )

    problem = cp.Problem(
        cp.Maximize(margin),
        constraints,
    )

    print(
        "full_solve_start:",
        name,
        configuration,
        flush=True,
    )

    started = time.perf_counter()

    objective = problem.solve(
        solver="SCS",
        eps=configuration["eps"],
        max_iters=configuration[
            "max_iters"
        ],
        scale=configuration["scale"],
        normalize=True,
        acceleration_lookback=20,
        verbose=False,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if gram.value is None:
        raise RuntimeError(
            f"no Gram value returned for {name}"
        )

    gram_value = np.array(
        gram.value,
        dtype=np.float64,
    )

    gram_value = 0.5 * (
        gram_value
        + gram_value.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        gram_value
    )

    kernel_basis = eigenvectors[
        :,
        :KERNEL_DIMENSION,
    ]

    face_basis = eigenvectors[
        :,
        KERNEL_DIMENSION:,
    ]

    kernel_projector = (
        kernel_basis
        @ kernel_basis.T
    )

    coefficients = gram_coefficients(
        gram_value,
        buckets,
    )

    residual = (
        coefficients - target
    )

    margin_value = (
        float(margin.value)
        if margin.value is not None
        else float("nan")
    )

    result = {
        "name": name,
        "status": problem.status,
        "objective": (
            float(objective)
            if objective is not None
            else float("nan")
        ),
        "margin": margin_value,
        "elapsed_seconds": elapsed,
        "gram": gram_value,
        "eigenvalues": eigenvalues,
        "kernel_basis": kernel_basis,
        "face_basis": face_basis,
        "kernel_projector": (
            kernel_projector
        ),
        "maximum_coefficient_residual": (
            max_abs(residual)
        ),
        "kernel_maximum_absolute_eigenvalue": float(
            np.max(
                np.abs(
                    eigenvalues[
                        :KERNEL_DIMENSION
                    ]
                )
            )
        ),
        "first_face_eigenvalue": float(
            eigenvalues[
                KERNEL_DIMENSION
            ]
        ),
        "numerical_rank": int(
            np.count_nonzero(
                eigenvalues
                > KERNEL_EIGENVALUE_TOLERANCE
            )
        ),
    }

    print(
        "full_solve_result:",
        name,
        "status:",
        result["status"],
        "margin:",
        result["margin"],
        "coefficient_residual:",
        result[
            "maximum_coefficient_residual"
        ],
        "kernel_max_abs_eig:",
        result[
            "kernel_maximum_absolute_eigenvalue"
        ],
        "first_face_eig:",
        result[
            "first_face_eigenvalue"
        ],
        "rank:",
        result["numerical_rank"],
        "elapsed_seconds:",
        elapsed,
        flush=True,
    )

    return result


def principal_angles_degrees(
    first_basis: np.ndarray,
    second_basis: np.ndarray,
) -> np.ndarray:
    singular_values = np.linalg.svd(
        first_basis.T
        @ second_basis,
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


def stabilized_kernel(
    projectors: list[np.ndarray],
) -> dict:
    average_projector = np.mean(
        np.stack(
            projectors,
            axis=0,
        ),
        axis=0,
    )

    average_projector = 0.5 * (
        average_projector
        + average_projector.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        average_projector
    )

    kernel_basis = eigenvectors[
        :,
        -KERNEL_DIMENSION:,
    ]

    kernel_projector = (
        kernel_basis
        @ kernel_basis.T
    )

    face_basis = eigenvectors[
        :,
        :FACE_DIMENSION,
    ]

    deviations = np.array(
        [
            np.linalg.norm(
                projector
                - kernel_projector,
                ord=2,
            )
            for projector in projectors
        ],
        dtype=np.float64,
    )

    return {
        "average_projector": (
            average_projector
        ),
        "average_projector_eigenvalues": (
            eigenvalues
        ),
        "kernel_basis": (
            kernel_basis
        ),
        "kernel_projector": (
            kernel_projector
        ),
        "face_basis": face_basis,
        "projector_deviations": (
            deviations
        ),
    }


def rationalize_projector(
    projector: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []

    for denominator_limit in (
        PROJECTOR_DENOMINATOR_LIMITS
    ):
        rational = np.empty_like(
            projector
        )

        for row in range(
            projector.shape[0]
        ):
            for column in range(
                projector.shape[1]
            ):
                rational[
                    row,
                    column,
                ] = float(
                    Fraction(
                        float(
                            projector[
                                row,
                                column,
                            ]
                        )
                    ).limit_denominator(
                        denominator_limit
                    )
                )

        rational = 0.5 * (
            rational
            + rational.T
        )

        approximation_error = max_abs(
            rational - projector
        )

        idempotence_error = max_abs(
            rational @ rational
            - rational
        )

        eigenvalues = np.linalg.eigvalsh(
            rational
        )

        trace_error = abs(
            float(np.trace(rational))
            - KERNEL_DIMENSION
        )

        row = {
            "denominator_limit": (
                denominator_limit
            ),
            "maximum_approximation_error": (
                approximation_error
            ),
            "maximum_idempotence_error": (
                idempotence_error
            ),
            "trace": float(
                np.trace(rational)
            ),
            "trace_error_from_10": (
                trace_error
            ),
            "minimum_eigenvalue": float(
                eigenvalues[0]
            ),
            "maximum_eigenvalue": float(
                eigenvalues[-1]
            ),
            "near_zero_eigenvalue_count": int(
                np.count_nonzero(
                    np.abs(eigenvalues)
                    < 1e-6
                )
            ),
            "near_one_eigenvalue_count": int(
                np.count_nonzero(
                    np.abs(
                        eigenvalues - 1.0
                    )
                    < 1e-6
                )
            ),
            "matrix": rational,
        }

        rows.append(row)

    best = min(
        rows,
        key=lambda row: (
            row[
                "maximum_idempotence_error"
            ],
            row[
                "maximum_approximation_error"
            ],
        ),
    )

    return rows, best


def solve_reduced_face(
    target: np.ndarray,
    buckets: list[
        list[tuple[int, int, float]]
    ],
    face_basis: np.ndarray,
) -> dict:
    full_size = face_basis.shape[0]
    reduced_size = face_basis.shape[1]

    print(
        "reduced_face_precompute_start:",
        "constraint_count:",
        len(buckets),
        "reduced_size:",
        reduced_size,
        flush=True,
    )

    reduced_coefficient_matrices = []

    for coefficient_index, bucket in enumerate(
        buckets
    ):
        full_coefficient_matrix = np.zeros(
            (full_size, full_size),
            dtype=np.float64,
        )

        for row, column, weight in bucket:
            if row == column:
                full_coefficient_matrix[
                    row,
                    column,
                ] += weight
            else:
                half_weight = (
                    weight / 2.0
                )

                full_coefficient_matrix[
                    row,
                    column,
                ] += half_weight

                full_coefficient_matrix[
                    column,
                    row,
                ] += half_weight

        reduced_matrix = (
            face_basis.T
            @ full_coefficient_matrix
            @ face_basis
        )

        reduced_matrix = 0.5 * (
            reduced_matrix
            + reduced_matrix.T
        )

        reduced_coefficient_matrices.append(
            reduced_matrix
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
                "reduced_face_precompute_progress:",
                f"{coefficient_index + 1}/{len(buckets)}",
                flush=True,
            )

    reduced_gram = cp.Variable(
        (
            reduced_size,
            reduced_size,
        ),
        symmetric=True,
        name="Y_e5_face",
    )

    margin = cp.Variable(
        name="margin_e5_face"
    )

    constraints = [
        reduced_gram
        - margin
        * np.eye(reduced_size)
        >> 0
    ]

    for target_index, matrix in enumerate(
        reduced_coefficient_matrices
    ):
        constraints.append(
            cp.sum(
                cp.multiply(
                    matrix,
                    reduced_gram,
                )
            )
            == target[target_index]
        )

    problem = cp.Problem(
        cp.Maximize(margin),
        constraints,
    )

    print(
        "reduced_face_solve_start:",
        "reduced_size:",
        reduced_size,
        "constraint_count:",
        len(
            reduced_coefficient_matrices
        ),
        flush=True,
    )

    started = time.perf_counter()

    objective = problem.solve(
        solver="SCS",
        eps=1e-9,
        max_iters=1_000_000,
        scale=10.0,
        normalize=True,
        acceleration_lookback=20,
        verbose=False,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if reduced_gram.value is None:
        raise RuntimeError(
            "reduced face solve returned no Gram value"
        )

    reduced_value = np.array(
        reduced_gram.value,
        dtype=np.float64,
    )

    reduced_value = 0.5 * (
        reduced_value
        + reduced_value.T
    )

    full_value = (
        face_basis
        @ reduced_value
        @ face_basis.T
    )

    full_value = 0.5 * (
        full_value
        + full_value.T
    )

    reduced_eigenvalues = np.linalg.eigvalsh(
        reduced_value
    )

    full_eigenvalues = np.linalg.eigvalsh(
        full_value
    )

    reduced_coefficients = np.array(
        [
            np.sum(
                matrix
                * reduced_value
            )
            for matrix in (
                reduced_coefficient_matrices
            )
        ],
        dtype=np.float64,
    )

    reduced_residual = (
        reduced_coefficients
        - target
    )

    full_residual = (
        gram_coefficients(
            full_value,
            buckets,
        )
        - target
    )

    coefficient_consistency = max_abs(
        reduced_coefficients
        - gram_coefficients(
            full_value,
            buckets,
        )
    )

    result = {
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
        "reduced_gram": reduced_value,
        "full_gram": full_value,
        "reduced_eigenvalues": (
            reduced_eigenvalues
        ),
        "full_eigenvalues": (
            full_eigenvalues
        ),
        "maximum_coefficient_residual": max(
            max_abs(
                reduced_residual
            ),
            max_abs(
                full_residual
            ),
        ),
        "reduced_full_coefficient_consistency": (
            coefficient_consistency
        ),
        "minimum_reduced_eigenvalue": float(
            reduced_eigenvalues[0]
        ),
        "minimum_full_eigenvalue": float(
            full_eigenvalues[0]
        ),
        "first_positive_full_eigenvalue": float(
            full_eigenvalues[
                KERNEL_DIMENSION
            ]
        ),
        "full_numerical_rank": int(
            np.count_nonzero(
                full_eigenvalues
                > KERNEL_EIGENVALUE_TOLERANCE
            )
        ),
    }

    print(
        "reduced_face_solve_result:",
        "status:",
        result["status"],
        "margin:",
        result["margin"],
        "coefficient_residual:",
        result[
            "maximum_coefficient_residual"
        ],
        "coefficient_consistency:",
        result[
            "reduced_full_coefficient_consistency"
        ],
        "min_reduced_eig:",
        result[
            "minimum_reduced_eigenvalue"
        ],
        "first_positive_full_eig:",
        result[
            "first_positive_full_eigenvalue"
        ],
        "full_rank:",
        result[
            "full_numerical_rank"
        ],
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

    SOLVE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANGLE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROJECTOR_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RATIONAL_CSV_OUT.parent.mkdir(
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

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    elementary = load_module(
        "gap_elementary_044",
        ELEMENTARY_SCRIPT_PATH,
    )

    axis_lines = np.array(
        orientation_data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    moments = (
        elementary.construct_register_moments(
            axis_lines
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

    saved_gram = np.array(
        source_data["e5_gram"],
        dtype=np.float64,
    )

    saved_gram = 0.5 * (
        saved_gram
        + saved_gram.T
    )

    saved_eigenvalues, saved_eigenvectors = (
        np.linalg.eigh(
            saved_gram
        )
    )

    saved_kernel_basis = (
        saved_eigenvectors[
            :,
            :KERNEL_DIMENSION,
        ]
    )

    saved_projector = (
        saved_kernel_basis
        @ saved_kernel_basis.T
    )

    full_results = []

    for configuration in (
        SOLVE_CONFIGURATIONS
    ):
        full_results.append(
            solve_full_gram(
                configuration["name"],
                target,
                buckets,
                len(
                    monomial_exponents
                ),
                configuration,
            )
        )

    projectors = [
        saved_projector
    ] + [
        result[
            "kernel_projector"
        ]
        for result in full_results
    ]

    stabilized = stabilized_kernel(
        projectors
    )

    angle_rows = []

    all_kernel_bases = [
        (
            "saved_048",
            saved_kernel_basis,
        )
    ] + [
        (
            result["name"],
            result[
                "kernel_basis"
            ],
        )
        for result in full_results
    ]

    for first_index in range(
        len(all_kernel_bases)
    ):
        for second_index in range(
            first_index + 1,
            len(all_kernel_bases),
        ):
            first_name, first_basis = (
                all_kernel_bases[
                    first_index
                ]
            )

            second_name, second_basis = (
                all_kernel_bases[
                    second_index
                ]
            )

            angles = principal_angles_degrees(
                first_basis,
                second_basis,
            )

            angle_rows.append(
                {
                    "first_solve": (
                        first_name
                    ),
                    "second_solve": (
                        second_name
                    ),
                    "minimum_angle_degrees": float(
                        np.min(angles)
                    ),
                    "maximum_angle_degrees": float(
                        np.max(angles)
                    ),
                    "mean_angle_degrees": float(
                        np.mean(angles)
                    ),
                }
            )

    rational_rows, best_rational = (
        rationalize_projector(
            stabilized[
                "kernel_projector"
            ]
        )
    )

    reduced_result = solve_reduced_face(
        target,
        buckets,
        stabilized["face_basis"],
    )

    solve_rows = []

    saved_residual = max_abs(
        gram_coefficients(
            saved_gram,
            buckets,
        )
        - target
    )

    solve_rows.append(
        {
            "solve_name": (
                "saved_048"
            ),
            "status": (
                "saved"
            ),
            "margin": float(
                saved_eigenvalues[0]
            ),
            "elapsed_seconds": 0.0,
            "maximum_coefficient_residual": (
                saved_residual
            ),
            "kernel_maximum_absolute_eigenvalue": float(
                np.max(
                    np.abs(
                        saved_eigenvalues[
                            :KERNEL_DIMENSION
                        ]
                    )
                )
            ),
            "first_face_eigenvalue": float(
                saved_eigenvalues[
                    KERNEL_DIMENSION
                ]
            ),
            "numerical_rank": int(
                np.count_nonzero(
                    saved_eigenvalues
                    > KERNEL_EIGENVALUE_TOLERANCE
                )
            ),
        }
    )

    for result in full_results:
        solve_rows.append(
            {
                "solve_name": (
                    result["name"]
                ),
                "status": (
                    result["status"]
                ),
                "margin": (
                    result["margin"]
                ),
                "elapsed_seconds": (
                    result[
                        "elapsed_seconds"
                    ]
                ),
                "maximum_coefficient_residual": (
                    result[
                        "maximum_coefficient_residual"
                    ]
                ),
                "kernel_maximum_absolute_eigenvalue": (
                    result[
                        "kernel_maximum_absolute_eigenvalue"
                    ]
                ),
                "first_face_eigenvalue": (
                    result[
                        "first_face_eigenvalue"
                    ]
                ),
                "numerical_rank": (
                    result[
                        "numerical_rank"
                    ]
                ),
            }
        )

    projector_rows = []

    projector = stabilized[
        "kernel_projector"
    ]

    for row in range(
        projector.shape[0]
    ):
        for column in range(
            row + 1
        ):
            projector_rows.append(
                {
                    "row": row,
                    "column": column,
                    "row_monomial": (
                        exponent_label(
                            monomial_exponents[
                                row
                            ]
                        )
                    ),
                    "column_monomial": (
                        exponent_label(
                            monomial_exponents[
                                column
                            ]
                        )
                    ),
                    "projector_entry": float(
                        projector[
                            row,
                            column,
                        ]
                    ),
                    "best_rational_entry": float(
                        best_rational[
                            "matrix"
                        ][
                            row,
                            column,
                        ]
                    ),
                }
            )

    rational_csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "matrix"
        }
        for row in rational_rows
    ]


    maximum_pairwise_angle = max(
        row[
            "maximum_angle_degrees"
        ]
        for row in angle_rows
    )

    maximum_projector_deviation = float(
        np.max(
            stabilized[
                "projector_deviations"
            ]
        )
    )

    full_solve_completed = all(
        result["status"]
        in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        )
        for result in full_results
    )

    kernel_dimension_stable = all(
        result["numerical_rank"]
        == FACE_DIMENSION
        for result in full_results
    )

    reduced_face_interior = (
        reduced_result["status"]
        in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        )
        and reduced_result[
            "maximum_coefficient_residual"
        ]
        < COEFFICIENT_TOLERANCE
        and reduced_result["margin"]
        > REDUCED_MARGIN_TOLERANCE
        and reduced_result[
            "minimum_reduced_eigenvalue"
        ]
        > REDUCED_MARGIN_TOLERANCE
    )

    checks = {
        "input_048_audit_pass": (
            source_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_048_e5_boundary_candidate_available": (
            source_receipt.get(
                "earned_interpretation",
                {},
            ).get(
                "e5_has_floating_interior_sos_candidate"
            )
            is False
        ),
        "e5_gram_size_is_56": (
            saved_gram.shape
            == (56, 56)
        ),
        "expected_face_dimension_is_46": (
            FACE_DIMENSION
            + KERNEL_DIMENSION
            == 56
        ),
        "all_tighter_full_solves_completed": (
            full_solve_completed
        ),
        "all_full_solve_coefficient_residuals_bounded": all(
            result[
                "maximum_coefficient_residual"
            ]
            < COEFFICIENT_TOLERANCE
            for result in full_results
        ),
        "numerical_kernel_dimension_stable_at_10": (
            kernel_dimension_stable
        ),
        "pairwise_kernel_angles_small": (
            maximum_pairwise_angle
            < PRINCIPAL_ANGLE_TOLERANCE_DEGREES
        ),
        "kernel_projector_stable": (
            maximum_projector_deviation
            < PROJECTOR_STABILITY_TOLERANCE
        ),
        "reduced_face_solve_completed": (
            reduced_result["status"]
            in (
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            )
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    if (
        audit_pass
        and reduced_face_interior
    ):
        verdict = (
            "stable_ten_dimensional_e5_kernel_and_interior_reduced_face_found"
        )
    elif audit_pass:
        verdict = (
            "stable_ten_dimensional_e5_kernel_found_reduced_face_interior_open"
        )
    else:
        verdict = (
            "e5_gram_face_probe_failed"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_e5_gram_face_probe_050"
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
        "full_solve_count": len(
            full_results
        ),
        "maximum_pairwise_kernel_angle_degrees": (
            maximum_pairwise_angle
        ),
        "maximum_kernel_projector_deviation": (
            maximum_projector_deviation
        ),
        "stabilized_projector": {
            "trace": float(
                np.trace(
                    stabilized[
                        "kernel_projector"
                    ]
                )
            ),
            "idempotence_residual": max_abs(
                stabilized[
                    "kernel_projector"
                ]
                @ stabilized[
                    "kernel_projector"
                ]
                - stabilized[
                    "kernel_projector"
                ]
            ),
            "average_projector_eigenvalues": (
                stabilized[
                    "average_projector_eigenvalues"
                ]
            ),
        },
        "best_rational_projector_probe": {
            key: value
            for key, value
            in best_rational.items()
            if key != "matrix"
        },
        "reduced_face_result": {
            key: value
            for key, value
            in reduced_result.items()
            if key not in (
                "reduced_gram",
                "full_gram",
                "reduced_eigenvalues",
                "full_eigenvalues",
            )
        },
        "reduced_face_interior_candidate": (
            reduced_face_interior
        ),
        "checks": checks,
        "earned_interpretation": {
            "e5_kernel_dimension_stable": (
                kernel_dimension_stable
            ),
            "stable_numerical_face_identified": (
                audit_pass
            ),
            "e5_is_interior_on_reduced_face": (
                reduced_face_interior
            ),
            "exact_rational_kernel_found": (
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
            "numerical_facial_reduction_probe_only": (
                True
            ),
            "floating_kernel_basis_not_exact": (
                True
            ),
            "rational_projector_probe_not_certificate": (
                True
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "solve_csv": str(
                SOLVE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "angle_csv": str(
                ANGLE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "projector_csv": str(
                PROJECTOR_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "rational_csv": str(
                RATIONAL_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "probe_npz": str(
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
            SOLVE_CSV_OUT,
            solve_rows,
        ),
        (
            ANGLE_CSV_OUT,
            angle_rows,
        ),
        (
            PROJECTOR_CSV_OUT,
            projector_rows,
        ),
        (
            RATIONAL_CSV_OUT,
            rational_csv_rows,
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
        monomial_exponents=np.array(
            monomial_exponents,
            dtype=np.int64,
        ),
        target_coefficients=target,
        stabilized_kernel_basis=(
            stabilized[
                "kernel_basis"
            ]
        ),
        stabilized_kernel_projector=(
            stabilized[
                "kernel_projector"
            ]
        ),
        stabilized_face_basis=(
            stabilized[
                "face_basis"
            ]
        ),
        reduced_gram=(
            reduced_result[
                "reduced_gram"
            ]
        ),
        reduced_full_gram=(
            reduced_result[
                "full_gram"
            ]
        ),
        reduced_eigenvalues=(
            reduced_result[
                "reduced_eigenvalues"
            ]
        ),
        reduced_full_eigenvalues=(
            reduced_result[
                "full_eigenvalues"
            ]
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "maximum_pairwise_kernel_angle_degrees:",
        maximum_pairwise_angle,
    )
    print(
        "maximum_kernel_projector_deviation:",
        maximum_projector_deviation,
    )
    print(
        "best_rational_projector_denominator_limit:",
        best_rational[
            "denominator_limit"
        ],
    )
    print(
        "best_rational_projector_idempotence_error:",
        best_rational[
            "maximum_idempotence_error"
        ],
    )
    print(
        "reduced_face_margin:",
        reduced_result["margin"],
    )
    print(
        "reduced_face_minimum_eigenvalue:",
        reduced_result[
            "minimum_reduced_eigenvalue"
        ],
    )
    print(
        "reduced_face_coefficient_residual:",
        reduced_result[
            "maximum_coefficient_residual"
        ],
    )
    print(
        "reduced_face_interior_candidate:",
        reduced_face_interior,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SOLVE_CSV_OUT)
    print("wrote:", ANGLE_CSV_OUT)
    print("wrote:", PROJECTOR_CSV_OUT)
    print("wrote:", RATIONAL_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
