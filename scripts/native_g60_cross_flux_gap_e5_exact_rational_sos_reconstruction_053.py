#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import qr
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]

SOURCE_052 = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_e5_rational_face_feasibility_052.py"
)

NPZ_052 = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_e5_rational_face_feasibility_052.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_e5_exact_rational_sos_reconstruction_053.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_exact_rational_sos_reconstruction_summary_053.csv"
)

REDUCED_GRAM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_e5_exact_rational_reduced_gram_053.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_e5_exact_rational_sos_reconstruction_053.npz"
)

FACE_DIMENSION = 46
FULL_DIMENSION = 56
COEFFICIENT_COUNT = 286
SYMMETRIC_VARIABLE_COUNT = 1081

TARGET_DENOMINATOR_LIMIT = 10**12
FREE_DENOMINATOR_LIMITS = [
    10**6,
    10**7,
    10**8,
    10**9,
    10**10,
]

TARGET_RECONSTRUCTION_TOLERANCE = 5e-15
POSITIVE_EIGENVALUE_TOLERANCE = 1e-10


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction_from_float(
    value: float,
    denominator_limit: int,
) -> Fraction:
    if not np.isfinite(value):
        raise ValueError(
            f"cannot rationalize nonfinite value: {value}"
        )

    return Fraction(
        float(value)
    ).limit_denominator(
        denominator_limit
    )


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


def symmetric_pairs(
    size: int,
) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(size)
        for column in range(row + 1)
    ]


def coefficient_buckets_exact(
    monomial_exponents,
    target_exponents,
):
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
            exponent = tuple(
                monomial_exponents[row][index]
                + monomial_exponents[column][index]
                for index in range(4)
            )

            weight = (
                Fraction(1, 1)
                if row == column
                else Fraction(2, 1)
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


def exact_full_coefficient_matrix(
    bucket,
    size: int,
):
    matrix = [
        [
            Fraction(0, 1)
            for _ in range(size)
        ]
        for _ in range(size)
    ]

    for row, column, weight in bucket:
        if row == column:
            matrix[row][column] += weight
        else:
            half = weight / 2
            matrix[row][column] += half
            matrix[column][row] += half

    return matrix


def exact_reduced_matrix(
    face_basis,
    full_matrix,
):
    reduced = [
        [
            Fraction(0, 1)
            for _ in range(
                FACE_DIMENSION
            )
        ]
        for _ in range(
            FACE_DIMENSION
        )
    ]

    active_entries = []

    for row in range(FULL_DIMENSION):
        for column in range(FULL_DIMENSION):
            value = full_matrix[row][column]
            if value:
                active_entries.append(
                    (
                        row,
                        column,
                        value,
                    )
                )

    for left in range(FACE_DIMENSION):
        for right in range(
            left + 1
        ):
            total = Fraction(0, 1)

            for row, column, value in active_entries:
                total += (
                    face_basis[row][left]
                    * value
                    * face_basis[column][right]
                )

            reduced[left][right] = total
            reduced[right][left] = total

    return reduced


def exact_symmetric_matrix_vector(
    matrix,
    pairs,
):
    values = []

    for row, column in pairs:
        if row == column:
            values.append(
                matrix[row][column]
            )
        else:
            values.append(
                2 * matrix[row][column]
            )

    return values


def build_exact_design(
    face_basis,
    buckets,
):
    pairs = symmetric_pairs(
        FACE_DIMENSION
    )

    design_rows = []

    print(
        "exact_operator_precompute_start:",
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
            exact_full_coefficient_matrix(
                bucket,
                FULL_DIMENSION,
            )
        )

        reduced_matrix = exact_reduced_matrix(
            face_basis,
            full_matrix,
        )

        design_rows.append(
            exact_symmetric_matrix_vector(
                reduced_matrix,
                pairs,
            )
        )

        if (
            coefficient_index == 0
            or (
                coefficient_index + 1
            ) % 25 == 0
            or (
                coefficient_index + 1
                == len(buckets)
            )
        ):
            print(
                "exact_operator_precompute_progress:",
                f"{coefficient_index + 1}/{len(buckets)}",
                flush=True,
            )

    return {
        "pairs": pairs,
        "design": design_rows,
    }


def vector_to_fraction_matrix(
    vector,
    pairs,
):
    matrix = [
        [
            Fraction(0, 1)
            for _ in range(
                FACE_DIMENSION
            )
        ]
        for _ in range(
            FACE_DIMENSION
        )
    ]

    for value, (row, column) in zip(
        vector,
        pairs,
    ):
        matrix[row][column] = value
        matrix[column][row] = value

    return matrix


def exact_matvec(
    design,
    vector,
):
    return [
        sum(
            (
                coefficient * value
                for coefficient, value in zip(
                    row,
                    vector,
                )
            ),
            Fraction(0, 1),
        )
        for row in design
    ]


def maximum_float_difference(
    first,
    second,
):
    if not first:
        return 0.0

    return max(
        abs(
            float(a)
            - float(b)
        )
        for a, b in zip(
            first,
            second,
        )
    )


def choose_independent_coordinates(
    design_float: np.ndarray,
    rank: int,
):
    _, _, column_pivots = qr(
        design_float,
        mode="economic",
        pivoting=True,
    )

    dependent_columns = [
        int(value)
        for value in column_pivots[:rank]
    ]

    dependent_matrix = design_float[
        :,
        dependent_columns,
    ]

    _, _, row_pivots = qr(
        dependent_matrix.T,
        mode="economic",
        pivoting=True,
    )

    independent_rows = [
        int(value)
        for value in row_pivots[:rank]
    ]

    dependent_set = set(
        dependent_columns
    )

    free_columns = [
        index
        for index in range(
            design_float.shape[1]
        )
        if index not in dependent_set
    ]

    return {
        "dependent_columns": dependent_columns,
        "free_columns": free_columns,
        "independent_rows": independent_rows,
    }


def solve_exact_dependent_coordinates(
    design,
    target,
    floating_vector,
    coordinate_partition,
    denominator_limit,
):
    dependent_columns = (
        coordinate_partition[
            "dependent_columns"
        ]
    )
    free_columns = (
        coordinate_partition[
            "free_columns"
        ]
    )
    independent_rows = (
        coordinate_partition[
            "independent_rows"
        ]
    )

    exact_vector = [
        None
        for _ in range(
            SYMMETRIC_VARIABLE_COUNT
        )
    ]

    for column in free_columns:
        exact_vector[column] = (
            fraction_from_float(
                floating_vector[column],
                denominator_limit,
            )
        )

    right_hand_side = []

    for row in independent_rows:
        free_contribution = sum(
            (
                design[row][column]
                * exact_vector[column]
                for column in free_columns
            ),
            Fraction(0, 1),
        )

        right_hand_side.append(
            target[row]
            - free_contribution
        )

    square_matrix = sp.MutableDenseMatrix(
        [
            [
                sp.Rational(
                    design[row][column].numerator,
                    design[row][column].denominator,
                )
                for column in dependent_columns
            ]
            for row in independent_rows
        ]
    )

    square_rhs = sp.MutableDenseMatrix(
        [
            sp.Rational(
                value.numerator,
                value.denominator,
            )
            for value in right_hand_side
        ]
    )

    print(
        "exact_dependent_solve_start:",
        "size:",
        square_matrix.shape,
        "free_denominator_limit:",
        denominator_limit,
        flush=True,
    )

    solution = square_matrix.inv().multiply(
        square_rhs
    )

    for index, column in enumerate(
        dependent_columns
    ):
        value = solution[index]
        exact_vector[column] = Fraction(
            int(value.p),
            int(value.q),
        )

    if any(
        value is None
        for value in exact_vector
    ):
        raise RuntimeError(
            "exact coordinate vector remained incomplete"
        )

    prediction = exact_matvec(
        design,
        exact_vector,
    )

    residual = [
        predicted - expected
        for predicted, expected in zip(
            prediction,
            target,
        )
    ]

    exact_coefficient_match = all(
        value == 0
        for value in residual
    )

    return {
        "vector": exact_vector,
        "prediction": prediction,
        "residual": residual,
        "exact_coefficient_match": (
            exact_coefficient_match
        ),
    }


def main() -> None:
    started = time.perf_counter()

    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    SUMMARY_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REDUCED_GRAM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_052 = load_module(
        "native_g60_e5_face_052",
        SOURCE_052,
    )

    exact_face = (
        source_052.load_exact_face_basis()
    )

    face_float = fraction_matrix_float(
        exact_face
    )

    with np.load(
        NPZ_052,
        allow_pickle=False,
    ) as archive:
        design_float_saved = np.array(
            archive["reduced_design"],
            dtype=np.float64,
        )
        target_float = np.array(
            archive["target_coefficients"],
            dtype=np.float64,
        )
        sdp_gram = np.array(
            archive["sdp_gram"],
            dtype=np.float64,
        )

    target_exact = [
        fraction_from_float(
            value,
            TARGET_DENOMINATOR_LIMIT,
        )
        for value in target_float
    ]

    target_reconstruction_error = max(
        abs(
            float(exact)
            - float(observed)
        )
        for exact, observed in zip(
            target_exact,
            target_float,
        )
    )

    print(
        "target_rational_reconstruction:",
        "maximum_error:",
        target_reconstruction_error,
        "maximum_denominator:",
        max(
            value.denominator
            for value in target_exact
        ),
        flush=True,
    )

    target_exponents = (
        source_052.homogeneous_exponents(
            10
        )
    )

    monomial_exponents = (
        source_052.homogeneous_exponents(
            5
        )
    )

    buckets = coefficient_buckets_exact(
        monomial_exponents,
        target_exponents,
    )

    exact_operator = build_exact_design(
        exact_face,
        buckets,
    )

    exact_design = exact_operator[
        "design"
    ]
    pairs = exact_operator["pairs"]

    exact_design_float = np.array(
        [
            [
                float(value)
                for value in row
            ]
            for row in exact_design
        ],
        dtype=np.float64,
    )

    exact_design_matches_saved = (
        float(
            np.max(
                np.abs(
                    exact_design_float
                    - design_float_saved
                )
            )
        )
    )

    design_rank = int(
        np.linalg.matrix_rank(
            exact_design_float,
            tol=1e-11,
        )
    )

    print(
        "exact_design_check:",
        "shape:",
        exact_design_float.shape,
        "rank:",
        design_rank,
        "saved_design_maximum_difference:",
        exact_design_matches_saved,
        flush=True,
    )

    coordinate_partition = (
        choose_independent_coordinates(
            exact_design_float,
            design_rank,
        )
    )

    print(
        "coordinate_partition:",
        "dependent_count:",
        len(
            coordinate_partition[
                "dependent_columns"
            ]
        ),
        "free_count:",
        len(
            coordinate_partition[
                "free_columns"
            ]
        ),
        "independent_row_count:",
        len(
            coordinate_partition[
                "independent_rows"
            ]
        ),
        flush=True,
    )

    floating_vector = np.array(
        [
            sdp_gram[row, column]
            for row, column in pairs
        ],
        dtype=np.float64,
    )

    candidate_records = []
    winning_candidate = None

    for denominator_limit in (
        FREE_DENOMINATOR_LIMITS
    ):
        candidate_started = (
            time.perf_counter()
        )

        try:
            candidate = (
                solve_exact_dependent_coordinates(
                    exact_design,
                    target_exact,
                    floating_vector,
                    coordinate_partition,
                    denominator_limit,
                )
            )

            exact_matrix = (
                vector_to_fraction_matrix(
                    candidate["vector"],
                    pairs,
                )
            )

            exact_matrix_float = (
                fraction_matrix_float(
                    exact_matrix
                )
            )

            eigenvalues = np.linalg.eigvalsh(
                exact_matrix_float
            )

            maximum_entry_change = float(
                np.max(
                    np.abs(
                        exact_matrix_float
                        - sdp_gram
                    )
                )
            )

            record = {
                "denominator_limit": (
                    denominator_limit
                ),
                "solve_succeeded": True,
                "exact_coefficient_match": (
                    candidate[
                        "exact_coefficient_match"
                    ]
                ),
                "minimum_eigenvalue": float(
                    eigenvalues[0]
                ),
                "maximum_eigenvalue": float(
                    eigenvalues[-1]
                ),
                "numerical_rank": int(
                    np.linalg.matrix_rank(
                        exact_matrix_float,
                        tol=1e-10,
                    )
                ),
                "maximum_entry_change": (
                    maximum_entry_change
                ),
                "maximum_numerator_bits": max(
                    abs(value.numerator).bit_length()
                    for value in candidate[
                        "vector"
                    ]
                ),
                "maximum_denominator_bits": max(
                    value.denominator.bit_length()
                    for value in candidate[
                        "vector"
                    ]
                ),
                "elapsed_seconds": (
                    time.perf_counter()
                    - candidate_started
                ),
            }

            print(
                "candidate_result:",
                "denominator_limit:",
                denominator_limit,
                "exact_match:",
                record[
                    "exact_coefficient_match"
                ],
                "minimum_eigenvalue:",
                record[
                    "minimum_eigenvalue"
                ],
                "maximum_entry_change:",
                record[
                    "maximum_entry_change"
                ],
                "maximum_denominator_bits:",
                record[
                    "maximum_denominator_bits"
                ],
                flush=True,
            )

            candidate_records.append(
                record
            )

            if (
                record[
                    "exact_coefficient_match"
                ]
                and record[
                    "minimum_eigenvalue"
                ]
                > POSITIVE_EIGENVALUE_TOLERANCE
            ):
                winning_candidate = {
                    **candidate,
                    "matrix": exact_matrix,
                    "matrix_float": (
                        exact_matrix_float
                    ),
                    "eigenvalues": eigenvalues,
                    "record": record,
                }
                break

        except Exception as exc:
            record = {
                "denominator_limit": (
                    denominator_limit
                ),
                "solve_succeeded": False,
                "error": repr(exc),
                "elapsed_seconds": (
                    time.perf_counter()
                    - candidate_started
                ),
            }
            candidate_records.append(
                record
            )

            print(
                "candidate_failure:",
                "denominator_limit:",
                denominator_limit,
                "error:",
                repr(exc),
                flush=True,
            )

    exact_affine_certificate_found = (
        winning_candidate is not None
    )

    target_reconstruction_pass = (
        target_reconstruction_error
        < TARGET_RECONSTRUCTION_TOLERANCE
    )

    audit_pass = (
        exact_design_float.shape
        == (
            COEFFICIENT_COUNT,
            SYMMETRIC_VARIABLE_COUNT,
        )
        and design_rank == 273
        and len(
            coordinate_partition[
                "free_columns"
            ]
        )
        == 808
        and exact_design_matches_saved
        < 1e-14
        and target_reconstruction_pass
    )

    theorem_pass = False

    verdict = (
        "exact_rational_e5_affine_positive_candidate_reconstructed_ldl_open"
        if (
            audit_pass
            and exact_affine_certificate_found
        )
        else (
            "exact_rational_e5_affine_reconstruction_not_resolved"
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_e5_exact_rational_sos_reconstruction_053"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "target_reconstruction": {
            "denominator_limit": (
                TARGET_DENOMINATOR_LIMIT
            ),
            "maximum_float_error": (
                target_reconstruction_error
            ),
            "pass": (
                target_reconstruction_pass
            ),
        },
        "exact_design": {
            "shape": list(
                exact_design_float.shape
            ),
            "rank": design_rank,
            "saved_design_maximum_difference": (
                exact_design_matches_saved
            ),
            "dependent_coordinate_count": len(
                coordinate_partition[
                    "dependent_columns"
                ]
            ),
            "free_coordinate_count": len(
                coordinate_partition[
                    "free_columns"
                ]
            ),
        },
        "candidate_records": (
            candidate_records
        ),
        "exact_affine_positive_candidate_found": (
            exact_affine_certificate_found
        ),
        "exact_rational_ldl_attempted": False,
        "exact_rational_sos_certificate_found": (
            False
        ),
        "global_gap_psd_proved": False,
        "global_one_third_bound_proved": (
            False
        ),
        "elapsed_seconds": elapsed_seconds,
        "boundary": {
            "exact_target_identity_requires_canonical_source_alignment": (
                True
            ),
            "exact_rational_ldl_still_open": (
                True
            ),
            "complete_e5_sos_theorem_not_yet_claimed": (
                True
            ),
            "physical_claim": False,
        },
    }

    if winning_candidate is not None:
        record = winning_candidate[
            "record"
        ]
        payload[
            "winning_candidate"
        ] = record

        gram_rows = []

        for row in range(
            FACE_DIMENSION
        ):
            for column in range(
                FACE_DIMENSION
            ):
                value = winning_candidate[
                    "matrix"
                ][row][column]

                gram_rows.append(
                    {
                        "row": row,
                        "column": column,
                        "numerator": (
                            value.numerator
                        ),
                        "denominator": (
                            value.denominator
                        ),
                        "float_value": (
                            float(value)
                        ),
                    }
                )

        with REDUCED_GRAM_CSV_OUT.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "row",
                    "column",
                    "numerator",
                    "denominator",
                    "float_value",
                ],
            )
            writer.writeheader()
            writer.writerows(
                gram_rows
            )

        np.savez_compressed(
            NPZ_OUT,
            reduced_gram=(
                winning_candidate[
                    "matrix_float"
                ]
            ),
            reduced_gram_eigenvalues=(
                winning_candidate[
                    "eigenvalues"
                ]
            ),
            dependent_columns=np.array(
                coordinate_partition[
                    "dependent_columns"
                ],
                dtype=np.int64,
            ),
            free_columns=np.array(
                coordinate_partition[
                    "free_columns"
                ],
                dtype=np.int64,
            ),
            independent_rows=np.array(
                coordinate_partition[
                    "independent_rows"
                ],
                dtype=np.int64,
            ),
        )

    with JSON_OUT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    summary_row = {
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "design_rank": design_rank,
        "design_nullity": (
            SYMMETRIC_VARIABLE_COUNT
            - design_rank
        ),
        "target_reconstruction_error": (
            target_reconstruction_error
        ),
        "exact_affine_positive_candidate_found": (
            exact_affine_certificate_found
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    with SUMMARY_CSV_OUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary_row
            ),
        )
        writer.writeheader()
        writer.writerow(
            summary_row
        )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "exact_affine_positive_candidate_found:",
        exact_affine_certificate_found,
    )
    print(
        "exact_rational_ldl_attempted:",
        False,
    )
    print(
        "elapsed_seconds:",
        elapsed_seconds,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)

    if winning_candidate is not None:
        print(
            "winning_denominator_limit:",
            winning_candidate[
                "record"
            ][
                "denominator_limit"
            ],
        )
        print(
            "winning_minimum_eigenvalue:",
            winning_candidate[
                "record"
            ][
                "minimum_eigenvalue"
            ],
        )
        print(
            "winning_maximum_entry_change:",
            winning_candidate[
                "record"
            ][
                "maximum_entry_change"
            ],
        )
        print("wrote:", REDUCED_GRAM_CSV_OUT)
        print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
