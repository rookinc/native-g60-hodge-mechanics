from __future__ import annotations

import csv
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_elementary_invariants_044.py"
)

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_scalar_sos_probe_045.json"
)

SOURCE_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_scalar_sos_probe_045.npz"
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
    / "native_g60_cross_flux_gap_exact_sos_reconstruction_046.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_sos_summary_046.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_sos_gram_entries_046.csv"
)

LDL_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_sos_ldl_pivots_046.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_exact_sos_reconstruction_046.npz"
)

TARGET_DENOMINATOR_LIMITS = (
    1_000_000,
    100_000_000,
    10_000_000_000,
)

GRAM_DENOMINATOR_LIMITS = (
    100_000,
    1_000_000,
    10_000_000,
)

TARGET_RATIONALIZATION_TOLERANCE = 5e-12
FLOATING_PROJECTION_TOLERANCE = 5e-11
FLOATING_PSD_TOLERANCE = 5e-10

INVARIANT_DEGREES = (
    2,
    3,
)


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


def load_source_module():
    spec = importlib.util.spec_from_file_location(
        "gap_elementary_044",
        SOURCE_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load source script 044"
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


def lower_entries(
    matrix: np.ndarray,
) -> np.ndarray:
    values = []

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            row + 1
        ):
            values.append(
                matrix[row, column]
            )

    return np.array(
        values,
        dtype=np.float64,
    )


def lower_pairs(
    size: int,
) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(size)
        for column in range(row + 1)
    ]


def coefficient_buckets(
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
) -> list[list[tuple[int, int]]]:
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

    cursor = 0

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
                1
                if row == column
                else 2
            )

            buckets[
                target_index[exponent]
            ].append(
                (cursor, weight)
            )

            cursor += 1

    return buckets


def project_to_float_slice(
    entries: np.ndarray,
    target: np.ndarray,
    buckets: list[
        list[tuple[int, int]]
    ],
) -> np.ndarray:
    projected = np.array(
        entries,
        dtype=np.float64,
        copy=True,
    )

    for target_index, bucket in enumerate(
        buckets
    ):
        indices = np.array(
            [
                index
                for index, _ in bucket
            ],
            dtype=np.int64,
        )

        weights = np.array(
            [
                weight
                for _, weight in bucket
            ],
            dtype=np.float64,
        )

        current = float(
            np.dot(
                weights,
                projected[indices],
            )
        )

        correction = (
            target[target_index]
            - current
        )

        denominator = float(
            np.dot(weights, weights)
        )

        projected[indices] += (
            correction
            * weights
            / denominator
        )

    return projected


def coefficient_values_float(
    entries: np.ndarray,
    buckets: list[
        list[tuple[int, int]]
    ],
) -> np.ndarray:
    values = np.zeros(
        len(buckets),
        dtype=np.float64,
    )

    for target_index, bucket in enumerate(
        buckets
    ):
        values[target_index] = sum(
            weight * entries[index]
            for index, weight in bucket
        )

    return values


def rationalize_vector(
    values: np.ndarray,
    denominator_limit: int,
) -> list[Fraction]:
    return [
        Fraction(
            float(value)
        ).limit_denominator(
            denominator_limit
        )
        for value in values
    ]


def exact_bucket_reconstruction(
    projected: np.ndarray,
    rational_target: list[Fraction],
    buckets: list[
        list[tuple[int, int]]
    ],
    denominator_limit: int,
) -> list[Fraction]:
    entries = rationalize_vector(
        projected,
        denominator_limit,
    )

    for target_index, bucket in enumerate(
        buckets
    ):
        correction_index, correction_weight = max(
            bucket,
            key=lambda item: (
                item[1],
                abs(
                    projected[
                        item[0]
                    ]
                ),
            ),
        )

        subtotal = Fraction(0, 1)

        for entry_index, weight in bucket:
            if entry_index == correction_index:
                continue

            subtotal += (
                weight
                * entries[entry_index]
            )

        entries[correction_index] = (
            rational_target[target_index]
            - subtotal
        ) / correction_weight

    return entries


def exact_coefficient_values(
    entries: list[Fraction],
    buckets: list[
        list[tuple[int, int]]
    ],
) -> list[Fraction]:
    return [
        sum(
            (
                weight * entries[index]
                for index, weight in bucket
            ),
            Fraction(0, 1),
        )
        for bucket in buckets
    ]


def rational_matrix_from_lower(
    entries: list[Fraction],
    size: int,
) -> list[list[Fraction]]:
    matrix = [
        [
            Fraction(0, 1)
            for _ in range(size)
        ]
        for _ in range(size)
    ]

    cursor = 0

    for row in range(size):
        for column in range(
            row + 1
        ):
            value = entries[cursor]

            matrix[row][column] = value
            matrix[column][row] = value

            cursor += 1

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


def exact_ldl_positive_definite(
    matrix: list[list[Fraction]],
) -> dict:
    size = len(matrix)

    lower = [
        [
            Fraction(0, 1)
            for _ in range(size)
        ]
        for _ in range(size)
    ]

    diagonal = [
        Fraction(0, 1)
        for _ in range(size)
    ]

    for row in range(size):
        lower[row][row] = Fraction(
            1,
            1,
        )

        pivot = matrix[row][row]

        for previous in range(row):
            pivot -= (
                lower[row][previous]
                * lower[row][previous]
                * diagonal[previous]
            )

        diagonal[row] = pivot

        if pivot <= 0:
            return {
                "positive_definite": False,
                "failed_pivot": row,
                "lower": lower,
                "diagonal": diagonal,
            }

        for next_row in range(
            row + 1,
            size,
        ):
            numerator = matrix[
                next_row
            ][row]

            for previous in range(row):
                numerator -= (
                    lower[next_row][previous]
                    * lower[row][previous]
                    * diagonal[previous]
                )

            lower[next_row][row] = (
                numerator / pivot
            )

    return {
        "positive_definite": True,
        "failed_pivot": None,
        "lower": lower,
        "diagonal": diagonal,
    }


def reconstruct_exact_gram(
    floating_gram: np.ndarray,
    target: np.ndarray,
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
) -> dict:
    size = floating_gram.shape[0]

    buckets = coefficient_buckets(
        monomial_exponents,
        target_exponents,
    )

    floating_entries = lower_entries(
        floating_gram
    )

    projected = project_to_float_slice(
        floating_entries,
        target,
        buckets,
    )

    projected_coefficients = (
        coefficient_values_float(
            projected,
            buckets,
        )
    )

    candidates = []

    for target_limit in (
        TARGET_DENOMINATOR_LIMITS
    ):
        rational_target = rationalize_vector(
            target,
            target_limit,
        )

        target_error = max(
            abs(
                float(exact) - value
            )
            for exact, value in zip(
                rational_target,
                target,
            )
        )

        for gram_limit in (
            GRAM_DENOMINATOR_LIMITS
        ):
            exact_entries = (
                exact_bucket_reconstruction(
                    projected,
                    rational_target,
                    buckets,
                    gram_limit,
                )
            )

            exact_coefficients = (
                exact_coefficient_values(
                    exact_entries,
                    buckets,
                )
            )

            coefficient_exact = (
                exact_coefficients
                == rational_target
            )

            exact_matrix = (
                rational_matrix_from_lower(
                    exact_entries,
                    size,
                )
            )

            float_matrix = (
                fraction_matrix_float(
                    exact_matrix
                )
            )

            eigenvalues = np.linalg.eigvalsh(
                float_matrix
            )

            ldl = (
                exact_ldl_positive_definite(
                    exact_matrix
                )
            )

            distance = max_abs(
                float_matrix
                - floating_gram
            )

            projected_distance = max_abs(
                float_matrix
                - fraction_matrix_float(
                    rational_matrix_from_lower(
                        [
                            Fraction(float(value))
                            for value in projected
                        ],
                        size,
                    )
                )
            )

            candidates.append(
                {
                    "target_denominator_limit": (
                        target_limit
                    ),
                    "gram_denominator_limit": (
                        gram_limit
                    ),
                    "target_error": (
                        target_error
                    ),
                    "coefficient_exact": (
                        coefficient_exact
                    ),
                    "exact_entries": (
                        exact_entries
                    ),
                    "exact_target": (
                        rational_target
                    ),
                    "exact_matrix": (
                        exact_matrix
                    ),
                    "float_matrix": (
                        float_matrix
                    ),
                    "minimum_float_eigenvalue": float(
                        eigenvalues[0]
                    ),
                    "maximum_float_eigenvalue": float(
                        eigenvalues[-1]
                    ),
                    "distance_from_saved_gram": (
                        distance
                    ),
                    "distance_from_projected_gram": (
                        projected_distance
                    ),
                    "ldl": ldl,
                    "certificate_pass": (
                        target_error
                        < TARGET_RATIONALIZATION_TOLERANCE
                        and coefficient_exact
                        and ldl[
                            "positive_definite"
                        ]
                    ),
                }
            )

    best = min(
        candidates,
        key=lambda item: (
            not item["certificate_pass"],
            item["target_error"],
            item[
                "distance_from_saved_gram"
            ],
            -item[
                "minimum_float_eigenvalue"
            ],
        ),
    )

    return {
        "buckets": buckets,
        "projected_entries": (
            projected
        ),
        "projected_coefficient_residual": (
            max_abs(
                projected_coefficients
                - target
            )
        ),
        "candidates": candidates,
        "best": best,
    }


def exponent_label(
    exponent: tuple[int, int, int, int],
) -> str:
    factors = []

    for index, power in enumerate(
        exponent
    ):
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


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LDL_CSV_OUT.parent.mkdir(
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

    source = load_source_module()

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

    moments = source.construct_register_moments(
        axis_lines
    )

    entries = source.construct_gap_entries(
        slices
    )

    results = {}
    summary_rows = []
    gram_rows = []
    ldl_rows = []

    for invariant_degree in (
        INVARIANT_DEGREES
    ):
        print(
            "exact_sos_target:",
            f"e{invariant_degree}",
            "constructing coefficient target",
            flush=True,
        )

        polynomial_degree = (
            2 * invariant_degree
        )

        target_polynomial = (
            source.elementary_polynomial(
                entries,
                invariant_degree,
            )
        )

        target_exponents = (
            source.degree_exponents(
                polynomial_degree
            )
        )

        target = source.polynomial_vector(
            target_polynomial,
            target_exponents,
        )

        monomial_exponents = (
            homogeneous_exponents(
                invariant_degree
            )
        )

        floating_gram = np.array(
            source_data[
                f"e{invariant_degree}_gram"
            ],
            dtype=np.float64,
        )

        result = reconstruct_exact_gram(
            floating_gram,
            target,
            monomial_exponents,
            target_exponents,
        )

        results[invariant_degree] = result

        best = result["best"]

        exact_certificate = bool(
            best["certificate_pass"]
        )

        summary_rows.append(
            {
                "elementary_invariant": (
                    f"e{invariant_degree}"
                ),
                "gram_size": (
                    floating_gram.shape[0]
                ),
                "coefficient_count": (
                    len(target)
                ),
                "target_denominator_limit": (
                    best[
                        "target_denominator_limit"
                    ]
                ),
                "gram_denominator_limit": (
                    best[
                        "gram_denominator_limit"
                    ]
                ),
                "target_rationalization_error": (
                    best["target_error"]
                ),
                "projected_coefficient_residual": (
                    result[
                        "projected_coefficient_residual"
                    ]
                ),
                "exact_coefficient_equality": (
                    best[
                        "coefficient_exact"
                    ]
                ),
                "exact_ldl_positive_definite": (
                    best["ldl"][
                        "positive_definite"
                    ]
                ),
                "minimum_float_eigenvalue": (
                    best[
                        "minimum_float_eigenvalue"
                    ]
                ),
                "maximum_float_eigenvalue": (
                    best[
                        "maximum_float_eigenvalue"
                    ]
                ),
                "maximum_entry_change_from_saved_gram": (
                    best[
                        "distance_from_saved_gram"
                    ]
                ),
                "exact_rational_sos_certificate": (
                    exact_certificate
                ),
            }
        )

        pairs = lower_pairs(
            floating_gram.shape[0]
        )

        for entry_index, (
            row,
            column,
        ) in enumerate(pairs):
            exact_value = best[
                "exact_entries"
            ][entry_index]

            gram_rows.append(
                {
                    "elementary_invariant": (
                        f"e{invariant_degree}"
                    ),
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
                    "numerator": (
                        exact_value.numerator
                    ),
                    "denominator": (
                        exact_value.denominator
                    ),
                    "exact_value": str(
                        exact_value
                    ),
                    "floating_value": float(
                        exact_value
                    ),
                }
            )

        for pivot_index, pivot in enumerate(
            best["ldl"]["diagonal"]
        ):
            ldl_rows.append(
                {
                    "elementary_invariant": (
                        f"e{invariant_degree}"
                    ),
                    "pivot_index": (
                        pivot_index
                    ),
                    "pivot_numerator": (
                        pivot.numerator
                    ),
                    "pivot_denominator": (
                        pivot.denominator
                    ),
                    "pivot_exact": str(
                        pivot
                    ),
                    "pivot_float": float(
                        pivot
                    ),
                    "pivot_positive": (
                        pivot > 0
                    ),
                }
            )

        print(
            "exact_sos_result:",
            f"e{invariant_degree}",
            "target_error:",
            best["target_error"],
            "entry_change:",
            best[
                "distance_from_saved_gram"
            ],
            "min_float_eig:",
            best[
                "minimum_float_eigenvalue"
            ],
            "exact_ldl_pd:",
            best["ldl"][
                "positive_definite"
            ],
            "certificate:",
            exact_certificate,
            flush=True,
        )

    certificate_count = sum(
        bool(
            row[
                "exact_rational_sos_certificate"
            ]
        )
        for row in summary_rows
    )

    checks = {
        "input_045_audit_pass": (
            source_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_045_has_two_floating_candidates": (
            source_receipt.get(
                "candidate_count"
            )
            == 2
        ),
        "e2_reconstruction_completed": (
            2 in results
        ),
        "e3_reconstruction_completed": (
            3 in results
        ),
        "all_float_slice_projections_close": all(
            row[
                "projected_coefficient_residual"
            ]
            < FLOATING_PROJECTION_TOLERANCE
            for row in summary_rows
        ),
        "all_reconstructed_grams_numerically_psd": all(
            row[
                "minimum_float_eigenvalue"
            ]
            >= -FLOATING_PSD_TOLERANCE
            for row in summary_rows
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = (
        audit_pass
        and certificate_count == 2
    )

    if theorem_pass:
        verdict = (
            "exact_rational_sos_certificates_found_for_e2_and_e3"
        )
    elif audit_pass and certificate_count == 1:
        verdict = (
            "exact_rational_sos_certificate_found_for_one_invariant"
        )
    elif audit_pass:
        verdict = (
            "floating_gram_interior_confirmed_exact_reconstruction_open"
        )
    else:
        verdict = (
            "exact_sos_reconstruction_audit_failed"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_exact_sos_reconstruction_046"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "certificate_count": (
            certificate_count
        ),
        "results": summary_rows,
        "checks": checks,
        "earned_interpretation": {
            "exact_rational_sos_certificate_for_e2": (
                summary_rows[0][
                    "exact_rational_sos_certificate"
                ]
            ),
            "exact_rational_sos_certificate_for_e3": (
                summary_rows[1][
                    "exact_rational_sos_certificate"
                ]
            ),
            "e2_globally_nonnegative_proved": (
                summary_rows[0][
                    "exact_rational_sos_certificate"
                ]
            ),
            "e3_globally_nonnegative_proved": (
                summary_rows[1][
                    "exact_rational_sos_certificate"
                ]
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "termux_economical_reconstruction": (
                True
            ),
            "e4_and_e5_not_attempted": (
                True
            ),
            "full_completion_check_deferred_to_m3_pro": (
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
            "gram_csv": str(
                GRAM_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "ldl_csv": str(
                LDL_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "reconstruction_npz": str(
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
            GRAM_CSV_OUT,
            gram_rows,
        ),
        (
            LDL_CSV_OUT,
            ldl_rows,
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
        e2_exact_gram_float=(
            results[2]["best"][
                "float_matrix"
            ]
        ),
        e3_exact_gram_float=(
            results[3]["best"][
                "float_matrix"
            ]
        ),
        e2_exact_ldl_pivots_float=np.array(
            [
                float(value)
                for value in results[2][
                    "best"
                ]["ldl"]["diagonal"]
            ],
            dtype=np.float64,
        ),
        e3_exact_ldl_pivots_float=np.array(
            [
                float(value)
                for value in results[3][
                    "best"
                ]["ldl"]["diagonal"]
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "certificate_count:",
        certificate_count,
    )

    for row in summary_rows:
        print(
            row[
                "elementary_invariant"
            ],
            "target_error:",
            row[
                "target_rationalization_error"
            ],
            "entry_change:",
            row[
                "maximum_entry_change_from_saved_gram"
            ],
            "minimum_float_eigenvalue:",
            row[
                "minimum_float_eigenvalue"
            ],
            "exact_ldl_pd:",
            row[
                "exact_ldl_positive_definite"
            ],
            "certificate:",
            row[
                "exact_rational_sos_certificate"
            ],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", LDL_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
