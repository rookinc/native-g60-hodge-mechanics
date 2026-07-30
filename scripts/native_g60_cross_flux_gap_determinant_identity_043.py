from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]

GAP_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_one_third_gap_probe_042.json"
)

GAP_REGRESSION_CSV_PATH = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_one_third_gap_regressions_042.csv"
)

ORIENTATION_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_covariant_orientation_035.json"
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
    / "native_g60_cross_flux_gap_determinant_identity_043.json"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_coefficients_043.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_probes_043.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_determinant_identity_043.npz"
)

MODEL_NAME = (
    "gap_determinant__moments_quadratic"
)

RATIONAL_DENOMINATOR_LIMIT = 10_000_000
COEFFICIENT_TOLERANCE = 5e-9
PROBE_TOLERANCE = 5e-12
PROBE_COUNT = 4096
RANDOM_SEED = 46043

Exponent = tuple[int, int, int, int]
Polynomial = dict[Exponent, float]


S4_SYMBOL = sp.Symbol("S4")
S6_SYMBOL = sp.Symbol("S6")
S8_SYMBOL = sp.Symbol("S8")
R2_SYMBOL = sp.Symbol("R2")
N2_SYMBOL = sp.Symbol("N2")


FEATURE_SYMBOLS = [
    sp.Integer(1),
    S4_SYMBOL,
    S6_SYMBOL,
    S8_SYMBOL,
    R2_SYMBOL,
    S4_SYMBOL**2,
    S4_SYMBOL * S6_SYMBOL,
    S4_SYMBOL * R2_SYMBOL,
    S6_SYMBOL**2,
    R2_SYMBOL**2,
]


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, sp.Basic):
        return str(value)

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


def rationalize(value: float) -> Fraction:
    return Fraction(
        float(value)
    ).limit_denominator(
        RATIONAL_DENOMINATOR_LIMIT
    )


def exponent_add(
    first: Exponent,
    second: Exponent,
) -> Exponent:
    return tuple(
        first[index] + second[index]
        for index in range(4)
    )


def polynomial_add(
    first: Polynomial,
    second: Polynomial,
    scale: float = 1.0,
) -> Polynomial:
    result = defaultdict(float)

    for exponent, value in first.items():
        result[exponent] += value

    for exponent, value in second.items():
        result[exponent] += (
            scale * value
        )

    return {
        exponent: value
        for exponent, value in result.items()
        if abs(value) > 1e-18
    }


def polynomial_scale(
    polynomial: Polynomial,
    scale: float,
) -> Polynomial:
    return {
        exponent: scale * value
        for exponent, value in polynomial.items()
    }


def polynomial_multiply(
    first: Polynomial,
    second: Polynomial,
) -> Polynomial:
    result = defaultdict(float)

    for first_exponent, first_value in first.items():
        for second_exponent, second_value in second.items():
            result[
                exponent_add(
                    first_exponent,
                    second_exponent,
                )
            ] += (
                first_value
                * second_value
            )

    return dict(result)


def polynomial_power(
    polynomial: Polynomial,
    power: int,
) -> Polynomial:
    result: Polynomial = {
        (0, 0, 0, 0): 1.0
    }

    for _ in range(power):
        result = polynomial_multiply(
            result,
            polynomial,
        )

    return result


def polynomial_evaluate(
    polynomial: Polynomial,
    point: np.ndarray,
) -> float:
    total = 0.0

    for exponent, coefficient in polynomial.items():
        term = coefficient

        for index, power in enumerate(exponent):
            term *= point[index] ** power

        total += term

    return float(total)


def degree_exponents(
    degree: int,
) -> list[Exponent]:
    values = []

    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(
                degree + 1 - a - b
            ):
                d = degree - a - b - c
                values.append((a, b, c, d))

    values.sort(reverse=True)

    return values


def monomial_label(
    exponent: Exponent,
) -> str:
    factors = []

    for index, power in enumerate(exponent):
        if power == 0:
            continue

        if power == 1:
            factors.append(f"f{index}")
        else:
            factors.append(
                f"f{index}^{power}"
            )

    return "*".join(factors) or "1"


def load_regression_row() -> dict:
    with GAP_REGRESSION_CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if row["model"] == MODEL_NAME:
                return row

    raise RuntimeError(
        f"regression model not found: {MODEL_NAME}"
    )


def reconstruct_regression() -> dict:
    row = load_regression_row()

    floating_coefficients = np.array(
        json.loads(
            row["coefficients"]
        ),
        dtype=np.float64,
    )

    if len(floating_coefficients) != len(
        FEATURE_SYMBOLS
    ):
        raise RuntimeError(
            "unexpected regression coefficient count: "
            f"{len(floating_coefficients)}"
        )

    rational_coefficients = [
        rationalize(value)
        for value in floating_coefficients
    ]

    expression = sp.expand(
        sum(
            sp.Rational(
                coefficient.numerator,
                coefficient.denominator,
            )
            * feature
            for coefficient, feature in zip(
                rational_coefficients,
                FEATURE_SYMBOLS,
            )
        )
    )

    unit_residual_identity = (
        -sp.Rational(625, 1152)
        + sp.Rational(55, 48)
        * S4_SYMBOL
        + S6_SYMBOL
        - S4_SYMBOL**2
    )

    reduced_expression = sp.factor(
        sp.expand(
            expression.subs(
                R2_SYMBOL,
                unit_residual_identity,
            )
        )
    )

    reduced_polynomial = sp.Poly(
        sp.expand(
            reduced_expression
        ),
        S4_SYMBOL,
        S6_SYMBOL,
        S8_SYMBOL,
    )

    terms = []

    homogeneous_compatible = True

    for powers, coefficient in (
        reduced_polynomial.terms()
    ):
        s4_power, s6_power, s8_power = powers

        weighted_degree = (
            4 * s4_power
            + 6 * s6_power
            + 8 * s8_power
        )

        remaining_degree = (
            12 - weighted_degree
        )

        compatible = (
            remaining_degree >= 0
            and remaining_degree % 2 == 0
        )

        if not compatible:
            homogeneous_compatible = False

        terms.append(
            {
                "s4_power": s4_power,
                "s6_power": s6_power,
                "s8_power": s8_power,
                "coefficient": coefficient,
                "weighted_degree": (
                    weighted_degree
                ),
                "n2_power": (
                    remaining_degree // 2
                    if compatible
                    else None
                ),
                "homogeneous_compatible": (
                    compatible
                ),
            }
        )

    return {
        "source_row": row,
        "floating_coefficients": (
            floating_coefficients
        ),
        "rational_coefficients": (
            rational_coefficients
        ),
        "expression_before_elimination": (
            expression
        ),
        "reduced_expression": (
            reduced_expression
        ),
        "reduced_expanded": sp.expand(
            reduced_expression
        ),
        "reduced_terms": terms,
        "homogeneous_compatible": (
            homogeneous_compatible
        ),
    }


def construct_register_polynomials(
    axis_lines: np.ndarray,
) -> dict[str, Polynomial]:
    n2: Polynomial = {
        (2, 0, 0, 0): 1.0,
        (0, 2, 0, 0): 1.0,
        (0, 0, 2, 0): 1.0,
        (0, 0, 0, 2): 1.0,
    }

    s4: Polynomial = {}
    s6: Polynomial = {}
    s8: Polynomial = {}

    for axis in axis_lines:
        linear: Polynomial = {}

        for index, value in enumerate(axis):
            exponent = [0, 0, 0, 0]
            exponent[index] = 1
            linear[tuple(exponent)] = float(
                value
            )

        s4 = polynomial_add(
            s4,
            polynomial_power(
                linear,
                4,
            ),
        )

        s6 = polynomial_add(
            s6,
            polynomial_power(
                linear,
                6,
            ),
        )

        s8 = polynomial_add(
            s8,
            polynomial_power(
                linear,
                8,
            ),
        )

    return {
        "n2": n2,
        "s4": s4,
        "s6": s6,
        "s8": s8,
    }


def build_candidate_polynomial(
    reconstruction: dict,
    register_polynomials: dict[
        str,
        Polynomial,
    ],
) -> Polynomial:
    if not reconstruction[
        "homogeneous_compatible"
    ]:
        return {}

    result: Polynomial = {}

    for term in reconstruction[
        "reduced_terms"
    ]:
        coefficient = float(
            term["coefficient"]
        )

        current: Polynomial = {
            (0, 0, 0, 0): 1.0
        }

        current = polynomial_multiply(
            current,
            polynomial_power(
                register_polynomials["s4"],
                term["s4_power"],
            ),
        )

        current = polynomial_multiply(
            current,
            polynomial_power(
                register_polynomials["s6"],
                term["s6_power"],
            ),
        )

        current = polynomial_multiply(
            current,
            polynomial_power(
                register_polynomials["s8"],
                term["s8_power"],
            ),
        )

        current = polynomial_multiply(
            current,
            polynomial_power(
                register_polynomials["n2"],
                term["n2_power"],
            ),
        )

        result = polynomial_add(
            result,
            polynomial_scale(
                current,
                coefficient,
            ),
        )

    return result


def construct_gap_entry_polynomials(
    slices: np.ndarray,
) -> list[list[Polynomial]]:
    n2: Polynomial = {
        (2, 0, 0, 0): 1.0,
        (0, 2, 0, 0): 1.0,
        (0, 0, 2, 0): 1.0,
        (0, 0, 0, 2): 1.0,
    }

    entries: list[list[Polynomial]] = [
        [
            {}
            for _ in range(6)
        ]
        for _ in range(6)
    ]

    for row in range(6):
        for column in range(6):
            polynomial: Polynomial = {}

            if row == column:
                polynomial = polynomial_add(
                    polynomial,
                    polynomial_scale(
                        n2,
                        1.0 / 9.0,
                    ),
                )

            for first in range(4):
                for second in range(4):
                    coefficient = -float(
                        np.dot(
                            slices[
                                first,
                                :,
                                row,
                            ],
                            slices[
                                second,
                                :,
                                column,
                            ],
                        )
                    )

                    if abs(coefficient) < 1e-18:
                        continue

                    exponent = [0, 0, 0, 0]
                    exponent[first] += 1
                    exponent[second] += 1

                    polynomial[
                        tuple(exponent)
                    ] = (
                        polynomial.get(
                            tuple(exponent),
                            0.0,
                        )
                        + coefficient
                    )

            entries[row][column] = (
                polynomial
            )

    return entries


def permutation_sign(
    permutation: tuple[int, ...],
) -> int:
    inversions = 0

    for first in range(len(permutation)):
        for second in range(
            first + 1,
            len(permutation),
        ):
            if (
                permutation[first]
                > permutation[second]
            ):
                inversions += 1

    return (
        -1
        if inversions % 2
        else 1
    )


def determinant_polynomial(
    entries: list[list[Polynomial]],
) -> Polynomial:
    result: Polynomial = {}

    all_permutations = list(
        permutations(range(6))
    )

    for index, permutation in enumerate(
        all_permutations,
        start=1,
    ):
        term: Polynomial = {
            (0, 0, 0, 0): 1.0
        }

        for row, column in enumerate(
            permutation
        ):
            term = polynomial_multiply(
                term,
                entries[row][column],
            )

            if not term:
                break

        if term:
            result = polynomial_add(
                result,
                term,
                scale=permutation_sign(
                    permutation
                ),
            )

        if (
            index == 1
            or index % 120 == 0
            or index == len(
                all_permutations
            )
        ):
            print(
                "determinant_progress:",
                f"{index}/{len(all_permutations)}",
                "current_term_count:",
                len(result),
                flush=True,
            )

    return result


def compare_polynomials(
    determinant: Polynomial,
    candidate: Polynomial,
) -> tuple[
    list[dict],
    dict,
]:
    exponents = degree_exponents(12)

    rows = []
    maximum_residual = 0.0
    residual_norm_squared = 0.0

    for coefficient_id, exponent in enumerate(
        exponents
    ):
        determinant_value = determinant.get(
            exponent,
            0.0,
        )

        candidate_value = candidate.get(
            exponent,
            0.0,
        )

        residual = (
            determinant_value
            - candidate_value
        )

        maximum_residual = max(
            maximum_residual,
            abs(residual),
        )

        residual_norm_squared += (
            residual**2
        )

        rows.append(
            {
                "coefficient_id": (
                    coefficient_id
                ),
                "monomial": (
                    monomial_label(
                        exponent
                    )
                ),
                "f0_power": exponent[0],
                "f1_power": exponent[1],
                "f2_power": exponent[2],
                "f3_power": exponent[3],
                "determinant_coefficient": (
                    determinant_value
                ),
                "candidate_coefficient": (
                    candidate_value
                ),
                "residual": residual,
                "coefficient_pass": (
                    abs(residual)
                    < COEFFICIENT_TOLERANCE
                ),
            }
        )

    return rows, {
        "degree": 12,
        "monomial_count": len(
            exponents
        ),
        "maximum_coefficient_residual": (
            maximum_residual
        ),
        "coefficient_residual_l2": (
            math.sqrt(
                residual_norm_squared
            )
        ),
        "all_coefficients_pass": all(
            row["coefficient_pass"]
            for row in rows
        ),
    }


def direct_gap_determinant(
    slices: np.ndarray,
    point: np.ndarray,
) -> float:
    matrix = np.einsum(
        "r,rab->ab",
        point,
        slices,
    )

    gram = matrix.T @ matrix

    gap = (
        float(
            np.dot(point, point)
        )
        / 9.0
        * np.eye(6)
        - gram
    )

    return float(
        np.linalg.det(gap)
    )


def probe_identity(
    slices: np.ndarray,
    candidate: Polynomial,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []
    maximum_residual = 0.0

    for probe_id in range(PROBE_COUNT):
        point = rng.normal(size=4)

        direct = direct_gap_determinant(
            slices,
            point,
        )

        predicted = polynomial_evaluate(
            candidate,
            point,
        )

        residual = abs(
            direct - predicted
        )

        maximum_residual = max(
            maximum_residual,
            residual,
        )

        if probe_id < 1024:
            rows.append(
                {
                    "probe_id": probe_id,
                    "point_norm_squared": float(
                        np.dot(point, point)
                    ),
                    "direct_determinant": direct,
                    "predicted_determinant": (
                        predicted
                    ),
                    "absolute_residual": (
                        residual
                    ),
                }
            )

    return rows, {
        "probe_count": PROBE_COUNT,
        "maximum_probe_residual": (
            maximum_residual
        ),
        "all_probes_pass": (
            maximum_residual
            < PROBE_TOLERANCE
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COEFFICIENT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROBE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gap_receipt = json.loads(
        GAP_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    orientation_receipt = json.loads(
        ORIENTATION_JSON_PATH.read_text(
            encoding="utf-8"
        )
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

    reconstruction = (
        reconstruct_regression()
    )

    register_polynomials = (
        construct_register_polynomials(
            axis_lines
        )
    )

    candidate = build_candidate_polynomial(
        reconstruction,
        register_polynomials,
    )

    gap_entries = (
        construct_gap_entry_polynomials(
            slices
        )
    )

    determinant = determinant_polynomial(
        gap_entries
    )

    coefficient_rows, coefficient_summary = (
        compare_polynomials(
            determinant,
            candidate,
        )
    )

    probe_rows, probe_summary = (
        probe_identity(
            slices,
            candidate,
        )
    )

    checks = {
        "input_042_audit_pass": (
            gap_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_035_theorem_pass": (
            orientation_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "regression_model_found": (
            reconstruction[
                "source_row"
            ]["model"]
            == MODEL_NAME
        ),
        "reduced_expression_is_homogeneous_compatible": (
            reconstruction[
                "homogeneous_compatible"
            ]
        ),
        "degree_twelve_monomial_count_is_455": (
            coefficient_summary[
                "monomial_count"
            ]
            == 455
        ),
        "all_coefficient_matrices_match": (
            coefficient_summary[
                "all_coefficients_pass"
            ]
        ),
        "probe_identity_pass": (
            probe_summary[
                "all_probes_pass"
            ]
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_gap_determinant_identity_exact"
        if theorem_pass
        else "native_g60_cross_flux_gap_determinant_identity_not_resolved"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_determinant_identity_043"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "regression_reconstruction": {
            "model": MODEL_NAME,
            "floating_coefficients": (
                reconstruction[
                    "floating_coefficients"
                ]
            ),
            "rational_coefficients": [
                str(value)
                for value in reconstruction[
                    "rational_coefficients"
                ]
            ],
            "expression_before_elimination": str(
                reconstruction[
                    "expression_before_elimination"
                ]
            ),
            "reduced_expression": str(
                reconstruction[
                    "reduced_expression"
                ]
            ),
            "reduced_expanded": str(
                reconstruction[
                    "reduced_expanded"
                ]
            ),
            "homogeneous_compatible": (
                reconstruction[
                    "homogeneous_compatible"
                ]
            ),
            "homogeneous_terms": [
                {
                    **{
                        key: value
                        for key, value in term.items()
                        if key != "coefficient"
                    },
                    "coefficient": str(
                        term["coefficient"]
                    ),
                }
                for term in reconstruction[
                    "reduced_terms"
                ]
            ],
        },
        "coefficient_certificate": (
            coefficient_summary
        ),
        "probe_certificate": (
            probe_summary
        ),
        "checks": checks,
        "earned_interpretation": {
            "gap_determinant_has_exact_native_moment_formula": (
                theorem_pass
            ),
            "gap_matrix_positive_semidefinite_proved": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
        },
        "boundary": {
            "determinant_identity_proved": (
                theorem_pass
            ),
            "determinant_nonnegative_globally_proved": (
                False
            ),
            "all_gap_elementary_invariants_nonnegative_proved": (
                False
            ),
            "global_gap_psd_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "coefficient_csv": str(
                COEFFICIENT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "identity_npz": str(
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

    with COEFFICIENT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                coefficient_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            coefficient_rows
        )

    with PROBE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                probe_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            probe_rows
        )

    degree_twelve = degree_exponents(12)

    np.savez_compressed(
        NPZ_OUT,
        degree_twelve_exponents=np.array(
            degree_twelve,
            dtype=np.int64,
        ),
        determinant_coefficients=np.array(
            [
                determinant.get(
                    exponent,
                    0.0,
                )
                for exponent in degree_twelve
            ],
            dtype=np.float64,
        ),
        candidate_coefficients=np.array(
            [
                candidate.get(
                    exponent,
                    0.0,
                )
                for exponent in degree_twelve
            ],
            dtype=np.float64,
        ),
        coefficient_residuals=np.array(
            [
                determinant.get(
                    exponent,
                    0.0,
                )
                - candidate.get(
                    exponent,
                    0.0,
                )
                for exponent in degree_twelve
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "rational_regression_coefficients:",
        [
            str(value)
            for value in reconstruction[
                "rational_coefficients"
            ]
        ],
    )
    print(
        "reduced_expression:",
        reconstruction[
            "reduced_expression"
        ],
    )
    print(
        "homogeneous_compatible:",
        reconstruction[
            "homogeneous_compatible"
        ],
    )
    print(
        "coefficient_summary:",
        coefficient_summary,
    )
    print(
        "probe_summary:",
        probe_summary,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
