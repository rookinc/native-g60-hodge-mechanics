from __future__ import annotations

import csv
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FAILED_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_determinant_identity_043.json"
)

FAILED_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_determinant_identity_043.npz"
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
    / "native_g60_cross_flux_gap_determinant_coefficient_fit_043b.json"
)

BASIS_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_basis_fit_043b.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_coefficient_fit_043b.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_fit_probes_043b.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_determinant_coefficient_fit_043b.npz"
)

RATIONAL_DENOMINATOR_LIMIT = 100_000_000
COEFFICIENT_TOLERANCE = 5e-10
PROBE_TOLERANCE = 5e-10

RANDOM_SEED = 460431
PROBE_COUNT = 4096

Exponent = tuple[int, int, int, int]
Polynomial = dict[Exponent, float]


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

    return dict(result)


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


def polynomial_vector(
    polynomial: Polynomial,
    exponents: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            polynomial.get(
                tuple(
                    int(value)
                    for value in exponent
                ),
                0.0,
            )
            for exponent in exponents
        ],
        dtype=np.float64,
    )


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

            linear[
                tuple(exponent)
            ] = float(value)

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

    basis = {
        "N2^6": polynomial_power(
            n2,
            6,
        ),
        "N2^4*S4": polynomial_multiply(
            polynomial_power(
                n2,
                4,
            ),
            s4,
        ),
        "N2^3*S6": polynomial_multiply(
            polynomial_power(
                n2,
                3,
            ),
            s6,
        ),
        "N2^2*S4^2": polynomial_multiply(
            polynomial_power(
                n2,
                2,
            ),
            polynomial_power(
                s4,
                2,
            ),
        ),
        "N2^2*S8": polynomial_multiply(
            polynomial_power(
                n2,
                2,
            ),
            s8,
        ),
        "N2*S4*S6": polynomial_multiply(
            n2,
            polynomial_multiply(
                s4,
                s6,
            ),
        ),
        "S4^3": polynomial_power(
            s4,
            3,
        ),
        "S6^2": polynomial_power(
            s6,
            2,
        ),
    }

    return {
        "n2": n2,
        "s4": s4,
        "s6": s6,
        "s8": s8,
        "basis": basis,
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
            np.dot(
                point,
                point,
            )
        )
        / 9.0
        * np.eye(6)
        - gram
    )

    return float(
        np.linalg.det(gap)
    )


def fit_coefficient_space(
    determinant_coefficients: np.ndarray,
    exponents: np.ndarray,
    basis: dict[str, Polynomial],
) -> dict:
    basis_names = list(
        basis
    )

    design = np.column_stack(
        [
            polynomial_vector(
                basis[name],
                exponents,
            )
            for name in basis_names
        ]
    )

    coefficients, residuals, rank, singular_values = (
        np.linalg.lstsq(
            design,
            determinant_coefficients,
            rcond=None,
        )
    )

    floating_prediction = (
        design @ coefficients
    )

    floating_residual = (
        determinant_coefficients
        - floating_prediction
    )

    rational_coefficients = [
        rationalize(value)
        for value in coefficients
    ]

    rational_values = np.array(
        [
            float(value)
            for value in rational_coefficients
        ],
        dtype=np.float64,
    )

    rational_prediction = (
        design @ rational_values
    )

    rational_residual = (
        determinant_coefficients
        - rational_prediction
    )

    candidate: Polynomial = {}

    for coefficient, name in zip(
        rational_values,
        basis_names,
    ):
        candidate = polynomial_add(
            candidate,
            basis[name],
            scale=coefficient,
        )

    return {
        "basis_names": basis_names,
        "design": design,
        "design_rank": int(rank),
        "design_singular_values": (
            singular_values
        ),
        "floating_coefficients": (
            coefficients
        ),
        "rational_coefficients": (
            rational_coefficients
        ),
        "rational_values": (
            rational_values
        ),
        "floating_residual": (
            floating_residual
        ),
        "rational_residual": (
            rational_residual
        ),
        "floating_maximum_residual": (
            max_abs(
                floating_residual
            )
        ),
        "rational_maximum_residual": (
            max_abs(
                rational_residual
            )
        ),
        "candidate_polynomial": (
            candidate
        ),
        "floating_fit_pass": (
            max_abs(
                floating_residual
            )
            < COEFFICIENT_TOLERANCE
        ),
        "rational_fit_pass": (
            max_abs(
                rational_residual
            )
            < COEFFICIENT_TOLERANCE
        ),
    }


def formula_text(
    names: list[str],
    coefficients: list[Fraction],
) -> str:
    terms = []

    for name, coefficient in zip(
        names,
        coefficients,
    ):
        if coefficient == 0:
            continue

        terms.append(
            f"({coefficient})*{name}"
        )

    return " + ".join(terms) or "0"


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BASIS_CSV_OUT.parent.mkdir(
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

    failed_receipt = json.loads(
        FAILED_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    failed_data = np.load(
        FAILED_NPZ_PATH
    )

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    exponents = np.array(
        failed_data[
            "degree_twelve_exponents"
        ],
        dtype=np.int64,
    )

    determinant_coefficients = np.array(
        failed_data[
            "determinant_coefficients"
        ],
        dtype=np.float64,
    )

    axis_lines = np.array(
        orientation_data[
            "axis_lines"
        ],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data[
            "slices"
        ],
        dtype=np.float64,
    )

    register = construct_register_polynomials(
        axis_lines
    )

    fit = fit_coefficient_space(
        determinant_coefficients,
        exponents,
        register["basis"],
    )

    formula = formula_text(
        fit["basis_names"],
        fit["rational_coefficients"],
    )

    coefficient_rows = []

    rational_prediction = (
        determinant_coefficients
        - fit[
            "rational_residual"
        ]
    )

    for coefficient_id, exponent in enumerate(
        exponents
    ):
        coefficient_rows.append(
            {
                "coefficient_id": (
                    coefficient_id
                ),
                "f0_power": int(
                    exponent[0]
                ),
                "f1_power": int(
                    exponent[1]
                ),
                "f2_power": int(
                    exponent[2]
                ),
                "f3_power": int(
                    exponent[3]
                ),
                "determinant_coefficient": float(
                    determinant_coefficients[
                        coefficient_id
                    ]
                ),
                "candidate_coefficient": float(
                    rational_prediction[
                        coefficient_id
                    ]
                ),
                "residual": float(
                    fit[
                        "rational_residual"
                    ][coefficient_id]
                ),
                "coefficient_pass": (
                    abs(
                        fit[
                            "rational_residual"
                        ][coefficient_id]
                    )
                    < COEFFICIENT_TOLERANCE
                ),
            }
        )

    basis_rows = []

    for name, floating, rational in zip(
        fit["basis_names"],
        fit["floating_coefficients"],
        fit["rational_coefficients"],
    ):
        basis_rows.append(
            {
                "basis_term": name,
                "floating_coefficient": float(
                    floating
                ),
                "rational_coefficient": str(
                    rational
                ),
                "rational_value": float(
                    rational
                ),
                "coefficient_difference": abs(
                    float(floating)
                    - float(rational)
                ),
            }
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    probe_rows = []
    maximum_probe_residual = 0.0

    for probe_id in range(PROBE_COUNT):
        point = rng.normal(
            size=4
        )

        direct = direct_gap_determinant(
            slices,
            point,
        )

        predicted = polynomial_evaluate(
            fit[
                "candidate_polynomial"
            ],
            point,
        )

        residual = abs(
            direct - predicted
        )

        maximum_probe_residual = max(
            maximum_probe_residual,
            residual,
        )

        if probe_id < 1024:
            probe_rows.append(
                {
                    "probe_id": probe_id,
                    "point_norm_squared": float(
                        np.dot(
                            point,
                            point,
                        )
                    ),
                    "direct_determinant": (
                        direct
                    ),
                    "predicted_determinant": (
                        predicted
                    ),
                    "absolute_residual": (
                        residual
                    ),
                }
            )

    checks = {
        "input_043_failure_recorded": (
            failed_receipt.get(
                "artifact_id"
            )
            == (
                "native_g60_cross_flux_gap_"
                "determinant_identity_043"
            )
        ),
        "degree_twelve_coefficient_count_is_455": (
            len(exponents) == 455
        ),
        "weighted_basis_count_is_8": (
            len(
                fit["basis_names"]
            )
            == 8
        ),
        "coefficient_design_has_full_column_rank": (
            fit["design_rank"] == 8
        ),
        "floating_coefficient_fit_pass": (
            fit[
                "floating_fit_pass"
            ]
        ),
        "rational_coefficient_fit_pass": (
            fit[
                "rational_fit_pass"
            ]
        ),
        "all_direct_probes_pass": (
            maximum_probe_residual
            < PROBE_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    if theorem_pass:
        verdict = (
            "native_g60_cross_flux_gap_"
            "determinant_native_moment_identity_exact"
        )
    elif fit["floating_fit_pass"]:
        verdict = (
            "native_g60_cross_flux_gap_"
            "determinant_moment_space_fit_unrationalized"
        )
    else:
        verdict = (
            "native_g60_cross_flux_gap_"
            "determinant_outside_tested_scalar_moment_space"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_"
            "determinant_coefficient_fit_043b"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "tested_basis": (
            fit["basis_names"]
        ),
        "coefficient_fit": {
            "design_shape": list(
                fit["design"].shape
            ),
            "design_rank": (
                fit["design_rank"]
            ),
            "design_singular_values": (
                fit[
                    "design_singular_values"
                ]
            ),
            "floating_coefficients": (
                fit[
                    "floating_coefficients"
                ]
            ),
            "rational_coefficients": [
                str(value)
                for value in fit[
                    "rational_coefficients"
                ]
            ],
            "floating_maximum_residual": (
                fit[
                    "floating_maximum_residual"
                ]
            ),
            "rational_maximum_residual": (
                fit[
                    "rational_maximum_residual"
                ]
            ),
            "formula": formula,
        },
        "probe_summary": {
            "probe_count": (
                PROBE_COUNT
            ),
            "maximum_probe_residual": (
                maximum_probe_residual
            ),
            "all_probes_pass": (
                maximum_probe_residual
                < PROBE_TOLERANCE
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "sample_regression_failure_corrected": (
                True
            ),
            "determinant_belongs_to_tested_scalar_moment_algebra": (
                fit[
                    "floating_fit_pass"
                ]
            ),
            "exact_rational_native_moment_formula_proved": (
                theorem_pass
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "coefficient_space_fit_completed": (
                True
            ),
            "determinant_identity_proved": (
                theorem_pass
            ),
            "determinant_nonnegative_proved": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "basis_csv": str(
                BASIS_CSV_OUT.relative_to(
                    ROOT
                )
            ),
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
            "fit_npz": str(
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
            BASIS_CSV_OUT,
            basis_rows,
        ),
        (
            COEFFICIENT_CSV_OUT,
            coefficient_rows,
        ),
        (
            PROBE_CSV_OUT,
            probe_rows,
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
        degree_twelve_exponents=(
            exponents
        ),
        determinant_coefficients=(
            determinant_coefficients
        ),
        basis_names=np.array(
            fit["basis_names"],
            dtype=object,
        ),
        floating_coefficients=(
            fit[
                "floating_coefficients"
            ]
        ),
        rational_coefficients=(
            fit[
                "rational_values"
            ]
        ),
        floating_residual=(
            fit[
                "floating_residual"
            ]
        ),
        rational_residual=(
            fit[
                "rational_residual"
            ]
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "design_shape:",
        fit["design"].shape,
    )
    print(
        "design_rank:",
        fit["design_rank"],
    )
    print(
        "floating_coefficients:",
        fit[
            "floating_coefficients"
        ].tolist(),
    )
    print(
        "rational_coefficients:",
        [
            str(value)
            for value in fit[
                "rational_coefficients"
            ]
        ],
    )
    print(
        "formula:",
        formula,
    )
    print(
        "floating_maximum_residual:",
        fit[
            "floating_maximum_residual"
        ],
    )
    print(
        "rational_maximum_residual:",
        fit[
            "rational_maximum_residual"
        ],
    )
    print(
        "maximum_probe_residual:",
        maximum_probe_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", BASIS_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
