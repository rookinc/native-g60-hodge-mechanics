from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

COVARIANT_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_register_covariant_033.json"
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
    / "native_g60_cross_flux_covariant_identity_034.json"
)

IDENTITY_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_identities_034.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_identity_coefficients_034.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_identity_probes_034.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_covariant_identity_034.npz"
)

COEFFICIENT_TOLERANCE = 2e-9
PROBE_TOLERANCE = 2e-9

RATIONAL_DENOMINATOR_LIMIT = 1000000

RANDOM_SEED = 46034
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


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0.0:
        raise RuntimeError(
            "cannot normalize zero vector"
        )

    return vector / norm


def rationalize(value: float) -> Fraction:
    return Fraction(
        float(value)
    ).limit_denominator(
        RATIONAL_DENOMINATOR_LIMIT
    )


def rational_string(value: float) -> str:
    return str(
        rationalize(value)
    )


def exponent_add(
    first: Exponent,
    second: Exponent,
) -> Exponent:
    return tuple(
        first[index] + second[index]
        for index in range(4)
    )


def degree_exponents(
    degree: int,
) -> list[Exponent]:
    values = []

    for exponent in product(
        range(degree + 1),
        repeat=4,
    ):
        if sum(exponent) == degree:
            values.append(
                tuple(
                    int(value)
                    for value in exponent
                )
            )

    values.sort(reverse=True)

    return values


def monomial_label(
    exponent: Exponent,
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


def polynomial_add(
    first: Polynomial,
    second: Polynomial,
    second_scale: float = 1.0,
) -> Polynomial:
    result = defaultdict(float)

    for exponent, value in first.items():
        result[exponent] += value

    for exponent, value in second.items():
        result[exponent] += (
            second_scale * value
        )

    return {
        exponent: value
        for exponent, value in result.items()
        if abs(value) > 1e-16
    }


def polynomial_scale(
    polynomial: Polynomial,
    scale: float,
) -> Polynomial:
    return {
        exponent: scale * value
        for exponent, value in (
            polynomial.items()
        )
    }


def polynomial_multiply(
    first: Polynomial,
    second: Polynomial,
) -> Polynomial:
    result = defaultdict(float)

    for first_exponent, first_value in (
        first.items()
    ):
        for second_exponent, second_value in (
            second.items()
        ):
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


def linear_form(
    coefficients: np.ndarray,
) -> Polynomial:
    result = {}

    for index, value in enumerate(
        coefficients
    ):
        exponent = [0, 0, 0, 0]
        exponent[index] = 1

        result[
            tuple(exponent)
        ] = float(value)

    return result


def polynomial_vector(
    polynomial: Polynomial,
    exponents: list[Exponent],
) -> np.ndarray:
    return np.array(
        [
            polynomial.get(
                exponent,
                0.0,
            )
            for exponent in exponents
        ],
        dtype=np.float64,
    )


def evaluate_polynomial(
    polynomial: Polynomial,
    point: np.ndarray,
) -> float:
    total = 0.0

    for exponent, coefficient in (
        polynomial.items()
    ):
        term = coefficient

        for index, power in enumerate(
            exponent
        ):
            term *= point[index] ** power

        total += term

    return float(total)


def construct_native_polynomials(
    axis_lines: np.ndarray,
) -> dict[str, Polynomial]:
    line_forms = [
        linear_form(axis)
        for axis in axis_lines
    ]

    norm_squared: Polynomial = {
        (2, 0, 0, 0): 1.0,
        (0, 2, 0, 0): 1.0,
        (0, 0, 2, 0): 1.0,
        (0, 0, 0, 2): 1.0,
    }

    norm_fourth = polynomial_power(
        norm_squared,
        2,
    )

    norm_sixth = polynomial_power(
        norm_squared,
        3,
    )

    s4: Polynomial = {}
    s6: Polynomial = {}

    line_squares = []
    line_cubes = []

    for form in line_forms:
        square = polynomial_power(
            form,
            2,
        )

        cube = polynomial_power(
            form,
            3,
        )

        fourth = polynomial_power(
            form,
            4,
        )

        sixth = polynomial_power(
            form,
            6,
        )

        line_squares.append(square)
        line_cubes.append(cube)

        s4 = polynomial_add(
            s4,
            fourth,
        )

        s6 = polynomial_add(
            s6,
            sixth,
        )

    trace_c_squared: Polynomial = {}
    directional_c_squared: Polynomial = {}

    gram = (
        axis_lines
        @ axis_lines.T
    )

    for first in range(
        len(axis_lines)
    ):
        for second in range(
            len(axis_lines)
        ):
            squared_weight_product = (
                polynomial_multiply(
                    line_squares[first],
                    line_squares[second],
                )
            )

            trace_c_squared = polynomial_add(
                trace_c_squared,
                squared_weight_product,
                second_scale=(
                    gram[first, second] ** 2
                ),
            )

            cubic_weight_product = (
                polynomial_multiply(
                    line_cubes[first],
                    line_cubes[second],
                )
            )

            directional_c_squared = (
                polynomial_add(
                    directional_c_squared,
                    cubic_weight_product,
                    second_scale=(
                        gram[first, second]
                    ),
                )
            )

    norm_squared_times_s4 = (
        polynomial_multiply(
            norm_squared,
            s4,
        )
    )

    return {
        "norm_squared": norm_squared,
        "norm_fourth": norm_fourth,
        "norm_sixth": norm_sixth,
        "s4": s4,
        "s6": s6,
        "norm_squared_times_s4": (
            norm_squared_times_s4
        ),
        "trace_c_squared": (
            trace_c_squared
        ),
        "directional_c_squared": (
            directional_c_squared
        ),
    }


def solve_polynomial_identity(
    identity_id: str,
    target: Polynomial,
    basis: list[tuple[str, Polynomial]],
    degree: int,
) -> dict:
    exponents = degree_exponents(
        degree
    )

    target_vector = polynomial_vector(
        target,
        exponents,
    )

    design = np.column_stack(
        [
            polynomial_vector(
                polynomial,
                exponents,
            )
            for _, polynomial in basis
        ]
    )

    coefficients, _, rank, singular_values = (
        np.linalg.lstsq(
            design,
            target_vector,
            rcond=None,
        )
    )

    rational_coefficients = [
        rationalize(value)
        for value in coefficients
    ]

    rational_float_coefficients = np.array(
        [
            float(value)
            for value in rational_coefficients
        ],
        dtype=np.float64,
    )

    reconstructed = (
        design
        @ rational_float_coefficients
    )

    residual = (
        target_vector
        - reconstructed
    )

    floating_prediction = (
        design @ coefficients
    )

    floating_residual = (
        target_vector
        - floating_prediction
    )

    return {
        "identity_id": identity_id,
        "degree": degree,
        "monomial_count": len(
            exponents
        ),
        "basis_names": [
            name
            for name, _ in basis
        ],
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
        "rational_float_coefficients": (
            rational_float_coefficients
        ),
        "floating_maximum_residual": (
            max_abs(
                floating_residual
            )
        ),
        "rational_maximum_residual": (
            max_abs(
                residual
            )
        ),
        "exponents": exponents,
        "target_vector": target_vector,
        "reconstructed_vector": (
            reconstructed
        ),
        "residual_vector": residual,
        "identity_pass": (
            max_abs(residual)
            < COEFFICIENT_TOLERANCE
        ),
    }


def identity_formula(
    left: str,
    basis_names: list[str],
    coefficients: list[Fraction],
) -> str:
    terms = []

    for name, coefficient in zip(
        basis_names,
        coefficients,
    ):
        if coefficient == 0:
            continue

        coefficient_text = str(
            coefficient
        )

        terms.append(
            f"({coefficient_text}) {name}"
        )

    return (
        left
        + " = "
        + " + ".join(terms)
    )


def build_coefficient_rows(
    identities: list[dict],
) -> list[dict]:
    rows = []

    for identity in identities:
        for index, exponent in enumerate(
            identity["exponents"]
        ):
            rows.append(
                {
                    "identity_id": (
                        identity[
                            "identity_id"
                        ]
                    ),
                    "degree": (
                        identity["degree"]
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
                    "target_coefficient": float(
                        identity[
                            "target_vector"
                        ][index]
                    ),
                    "reconstructed_coefficient": float(
                        identity[
                            "reconstructed_vector"
                        ][index]
                    ),
                    "residual": float(
                        identity[
                            "residual_vector"
                        ][index]
                    ),
                    "coefficient_pass": (
                        abs(
                            identity[
                                "residual_vector"
                            ][index]
                        )
                        < COEFFICIENT_TOLERANCE
                    ),
                }
            )

    return rows


def probe_identities(
    polynomials: dict[str, Polynomial],
    quartic_identity: dict,
    sextic_identity: dict,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    maximum_quartic_residual = 0.0
    maximum_sextic_residual = 0.0
    maximum_f_c_f_residual = 0.0

    quartic_coefficients = (
        quartic_identity[
            "rational_float_coefficients"
        ]
    )

    sextic_coefficients = (
        sextic_identity[
            "rational_float_coefficients"
        ]
    )

    for probe_id in range(PROBE_COUNT):
        point = normalized(
            rng.normal(size=4)
        )

        norm_fourth = evaluate_polynomial(
            polynomials[
                "norm_fourth"
            ],
            point,
        )

        norm_sixth = evaluate_polynomial(
            polynomials[
                "norm_sixth"
            ],
            point,
        )

        s4 = evaluate_polynomial(
            polynomials["s4"],
            point,
        )

        s6 = evaluate_polynomial(
            polynomials["s6"],
            point,
        )

        trace_c_squared = (
            evaluate_polynomial(
                polynomials[
                    "trace_c_squared"
                ],
                point,
            )
        )

        directional_c_squared = (
            evaluate_polynomial(
                polynomials[
                    "directional_c_squared"
                ],
                point,
            )
        )

        norm_squared_times_s4 = (
            evaluate_polynomial(
                polynomials[
                    "norm_squared_times_s4"
                ],
                point,
            )
        )

        quartic_prediction = (
            quartic_coefficients[0]
            * norm_fourth
            + quartic_coefficients[1]
            * s4
        )

        sextic_prediction = (
            sextic_coefficients[0]
            * norm_sixth
            + sextic_coefficients[1]
            * norm_squared_times_s4
            + sextic_coefficients[2]
            * s6
        )

        quartic_residual = abs(
            trace_c_squared
            - quartic_prediction
        )

        sextic_residual = abs(
            directional_c_squared
            - sextic_prediction
        )

        axis_lines = polynomials[
            "_axis_lines"
        ]

        overlaps = (
            axis_lines @ point
        )

        covariant = np.einsum(
            "i,ia,ib->ab",
            overlaps**2,
            axis_lines,
            axis_lines,
        )

        direct_f_c_f = float(
            point
            @ covariant
            @ point
        )

        f_c_f_residual = abs(
            direct_f_c_f - s4
        )

        maximum_quartic_residual = max(
            maximum_quartic_residual,
            quartic_residual,
        )

        maximum_sextic_residual = max(
            maximum_sextic_residual,
            sextic_residual,
        )

        maximum_f_c_f_residual = max(
            maximum_f_c_f_residual,
            f_c_f_residual,
        )

        if probe_id < 1024:
            rows.append(
                {
                    "probe_id": probe_id,
                    "trace_c_squared": (
                        trace_c_squared
                    ),
                    "quartic_prediction": (
                        quartic_prediction
                    ),
                    "quartic_residual": (
                        quartic_residual
                    ),
                    "directional_c_squared": (
                        directional_c_squared
                    ),
                    "sextic_prediction": (
                        sextic_prediction
                    ),
                    "sextic_residual": (
                        sextic_residual
                    ),
                    "f_transpose_c_f": (
                        direct_f_c_f
                    ),
                    "s4": s4,
                    "f_transpose_c_f_residual": (
                        f_c_f_residual
                    ),
                }
            )

    summary = {
        "probe_count": PROBE_COUNT,
        "maximum_quartic_residual": (
            maximum_quartic_residual
        ),
        "maximum_sextic_residual": (
            maximum_sextic_residual
        ),
        "maximum_f_transpose_c_f_residual": (
            maximum_f_c_f_residual
        ),
        "all_probes_pass": (
            maximum_quartic_residual
            < PROBE_TOLERANCE
            and maximum_sextic_residual
            < PROBE_TOLERANCE
            and maximum_f_c_f_residual
            < PROBE_TOLERANCE
        ),
    }

    return rows, summary


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    IDENTITY_CSV_OUT.parent.mkdir(
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

    covariant_receipt = json.loads(
        COVARIANT_JSON_PATH.read_text(
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

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis line shape: {axis_lines.shape}"
        )

    polynomials = (
        construct_native_polynomials(
            axis_lines
        )
    )

    polynomials["_axis_lines"] = (
        axis_lines
    )

    quartic_identity = (
        solve_polynomial_identity(
            "trace_c_squared",
            polynomials[
                "trace_c_squared"
            ],
            [
                (
                    "norm_fourth",
                    polynomials[
                        "norm_fourth"
                    ],
                ),
                (
                    "S4",
                    polynomials["s4"],
                ),
            ],
            degree=4,
        )
    )

    sextic_identity = (
        solve_polynomial_identity(
            "f_transpose_c_squared_f",
            polynomials[
                "directional_c_squared"
            ],
            [
                (
                    "norm_sixth",
                    polynomials[
                        "norm_sixth"
                    ],
                ),
                (
                    "norm_squared_times_S4",
                    polynomials[
                        "norm_squared_times_s4"
                    ],
                ),
                (
                    "S6",
                    polynomials["s6"],
                ),
            ],
            degree=6,
        )
    )

    identities = [
        quartic_identity,
        sextic_identity,
    ]

    coefficient_rows = (
        build_coefficient_rows(
            identities
        )
    )

    probe_rows, probe_summary = (
        probe_identities(
            polynomials,
            quartic_identity,
            sextic_identity,
        )
    )

    identity_rows = []

    for identity in identities:
        formula = identity_formula(
            (
                "tr(C(f)^2)"
                if identity[
                    "identity_id"
                ]
                == "trace_c_squared"
                else "f^T C(f)^2 f"
            ),
            identity["basis_names"],
            identity[
                "rational_coefficients"
            ],
        )

        identity_rows.append(
            {
                "identity_id": (
                    identity[
                        "identity_id"
                    ]
                ),
                "degree": (
                    identity["degree"]
                ),
                "formula": formula,
                "basis_names": json.dumps(
                    identity[
                        "basis_names"
                    ]
                ),
                "floating_coefficients": json.dumps(
                    identity[
                        "floating_coefficients"
                    ].tolist()
                ),
                "rational_coefficients": json.dumps(
                    [
                        str(value)
                        for value in identity[
                            "rational_coefficients"
                        ]
                    ]
                ),
                "design_rank": (
                    identity[
                        "design_rank"
                    ]
                ),
                "monomial_count": (
                    identity[
                        "monomial_count"
                    ]
                ),
                "floating_maximum_residual": (
                    identity[
                        "floating_maximum_residual"
                    ]
                ),
                "rational_maximum_residual": (
                    identity[
                        "rational_maximum_residual"
                    ]
                ),
                "identity_pass": (
                    identity[
                        "identity_pass"
                    ]
                ),
            }
        )

    checks = {
        "input_033_audit_pass": (
            covariant_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "axis_register_shape_is_10_by_4": (
            axis_lines.shape
            == (10, 4)
        ),
        "quartic_identity_coefficientwise_exact": (
            quartic_identity[
                "identity_pass"
            ]
        ),
        "sextic_identity_coefficientwise_exact": (
            sextic_identity[
                "identity_pass"
            ]
        ),
        "quartic_monomial_count_is_35": (
            quartic_identity[
                "monomial_count"
            ]
            == 35
        ),
        "sextic_monomial_count_is_84": (
            sextic_identity[
                "monomial_count"
            ]
            == 84
        ),
        "probe_audit_pass": (
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
        "native_g60_cross_flux_covariant_contraction_identities_exact"
        if theorem_pass
        else "native_g60_cross_flux_covariant_identity_not_resolved"
    )

    quartic_formula = identity_rows[0][
        "formula"
    ]

    sextic_formula = identity_rows[1][
        "formula"
    ]

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_covariant_identity_034"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "definition": {
            "covariant": (
                "C(f)=sum_i <f,q_i>^2 q_i q_i^T"
            ),
            "S4": (
                "S4(f)=sum_i <f,q_i>^4"
            ),
            "S6": (
                "S6(f)=sum_i <f,q_i>^6"
            ),
        },
        "identities": {
            "f_transpose_c_f": (
                "f^T C(f) f = S4(f)"
            ),
            "trace_c_squared": (
                quartic_formula
            ),
            "f_transpose_c_squared_f": (
                sextic_formula
            ),
        },
        "coefficient_certificates": {
            "quartic": {
                "degree": 4,
                "monomial_count": (
                    quartic_identity[
                        "monomial_count"
                    ]
                ),
                "rational_coefficients": [
                    str(value)
                    for value in quartic_identity[
                        "rational_coefficients"
                    ]
                ],
                "maximum_residual": (
                    quartic_identity[
                        "rational_maximum_residual"
                    ]
                ),
            },
            "sextic": {
                "degree": 6,
                "monomial_count": (
                    sextic_identity[
                        "monomial_count"
                    ]
                ),
                "rational_coefficients": [
                    str(value)
                    for value in sextic_identity[
                        "rational_coefficients"
                    ]
                ],
                "maximum_residual": (
                    sextic_identity[
                        "rational_maximum_residual"
                    ]
                ),
            },
        },
        "probe_summary": probe_summary,
        "checks": checks,
        "earned_interpretation": {
            "trace_c_squared_adds_no_information_beyond_s4_on_unit_sphere": (
                theorem_pass
            ),
            "directional_c_squared_reduces_to_scalar_register_moments": (
                theorem_pass
            ),
            "full_covariant_matrix_remains_richer_than_its_scalar_contractions": (
                True
            ),
            "global_operator_norm_bound_proved": (
                False
            ),
        },
        "boundary": {
            "quartic_covariant_identity_proved": (
                theorem_pass
            ),
            "sextic_directional_identity_proved": (
                theorem_pass
            ),
            "full_covariant_determined_by_scalar_moments": (
                False
            ),
            "operator_norm_reconstructed_exactly": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "identity_csv": str(
                IDENTITY_CSV_OUT.relative_to(
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

    with IDENTITY_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                identity_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            identity_rows
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

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        quartic_coefficients=(
            quartic_identity[
                "rational_float_coefficients"
            ]
        ),
        sextic_coefficients=(
            sextic_identity[
                "rational_float_coefficients"
            ]
        ),
        quartic_residual_vector=(
            quartic_identity[
                "residual_vector"
            ]
        ),
        sextic_residual_vector=(
            sextic_identity[
                "residual_vector"
            ]
        ),
        quartic_exponents=np.array(
            quartic_identity[
                "exponents"
            ],
            dtype=np.int64,
        ),
        sextic_exponents=np.array(
            sextic_identity[
                "exponents"
            ],
            dtype=np.int64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "quartic_formula:",
        quartic_formula,
    )
    print(
        "quartic_rational_coefficients:",
        [
            str(value)
            for value in quartic_identity[
                "rational_coefficients"
            ]
        ],
    )
    print(
        "quartic_coefficient_residual:",
        quartic_identity[
            "rational_maximum_residual"
        ],
    )
    print(
        "sextic_formula:",
        sextic_formula,
    )
    print(
        "sextic_rational_coefficients:",
        [
            str(value)
            for value in sextic_identity[
                "rational_coefficients"
            ]
        ],
    )
    print(
        "sextic_coefficient_residual:",
        sextic_identity[
            "rational_maximum_residual"
        ],
    )
    print(
        "probe_summary:",
        probe_summary,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", IDENTITY_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
