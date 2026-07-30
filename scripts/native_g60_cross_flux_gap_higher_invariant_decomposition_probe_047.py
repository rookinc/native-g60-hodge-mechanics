from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_exact_sos_reconstruction_046.json"
)

CANONICAL_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_elementary_canonical_certificate_044c.json"
)

ORIENTATION_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_covariant_orientation_035.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_higher_invariant_decomposition_probe_047.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_higher_invariant_decomposition_summary_047.csv"
)

CANDIDATE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_higher_invariant_decomposition_candidates_047.csv"
)

SAMPLE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_higher_invariant_decomposition_samples_047.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_higher_invariant_decomposition_probe_047.npz"
)

TRAINING_SAMPLE_COUNT = 5000
SAVED_SAMPLE_COUNT = 2000

COEFFICIENT_DENOMINATOR_LIMITS = (
    12,
    24,
    60,
    120,
    360,
    1080,
    10000,
)

NONNEGATIVITY_TOLERANCE = 2e-12
RANDOM_SEED = 46047


E1 = {
    "N2": Fraction(5, 12),
}

E2 = {
    "N2^2": Fraction(271, 3456),
    "S4": Fraction(-1, 120),
}

E3 = {
    "N2^3": Fraction(6107, 746496),
    "N2*S4": Fraction(-1, 540),
    "S6": Fraction(-1, 1800),
}

E4 = {
    "N2^4": Fraction(70403, 143327232),
    "N2^2*S4": Fraction(-47, 276480),
    "N2*S6": Fraction(-1, 8640),
    "S4^2": Fraction(1, 57600),
}

E5 = {
    "N2^5": Fraction(40051, 2579890176),
    "N2^3*S4": Fraction(-205, 35831808),
    "N2^2*S6": Fraction(-121, 12441600),
    "N2*S4^2": Fraction(1, 2073600),
    "S4*S6": Fraction(1, 432000),
}


MONOMIAL_EXPONENTS = {
    "N2": (1, 0, 0),
    "N2^2": (2, 0, 0),
    "N2^3": (3, 0, 0),
    "N2^4": (4, 0, 0),
    "N2^5": (5, 0, 0),
    "S4": (0, 1, 0),
    "S6": (0, 0, 1),
    "N2*S4": (1, 1, 0),
    "N2^2*S4": (2, 1, 0),
    "N2^3*S4": (3, 1, 0),
    "N2*S6": (1, 0, 1),
    "N2^2*S6": (2, 0, 1),
    "S4^2": (0, 2, 0),
    "N2*S4^2": (1, 2, 0),
    "S4*S6": (0, 1, 1),
}


EXPONENT_NAMES = {
    value: key
    for key, value in MONOMIAL_EXPONENTS.items()
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


def exponent_dictionary(
    formula: dict[str, Fraction],
) -> dict[tuple[int, int, int], Fraction]:
    return {
        MONOMIAL_EXPONENTS[name]: coefficient
        for name, coefficient in formula.items()
    }


def polynomial_product(
    first: dict[str, Fraction],
    second: dict[str, Fraction],
) -> dict[tuple[int, int, int], Fraction]:
    result: dict[
        tuple[int, int, int],
        Fraction,
    ] = {}

    for first_name, first_value in first.items():
        for second_name, second_value in second.items():
            first_exponent = MONOMIAL_EXPONENTS[
                first_name
            ]

            second_exponent = MONOMIAL_EXPONENTS[
                second_name
            ]

            exponent = tuple(
                first_exponent[index]
                + second_exponent[index]
                for index in range(3)
            )

            result[exponent] = (
                result.get(
                    exponent,
                    Fraction(0, 1),
                )
                + first_value * second_value
            )

    return {
        exponent: coefficient
        for exponent, coefficient in result.items()
        if coefficient != 0
    }


def polynomial_linear_combination(
    terms: list[
        tuple[
            Fraction,
            dict[tuple[int, int, int], Fraction],
        ]
    ],
) -> dict[tuple[int, int, int], Fraction]:
    result: dict[
        tuple[int, int, int],
        Fraction,
    ] = {}

    for scale, polynomial in terms:
        for exponent, coefficient in polynomial.items():
            result[exponent] = (
                result.get(
                    exponent,
                    Fraction(0, 1),
                )
                + scale * coefficient
            )

    return {
        exponent: coefficient
        for exponent, coefficient in result.items()
        if coefficient != 0
    }


def formula_text(
    polynomial: dict[
        tuple[int, int, int],
        Fraction,
    ],
) -> str:
    terms = []

    for exponent in sorted(
        polynomial,
        reverse=True,
    ):
        coefficient = polynomial[exponent]

        name = EXPONENT_NAMES.get(
            exponent,
            (
                f"N2^{exponent[0]}"
                f"*S4^{exponent[1]}"
                f"*S6^{exponent[2]}"
            ),
        )

        terms.append(
            f"({coefficient})*{name}"
        )

    return " + ".join(terms) or "0"


def invariant_values(
    s4: np.ndarray,
    s6: np.ndarray,
) -> dict[str, np.ndarray]:
    ones = np.ones_like(
        s4,
        dtype=np.float64,
    )

    e1 = (
        5.0 / 12.0
        * ones
    )

    e2 = (
        271.0 / 3456.0
        - s4 / 120.0
    )

    e3 = (
        6107.0 / 746496.0
        - s4 / 540.0
        - s6 / 1800.0
    )

    e4 = (
        70403.0 / 143327232.0
        - 47.0
        * s4
        / 276480.0
        - s6 / 8640.0
        + s4**2 / 57600.0
    )

    e5 = (
        40051.0 / 2579890176.0
        - 205.0
        * s4
        / 35831808.0
        - 121.0
        * s6
        / 12441600.0
        + s4**2 / 2073600.0
        + s4 * s6 / 432000.0
    )

    return {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "e5": e5,
    }


def select_training_indices(
    target: np.ndarray,
    count: int,
) -> np.ndarray:
    if len(target) <= count:
        return np.arange(
            len(target),
            dtype=np.int64,
        )

    sorted_indices = np.argsort(
        target
    )

    positions = np.linspace(
        0,
        len(target) - 1,
        count,
        dtype=np.int64,
    )

    return sorted_indices[
        positions
    ]


def solve_capture_lp(
    target: np.ndarray,
    carriers: np.ndarray,
    training_indices: np.ndarray,
) -> dict:
    training_target = target[
        training_indices
    ]

    training_carriers = carriers[
        training_indices
    ]

    average_carrier = np.mean(
        training_carriers,
        axis=0,
    )

    result = linprog(
        c=-average_carrier,
        A_ub=training_carriers,
        b_ub=training_target,
        bounds=[
            (0.0, None)
            for _ in range(
                carriers.shape[1]
            )
        ],
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "linear program failed: "
            f"{result.message}"
        )

    coefficients = np.array(
        result.x,
        dtype=np.float64,
    )

    residual = (
        target
        - carriers @ coefficients
    )

    return {
        "coefficients": coefficients,
        "residual": residual,
        "minimum_residual": float(
            np.min(residual)
        ),
        "mean_residual": float(
            np.mean(residual)
        ),
        "captured_mean": float(
            np.mean(
                carriers @ coefficients
            )
        ),
        "training_minimum_residual": float(
            np.min(
                training_target
                - training_carriers
                @ coefficients
            )
        ),
        "status": int(
            result.status
        ),
        "message": result.message,
    }


def rational_candidates(
    floating_coefficients: np.ndarray,
) -> list[
    tuple[
        int,
        tuple[Fraction, ...],
    ]
]:
    candidates = []

    for denominator_limit in (
        COEFFICIENT_DENOMINATOR_LIMITS
    ):
        coefficients = tuple(
            Fraction(float(value))
            .limit_denominator(
                denominator_limit
            )
            for value in floating_coefficients
        )

        candidates.append(
            (
                denominator_limit,
                coefficients,
            )
        )

    candidates.append(
        (
            0,
            tuple(
                Fraction(0, 1)
                for _ in floating_coefficients
            ),
        )
    )

    unique = {}

    for denominator_limit, coefficients in candidates:
        unique[coefficients] = (
            denominator_limit,
            coefficients,
        )

    return list(
        unique.values()
    )


def score_rational_candidates(
    target: np.ndarray,
    carriers: np.ndarray,
    floating_coefficients: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []

    for (
        denominator_limit,
        coefficients,
    ) in rational_candidates(
        floating_coefficients
    ):
        values = np.array(
            [
                float(value)
                for value in coefficients
            ],
            dtype=np.float64,
        )

        residual = (
            target
            - carriers @ values
        )

        row = {
            "denominator_limit": (
                denominator_limit
            ),
            "coefficient_0": str(
                coefficients[0]
            ),
            "coefficient_1": str(
                coefficients[1]
            ),
            "minimum_residual": float(
                np.min(residual)
            ),
            "maximum_residual": float(
                np.max(residual)
            ),
            "mean_residual": float(
                np.mean(residual)
            ),
            "captured_mean": float(
                np.mean(
                    carriers @ values
                )
            ),
            "negative_residual_count": int(
                np.count_nonzero(
                    residual
                    < -NONNEGATIVITY_TOLERANCE
                )
            ),
            "sample_nonnegative": bool(
                np.min(residual)
                >= -NONNEGATIVITY_TOLERANCE
            ),
            "coefficients": coefficients,
            "values": values,
            "residual": residual,
        }

        rows.append(row)

    viable = [
        row
        for row in rows
        if row["sample_nonnegative"]
    ]

    pool = (
        viable
        if viable
        else rows
    )

    best = max(
        pool,
        key=lambda row: (
            row["captured_mean"],
            row["minimum_residual"],
            -row[
                "denominator_limit"
            ],
        ),
    )

    return rows, best


def residual_formula(
    target_formula: dict[str, Fraction],
    carrier_formulas: list[
        dict[
            tuple[int, int, int],
            Fraction,
        ]
    ],
    coefficients: tuple[Fraction, ...],
) -> dict[
    tuple[int, int, int],
    Fraction,
]:
    target = exponent_dictionary(
        target_formula
    )

    terms = [
        (
            Fraction(1, 1),
            target,
        )
    ]

    for coefficient, carrier in zip(
        coefficients,
        carrier_formulas,
    ):
        terms.append(
            (
                -coefficient,
                carrier,
            )
        )

    return polynomial_linear_combination(
        terms
    )


def run_decomposition(
    label: str,
    target: np.ndarray,
    carriers: np.ndarray,
    target_formula: dict[str, Fraction],
    carrier_formulas: list[
        dict[
            tuple[int, int, int],
            Fraction,
        ]
    ],
    carrier_names: tuple[str, str],
) -> dict:
    training_indices = (
        select_training_indices(
            target,
            TRAINING_SAMPLE_COUNT,
        )
    )

    floating = solve_capture_lp(
        target,
        carriers,
        training_indices,
    )

    candidate_rows, best = (
        score_rational_candidates(
            target,
            carriers,
            floating[
                "coefficients"
            ],
        )
    )

    residual_polynomial = (
        residual_formula(
            target_formula,
            carrier_formulas,
            best["coefficients"],
        )
    )

    support_size = len(
        residual_polynomial
    )

    original_support_size = len(
        exponent_dictionary(
            target_formula
        )
    )

    print(
        "decomposition_result:",
        label,
        "floating_coefficients:",
        floating[
            "coefficients"
        ].tolist(),
        "rational_coefficients:",
        [
            str(value)
            for value in best[
                "coefficients"
            ]
        ],
        "minimum_residual:",
        best["minimum_residual"],
        "negative_count:",
        best[
            "negative_residual_count"
        ],
        "support:",
        f"{support_size}/{original_support_size}",
        flush=True,
    )

    return {
        "label": label,
        "carrier_names": (
            carrier_names
        ),
        "training_indices": (
            training_indices
        ),
        "floating": floating,
        "candidate_rows": (
            candidate_rows
        ),
        "best": best,
        "residual_polynomial": (
            residual_polynomial
        ),
        "residual_formula": (
            formula_text(
                residual_polynomial
            )
        ),
        "support_size": (
            support_size
        ),
        "original_support_size": (
            original_support_size
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANDIDATE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_CSV_OUT.parent.mkdir(
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

    canonical_receipt = json.loads(
        CANONICAL_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    directions = np.array(
        orientation_data[
            "random_directions"
        ],
        dtype=np.float64,
    )

    s4 = np.array(
        orientation_data["s4"],
        dtype=np.float64,
    )

    s6 = np.array(
        orientation_data["s6"],
        dtype=np.float64,
    )

    values = invariant_values(
        s4,
        s6,
    )

    e2_squared_formula = polynomial_product(
        E2,
        E2,
    )

    e1_e3_formula = polynomial_product(
        E1,
        E3,
    )

    e2_e3_formula = polynomial_product(
        E2,
        E3,
    )

    e1_e4_formula = polynomial_product(
        E1,
        E4,
    )

    result_e4 = run_decomposition(
        "e4",
        values["e4"],
        np.column_stack(
            [
                values["e2"] ** 2,
                values["e1"]
                * values["e3"],
            ]
        ),
        E4,
        [
            e2_squared_formula,
            e1_e3_formula,
        ],
        (
            "e2^2",
            "e1*e3",
        ),
    )

    result_e5 = run_decomposition(
        "e5",
        values["e5"],
        np.column_stack(
            [
                values["e2"]
                * values["e3"],
                values["e1"]
                * values["e4"],
            ]
        ),
        E5,
        [
            e2_e3_formula,
            e1_e4_formula,
        ],
        (
            "e2*e3",
            "e1*e4",
        ),
    )

    results = [
        result_e4,
        result_e5,
    ]

    summary_rows = []
    candidate_rows = []

    for result in results:
        best = result["best"]

        summary_rows.append(
            {
                "target": result["label"],
                "carrier_0": (
                    result[
                        "carrier_names"
                    ][0]
                ),
                "carrier_1": (
                    result[
                        "carrier_names"
                    ][1]
                ),
                "floating_coefficient_0": (
                    result[
                        "floating"
                    ][
                        "coefficients"
                    ][0]
                ),
                "floating_coefficient_1": (
                    result[
                        "floating"
                    ][
                        "coefficients"
                    ][1]
                ),
                "rational_coefficient_0": str(
                    best[
                        "coefficients"
                    ][0]
                ),
                "rational_coefficient_1": str(
                    best[
                        "coefficients"
                    ][1]
                ),
                "minimum_residual": (
                    best[
                        "minimum_residual"
                    ]
                ),
                "maximum_residual": (
                    best[
                        "maximum_residual"
                    ]
                ),
                "mean_residual": (
                    best["mean_residual"]
                ),
                "negative_residual_count": (
                    best[
                        "negative_residual_count"
                    ]
                ),
                "sample_nonnegative": (
                    best[
                        "sample_nonnegative"
                    ]
                ),
                "original_support_size": (
                    result[
                        "original_support_size"
                    ]
                ),
                "residual_support_size": (
                    result[
                        "support_size"
                    ]
                ),
                "residual_formula": (
                    result[
                        "residual_formula"
                    ]
                ),
            }
        )

        for row in result[
            "candidate_rows"
        ]:
            candidate_rows.append(
                {
                    "target": (
                        result["label"]
                    ),
                    "denominator_limit": (
                        row[
                            "denominator_limit"
                        ]
                    ),
                    "coefficient_0": (
                        row[
                            "coefficient_0"
                        ]
                    ),
                    "coefficient_1": (
                        row[
                            "coefficient_1"
                        ]
                    ),
                    "minimum_residual": (
                        row[
                            "minimum_residual"
                        ]
                    ),
                    "maximum_residual": (
                        row[
                            "maximum_residual"
                        ]
                    ),
                    "mean_residual": (
                        row[
                            "mean_residual"
                        ]
                    ),
                    "captured_mean": (
                        row[
                            "captured_mean"
                        ]
                    ),
                    "negative_residual_count": (
                        row[
                            "negative_residual_count"
                        ]
                    ),
                    "sample_nonnegative": (
                        row[
                            "sample_nonnegative"
                        ]
                    ),
                }
            )

    saved_indices = np.linspace(
        0,
        len(directions) - 1,
        SAVED_SAMPLE_COUNT,
        dtype=np.int64,
    )

    sample_rows = []

    for sample_id, source_index in enumerate(
        saved_indices
    ):
        sample_rows.append(
            {
                "sample_id": sample_id,
                "source_index": int(
                    source_index
                ),
                "s4": float(
                    s4[source_index]
                ),
                "s6": float(
                    s6[source_index]
                ),
                "e4": float(
                    values["e4"][
                        source_index
                    ]
                ),
                "e4_residual": float(
                    result_e4[
                        "best"
                    ][
                        "residual"
                    ][source_index]
                ),
                "e5": float(
                    values["e5"][
                        source_index
                    ]
                ),
                "e5_residual": float(
                    result_e5[
                        "best"
                    ][
                        "residual"
                    ][source_index]
                ),
            }
        )

    checks = {
        "input_046_theorem_pass": (
            source_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "input_044c_theorem_pass": (
            canonical_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "stored_sample_count_is_50000": (
            len(directions) == 50000
        ),
        "e4_decomposition_completed": (
            result_e4 is not None
        ),
        "e5_decomposition_completed": (
            result_e5 is not None
        ),
        "rational_coefficients_nonnegative": all(
            coefficient >= 0
            for result in results
            for coefficient in result[
                "best"
            ]["coefficients"]
        ),
    }

    audit_pass = all(
        checks.values()
    )

    sample_nonnegative_count = sum(
        bool(
            result["best"]["sample_nonnegative"]
            and any(
                coefficient > 0
                for coefficient in result[
                    "best"
                ]["coefficients"]
            )
        )
        for result in results
    )

    theorem_pass = False

    if (
        audit_pass
        and sample_nonnegative_count == 2
    ):
        verdict = (
            "sample_nonnegative_product_decompositions_found_for_e4_and_e5"
        )
    elif (
        audit_pass
        and sample_nonnegative_count == 1
    ):
        verdict = (
            "sample_nonnegative_product_decomposition_found_for_one_higher_invariant"
        )
    elif audit_pass:
        verdict = (
            "no_nontrivial_sample_nonnegative_product_decomposition_found"
        )
    else:
        verdict = (
            "higher_invariant_decomposition_probe_failed"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_higher_invariant_decomposition_probe_047"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "sample_count": len(
            directions
        ),
        "training_sample_count": (
            TRAINING_SAMPLE_COUNT
        ),
        "sample_nonnegative_decomposition_count": (
            sample_nonnegative_count
        ),
        "results": summary_rows,
        "checks": checks,
        "earned_interpretation": {
            "e4_product_decomposition_sample_nonnegative": (
                result_e4["best"]["sample_nonnegative"]
                and any(
                    coefficient > 0
                    for coefficient in result_e4[
                        "best"
                    ]["coefficients"]
                )
            ),
            "e5_product_decomposition_sample_nonnegative": (
                result_e5["best"]["sample_nonnegative"]
                and any(
                    coefficient > 0
                    for coefficient in result_e5[
                        "best"
                    ]["coefficients"]
                )
            ),
            "residual_formulas_exact_in_native_moment_coordinates": (
                True
            ),
            "residual_global_nonnegativity_proved": (
                False
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "termux_economical_probe": (
                True
            ),
            "sample_nonnegativity_only": (
                True
            ),
            "no_residual_sos_certificate": (
                True
            ),
            "m3_pro_completion_check_deferred": (
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
            "candidate_csv": str(
                CANDIDATE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "sample_csv": str(
                SAMPLE_CSV_OUT.relative_to(
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
            SUMMARY_CSV_OUT,
            summary_rows,
        ),
        (
            CANDIDATE_CSV_OUT,
            candidate_rows,
        ),
        (
            SAMPLE_CSV_OUT,
            sample_rows,
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
        directions=directions,
        s4=s4,
        s6=s6,
        e4=values["e4"],
        e5=values["e5"],
        e4_residual=(
            result_e4[
                "best"
            ]["residual"]
        ),
        e5_residual=(
            result_e5[
                "best"
            ]["residual"]
        ),
        e4_coefficients=np.array(
            [
                float(value)
                for value in result_e4[
                    "best"
                ]["coefficients"]
            ],
            dtype=np.float64,
        ),
        e5_coefficients=np.array(
            [
                float(value)
                for value in result_e5[
                    "best"
                ]["coefficients"]
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "sample_nonnegative_decomposition_count:",
        sample_nonnegative_count,
    )

    for row in summary_rows:
        print(
            row["target"],
            "coefficients:",
            [
                row[
                    "rational_coefficient_0"
                ],
                row[
                    "rational_coefficient_1"
                ],
            ],
            "minimum_residual:",
            row[
                "minimum_residual"
            ],
            "negative_count:",
            row[
                "negative_residual_count"
            ],
            "support:",
            (
                row[
                    "residual_support_size"
                ],
                row[
                    "original_support_size"
                ],
            ),
        )
        print(
            row["target"],
            "residual_formula:",
            row[
                "residual_formula"
            ],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)
    print("wrote:", CANDIDATE_CSV_OUT)
    print("wrote:", SAMPLE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
