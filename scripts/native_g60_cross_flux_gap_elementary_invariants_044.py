from __future__ import annotations

import csv
import json
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DETERMINANT_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_determinant_scaled_reconstruction_043c.json"
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

GAP_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_one_third_gap_probe_042.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_elementary_invariants_044.json"
)

INVARIANT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_elementary_invariants_044.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_elementary_coefficients_044.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_elementary_probes_044.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_elementary_invariants_044.npz"
)

RATIO_DENOMINATOR_LIMITS = (
    1_000,
    10_000,
    100_000,
    1_000_000,
)

SCALE_DENOMINATOR_LIMIT = 10_000_000_000_000

COEFFICIENT_TOLERANCE = 5e-10
PROBE_TOLERANCE = 5e-9
PSD_SAMPLE_TOLERANCE = 2e-10

RANDOM_SEED = 46044
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

    return float(np.max(np.abs(array)))


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
        result[exponent] += scale * value

    return dict(result)


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
                values.append(
                    (a, b, c, d)
                )

    values.sort(reverse=True)

    return values


def construct_register_moments(
    axis_lines: np.ndarray,
) -> dict[str, Polynomial]:
    n2: Polynomial = {
        (2, 0, 0, 0): 1.0,
        (0, 2, 0, 0): 1.0,
        (0, 0, 2, 0): 1.0,
        (0, 0, 0, 2): 1.0,
    }

    moments: dict[str, Polynomial] = {
        "N2": n2,
    }

    for power in (
        4,
        6,
        8,
        10,
    ):
        moment: Polynomial = {}

        for axis in axis_lines:
            linear: Polynomial = {}

            for index, value in enumerate(axis):
                exponent = [0, 0, 0, 0]
                exponent[index] = 1

                linear[
                    tuple(exponent)
                ] = float(value)

            moment = polynomial_add(
                moment,
                polynomial_power(
                    linear,
                    power,
                ),
            )

        moments[
            f"S{power}"
        ] = moment

    return moments


def invariant_basis(
    degree: int,
    moments: dict[str, Polynomial],
) -> dict[str, Polynomial]:
    n2 = moments["N2"]
    s4 = moments["S4"]
    s6 = moments["S6"]
    s8 = moments["S8"]
    s10 = moments["S10"]

    if degree == 2:
        return {
            "N2": n2,
        }

    if degree == 4:
        return {
            "N2^2": polynomial_power(
                n2,
                2,
            ),
            "S4": s4,
        }

    if degree == 6:
        return {
            "N2^3": polynomial_power(
                n2,
                3,
            ),
            "N2*S4": polynomial_multiply(
                n2,
                s4,
            ),
            "S6": s6,
        }

    if degree == 8:
        return {
            "N2^4": polynomial_power(
                n2,
                4,
            ),
            "N2^2*S4": polynomial_multiply(
                polynomial_power(
                    n2,
                    2,
                ),
                s4,
            ),
            "N2*S6": polynomial_multiply(
                n2,
                s6,
            ),
            "S4^2": polynomial_power(
                s4,
                2,
            ),
            "S8": s8,
        }

    if degree == 10:
        return {
            "N2^5": polynomial_power(
                n2,
                5,
            ),
            "N2^3*S4": polynomial_multiply(
                polynomial_power(
                    n2,
                    3,
                ),
                s4,
            ),
            "N2^2*S6": polynomial_multiply(
                polynomial_power(
                    n2,
                    2,
                ),
                s6,
            ),
            "N2*S4^2": polynomial_multiply(
                n2,
                polynomial_power(
                    s4,
                    2,
                ),
            ),
            "N2*S8": polynomial_multiply(
                n2,
                s8,
            ),
            "S4*S6": polynomial_multiply(
                s4,
                s6,
            ),
            "S10": s10,
        }

    raise RuntimeError(
        f"unsupported degree: {degree}"
    )


def construct_gap_entries(
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

                    key = tuple(exponent)

                    polynomial[key] = (
                        polynomial.get(
                            key,
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


def principal_minor_polynomial(
    entries: list[list[Polynomial]],
    indices: tuple[int, ...],
) -> Polynomial:
    result: Polynomial = {}

    for permutation in permutations(
        range(len(indices))
    ):
        term: Polynomial = {
            (0, 0, 0, 0): 1.0
        }

        for local_row, local_column in enumerate(
            permutation
        ):
            row = indices[local_row]
            column = indices[local_column]

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

    return result


def elementary_polynomial(
    entries: list[list[Polynomial]],
    degree: int,
) -> Polynomial:
    result: Polynomial = {}

    subsets = list(
        combinations(
            range(6),
            degree,
        )
    )

    for subset_id, subset in enumerate(
        subsets,
        start=1,
    ):
        result = polynomial_add(
            result,
            principal_minor_polynomial(
                entries,
                subset,
            ),
        )

        if (
            subset_id == 1
            or subset_id == len(subsets)
        ):
            print(
                "elementary_progress:",
                f"e{degree}",
                f"{subset_id}/{len(subsets)}",
                "term_count:",
                len(result),
                flush=True,
            )

    return result


def scaled_solution(
    design: np.ndarray,
    target: np.ndarray,
) -> dict:
    column_norms = np.linalg.norm(
        design,
        axis=0,
    )

    scaled_design = (
        design / column_norms
    )

    scaled_coefficients, _, rank, singular_values = (
        np.linalg.lstsq(
            scaled_design,
            target,
            rcond=None,
        )
    )

    coefficients = (
        scaled_coefficients
        / column_norms
    )

    residual = (
        target
        - design @ coefficients
    )

    return {
        "column_norms": column_norms,
        "scaled_design": scaled_design,
        "coefficients": coefficients,
        "rank": int(rank),
        "singular_values": singular_values,
        "condition_number": float(
            singular_values[0]
            / singular_values[-1]
        ),
        "residual": residual,
        "maximum_residual": max_abs(
            residual
        ),
    }


def rational_direction(
    coefficients: np.ndarray,
    anchor: int,
    denominator_limit: int,
) -> np.ndarray | None:
    anchor_value = float(
        coefficients[anchor]
    )

    if abs(anchor_value) < 1e-20:
        return None

    ratios = (
        coefficients / anchor_value
    )

    direction = np.array(
        [
            float(
                Fraction(float(value))
                .limit_denominator(
                    denominator_limit
                )
            )
            if abs(value) > 1e-12
            else 0.0
            for value in ratios
        ],
        dtype=np.float64,
    )

    direction[anchor] = 1.0

    return direction


def optimize_scale(
    design: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
) -> tuple[Fraction, np.ndarray]:
    image = design @ direction

    denominator = float(
        np.dot(
            image,
            image,
        )
    )

    floating_scale = float(
        np.dot(
            image,
            target,
        )
        / denominator
    )

    rational_scale = Fraction(
        floating_scale
    ).limit_denominator(
        SCALE_DENOMINATOR_LIMIT
    )

    candidate = (
        float(rational_scale)
        * direction
    )

    return rational_scale, candidate


def reconstruct_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
) -> dict:
    candidates = []

    for anchor in range(
        len(coefficients)
    ):
        if abs(
            coefficients[anchor]
        ) < 1e-20:
            continue

        for denominator_limit in (
            RATIO_DENOMINATOR_LIMITS
        ):
            direction = rational_direction(
                coefficients,
                anchor,
                denominator_limit,
            )

            if direction is None:
                continue

            rational_scale, candidate = (
                optimize_scale(
                    design,
                    target,
                    direction,
                )
            )

            residual = (
                target
                - design @ candidate
            )

            candidates.append(
                {
                    "anchor": anchor,
                    "denominator_limit": (
                        denominator_limit
                    ),
                    "direction": direction,
                    "scale": rational_scale,
                    "candidate": candidate,
                    "residual": residual,
                    "maximum_residual": (
                        max_abs(residual)
                    ),
                }
            )

    return min(
        candidates,
        key=lambda item: item[
            "maximum_residual"
        ],
    )


def formula_text(
    names: list[str],
    coefficients: np.ndarray,
    scale: Fraction,
    denominator_limit: int,
) -> str:
    terms = []

    scale_fraction = scale

    for name, coefficient in zip(
        names,
        coefficients,
    ):
        if abs(coefficient) < 1e-18:
            continue

        ratio = Fraction(
            float(
                coefficient
                / float(scale_fraction)
            )
        ).limit_denominator(
            denominator_limit
        )

        terms.append(
            f"({scale_fraction * ratio})*{name}"
        )

    return " + ".join(terms) or "0"


def build_candidate_polynomial(
    basis: dict[str, Polynomial],
    names: list[str],
    coefficients: np.ndarray,
) -> Polynomial:
    result: Polynomial = {}

    for name, coefficient in zip(
        names,
        coefficients,
    ):
        result = polynomial_add(
            result,
            basis[name],
            scale=float(coefficient),
        )

    return result


def elementary_values(
    eigenvalues: np.ndarray,
) -> np.ndarray:
    values = np.zeros(
        7,
        dtype=np.float64,
    )

    values[0] = 1.0

    for eigenvalue in eigenvalues:
        for degree in range(
            6,
            0,
            -1,
        ):
            values[degree] += (
                eigenvalue
                * values[
                    degree - 1
                ]
            )

    return values


def direct_gap_eigenvalues(
    slices: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
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

    return np.linalg.eigvalsh(
        gap
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    INVARIANT_CSV_OUT.parent.mkdir(
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

    determinant_receipt = json.loads(
        DETERMINANT_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    gap_data = np.load(
        GAP_NPZ_PATH
    )

    axis_lines = np.array(
        orientation_data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    sampled_gap_eigenvalues = np.array(
        gap_data[
            "sample_gap_eigenvalues"
        ],
        dtype=np.float64,
    )

    moments = construct_register_moments(
        axis_lines
    )

    entries = construct_gap_entries(
        slices
    )

    invariant_rows = []
    coefficient_rows = []
    candidate_polynomials = {}
    exact_coefficient_vectors = {}
    elementary_polynomials = {}

    global_coefficient_residual = 0.0

    for invariant_degree in range(
        1,
        6,
    ):
        polynomial_degree = (
            2 * invariant_degree
        )

        target_polynomial = (
            elementary_polynomial(
                entries,
                invariant_degree,
            )
        )

        elementary_polynomials[
            invariant_degree
        ] = target_polynomial

        exponents = degree_exponents(
            polynomial_degree
        )

        target_vector = polynomial_vector(
            target_polynomial,
            exponents,
        )

        basis = invariant_basis(
            polynomial_degree,
            moments,
        )

        basis_names = list(basis)

        design = np.column_stack(
            [
                polynomial_vector(
                    basis[name],
                    exponents,
                )
                for name in basis_names
            ]
        )

        floating = scaled_solution(
            design,
            target_vector,
        )

        reconstruction = (
            reconstruct_coefficients(
                design,
                target_vector,
                floating[
                    "coefficients"
                ],
            )
        )

        candidate = np.array(
            reconstruction[
                "candidate"
            ],
            dtype=np.float64,
        )

        candidate_polynomial = (
            build_candidate_polynomial(
                basis,
                basis_names,
                candidate,
            )
        )

        candidate_polynomials[
            invariant_degree
        ] = candidate_polynomial

        exact_coefficient_vectors[
            invariant_degree
        ] = candidate

        coefficient_residual = (
            reconstruction[
                "maximum_residual"
            ]
        )

        global_coefficient_residual = max(
            global_coefficient_residual,
            coefficient_residual,
        )

        formula = formula_text(
            basis_names,
            candidate,
            reconstruction["scale"],
            reconstruction[
                "denominator_limit"
            ],
        )

        invariant_pass = (
            floating[
                "maximum_residual"
            ]
            < COEFFICIENT_TOLERANCE
            and coefficient_residual
            < COEFFICIENT_TOLERANCE
        )

        invariant_rows.append(
            {
                "elementary_invariant": (
                    f"e{invariant_degree}"
                ),
                "matrix_order": (
                    invariant_degree
                ),
                "polynomial_degree": (
                    polynomial_degree
                ),
                "monomial_count": len(
                    exponents
                ),
                "basis_names": json.dumps(
                    basis_names
                ),
                "basis_count": len(
                    basis_names
                ),
                "design_rank": (
                    floating["rank"]
                ),
                "condition_number": (
                    floating[
                        "condition_number"
                    ]
                ),
                "floating_maximum_residual": (
                    floating[
                        "maximum_residual"
                    ]
                ),
                "rational_maximum_residual": (
                    coefficient_residual
                ),
                "anchor_basis": (
                    basis_names[
                        reconstruction[
                            "anchor"
                        ]
                    ]
                ),
                "ratio_denominator_limit": (
                    reconstruction[
                        "denominator_limit"
                    ]
                ),
                "common_scale": str(
                    reconstruction["scale"]
                ),
                "formula": formula,
                "invariant_pass": (
                    invariant_pass
                ),
            }
        )

        rational_scale = reconstruction[
            "scale"
        ]

        for basis_index, (
            name,
            floating_value,
            candidate_value,
        ) in enumerate(
            zip(
                basis_names,
                floating[
                    "coefficients"
                ],
                candidate,
            )
        ):
            coefficient_rows.append(
                {
                    "elementary_invariant": (
                        f"e{invariant_degree}"
                    ),
                    "basis_index": (
                        basis_index
                    ),
                    "basis_name": name,
                    "floating_coefficient": (
                        floating_value
                    ),
                    "candidate_coefficient": (
                        candidate_value
                    ),
                    "candidate_rational": str(
                        Fraction(
                            float(
                                candidate_value
                            )
                        ).limit_denominator(
                            SCALE_DENOMINATOR_LIMIT
                        )
                    ),
                    "common_scale": str(
                        rational_scale
                    ),
                }
            )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    probe_rows = []
    maximum_probe_residual = 0.0
    minimum_sampled_elementary = np.full(
        6,
        np.inf,
        dtype=np.float64,
    )

    for probe_id in range(PROBE_COUNT):
        point = rng.normal(size=4)

        eigenvalues = direct_gap_eigenvalues(
            slices,
            point,
        )

        direct_values = elementary_values(
            eigenvalues
        )

        row = {
            "probe_id": probe_id,
            "point_norm_squared": float(
                np.dot(point, point)
            ),
        }

        for invariant_degree in range(
            1,
            6,
        ):
            predicted = polynomial_evaluate(
                candidate_polynomials[
                    invariant_degree
                ],
                point,
            )

            direct = direct_values[
                invariant_degree
            ]

            residual = abs(
                direct - predicted
            )

            maximum_probe_residual = max(
                maximum_probe_residual,
                residual,
            )

            minimum_sampled_elementary[
                invariant_degree
            ] = min(
                minimum_sampled_elementary[
                    invariant_degree
                ],
                direct,
            )

            row[
                f"e{invariant_degree}_direct"
            ] = direct

            row[
                f"e{invariant_degree}_predicted"
            ] = predicted

            row[
                f"e{invariant_degree}_residual"
            ] = residual

        if probe_id < 1024:
            probe_rows.append(row)

    stored_minima = np.min(
        np.array(
            [
                elementary_values(values)
                for values in (
                    sampled_gap_eigenvalues
                )
            ],
            dtype=np.float64,
        ),
        axis=0,
    )

    checks = {
        "input_043c_theorem_pass": (
            determinant_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "five_elementary_invariants_derived": (
            len(invariant_rows) == 5
        ),
        "all_coefficient_fits_pass": all(
            row["invariant_pass"]
            for row in invariant_rows
        ),
        "global_coefficient_residual_resolved": (
            global_coefficient_residual
            < COEFFICIENT_TOLERANCE
        ),
        "all_direct_probes_pass": (
            maximum_probe_residual
            < PROBE_TOLERANCE
        ),
        "all_stored_sample_elementary_invariants_nonnegative": (
            np.min(
                stored_minima[1:]
            )
            >= -PSD_SAMPLE_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_gap_elementary_invariant_identities_exact"
        if theorem_pass
        else "native_g60_cross_flux_gap_elementary_invariants_not_resolved"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_elementary_invariants_044"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "definition": {
            "gap_matrix": (
                "G(f)=(||f||^2/9)I6-A(f)^T A(f)"
            ),
            "elementary_invariants": (
                "e_k(G) for k=1,...,6"
            ),
            "determinant": (
                "e6 supplied by exact square law from 043c"
            ),
        },
        "invariants": invariant_rows,
        "determinant_square": {
            "formula": (
                "e6=(1/298598400)"
                "*(N2*S4+(24/5)S6-(1655/216)N2^3)^2"
            ),
            "source_artifact": (
                "native_g60_cross_flux_gap_determinant_scaled_reconstruction_043c"
            ),
        },
        "probe_summary": {
            "probe_count": (
                PROBE_COUNT
            ),
            "maximum_probe_residual": (
                maximum_probe_residual
            ),
            "minimum_random_probe_elementary_values": (
                minimum_sampled_elementary
            ),
            "minimum_stored_sample_elementary_values": (
                stored_minima
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "exact_native_moment_formulas_for_e1_through_e5": (
                theorem_pass
            ),
            "determinant_is_exact_square": (
                theorem_pass
            ),
            "all_sampled_elementary_invariants_nonnegative": (
                checks[
                    "all_stored_sample_elementary_invariants_nonnegative"
                ]
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "elementary_invariant_identities_proved": (
                theorem_pass
            ),
            "global_nonnegativity_of_all_formulas_proved": (
                False
            ),
            "global_gap_psd_proved": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "invariant_csv": str(
                INVARIANT_CSV_OUT.relative_to(
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
            "invariant_npz": str(
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
            INVARIANT_CSV_OUT,
            invariant_rows,
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
        elementary_invariant_degrees=np.array(
            [
                2,
                4,
                6,
                8,
                10,
            ],
            dtype=np.int64,
        ),
        minimum_random_probe_elementary=(
            minimum_sampled_elementary
        ),
        minimum_stored_sample_elementary=(
            stored_minima
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)

    for row in invariant_rows:
        print(
            row["elementary_invariant"],
            "formula:",
            row["formula"],
            "condition:",
            row["condition_number"],
            "residual:",
            row[
                "rational_maximum_residual"
            ],
            "pass:",
            row["invariant_pass"],
        )

    print(
        "maximum_probe_residual:",
        maximum_probe_residual,
    )
    print(
        "minimum_stored_sample_elementary:",
        stored_minima.tolist(),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", INVARIANT_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
