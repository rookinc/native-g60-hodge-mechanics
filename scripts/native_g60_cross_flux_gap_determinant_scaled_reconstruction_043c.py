from __future__ import annotations

import csv
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_determinant_coefficient_fit_043b.json"
)

SOURCE_NPZ_PATH = (
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
    / "native_g60_cross_flux_gap_determinant_scaled_reconstruction_043c.json"
)

SEARCH_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_ratio_search_043c.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_scaled_coefficients_043c.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_determinant_scaled_probes_043c.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_determinant_scaled_reconstruction_043c.npz"
)

BASIS_NAMES = (
    "N2^6",
    "N2^4*S4",
    "N2^3*S6",
    "N2^2*S4^2",
    "N2^2*S8",
    "N2*S4*S6",
    "S4^3",
    "S6^2",
)

RATIO_DENOMINATOR_LIMITS = (
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
)

SCALE_DENOMINATOR_LIMIT = 10_000_000_000_000

COEFFICIENT_TOLERANCE = 5e-10
PROBE_TOLERANCE = 5e-9

RANDOM_SEED = 460432
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
            ] += first_value * second_value

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
                tuple(int(value) for value in exponent),
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


def construct_basis(
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
            polynomial_power(linear, 4),
        )

        s6 = polynomial_add(
            s6,
            polynomial_power(linear, 6),
        )

        s8 = polynomial_add(
            s8,
            polynomial_power(linear, 8),
        )

    return {
        "N2^6": polynomial_power(
            n2,
            6,
        ),
        "N2^4*S4": polynomial_multiply(
            polynomial_power(n2, 4),
            s4,
        ),
        "N2^3*S6": polynomial_multiply(
            polynomial_power(n2, 3),
            s6,
        ),
        "N2^2*S4^2": polynomial_multiply(
            polynomial_power(n2, 2),
            polynomial_power(s4, 2),
        ),
        "N2^2*S8": polynomial_multiply(
            polynomial_power(n2, 2),
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


def scaled_lstsq(
    design: np.ndarray,
    target: np.ndarray,
) -> dict:
    column_norms = np.linalg.norm(
        design,
        axis=0,
    )

    if np.any(column_norms == 0.0):
        raise RuntimeError(
            "zero basis column"
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
        "scaled_coefficients": (
            scaled_coefficients
        ),
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


def optimize_common_scale(
    design: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, Fraction, np.ndarray]:
    image = design @ direction

    denominator = float(
        np.dot(image, image)
    )

    if denominator == 0.0:
        raise RuntimeError(
            "zero rational direction image"
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

    return (
        floating_scale,
        rational_scale,
        candidate,
    )


def search_reconstructions(
    design: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []
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

            (
                floating_scale,
                rational_scale,
                candidate,
            ) = optimize_common_scale(
                design,
                target,
                direction,
            )

            residual = (
                target
                - design @ candidate
            )

            maximum_residual = max_abs(
                residual
            )

            row = {
                "anchor_index": anchor,
                "anchor_name": (
                    BASIS_NAMES[anchor]
                ),
                "ratio_denominator_limit": (
                    denominator_limit
                ),
                "floating_scale": (
                    floating_scale
                ),
                "rational_scale": str(
                    rational_scale
                ),
                "rational_direction": (
                    json.dumps(
                        [
                            str(
                                Fraction(float(value))
                                .limit_denominator(
                                    denominator_limit
                                )
                            )
                            for value in direction
                        ]
                    )
                ),
                "maximum_coefficient_residual": (
                    maximum_residual
                ),
                "coefficient_residual_l2": float(
                    np.linalg.norm(
                        residual
                    )
                ),
                "candidate_pass": (
                    maximum_residual
                    < COEFFICIENT_TOLERANCE
                ),
            }

            rows.append(row)

            candidates.append(
                {
                    "row": row,
                    "direction": direction,
                    "candidate": candidate,
                    "rational_scale": (
                        rational_scale
                    ),
                    "residual": residual,
                }
            )

    if not candidates:
        raise RuntimeError(
            "no reconstruction candidates"
        )

    best = min(
        candidates,
        key=lambda item: (
            item["row"][
                "maximum_coefficient_residual"
            ]
        ),
    )

    return rows, best


def build_candidate_polynomial(
    basis: dict[str, Polynomial],
    coefficients: np.ndarray,
) -> Polynomial:
    result: Polynomial = {}

    for name, coefficient in zip(
        BASIS_NAMES,
        coefficients,
    ):
        result = polynomial_add(
            result,
            basis[name],
            scale=float(coefficient),
        )

    return result


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


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEARCH_CSV_OUT.parent.mkdir(
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

    exponents = np.array(
        source_data[
            "degree_twelve_exponents"
        ],
        dtype=np.int64,
    )

    determinant_coefficients = np.array(
        source_data[
            "determinant_coefficients"
        ],
        dtype=np.float64,
    )

    axis_lines = np.array(
        orientation_data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    basis = construct_basis(
        axis_lines
    )

    design = np.column_stack(
        [
            polynomial_vector(
                basis[name],
                exponents,
            )
            for name in BASIS_NAMES
        ]
    )

    scaled_solution = scaled_lstsq(
        design,
        determinant_coefficients,
    )

    search_rows, best = (
        search_reconstructions(
            design,
            determinant_coefficients,
            scaled_solution[
                "coefficients"
            ],
        )
    )

    best_coefficients = np.array(
        best["candidate"],
        dtype=np.float64,
    )

    candidate_polynomial = (
        build_candidate_polynomial(
            basis,
            best_coefficients,
        )
    )

    coefficient_rows = []

    for index, name in enumerate(
        BASIS_NAMES
    ):
        coefficient_rows.append(
            {
                "basis_index": index,
                "basis_name": name,
                "scaled_lstsq_coefficient": float(
                    scaled_solution[
                        "coefficients"
                    ][index]
                ),
                "best_candidate_coefficient": float(
                    best_coefficients[index]
                ),
                "candidate_difference": abs(
                    float(
                        scaled_solution[
                            "coefficients"
                        ][index]
                    )
                    - float(
                        best_coefficients[index]
                    )
                ),
            }
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    probe_rows = []
    maximum_probe_residual = 0.0

    for probe_id in range(PROBE_COUNT):
        point = rng.normal(size=4)

        direct = direct_gap_determinant(
            slices,
            point,
        )

        predicted = polynomial_evaluate(
            candidate_polynomial,
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
                        np.dot(point, point)
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

    coefficient_pass = (
        best["row"][
            "maximum_coefficient_residual"
        ]
        < COEFFICIENT_TOLERANCE
    )

    probe_pass = (
        maximum_probe_residual
        < PROBE_TOLERANCE
    )

    checks = {
        "input_043b_record_present": (
            source_receipt.get(
                "artifact_id"
            )
            == (
                "native_g60_cross_flux_gap_"
                "determinant_coefficient_fit_043b"
            )
        ),
        "design_shape_is_455_by_8": (
            design.shape == (455, 8)
        ),
        "scaled_design_has_full_rank": (
            scaled_solution["rank"] == 8
        ),
        "scaled_floating_fit_is_exact": (
            scaled_solution[
                "maximum_residual"
            ]
            < COEFFICIENT_TOLERANCE
        ),
        "common_scale_rational_candidate_passes_coefficients": (
            coefficient_pass
        ),
        "common_scale_rational_candidate_passes_probes": (
            probe_pass
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    if theorem_pass:
        verdict = (
            "native_g60_cross_flux_gap_determinant_"
            "scaled_rational_reconstruction_exact"
        )
    elif scaled_solution[
        "maximum_residual"
    ] < COEFFICIENT_TOLERANCE:
        verdict = (
            "native_g60_cross_flux_gap_determinant_"
            "scaled_fit_exact_rational_reconstruction_open"
        )
    else:
        verdict = (
            "native_g60_cross_flux_gap_determinant_"
            "scaled_moment_fit_failed"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_determinant_"
            "scaled_reconstruction_043c"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "scaled_fit": {
            "design_shape": list(
                design.shape
            ),
            "rank": (
                scaled_solution["rank"]
            ),
            "condition_number": (
                scaled_solution[
                    "condition_number"
                ]
            ),
            "column_norms": (
                scaled_solution[
                    "column_norms"
                ]
            ),
            "floating_coefficients": (
                scaled_solution[
                    "coefficients"
                ]
            ),
            "maximum_residual": (
                scaled_solution[
                    "maximum_residual"
                ]
            ),
        },
        "rational_search": {
            "candidate_count": len(
                search_rows
            ),
            "best_anchor": (
                best["row"][
                    "anchor_name"
                ]
            ),
            "best_ratio_denominator_limit": (
                best["row"][
                    "ratio_denominator_limit"
                ]
            ),
            "best_rational_scale": str(
                best[
                    "rational_scale"
                ]
            ),
            "best_rational_direction": (
                best["row"][
                    "rational_direction"
                ]
            ),
            "best_coefficients": (
                best_coefficients
            ),
            "maximum_coefficient_residual": (
                best["row"][
                    "maximum_coefficient_residual"
                ]
            ),
        },
        "probe_summary": {
            "probe_count": (
                PROBE_COUNT
            ),
            "maximum_probe_residual": (
                maximum_probe_residual
            ),
            "all_probes_pass": (
                probe_pass
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "column_scaling_resolves_numerical_conditioning": (
                scaled_solution[
                    "maximum_residual"
                ]
                < COEFFICIENT_TOLERANCE
            ),
            "single_common_rational_scale_recovered": (
                theorem_pass
            ),
            "determinant_identity_proved": (
                theorem_pass
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "scaled_coefficient_fit_completed": (
                True
            ),
            "exact_rational_reconstruction_completed": (
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
            "search_csv": str(
                SEARCH_CSV_OUT.relative_to(
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
            SEARCH_CSV_OUT,
            search_rows,
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
                fieldnames=list(rows[0]),
            )

            writer.writeheader()
            writer.writerows(rows)

    np.savez_compressed(
        NPZ_OUT,
        exponents=exponents,
        determinant_coefficients=(
            determinant_coefficients
        ),
        design=design,
        column_norms=(
            scaled_solution[
                "column_norms"
            ]
        ),
        floating_coefficients=(
            scaled_solution[
                "coefficients"
            ]
        ),
        best_candidate_coefficients=(
            best_coefficients
        ),
        best_coefficient_residual=(
            best["residual"]
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "scaled_condition_number:",
        scaled_solution[
            "condition_number"
        ],
    )
    print(
        "scaled_floating_coefficients:",
        scaled_solution[
            "coefficients"
        ].tolist(),
    )
    print(
        "scaled_floating_residual:",
        scaled_solution[
            "maximum_residual"
        ],
    )
    print(
        "best_anchor:",
        best["row"][
            "anchor_name"
        ],
    )
    print(
        "best_ratio_denominator_limit:",
        best["row"][
            "ratio_denominator_limit"
        ],
    )
    print(
        "best_rational_scale:",
        best[
            "rational_scale"
        ],
    )
    print(
        "best_rational_direction:",
        best["row"][
            "rational_direction"
        ],
    )
    print(
        "best_coefficient_residual:",
        best["row"][
            "maximum_coefficient_residual"
        ],
    )
    print(
        "maximum_probe_residual:",
        maximum_probe_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SEARCH_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
