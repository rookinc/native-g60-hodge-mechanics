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
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_gap_elementary_invariants_044.json"
)

ORIENTATION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_covariant_orientation_035.npz"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_gap_elementary_normalization_044b.json"
)

FORMULA_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_gap_elementary_normalized_formulas_044b.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_gap_elementary_normalized_coefficients_044b.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_gap_elementary_normalized_probes_044b.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_gap_elementary_normalization_044b.npz"
)

RATIO_DENOMINATOR_LIMITS = (
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
)

SCALE_DENOMINATOR_LIMIT = 10_000_000_000_000

COEFFICIENT_TOLERANCE = 5e-10
PROBE_TOLERANCE = 5e-9

RANDOM_SEED = 460442
PROBE_COUNT = 4096


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
            if abs(value) > 1e-14
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
) -> tuple[
    Fraction,
    np.ndarray,
    np.ndarray,
]:
    image = design @ direction

    denominator = float(
        np.dot(image, image)
    )

    if denominator == 0.0:
        raise RuntimeError(
            "zero candidate direction"
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

    residual = (
        target
        - design @ candidate
    )

    return (
        rational_scale,
        candidate,
        residual,
    )


def normalized_formula(
    basis_names: list[str],
    candidate: np.ndarray,
    scale: Fraction,
    denominator_limit: int,
) -> str:
    terms = []

    for name, coefficient in zip(
        basis_names,
        candidate,
    ):
        if abs(coefficient) < 1e-18:
            continue

        ratio = Fraction(
            float(
                coefficient
                / float(scale)
            )
        ).limit_denominator(
            denominator_limit
        )

        exact_coefficient = (
            scale * ratio
        )

        terms.append(
            f"({exact_coefficient})*{name}"
        )

    return " + ".join(terms) or "0"


def search_normalization(
    design: np.ndarray,
    target: np.ndarray,
    floating_coefficients: np.ndarray,
    basis_names: list[str],
) -> dict:
    candidates = []

    for anchor in range(
        len(floating_coefficients)
    ):
        if abs(
            floating_coefficients[anchor]
        ) < 1e-20:
            continue

        for denominator_limit in (
            RATIO_DENOMINATOR_LIMITS
        ):
            direction = rational_direction(
                floating_coefficients,
                anchor,
                denominator_limit,
            )

            if direction is None:
                continue

            (
                scale,
                candidate,
                residual,
            ) = optimize_common_scale(
                design,
                target,
                direction,
            )

            candidates.append(
                {
                    "anchor": anchor,
                    "anchor_name": (
                        basis_names[anchor]
                    ),
                    "denominator_limit": (
                        denominator_limit
                    ),
                    "direction": direction,
                    "scale": scale,
                    "candidate": candidate,
                    "residual": residual,
                    "maximum_residual": (
                        max_abs(residual)
                    ),
                    "residual_l2": float(
                        np.linalg.norm(
                            residual
                        )
                    ),
                }
            )

    if not candidates:
        raise RuntimeError(
            "no normalization candidates"
        )

    return min(
        candidates,
        key=lambda item: (
            item["maximum_residual"],
            item["denominator_limit"],
        ),
    )


def candidate_polynomial(
    source,
    basis: dict,
    basis_names: list[str],
    coefficients: np.ndarray,
):
    result = {}

    for name, coefficient in zip(
        basis_names,
        coefficients,
    ):
        result = source.polynomial_add(
            result,
            basis[name],
            scale=float(coefficient),
        )

    return result


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FORMULA_CSV_OUT.parent.mkdir(
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

    formula_rows = []
    coefficient_rows = []

    candidate_polynomials = {}
    candidate_vectors = {}
    coefficient_residual_vectors = {}

    for invariant_degree in range(
        1,
        6,
    ):
        polynomial_degree = (
            2 * invariant_degree
        )

        target_polynomial = (
            source.elementary_polynomial(
                entries,
                invariant_degree,
            )
        )

        exponents = source.degree_exponents(
            polynomial_degree
        )

        target_vector = source.polynomial_vector(
            target_polynomial,
            exponents,
        )

        basis = source.invariant_basis(
            polynomial_degree,
            moments,
        )

        basis_names = list(basis)

        design = np.column_stack(
            [
                source.polynomial_vector(
                    basis[name],
                    exponents,
                )
                for name in basis_names
            ]
        )

        floating = source.scaled_solution(
            design,
            target_vector,
        )

        normalized = search_normalization(
            design,
            target_vector,
            floating["coefficients"],
            basis_names,
        )

        candidate = np.array(
            normalized["candidate"],
            dtype=np.float64,
        )

        formula = normalized_formula(
            basis_names,
            candidate,
            normalized["scale"],
            normalized[
                "denominator_limit"
            ],
        )

        current_polynomial = candidate_polynomial(
            source,
            basis,
            basis_names,
            candidate,
        )

        candidate_polynomials[
            invariant_degree
        ] = current_polynomial

        candidate_vectors[
            invariant_degree
        ] = candidate

        coefficient_residual_vectors[
            invariant_degree
        ] = normalized["residual"]

        invariant_pass = (
            floating["maximum_residual"]
            < COEFFICIENT_TOLERANCE
            and normalized[
                "maximum_residual"
            ]
            < COEFFICIENT_TOLERANCE
        )

        formula_rows.append(
            {
                "elementary_invariant": (
                    f"e{invariant_degree}"
                ),
                "polynomial_degree": (
                    polynomial_degree
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
                "normalized_maximum_residual": (
                    normalized[
                        "maximum_residual"
                    ]
                ),
                "anchor_basis": (
                    normalized[
                        "anchor_name"
                    ]
                ),
                "ratio_denominator_limit": (
                    normalized[
                        "denominator_limit"
                    ]
                ),
                "common_scale": str(
                    normalized["scale"]
                ),
                "formula": formula,
                "invariant_pass": (
                    invariant_pass
                ),
            }
        )

        for basis_index, (
            name,
            floating_value,
            candidate_value,
        ) in enumerate(
            zip(
                basis_names,
                floating["coefficients"],
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
                    "normalized_coefficient": (
                        candidate_value
                    ),
                    "normalized_rational": str(
                        Fraction(
                            float(
                                candidate_value
                            )
                        ).limit_denominator(
                            SCALE_DENOMINATOR_LIMIT
                        )
                    ),
                    "coefficient_difference": abs(
                        float(floating_value)
                        - float(candidate_value)
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

        eigenvalues = (
            source.direct_gap_eigenvalues(
                slices,
                point,
            )
        )

        direct_values = (
            source.elementary_values(
                eigenvalues
            )
        )

        current_row = {
            "probe_id": probe_id,
            "point_norm_squared": float(
                np.dot(point, point)
            ),
        }

        for invariant_degree in range(
            1,
            6,
        ):
            direct = direct_values[
                invariant_degree
            ]

            predicted = (
                source.polynomial_evaluate(
                    candidate_polynomials[
                        invariant_degree
                    ],
                    point,
                )
            )

            residual = abs(
                direct - predicted
            )

            maximum_probe_residual = max(
                maximum_probe_residual,
                residual,
            )

            current_row[
                f"e{invariant_degree}_direct"
            ] = direct

            current_row[
                f"e{invariant_degree}_predicted"
            ] = predicted

            current_row[
                f"e{invariant_degree}_residual"
            ] = residual

        if probe_id < 1024:
            probe_rows.append(
                current_row
            )

    checks = {
        "input_044_theorem_pass": (
            source_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "five_formulas_normalized": (
            len(formula_rows) == 5
        ),
        "all_normalized_coefficient_fits_pass": all(
            row["invariant_pass"]
            for row in formula_rows
        ),
        "all_direct_nonunit_probes_pass": (
            maximum_probe_residual
            < PROBE_TOLERANCE
        ),
        "e3_formula_is_compact": (
            "6107/746496"
            in formula_rows[2]["formula"]
            and "-1/540"
            in formula_rows[2]["formula"]
            and "-1/1800"
            in formula_rows[2]["formula"]
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_gap_elementary_formulas_normalized_exact"
        if theorem_pass
        else "native_g60_cross_flux_gap_elementary_normalization_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_elementary_normalization_044b"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "normalized_formulas": {
            row[
                "elementary_invariant"
            ]: row["formula"]
            for row in formula_rows
        },
        "formula_rows": formula_rows,
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
            "noncanonical_e3_reconstruction_replaced": (
                theorem_pass
            ),
            "all_e1_through_e5_formulas_have_common_scale_normalizations": (
                theorem_pass
            ),
            "elementary_identity_chain_is_ready_for_nonnegativity_analysis": (
                theorem_pass
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "elementary_formula_normalization_completed": (
                theorem_pass
            ),
            "global_nonnegativity_proved": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "formula_csv": str(
                FORMULA_CSV_OUT.relative_to(
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
            "normalization_npz": str(
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
            FORMULA_CSV_OUT,
            formula_rows,
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
        elementary_degrees=np.array(
            [1, 2, 3, 4, 5],
            dtype=np.int64,
        ),
        candidate_coefficient_vectors=np.array(
            [
                candidate_vectors[degree]
                for degree in range(
                    1,
                    6,
                )
            ],
            dtype=object,
        ),
        coefficient_residual_vectors=np.array(
            [
                coefficient_residual_vectors[
                    degree
                ]
                for degree in range(
                    1,
                    6,
                )
            ],
            dtype=object,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)

    for row in formula_rows:
        print(
            row["elementary_invariant"],
            "formula:",
            row["formula"],
            "residual:",
            row[
                "normalized_maximum_residual"
            ],
            "pass:",
            row["invariant_pass"],
        )

    print(
        "maximum_probe_residual:",
        maximum_probe_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", FORMULA_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
