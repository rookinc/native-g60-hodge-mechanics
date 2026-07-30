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
    / "native_g60_cross_flux_gap_elementary_invariants_044.json"
)

NORMALIZATION_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_elementary_normalization_044b.json"
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
    / "native_g60_cross_flux_gap_elementary_canonical_certificate_044c.json"
)

FORMULA_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_elementary_canonical_formulas_044c.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_elementary_canonical_coefficients_044c.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_elementary_canonical_probes_044c.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_elementary_canonical_certificate_044c.npz"
)

COEFFICIENT_TOLERANCE = 5e-10
PROBE_TOLERANCE = 5e-9

RANDOM_SEED = 460443
PROBE_COUNT = 4096


CANONICAL_FORMULAS = {
    1: {
        "N2": Fraction(5, 12),
    },
    2: {
        "N2^2": Fraction(271, 3456),
        "S4": Fraction(-1, 120),
    },
    3: {
        "N2^3": Fraction(6107, 746496),
        "N2*S4": Fraction(-1, 540),
        "S6": Fraction(-1, 1800),
    },
    4: {
        "N2^4": Fraction(70403, 143327232),
        "N2^2*S4": Fraction(-47, 276480),
        "N2*S6": Fraction(-1, 8640),
        "S4^2": Fraction(1, 57600),
        "S8": Fraction(0, 1),
    },
    5: {
        "N2^5": Fraction(40051, 2579890176),
        "N2^3*S4": Fraction(-205, 35831808),
        "N2^2*S6": Fraction(-121, 12441600),
        "N2*S4^2": Fraction(1, 2073600),
        "N2*S8": Fraction(0, 1),
        "S4*S6": Fraction(1, 432000),
        "S10": Fraction(0, 1),
    },
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


def formula_text(
    coefficient_map: dict[str, Fraction],
) -> str:
    terms = []

    for name, coefficient in coefficient_map.items():
        if coefficient == 0:
            continue

        terms.append(
            f"({coefficient})*{name}"
        )

    return " + ".join(terms) or "0"


def build_candidate_polynomial(
    source,
    basis: dict,
    coefficient_map: dict[str, Fraction],
):
    result = {}

    for name, coefficient in coefficient_map.items():
        if name not in basis:
            raise RuntimeError(
                f"canonical basis term absent: {name}"
            )

        result = source.polynomial_add(
            result,
            basis[name],
            scale=float(coefficient),
        )

    return result


def direct_certificate(
    source,
    entries,
    moments,
) -> tuple[
    list[dict],
    list[dict],
    dict[int, dict],
]:
    formula_rows = []
    coefficient_rows = []
    records = {}

    for invariant_degree in range(1, 6):
        polynomial_degree = (
            2 * invariant_degree
        )

        print(
            "certificate_progress:",
            f"e{invariant_degree}",
            "constructing target",
            flush=True,
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

        coefficient_map = (
            CANONICAL_FORMULAS[
                invariant_degree
            ]
        )

        candidate_polynomial = (
            build_candidate_polynomial(
                source,
                basis,
                coefficient_map,
            )
        )

        candidate_vector = (
            source.polynomial_vector(
                candidate_polynomial,
                exponents,
            )
        )

        residual = (
            target_vector
            - candidate_vector
        )

        maximum_residual = max_abs(
            residual
        )

        residual_l2 = float(
            np.linalg.norm(residual)
        )

        invariant_pass = (
            maximum_residual
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
                "monomial_count": len(
                    exponents
                ),
                "basis_names": json.dumps(
                    list(basis)
                ),
                "canonical_formula": (
                    formula_text(
                        coefficient_map
                    )
                ),
                "maximum_coefficient_residual": (
                    maximum_residual
                ),
                "coefficient_residual_l2": (
                    residual_l2
                ),
                "coefficient_certificate_pass": (
                    invariant_pass
                ),
            }
        )

        for coefficient_id, exponent in enumerate(
            exponents
        ):
            coefficient_rows.append(
                {
                    "elementary_invariant": (
                        f"e{invariant_degree}"
                    ),
                    "coefficient_id": (
                        coefficient_id
                    ),
                    "f0_power": exponent[0],
                    "f1_power": exponent[1],
                    "f2_power": exponent[2],
                    "f3_power": exponent[3],
                    "target_coefficient": float(
                        target_vector[
                            coefficient_id
                        ]
                    ),
                    "canonical_coefficient": float(
                        candidate_vector[
                            coefficient_id
                        ]
                    ),
                    "residual": float(
                        residual[
                            coefficient_id
                        ]
                    ),
                    "coefficient_pass": (
                        abs(
                            residual[
                                coefficient_id
                            ]
                        )
                        < COEFFICIENT_TOLERANCE
                    ),
                }
            )

        records[invariant_degree] = {
            "exponents": np.array(
                exponents,
                dtype=np.int64,
            ),
            "target_vector": (
                target_vector
            ),
            "candidate_vector": (
                candidate_vector
            ),
            "residual": residual,
            "candidate_polynomial": (
                candidate_polynomial
            ),
        }

        print(
            "certificate_progress:",
            f"e{invariant_degree}",
            "maximum_residual:",
            maximum_residual,
            "pass:",
            invariant_pass,
            flush=True,
        )

    return (
        formula_rows,
        coefficient_rows,
        records,
    )


def probe_certificates(
    source,
    slices: np.ndarray,
    records: dict[int, dict],
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []
    maximum_residual = 0.0
    maximum_by_invariant = {
        degree: 0.0
        for degree in range(1, 6)
    }

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
            direct = float(
                direct_values[
                    invariant_degree
                ]
            )

            predicted = float(
                source.polynomial_evaluate(
                    records[
                        invariant_degree
                    ][
                        "candidate_polynomial"
                    ],
                    point,
                )
            )

            residual = abs(
                direct - predicted
            )

            maximum_residual = max(
                maximum_residual,
                residual,
            )

            maximum_by_invariant[
                invariant_degree
            ] = max(
                maximum_by_invariant[
                    invariant_degree
                ],
                residual,
            )

            current_row[
                f"e{invariant_degree}_direct"
            ] = direct

            current_row[
                f"e{invariant_degree}_canonical"
            ] = predicted

            current_row[
                f"e{invariant_degree}_residual"
            ] = residual

        if probe_id < 1024:
            rows.append(current_row)

    return rows, {
        "probe_count": PROBE_COUNT,
        "maximum_probe_residual": (
            maximum_residual
        ),
        "maximum_probe_residual_by_invariant": {
            f"e{degree}": value
            for degree, value in (
                maximum_by_invariant.items()
            )
        },
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

    normalization_receipt = json.loads(
        NORMALIZATION_JSON_PATH.read_text(
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

    (
        formula_rows,
        coefficient_rows,
        records,
    ) = direct_certificate(
        source,
        entries,
        moments,
    )

    probe_rows, probe_summary = (
        probe_certificates(
            source,
            slices,
            records,
        )
    )

    e3_expected_formula = (
        "(6107/746496)*N2^3 + "
        "(-1/540)*N2*S4 + "
        "(-1/1800)*S6"
    )

    checks = {
        "input_044_theorem_pass": (
            source_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "input_044b_selection_failure_recorded": (
            normalization_receipt.get(
                "theorem_pass"
            )
            is False
            and normalization_receipt.get(
                "verdict"
            )
            == (
                "native_g60_cross_flux_"
                "gap_elementary_"
                "normalization_failed"
            )
        ),
        "five_canonical_formulas_tested": (
            len(formula_rows) == 5
        ),
        "all_coefficient_certificates_pass": all(
            row[
                "coefficient_certificate_pass"
            ]
            for row in formula_rows
        ),
        "canonical_e3_formula_locked": (
            formula_rows[2][
                "canonical_formula"
            ]
            == e3_expected_formula
        ),
        "all_nonunit_probes_pass": (
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
        "native_g60_cross_flux_gap_elementary_canonical_certificate_exact"
        if theorem_pass
        else "native_g60_cross_flux_gap_elementary_canonical_certificate_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_elementary_canonical_certificate_044c"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "canonical_formulas": {
            row[
                "elementary_invariant"
            ]: row[
                "canonical_formula"
            ]
            for row in formula_rows
        },
        "formula_certificates": (
            formula_rows
        ),
        "probe_summary": (
            probe_summary
        ),
        "checks": checks,
        "earned_interpretation": {
            "canonical_e1_through_e5_formulas_proved": (
                theorem_pass
            ),
            "noncanonical_e3_coordinates_replaced": (
                theorem_pass
            ),
            "automatic_normalization_failure_resolved_by_direct_certificate": (
                theorem_pass
            ),
            "complete_gap_characteristic_invariant_chain_available": (
                theorem_pass
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "canonical_elementary_identity_chain_proved": (
                theorem_pass
            ),
            "global_nonnegativity_of_e1_through_e5_proved": (
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
            "certificate_npz": str(
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
        exponents=np.array(
            [
                records[degree][
                    "exponents"
                ]
                for degree in range(
                    1,
                    6,
                )
            ],
            dtype=object,
        ),
        target_vectors=np.array(
            [
                records[degree][
                    "target_vector"
                ]
                for degree in range(
                    1,
                    6,
                )
            ],
            dtype=object,
        ),
        canonical_vectors=np.array(
            [
                records[degree][
                    "candidate_vector"
                ]
                for degree in range(
                    1,
                    6,
                )
            ],
            dtype=object,
        ),
        residual_vectors=np.array(
            [
                records[degree][
                    "residual"
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
            row["canonical_formula"],
            "maximum_coefficient_residual:",
            row[
                "maximum_coefficient_residual"
            ],
            "pass:",
            row[
                "coefficient_certificate_pass"
            ],
        )

    print(
        "maximum_probe_residual:",
        probe_summary[
            "maximum_probe_residual"
        ],
    )
    print(
        "maximum_probe_residual_by_invariant:",
        probe_summary[
            "maximum_probe_residual_by_invariant"
        ],
    )
    print("wrote:", JSON_OUT)
    print("wrote:", FORMULA_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
