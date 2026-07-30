from __future__ import annotations

import csv
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.set_int_max_str_digits(0)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ELEMENTARY_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_elementary_invariants_044.py"
)

RECONSTRUCTION_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_exact_sos_reconstruction_046.py"
)

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_higher_sdp_probe_048.json"
)

SOURCE_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_higher_sdp_probe_048.npz"
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
    / "native_g60_cross_flux_gap_exact_e4_sos_reconstruction_049.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_e4_sos_summary_049.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_e4_sos_gram_entries_049.csv"
)

LDL_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_e4_sos_ldl_pivots_049.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_exact_e4_sos_coefficients_049.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_exact_e4_sos_reconstruction_049.npz"
)

TARGET_DENOMINATOR_LIMITS = (
    1_000_000,
    100_000_000,
    10_000_000_000,
    1_000_000_000_000,
)

GRAM_DENOMINATOR_LIMITS = (
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
)

TARGET_RATIONALIZATION_TOLERANCE = 5e-12
FLOATING_PROJECTION_TOLERANCE = 5e-11
FLOATING_PSD_TOLERANCE = 5e-10


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load module: {path}"
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


def reconstruct_e4(
    reconstruction,
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

    buckets = reconstruction.coefficient_buckets(
        monomial_exponents,
        target_exponents,
    )

    floating_entries = (
        reconstruction.lower_entries(
            floating_gram
        )
    )

    projected = (
        reconstruction.project_to_float_slice(
            floating_entries,
            target,
            buckets,
        )
    )

    projected_coefficients = (
        reconstruction.coefficient_values_float(
            projected,
            buckets,
        )
    )

    candidates = []

    for target_limit in (
        TARGET_DENOMINATOR_LIMITS
    ):
        rational_target = (
            reconstruction.rationalize_vector(
                target,
                target_limit,
            )
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
                reconstruction.exact_bucket_reconstruction(
                    projected,
                    rational_target,
                    buckets,
                    gram_limit,
                )
            )

            exact_coefficients = (
                reconstruction.exact_coefficient_values(
                    exact_entries,
                    buckets,
                )
            )

            coefficient_exact = (
                exact_coefficients
                == rational_target
            )

            exact_matrix = (
                reconstruction.rational_matrix_from_lower(
                    exact_entries,
                    size,
                )
            )

            float_matrix = (
                reconstruction.fraction_matrix_float(
                    exact_matrix
                )
            )

            eigenvalues = np.linalg.eigvalsh(
                float_matrix
            )

            ldl = (
                reconstruction.exact_ldl_positive_definite(
                    exact_matrix
                )
            )

            distance = float(
                np.max(
                    np.abs(
                        float_matrix
                        - floating_gram
                    )
                )
            )

            certificate_pass = (
                target_error
                < TARGET_RATIONALIZATION_TOLERANCE
                and coefficient_exact
                and ldl[
                    "positive_definite"
                ]
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
                    "exact_coefficients": (
                        exact_coefficients
                    ),
                    "exact_matrix": (
                        exact_matrix
                    ),
                    "float_matrix": (
                        float_matrix
                    ),
                    "eigenvalues": (
                        eigenvalues
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
                    "ldl": ldl,
                    "certificate_pass": (
                        certificate_pass
                    ),
                }
            )

            print(
                "candidate:",
                "target_limit:",
                target_limit,
                "gram_limit:",
                gram_limit,
                "target_error:",
                target_error,
                "entry_change:",
                distance,
                "min_eig:",
                float(
                    eigenvalues[0]
                ),
                "ldl_pd:",
                ldl[
                    "positive_definite"
                ],
                "pass:",
                certificate_pass,
                flush=True,
            )

    best = min(
        candidates,
        key=lambda item: (
            not item[
                "certificate_pass"
            ],
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
        "projected_coefficient_residual": float(
            np.max(
                np.abs(
                    projected_coefficients
                    - target
                )
            )
        ),
        "candidates": candidates,
        "best": best,
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

    GRAM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LDL_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COEFFICIENT_CSV_OUT.parent.mkdir(
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

    elementary = load_module(
        "gap_elementary_044",
        ELEMENTARY_SCRIPT_PATH,
    )

    reconstruction = load_module(
        "gap_reconstruction_046",
        RECONSTRUCTION_SCRIPT_PATH,
    )

    axis_lines = np.array(
        orientation_data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    moments = (
        elementary.construct_register_moments(
            axis_lines
        )
    )

    entries = (
        elementary.construct_gap_entries(
            slices
        )
    )

    print(
        "exact_e4_target:",
        "constructing coefficient polynomial",
        flush=True,
    )

    target_polynomial = (
        elementary.elementary_polynomial(
            entries,
            4,
        )
    )

    target_exponents = (
        elementary.degree_exponents(
            8
        )
    )

    target = (
        elementary.polynomial_vector(
            target_polynomial,
            target_exponents,
        )
    )

    monomial_exponents = (
        homogeneous_exponents(
            4
        )
    )

    floating_gram = np.array(
        source_data["e4_gram"],
        dtype=np.float64,
    )

    result = reconstruct_e4(
        reconstruction,
        floating_gram,
        target,
        monomial_exponents,
        target_exponents,
    )

    best = result["best"]

    theorem_pass = bool(
        best["certificate_pass"]
    )

    checks = {
        "input_048_audit_pass": (
            source_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_048_e4_interior_candidate": (
            source_receipt.get(
                "earned_interpretation",
                {},
            ).get(
                "e4_has_floating_interior_sos_candidate"
            )
            is True
        ),
        "gram_size_is_35": (
            floating_gram.shape
            == (35, 35)
        ),
        "coefficient_count_is_165": (
            len(target) == 165
        ),
        "floating_slice_projection_closes": (
            result[
                "projected_coefficient_residual"
            ]
            < FLOATING_PROJECTION_TOLERANCE
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
        "reconstructed_gram_numerically_positive": (
            best[
                "minimum_float_eigenvalue"
            ]
            > FLOATING_PSD_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = (
        theorem_pass
        and audit_pass
    )

    verdict = (
        "exact_rational_sos_certificate_found_for_e4"
        if theorem_pass
        else "exact_e4_sos_reconstruction_not_resolved"
    )

    summary_rows = [
        {
            "target": "e4",
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
                theorem_pass
            ),
        }
    ]

    gram_rows = []

    pairs = reconstruction.lower_pairs(
        floating_gram.shape[0]
    )

    for entry_index, (
        row,
        column,
    ) in enumerate(pairs):
        value = best[
            "exact_entries"
        ][entry_index]

        gram_rows.append(
            {
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
                    value.numerator
                ),
                "denominator": (
                    value.denominator
                ),
                "exact_value": str(
                    value
                ),
                "floating_value": float(
                    value
                ),
            }
        )

    ldl_rows = []

    for pivot_index, pivot in enumerate(
        best["ldl"]["diagonal"]
    ):
        ldl_rows.append(
            {
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

    coefficient_rows = []

    for coefficient_id, exponent in enumerate(
        target_exponents
    ):
        exact_target = best[
            "exact_target"
        ][coefficient_id]

        exact_candidate = best[
            "exact_coefficients"
        ][coefficient_id]

        coefficient_rows.append(
            {
                "coefficient_id": (
                    coefficient_id
                ),
                "monomial": (
                    exponent_label(
                        exponent
                    )
                ),
                "target_exact": str(
                    exact_target
                ),
                "candidate_exact": str(
                    exact_candidate
                ),
                "exact_match": (
                    exact_target
                    == exact_candidate
                ),
                "target_float": float(
                    exact_target
                ),
                "candidate_float": float(
                    exact_candidate
                ),
            }
        )


    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_exact_e4_sos_reconstruction_049"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "definition": {
            "target": "e4(G)",
            "sos_form": (
                "e4=z4^T Q4 z4"
            ),
            "monomial_degree": 4,
            "gram_size": 35,
            "coefficient_count": 165,
        },
        "result": (
            summary_rows[0]
        ),
        "checks": checks,
        "earned_interpretation": {
            "exact_rational_sos_certificate_for_e4": (
                theorem_pass
            ),
            "e4_globally_positive_for_nonzero_f": (
                theorem_pass
            ),
            "remaining_open_elementary_invariant": (
                "e5"
                if theorem_pass
                else "e4_and_e5"
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "exact_e4_sos_certificate": (
                theorem_pass
            ),
            "e5_not_resolved": True,
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
            "coefficient_csv": str(
                COEFFICIENT_CSV_OUT.relative_to(
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
        (
            COEFFICIENT_CSV_OUT,
            coefficient_rows,
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
        exact_gram_float=(
            best["float_matrix"]
        ),
        exact_gram_eigenvalues=(
            best["eigenvalues"]
        ),
        exact_ldl_pivots_float=np.array(
            [
                float(value)
                for value in best[
                    "ldl"
                ]["diagonal"]
            ],
            dtype=np.float64,
        ),
        floating_source_gram=(
            floating_gram
        ),
        target_coefficients=(
            target
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "target_error:",
        best["target_error"],
    )
    print(
        "entry_change:",
        best[
            "distance_from_saved_gram"
        ],
    )
    print(
        "minimum_float_eigenvalue:",
        best[
            "minimum_float_eigenvalue"
        ],
    )
    print(
        "maximum_float_eigenvalue:",
        best[
            "maximum_float_eigenvalue"
        ],
    )
    print(
        "exact_ldl_positive_definite:",
        best["ldl"][
            "positive_definite"
        ],
    )
    print(
        "minimum_exact_ldl_pivot_float:",
        min(
            float(value)
            for value in best[
                "ldl"
            ]["diagonal"]
        ),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", LDL_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
