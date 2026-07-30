from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_elementary_invariants_044.py"
)

SOURCE_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_gap_elementary_canonical_certificate_044c.json"
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
    / "native_g60_cross_flux_gap_scalar_sos_probe_045.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_scalar_sos_summary_045.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_scalar_sos_gram_entries_045.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_scalar_sos_probe_045.npz"
)

RANDOM_SEED = 46045

START_COUNT = {
    2: 6,
    3: 6,
}

MAX_FUNCTION_EVALUATIONS = {
    2: 2500,
    3: 3000,
}

COEFFICIENT_TOLERANCE = 5e-10
PSD_TOLERANCE = 5e-10


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
    exponents = []

    for a in range(degree + 1):
        for b in range(
            degree + 1 - a
        ):
            for c in range(
                degree + 1 - a - b
            ):
                d = degree - a - b - c

                exponents.append(
                    (a, b, c, d)
                )

    exponents.sort(reverse=True)

    return exponents


def exponent_add(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return tuple(
        first[index] + second[index]
        for index in range(4)
    )


def lower_triangular_matrix(
    parameters: np.ndarray,
    size: int,
) -> np.ndarray:
    matrix = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    cursor = 0

    for row in range(size):
        width = row + 1

        matrix[
            row,
            :width,
        ] = parameters[
            cursor : cursor + width
        ]

        cursor += width

    return matrix


def lower_triangular_parameters(
    matrix: np.ndarray,
) -> np.ndarray:
    values = []

    for row in range(
        matrix.shape[0]
    ):
        values.extend(
            matrix[
                row,
                : row + 1,
            ]
        )

    return np.array(
        values,
        dtype=np.float64,
    )


def gram_coefficient_map(
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
) -> np.ndarray:
    target_index = {
        exponent: index
        for index, exponent in enumerate(
            target_exponents
        )
    }

    size = len(
        monomial_exponents
    )

    pair_count = (
        size * (size + 1) // 2
    )

    mapping = np.zeros(
        (
            len(target_exponents),
            pair_count,
        ),
        dtype=np.float64,
    )

    cursor = 0

    for row in range(size):
        for column in range(
            row + 1
        ):
            exponent = exponent_add(
                monomial_exponents[row],
                monomial_exponents[column],
            )

            coefficient = (
                1.0
                if row == column
                else 2.0
            )

            mapping[
                target_index[exponent],
                cursor,
            ] = coefficient

            cursor += 1

    return mapping


def symmetric_lower_entries(
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


def gram_polynomial_coefficients(
    lower_parameters: np.ndarray,
    size: int,
    coefficient_map: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    lower = lower_triangular_matrix(
        lower_parameters,
        size,
    )

    gram = lower @ lower.T

    gram_entries = symmetric_lower_entries(
        gram
    )

    coefficients = (
        coefficient_map
        @ gram_entries
    )

    return lower, gram, coefficients


def initial_lower_factor(
    size: int,
    target: np.ndarray,
    rng: np.random.Generator,
    start_id: int,
) -> np.ndarray:
    target_scale = max(
        float(
            np.linalg.norm(target)
        ),
        1e-12,
    )

    diagonal_scale = (
        target_scale
        / max(size, 1)
    ) ** 0.25

    lower = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    if start_id == 0:
        np.fill_diagonal(
            lower,
            diagonal_scale,
        )
    else:
        lower = rng.normal(
            scale=0.15 * diagonal_scale,
            size=(size, size),
        )

        lower = np.tril(lower)

        diagonal = (
            diagonal_scale
            * (
                0.75
                + 0.5
                * rng.random(size)
            )
        )

        np.fill_diagonal(
            lower,
            diagonal,
        )

    return lower_triangular_parameters(
        lower
    )


def solve_sos_candidate(
    target: np.ndarray,
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
    start_count: int,
    max_nfev: int,
    rng: np.random.Generator,
    label: str,
) -> dict:
    size = len(
        monomial_exponents
    )

    coefficient_map = (
        gram_coefficient_map(
            monomial_exponents,
            target_exponents,
        )
    )

    row_scales = np.maximum(
        np.abs(target),
        1e-8,
    )

    def residual_function(
        parameters: np.ndarray,
    ) -> np.ndarray:
        _, _, coefficients = (
            gram_polynomial_coefficients(
                parameters,
                size,
                coefficient_map,
            )
        )

        return (
            coefficients - target
        ) / row_scales

    start_records = []
    best = None

    for start_id in range(
        start_count
    ):
        initial = initial_lower_factor(
            size,
            target,
            rng,
            start_id,
        )

        result = least_squares(
            residual_function,
            initial,
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            verbose=0,
        )

        lower, gram, coefficients = (
            gram_polynomial_coefficients(
                result.x,
                size,
                coefficient_map,
            )
        )

        coefficient_residual = (
            coefficients - target
        )

        eigenvalues = np.linalg.eigvalsh(
            gram
        )

        record = {
            "start_id": start_id,
            "success": bool(
                result.success
            ),
            "status": int(
                result.status
            ),
            "nfev": int(
                result.nfev
            ),
            "cost": float(
                result.cost
            ),
            "maximum_coefficient_residual": (
                max_abs(
                    coefficient_residual
                )
            ),
            "coefficient_residual_l2": float(
                np.linalg.norm(
                    coefficient_residual
                )
            ),
            "minimum_gram_eigenvalue": float(
                eigenvalues[0]
            ),
            "maximum_gram_eigenvalue": float(
                eigenvalues[-1]
            ),
            "numerical_rank": int(
                np.count_nonzero(
                    eigenvalues
                    > PSD_TOLERANCE
                )
            ),
        }

        start_records.append(record)

        candidate = {
            "record": record,
            "lower": lower,
            "gram": gram,
            "coefficients": coefficients,
            "coefficient_residual": (
                coefficient_residual
            ),
            "eigenvalues": eigenvalues,
        }

        if best is None:
            best = candidate
        else:
            current_key = (
                record[
                    "maximum_coefficient_residual"
                ],
                -record[
                    "minimum_gram_eigenvalue"
                ],
            )

            best_key = (
                best["record"][
                    "maximum_coefficient_residual"
                ],
                -best["record"][
                    "minimum_gram_eigenvalue"
                ],
            )

            if current_key < best_key:
                best = candidate

        print(
            "sos_progress:",
            label,
            f"{start_id + 1}/{start_count}",
            "residual:",
            record[
                "maximum_coefficient_residual"
            ],
            "min_eig:",
            record[
                "minimum_gram_eigenvalue"
            ],
            "rank:",
            record[
                "numerical_rank"
            ],
            flush=True,
        )

    if best is None:
        raise RuntimeError(
            f"no SOS candidate produced for {label}"
        )

    return {
        "monomial_exponents": (
            monomial_exponents
        ),
        "target_exponents": (
            target_exponents
        ),
        "coefficient_map": (
            coefficient_map
        ),
        "start_records": (
            start_records
        ),
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

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = {}
    summary_rows = []
    gram_rows = []

    for invariant_degree in (
        2,
        3,
    ):
        polynomial_degree = (
            2 * invariant_degree
        )

        print(
            "sos_target:",
            f"e{invariant_degree}",
            "constructing exact coefficient target",
            flush=True,
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

        target_vector = (
            source.polynomial_vector(
                target_polynomial,
                target_exponents,
            )
        )

        monomial_degree = (
            invariant_degree
        )

        monomial_exponents = (
            homogeneous_exponents(
                monomial_degree
            )
        )

        result = solve_sos_candidate(
            target_vector,
            monomial_exponents,
            target_exponents,
            START_COUNT[
                invariant_degree
            ],
            MAX_FUNCTION_EVALUATIONS[
                invariant_degree
            ],
            rng,
            f"e{invariant_degree}",
        )

        results[
            invariant_degree
        ] = result

        best = result["best"]
        record = best["record"]

        coefficient_pass = (
            record[
                "maximum_coefficient_residual"
            ]
            < COEFFICIENT_TOLERANCE
        )

        psd_pass = (
            record[
                "minimum_gram_eigenvalue"
            ]
            >= -PSD_TOLERANCE
        )

        candidate_found = (
            coefficient_pass
            and psd_pass
        )

        summary_rows.append(
            {
                "elementary_invariant": (
                    f"e{invariant_degree}"
                ),
                "polynomial_degree": (
                    polynomial_degree
                ),
                "monomial_degree": (
                    monomial_degree
                ),
                "monomial_count": len(
                    monomial_exponents
                ),
                "gram_size": len(
                    monomial_exponents
                ),
                "symmetric_gram_variable_count": (
                    len(monomial_exponents)
                    * (
                        len(
                            monomial_exponents
                        )
                        + 1
                    )
                    // 2
                ),
                "coefficient_constraint_count": (
                    len(target_exponents)
                ),
                "start_count": (
                    START_COUNT[
                        invariant_degree
                    ]
                ),
                "best_start_id": (
                    record["start_id"]
                ),
                "best_nfev": (
                    record["nfev"]
                ),
                "maximum_coefficient_residual": (
                    record[
                        "maximum_coefficient_residual"
                    ]
                ),
                "coefficient_residual_l2": (
                    record[
                        "coefficient_residual_l2"
                    ]
                ),
                "minimum_gram_eigenvalue": (
                    record[
                        "minimum_gram_eigenvalue"
                    ]
                ),
                "maximum_gram_eigenvalue": (
                    record[
                        "maximum_gram_eigenvalue"
                    ]
                ),
                "numerical_rank": (
                    record[
                        "numerical_rank"
                    ]
                ),
                "floating_psd_gram_candidate_found": (
                    candidate_found
                ),
            }
        )

        gram = best["gram"]

        for row in range(
            gram.shape[0]
        ):
            for column in range(
                gram.shape[1]
            ):
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
                        "gram_entry": float(
                            gram[row, column]
                        ),
                    }
                )

    checks = {
        "input_044c_theorem_pass": (
            source_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "quartic_e2_probe_completed": (
            2 in results
        ),
        "sextic_e3_probe_completed": (
            3 in results
        ),
        "summary_row_count_is_2": (
            len(summary_rows) == 2
        ),
        "all_best_gram_matrices_numerically_psd": all(
            row[
                "minimum_gram_eigenvalue"
            ]
            >= -PSD_TOLERANCE
            for row in summary_rows
        ),
    }

    audit_pass = all(
        checks.values()
    )

    candidate_count = sum(
        bool(
            row[
                "floating_psd_gram_candidate_found"
            ]
        )
        for row in summary_rows
    )

    theorem_pass = False

    if audit_pass and candidate_count == 2:
        verdict = (
            "floating_psd_gram_candidates_found_for_e2_and_e3"
        )
    elif audit_pass and candidate_count > 0:
        verdict = (
            "partial_floating_psd_gram_candidate_found"
        )
    elif audit_pass:
        verdict = (
            "no_floating_psd_gram_candidate_found_in_tested_search"
        )
    else:
        verdict = (
            "scalar_sos_probe_failed"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_scalar_sos_probe_045"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "definition": {
            "target_invariants": [
                "e2(G)",
                "e3(G)",
            ],
            "quartic_form": (
                "e2=z2^T Q2 z2"
            ),
            "sextic_form": (
                "e3=z3^T Q3 z3"
            ),
            "gram_constraint": (
                "Qk=Lk Lk^T"
            ),
        },
        "results": summary_rows,
        "candidate_count": (
            candidate_count
        ),
        "checks": checks,
        "earned_interpretation": {
            "floating_psd_gram_candidate_for_e2": (
                summary_rows[0][
                    "floating_psd_gram_candidate_found"
                ]
            ),
            "floating_psd_gram_candidate_for_e3": (
                summary_rows[1][
                    "floating_psd_gram_candidate_found"
                ]
            ),
            "exact_rational_sos_certificate_found": (
                False
            ),
            "global_nonnegativity_proved": (
                False
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "nonlinear_gram_search_only": (
                True
            ),
            "floating_candidates_are_not_exact_certificates": (
                True
            ),
            "failure_would_only_cover_tested_multistart_search": (
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

    with SUMMARY_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    with GRAM_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                gram_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            gram_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        e2_monomial_exponents=np.array(
            results[2][
                "monomial_exponents"
            ],
            dtype=np.int64,
        ),
        e2_gram=results[2][
            "best"
        ]["gram"],
        e2_lower_factor=results[2][
            "best"
        ]["lower"],
        e2_target_coefficients=results[2][
            "best"
        ]["coefficients"]
        - results[2][
            "best"
        ]["coefficient_residual"],
        e2_candidate_coefficients=results[2][
            "best"
        ]["coefficients"],
        e2_coefficient_residual=results[2][
            "best"
        ]["coefficient_residual"],
        e3_monomial_exponents=np.array(
            results[3][
                "monomial_exponents"
            ],
            dtype=np.int64,
        ),
        e3_gram=results[3][
            "best"
        ]["gram"],
        e3_lower_factor=results[3][
            "best"
        ]["lower"],
        e3_target_coefficients=results[3][
            "best"
        ]["coefficients"]
        - results[3][
            "best"
        ]["coefficient_residual"],
        e3_candidate_coefficients=results[3][
            "best"
        ]["coefficients"],
        e3_coefficient_residual=results[3][
            "best"
        ]["coefficient_residual"],
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "candidate_count:",
        candidate_count,
    )

    for row in summary_rows:
        print(
            row[
                "elementary_invariant"
            ],
            "residual:",
            row[
                "maximum_coefficient_residual"
            ],
            "minimum_gram_eigenvalue:",
            row[
                "minimum_gram_eigenvalue"
            ],
            "rank:",
            row[
                "numerical_rank"
            ],
            "candidate_found:",
            row[
                "floating_psd_gram_candidate_found"
            ],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
