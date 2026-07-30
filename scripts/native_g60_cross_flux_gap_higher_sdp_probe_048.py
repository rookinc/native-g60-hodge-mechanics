from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "native_g60_cross_flux_gap_elementary_invariants_044.py"
)

REPLAY_JSON_PATHS = {
    "043c": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_gap_determinant_scaled_reconstruction_043c.json"
    ),
    "044c": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_gap_elementary_canonical_certificate_044c.json"
    ),
    "046": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_gap_exact_sos_reconstruction_046.json"
    ),
}

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
    / "native_g60_cross_flux_gap_higher_sdp_probe_048.json"
)

SUMMARY_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_higher_sdp_summary_048.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_higher_sdp_gram_entries_048.csv"
)

RESIDUAL_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_gap_higher_sdp_coefficient_residuals_048.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_gap_higher_sdp_probe_048.npz"
)

SOLVER = "SCS"

SCS_OPTIONS = {
    "eps": 1e-8,
    "max_iters": 250_000,
    "acceleration_lookback": 20,
    "normalize": True,
    "scale": 1.0,
    "verbose": True,
}

COEFFICIENT_TOLERANCE = 5e-7
PSD_TOLERANCE = 5e-7
POSITIVE_MARGIN_TOLERANCE = 1e-8

TARGETS = (
    "e4_residual_1_over_16",
    "e4",
    "e5",
)


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


def exponent_add(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return tuple(
        first[index] + second[index]
        for index in range(4)
    )


def exponent_label(
    exponent: tuple[int, int, int, int],
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


def construct_targets(
    source,
    entries,
    moments,
) -> dict[str, dict]:
    e1 = source.elementary_polynomial(
        entries,
        1,
    )

    e3 = source.elementary_polynomial(
        entries,
        3,
    )

    e4 = source.elementary_polynomial(
        entries,
        4,
    )

    e5 = source.elementary_polynomial(
        entries,
        5,
    )

    e1_e3 = source.polynomial_multiply(
        e1,
        e3,
    )

    e4_residual = source.polynomial_add(
        e4,
        e1_e3,
        scale=-1.0 / 16.0,
    )

    return {
        "e4_residual_1_over_16": {
            "polynomial": e4_residual,
            "polynomial_degree": 8,
            "monomial_degree": 4,
            "definition": (
                "e4-(1/16)e1e3"
            ),
        },
        "e4": {
            "polynomial": e4,
            "polynomial_degree": 8,
            "monomial_degree": 4,
            "definition": "e4",
        },
        "e5": {
            "polynomial": e5,
            "polynomial_degree": 10,
            "monomial_degree": 5,
            "definition": "e5",
        },
    }


def coefficient_buckets(
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
) -> list[list[tuple[int, int, float]]]:
    target_index = {
        exponent: index
        for index, exponent in enumerate(
            target_exponents
        )
    }

    buckets = [
        []
        for _ in target_exponents
    ]

    for row in range(
        len(monomial_exponents)
    ):
        for column in range(
            row + 1
        ):
            exponent = exponent_add(
                monomial_exponents[row],
                monomial_exponents[column],
            )

            weight = (
                1.0
                if row == column
                else 2.0
            )

            buckets[
                target_index[exponent]
            ].append(
                (
                    row,
                    column,
                    weight,
                )
            )

    return buckets


def gram_coefficients(
    gram: np.ndarray,
    buckets: list[
        list[tuple[int, int, float]]
    ],
) -> np.ndarray:
    return np.array(
        [
            sum(
                weight * gram[row, column]
                for row, column, weight
                in bucket
            )
            for bucket in buckets
        ],
        dtype=np.float64,
    )


def solve_target_sdp(
    label: str,
    target_vector: np.ndarray,
    monomial_exponents: list[
        tuple[int, int, int, int]
    ],
    target_exponents: list[
        tuple[int, int, int, int]
    ],
) -> dict:
    size = len(monomial_exponents)

    buckets = coefficient_buckets(
        monomial_exponents,
        target_exponents,
    )

    gram = cp.Variable(
        (size, size),
        symmetric=True,
        name=f"Q_{label}",
    )

    margin = cp.Variable(
        name=f"margin_{label}"
    )

    constraints = [
        gram
        - margin * np.eye(size)
        >> 0
    ]

    for target_index, bucket in enumerate(
        buckets
    ):
        expression = 0

        for row, column, weight in bucket:
            expression += (
                weight
                * gram[row, column]
            )

        constraints.append(
            expression
            == target_vector[
                target_index
            ]
        )

    problem = cp.Problem(
        cp.Maximize(margin),
        constraints,
    )

    print(
        "sdp_start:",
        label,
        "gram_size:",
        size,
        "coefficient_constraints:",
        len(target_vector),
        "solver:",
        SOLVER,
        flush=True,
    )

    started = time.perf_counter()

    objective = problem.solve(
        solver=SOLVER,
        **SCS_OPTIONS,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if gram.value is None:
        gram_value = np.full(
            (size, size),
            np.nan,
            dtype=np.float64,
        )
    else:
        gram_value = np.array(
            gram.value,
            dtype=np.float64,
        )

        gram_value = 0.5 * (
            gram_value
            + gram_value.T
        )

    candidate_coefficients = (
        gram_coefficients(
            gram_value,
            buckets,
        )
        if np.all(
            np.isfinite(gram_value)
        )
        else np.full_like(
            target_vector,
            np.nan,
        )
    )

    coefficient_residual = (
        candidate_coefficients
        - target_vector
    )

    if np.all(np.isfinite(gram_value)):
        eigenvalues = np.linalg.eigvalsh(
            gram_value
        )
    else:
        eigenvalues = np.full(
            size,
            np.nan,
            dtype=np.float64,
        )

    margin_value = (
        float(margin.value)
        if margin.value is not None
        else float("nan")
    )

    coefficient_pass = (
        np.all(
            np.isfinite(
                coefficient_residual
            )
        )
        and max_abs(
            coefficient_residual
        )
        < COEFFICIENT_TOLERANCE
    )

    psd_pass = (
        np.all(
            np.isfinite(eigenvalues)
        )
        and float(eigenvalues[0])
        >= -PSD_TOLERANCE
    )

    interior_candidate = (
        coefficient_pass
        and psd_pass
        and margin_value
        > POSITIVE_MARGIN_TOLERANCE
    )

    print(
        "sdp_result:",
        label,
        "status:",
        problem.status,
        "objective:",
        objective,
        "margin:",
        margin_value,
        "coefficient_residual:",
        max_abs(
            coefficient_residual
        ),
        "min_eig:",
        float(eigenvalues[0]),
        "rank:",
        int(
            np.count_nonzero(
                eigenvalues
                > PSD_TOLERANCE
            )
        ),
        "elapsed_seconds:",
        elapsed,
        "interior_candidate:",
        interior_candidate,
        flush=True,
    )

    return {
        "label": label,
        "problem_status": (
            problem.status
        ),
        "objective_value": (
            float(objective)
            if objective is not None
            else float("nan")
        ),
        "margin": margin_value,
        "elapsed_seconds": elapsed,
        "gram": gram_value,
        "eigenvalues": eigenvalues,
        "target_coefficients": (
            target_vector
        ),
        "candidate_coefficients": (
            candidate_coefficients
        ),
        "coefficient_residual": (
            coefficient_residual
        ),
        "maximum_coefficient_residual": (
            max_abs(
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
        "coefficient_pass": (
            coefficient_pass
        ),
        "psd_pass": psd_pass,
        "interior_candidate": (
            interior_candidate
        ),
        "monomial_exponents": (
            monomial_exponents
        ),
        "target_exponents": (
            target_exponents
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

    GRAM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESIDUAL_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_receipts = {
        label: json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        for label, path
        in REPLAY_JSON_PATHS.items()
    }

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

    moments = (
        source.construct_register_moments(
            axis_lines
        )
    )

    entries = source.construct_gap_entries(
        slices
    )

    target_records = construct_targets(
        source,
        entries,
        moments,
    )

    results = {}
    summary_rows = []
    gram_rows = []
    residual_rows = []

    for label in TARGETS:
        record = target_records[label]

        target_exponents = (
            source.degree_exponents(
                record[
                    "polynomial_degree"
                ]
            )
        )

        target_vector = (
            source.polynomial_vector(
                record["polynomial"],
                target_exponents,
            )
        )

        monomial_exponents = (
            homogeneous_exponents(
                record[
                    "monomial_degree"
                ]
            )
        )

        result = solve_target_sdp(
            label,
            target_vector,
            monomial_exponents,
            target_exponents,
        )

        results[label] = result

        summary_rows.append(
            {
                "target": label,
                "definition": (
                    record["definition"]
                ),
                "polynomial_degree": (
                    record[
                        "polynomial_degree"
                    ]
                ),
                "monomial_degree": (
                    record[
                        "monomial_degree"
                    ]
                ),
                "gram_size": (
                    result["gram"].shape[0]
                ),
                "coefficient_constraint_count": (
                    len(target_vector)
                ),
                "solver": SOLVER,
                "solver_status": (
                    result[
                        "problem_status"
                    ]
                ),
                "elapsed_seconds": (
                    result[
                        "elapsed_seconds"
                    ]
                ),
                "optimized_margin": (
                    result["margin"]
                ),
                "maximum_coefficient_residual": (
                    result[
                        "maximum_coefficient_residual"
                    ]
                ),
                "minimum_gram_eigenvalue": (
                    result[
                        "minimum_gram_eigenvalue"
                    ]
                ),
                "maximum_gram_eigenvalue": (
                    result[
                        "maximum_gram_eigenvalue"
                    ]
                ),
                "numerical_rank": (
                    result[
                        "numerical_rank"
                    ]
                ),
                "coefficient_pass": (
                    result[
                        "coefficient_pass"
                    ]
                ),
                "psd_pass": (
                    result["psd_pass"]
                ),
                "floating_interior_sos_candidate": (
                    result[
                        "interior_candidate"
                    ]
                ),
            }
        )

        gram = result["gram"]
        monomials = result[
            "monomial_exponents"
        ]

        for row in range(
            gram.shape[0]
        ):
            for column in range(
                row + 1
            ):
                gram_rows.append(
                    {
                        "target": label,
                        "row": row,
                        "column": column,
                        "row_monomial": (
                            exponent_label(
                                monomials[row]
                            )
                        ),
                        "column_monomial": (
                            exponent_label(
                                monomials[
                                    column
                                ]
                            )
                        ),
                        "gram_entry": float(
                            gram[
                                row,
                                column,
                            ]
                        ),
                    }
                )

        for coefficient_id, exponent in enumerate(
            result["target_exponents"]
        ):
            residual_rows.append(
                {
                    "target": label,
                    "coefficient_id": (
                        coefficient_id
                    ),
                    "monomial": (
                        exponent_label(
                            exponent
                        )
                    ),
                    "target_coefficient": float(
                        result[
                            "target_coefficients"
                        ][coefficient_id]
                    ),
                    "candidate_coefficient": float(
                        result[
                            "candidate_coefficients"
                        ][coefficient_id]
                    ),
                    "residual": float(
                        result[
                            "coefficient_residual"
                        ][coefficient_id]
                    ),
                }
            )

    replay_pass = all(
        replay_receipts[label].get(
            "theorem_pass"
        )
        is True
        for label in (
            "043c",
            "044c",
            "046",
        )
    )

    completed_count = sum(
        result["problem_status"]
        in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        )
        for result in results.values()
    )

    interior_candidate_count = sum(
        bool(
            result[
                "interior_candidate"
            ]
        )
        for result in results.values()
    )

    checks = {
        "m3_replay_receipts_pass": (
            replay_pass
        ),
        "all_three_sdp_problems_completed": (
            completed_count == 3
        ),
        "all_gram_values_finite": all(
            np.all(
                np.isfinite(
                    result["gram"]
                )
            )
            for result in results.values()
        ),
        "all_coefficient_residuals_bounded": all(
            result[
                "maximum_coefficient_residual"
            ]
            < COEFFICIENT_TOLERANCE
            for result in results.values()
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    if (
        audit_pass
        and interior_candidate_count == 3
    ):
        verdict = (
            "floating_interior_sos_candidates_found_for_e4_residual_e4_and_e5"
        )
    elif (
        audit_pass
        and interior_candidate_count > 0
    ):
        verdict = (
            "partial_floating_interior_sos_candidates_found_for_higher_gap_invariants"
        )
    elif audit_pass:
        verdict = (
            "no_floating_interior_sos_candidate_found_in_higher_sdp_probe"
        )
    else:
        verdict = (
            "higher_gap_sdp_probe_failed"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_gap_higher_sdp_probe_048"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "solver": {
            "name": SOLVER,
            "options": SCS_OPTIONS,
            "cvxpy_version": (
                cp.__version__
            ),
            "installed_solvers": (
                cp.installed_solvers()
            ),
        },
        "m3_replay_pass": replay_pass,
        "completed_target_count": (
            completed_count
        ),
        "interior_candidate_count": (
            interior_candidate_count
        ),
        "results": summary_rows,
        "checks": checks,
        "earned_interpretation": {
            "e4_residual_has_floating_interior_sos_candidate": (
                results[
                    "e4_residual_1_over_16"
                ][
                    "interior_candidate"
                ]
            ),
            "e4_has_floating_interior_sos_candidate": (
                results["e4"][
                    "interior_candidate"
                ]
            ),
            "e5_has_floating_interior_sos_candidate": (
                results["e5"][
                    "interior_candidate"
                ]
            ),
            "exact_rational_higher_sos_certificate_found": (
                False
            ),
            "global_gap_psd_proved": (
                False
            ),
        },
        "boundary": {
            "floating_sdp_probe_only": (
                True
            ),
            "symmetry_reduction_not_used": (
                True
            ),
            "exact_reconstruction_deferred": (
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
            "residual_csv": str(
                RESIDUAL_CSV_OUT.relative_to(
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
            GRAM_CSV_OUT,
            gram_rows,
        ),
        (
            RESIDUAL_CSV_OUT,
            residual_rows,
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
        e4_residual_gram=results[
            "e4_residual_1_over_16"
        ]["gram"],
        e4_residual_eigenvalues=results[
            "e4_residual_1_over_16"
        ]["eigenvalues"],
        e4_gram=results["e4"]["gram"],
        e4_eigenvalues=results[
            "e4"
        ]["eigenvalues"],
        e5_gram=results["e5"]["gram"],
        e5_eigenvalues=results[
            "e5"
        ]["eigenvalues"],
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "completed_target_count:",
        completed_count,
    )
    print(
        "interior_candidate_count:",
        interior_candidate_count,
    )

    for row in summary_rows:
        print(
            row["target"],
            "status:",
            row["solver_status"],
            "margin:",
            row[
                "optimized_margin"
            ],
            "coefficient_residual:",
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
            "elapsed_seconds:",
            row[
                "elapsed_seconds"
            ],
            "interior_candidate:",
            row[
                "floating_interior_sos_candidate"
            ],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", SUMMARY_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", RESIDUAL_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
