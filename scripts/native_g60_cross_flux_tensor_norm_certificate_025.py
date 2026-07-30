from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

OBSTRUCTION_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_coercivity_obstruction_024.json"
)

OBSTRUCTION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_coercivity_obstruction_024.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_tensor_norm_certificate_025.json"
)

MINOR_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_rank_one_minor_multipliers_025.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_tensor_norm_certificate_probes_025.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_tensor_norm_certificate_025.npz"
)

TARGET_NORM_SQUARED = 1.0 / 9.0

RANDOM_SEED = 46025
PROBE_COUNT = 4096

PSD_TOLERANCE = 5e-7
IDENTITY_TOLERANCE = 5e-7
RANK_ONE_PROBE_TOLERANCE = 5e-7


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def symmetric_entry_matrix(
    size: int,
    first: int,
    second: int,
    coefficient: float,
) -> np.ndarray:
    matrix = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    if first == second:
        matrix[first, second] = coefficient
    else:
        matrix[first, second] = 0.5 * coefficient
        matrix[second, first] = 0.5 * coefficient

    return matrix


def tensor_index(
    first: int,
    second: int,
) -> int:
    return 6 * first + second


def build_rank_one_minor_matrices() -> tuple[
    list[dict],
    list[np.ndarray],
]:
    records = []
    matrices = []

    for a, c in combinations(range(6), 2):
        for b, d in combinations(range(6), 2):
            ab = tensor_index(a, b)
            cd = tensor_index(c, d)
            ad = tensor_index(a, d)
            cb = tensor_index(c, b)

            matrix = (
                symmetric_entry_matrix(
                    36,
                    ab,
                    cd,
                    1.0,
                )
                - symmetric_entry_matrix(
                    36,
                    ad,
                    cb,
                    1.0,
                )
            )

            records.append(
                {
                    "minor_id": len(records),
                    "row_first": a,
                    "row_second": c,
                    "column_first": b,
                    "column_second": d,
                    "positive_first_index": ab,
                    "positive_second_index": cd,
                    "negative_first_index": ad,
                    "negative_second_index": cb,
                }
            )

            matrices.append(matrix)

    return records, matrices


def bilinear_flux(
    channel: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    return channel @ np.kron(u, v)


def normalized(
    vector: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        raise RuntimeError("zero vector")

    return vector / norm


def solve_sos_certificate_scipy(
    base_matrix: np.ndarray,
    minor_matrices: list[np.ndarray],
) -> dict:
    try:
        from scipy.optimize import minimize
        from scipy.special import logsumexp
    except ImportError:
        return {
            "solver_available": False,
            "status": "scipy_unavailable",
            "message": (
                "Neither cvxpy nor scipy is available."
            ),
        }

    minor_array = np.array(
        minor_matrices,
        dtype=np.float64,
    )

    multiplier_count = len(
        minor_matrices
    )

    temperatures = (
        5e-2,
        2e-2,
        1e-2,
        5e-3,
        2e-3,
        1e-3,
        5e-4,
        2e-4,
        1e-4,
    )

    multipliers = np.zeros(
        multiplier_count,
        dtype=np.float64,
    )

    best_multipliers = multipliers.copy()
    best_true_minimum = -float("inf")

    stage_rows = []
    total_evaluation_count = 0

    def build_gram(
        values: np.ndarray,
    ) -> np.ndarray:
        gram = (
            base_matrix
            + np.tensordot(
                values,
                minor_array,
                axes=(0, 0),
            )
        )

        return 0.5 * (
            gram + gram.T
        )

    for stage_id, temperature in enumerate(
        temperatures,
        start=1,
    ):
        stage_evaluation_count = 0

        def objective_and_gradient(
            values: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            nonlocal stage_evaluation_count
            nonlocal total_evaluation_count

            stage_evaluation_count += 1
            total_evaluation_count += 1

            gram = build_gram(values)

            eigenvalues, eigenvectors = np.linalg.eigh(
                gram
            )

            scaled = (
                -eigenvalues
                / temperature
            )

            smooth_minimum = (
                -temperature
                * logsumexp(scaled)
            )

            shifted = (
                scaled
                - np.max(scaled)
            )

            weights = np.exp(shifted)
            weights /= np.sum(weights)

            density = (
                eigenvectors
                @ np.diag(weights)
                @ eigenvectors.T
            )

            smooth_gradient = np.einsum(
                "ij,mij->m",
                density,
                minor_array,
            )

            # scipy minimizes, while we maximize smooth_minimum.
            return (
                -float(smooth_minimum),
                -smooth_gradient,
            )

        result = minimize(
            objective_and_gradient,
            multipliers,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 3000,
                "maxls": 100,
                "ftol": 1e-15,
                "gtol": 1e-11,
                "maxcor": 75,
            },
        )

        multipliers = np.array(
            result.x,
            dtype=np.float64,
        )

        gram = build_gram(
            multipliers
        )

        eigenvalues = np.linalg.eigvalsh(
            gram
        )

        true_minimum = float(
            eigenvalues[0]
        )

        minimum_multiplicity = int(
            np.count_nonzero(
                np.abs(
                    eigenvalues
                    - true_minimum
                )
                < 1e-7
            )
        )

        if true_minimum > best_true_minimum:
            best_true_minimum = true_minimum
            best_multipliers = (
                multipliers.copy()
            )

        stage_rows.append(
            {
                "stage_id": stage_id,
                "temperature": temperature,
                "success": bool(
                    result.success
                ),
                "status": int(
                    result.status
                ),
                "message": str(
                    result.message
                ),
                "iteration_count": int(
                    result.nit
                ),
                "evaluation_count": int(
                    result.nfev
                ),
                "true_minimum_eigenvalue": (
                    true_minimum
                ),
                "minimum_eigenvalue_multiplicity": (
                    minimum_multiplicity
                ),
                "smooth_objective": float(
                    -result.fun
                ),
                "gradient_max_abs": float(
                    np.max(
                        np.abs(
                            result.jac
                        )
                    )
                ),
                "multiplier_norm": float(
                    np.linalg.norm(
                        multipliers
                    )
                ),
            }
        )

        print(
            "scipy_sos_stage:",
            f"{stage_id}/{len(temperatures)}",
            "temperature:",
            temperature,
            "iterations:",
            int(result.nit),
            "true_min_eigenvalue:",
            true_minimum,
            "multiplicity:",
            minimum_multiplicity,
            flush=True,
        )

    gram_matrix = build_gram(
        best_multipliers
    )

    eigenvalues = np.linalg.eigvalsh(
        gram_matrix
    )

    return {
        "solver_available": True,
        "solver_backend": (
            "scipy_l_bfgs_b_soft_minimum_homotopy"
        ),
        "status": (
            "optimal_candidate"
            if eigenvalues[0] >= -PSD_TOLERANCE
            else "candidate_not_psd"
        ),
        "message": (
            "Smooth minimum-eigenvalue homotopy completed."
        ),
        "success": bool(
            all(
                row["success"]
                for row in stage_rows
            )
        ),
        "stage_rows": stage_rows,
        "iteration_count": int(
            sum(
                row["iteration_count"]
                for row in stage_rows
            )
        ),
        "evaluation_count": int(
            total_evaluation_count
        ),
        "objective_margin": float(
            eigenvalues[0]
        ),
        "multipliers": best_multipliers,
        "gram_matrix": gram_matrix,
        "gram_eigenvalues": eigenvalues,
        "minimum_gram_eigenvalue": float(
            eigenvalues[0]
        ),
        "maximum_gram_eigenvalue": float(
            eigenvalues[-1]
        ),
        "minimum_eigenvalue_multiplicity": int(
            np.count_nonzero(
                np.abs(
                    eigenvalues
                    - eigenvalues[0]
                )
                < 1e-7
            )
        ),
    }


def solve_sos_certificate(
    base_matrix: np.ndarray,
    minor_matrices: list[np.ndarray],
) -> dict:
    try:
        import cvxpy as cp
    except ImportError:
        return solve_sos_certificate_scipy(
            base_matrix,
            minor_matrices,
        )

    multiplier_count = len(
        minor_matrices
    )

    multipliers = cp.Variable(
        multiplier_count
    )

    margin = cp.Variable()

    correction = 0

    for index, matrix in enumerate(
        minor_matrices
    ):
        correction = (
            correction
            + multipliers[index]
            * matrix
        )

    gram_expression = (
        base_matrix + correction
    )

    constraints = [
        gram_expression
        - margin * np.eye(36)
        >> 0
    ]

    problem = cp.Problem(
        cp.Maximize(margin),
        constraints,
    )

    installed_solvers = set(
        cp.installed_solvers()
    )

    solver_attempts = []

    for solver_name in (
        "CLARABEL",
        "SCS",
    ):
        if solver_name not in installed_solvers:
            continue

        try:
            if solver_name == "CLARABEL":
                problem.solve(
                    solver=solver_name,
                    verbose=False,
                )
            else:
                problem.solve(
                    solver=solver_name,
                    verbose=False,
                    eps=1e-8,
                    max_iters=200000,
                )

            solver_attempts.append(
                {
                    "solver": solver_name,
                    "status": problem.status,
                    "value": problem.value,
                }
            )

            if problem.status in (
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            ):
                break

        except Exception as error:
            solver_attempts.append(
                {
                    "solver": solver_name,
                    "status": "exception",
                    "message": repr(error),
                }
            )

    if multipliers.value is None:
        return {
            "solver_available": True,
            "status": problem.status,
            "solver_attempts": solver_attempts,
            "message": (
                "No numerical multiplier vector was returned."
            ),
        }

    multiplier_values = np.array(
        multipliers.value,
        dtype=np.float64,
    )

    gram_matrix = base_matrix.copy()

    for value, matrix in zip(
        multiplier_values,
        minor_matrices,
    ):
        gram_matrix += value * matrix

    gram_matrix = 0.5 * (
        gram_matrix + gram_matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        gram_matrix
    )

    return {
        "solver_available": True,
        "status": problem.status,
        "solver_attempts": solver_attempts,
        "objective_margin": (
            None
            if margin.value is None
            else float(margin.value)
        ),
        "multipliers": multiplier_values,
        "gram_matrix": gram_matrix,
        "gram_eigenvalues": eigenvalues,
        "minimum_gram_eigenvalue": float(
            eigenvalues[0]
        ),
        "maximum_gram_eigenvalue": float(
            eigenvalues[-1]
        ),
    }


def verify_rank_one_identity(
    channel: np.ndarray,
    base_matrix: np.ndarray,
    gram_matrix: np.ndarray,
    minor_matrices: list[np.ndarray],
    multiplier_values: np.ndarray,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    probe_rows = []

    maximum_minor_residual = 0.0
    maximum_identity_residual = 0.0
    minimum_gap = float("inf")
    maximum_ratio = 0.0

    for probe_id in range(PROBE_COUNT):
        u = normalized(
            rng.normal(size=6)
        )

        v = normalized(
            rng.normal(size=6)
        )

        z = np.kron(u, v)

        flux = bilinear_flux(
            channel,
            u,
            v,
        )

        flux_norm_squared = float(
            np.dot(flux, flux)
        )

        direct_gap = (
            TARGET_NORM_SQUARED
            - flux_norm_squared
        )

        base_value = float(
            z @ base_matrix @ z
        )

        gram_value = float(
            z @ gram_matrix @ z
        )

        minor_values = np.array(
            [
                z @ matrix @ z
                for matrix in minor_matrices
            ],
            dtype=np.float64,
        )

        weighted_minor_value = float(
            np.dot(
                multiplier_values,
                minor_values,
            )
        )

        identity_residual = abs(
            gram_value
            - direct_gap
        )

        minor_residual = float(
            np.max(
                np.abs(minor_values)
            )
        )

        ratio = float(
            np.linalg.norm(flux)
        )

        maximum_minor_residual = max(
            maximum_minor_residual,
            minor_residual,
        )

        maximum_identity_residual = max(
            maximum_identity_residual,
            identity_residual,
        )

        minimum_gap = min(
            minimum_gap,
            direct_gap,
        )

        maximum_ratio = max(
            maximum_ratio,
            ratio,
        )

        probe_rows.append(
            {
                "probe_id": probe_id,
                "flux_norm": ratio,
                "flux_norm_squared": (
                    flux_norm_squared
                ),
                "direct_gap": direct_gap,
                "base_quadratic_value": (
                    base_value
                ),
                "gram_quadratic_value": (
                    gram_value
                ),
                "weighted_minor_value": (
                    weighted_minor_value
                ),
                "maximum_minor_value_abs": (
                    minor_residual
                ),
                "identity_residual": (
                    identity_residual
                ),
            }
        )

    summary = {
        "probe_count": PROBE_COUNT,
        "minimum_direct_gap": minimum_gap,
        "maximum_flux_norm": maximum_ratio,
        "maximum_minor_residual": (
            maximum_minor_residual
        ),
        "maximum_identity_residual": (
            maximum_identity_residual
        ),
    }

    return probe_rows, summary


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MINOR_CSV_OUT.parent.mkdir(
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

    obstruction_receipt = json.loads(
        OBSTRUCTION_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        OBSTRUCTION_NPZ_PATH
    )

    channel = np.array(
        data["channel"],
        dtype=np.float64,
    )

    witness_gain = float(
        np.array(
            data["witness_gain"],
            dtype=np.float64,
        )[0]
    )

    if channel.shape != (4, 36):
        raise RuntimeError(
            f"unexpected channel shape: {channel.shape}"
        )

    base_matrix = (
        TARGET_NORM_SQUARED
        * np.eye(36)
        - channel.T @ channel
    )

    base_eigenvalues = np.linalg.eigvalsh(
        0.5
        * (
            base_matrix
            + base_matrix.T
        )
    )

    minor_records, minor_matrices = (
        build_rank_one_minor_matrices()
    )

    search = solve_sos_certificate(
        base_matrix,
        minor_matrices,
    )

    solver_available = bool(
        search.get(
            "solver_available",
            False,
        )
    )

    certificate_found = (
        solver_available
        and "gram_matrix" in search
    )

    probe_rows = []
    probe_summary = None

    if certificate_found:
        gram_matrix = np.array(
            search["gram_matrix"],
            dtype=np.float64,
        )

        multiplier_values = np.array(
            search["multipliers"],
            dtype=np.float64,
        )

        probe_rows, probe_summary = (
            verify_rank_one_identity(
                channel,
                base_matrix,
                gram_matrix,
                minor_matrices,
                multiplier_values,
            )
        )

        for record, value in zip(
            minor_records,
            multiplier_values,
        ):
            record["multiplier"] = float(
                value
            )

        minimum_gram_eigenvalue = float(
            search[
                "minimum_gram_eigenvalue"
            ]
        )

        numerical_psd = (
            minimum_gram_eigenvalue
            >= -PSD_TOLERANCE
        )

        rank_one_identity_resolved = (
            probe_summary[
                "maximum_minor_residual"
            ]
            < IDENTITY_TOLERANCE
            and probe_summary[
                "maximum_identity_residual"
            ]
            < RANK_ONE_PROBE_TOLERANCE
        )
    else:
        gram_matrix = np.empty(
            (0, 0),
            dtype=np.float64,
        )

        multiplier_values = np.empty(
            0,
            dtype=np.float64,
        )

        minimum_gram_eigenvalue = None
        numerical_psd = False
        rank_one_identity_resolved = False

        for record in minor_records:
            record["multiplier"] = None

    witness_matches_one_third = abs(
        witness_gain - 1.0 / 3.0
    ) < 1e-10

    numerical_certificate_pass = (
        certificate_found
        and numerical_psd
        and rank_one_identity_resolved
        and witness_matches_one_third
    )

    exact_rational_certificate_recovered = False

    checks = {
        "input_024_theorem_pass": (
            obstruction_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "channel_shape_is_4_by_36": (
            channel.shape == (4, 36)
        ),
        "rank_one_minor_count_is_225": (
            len(minor_matrices) == 225
        ),
        "numerical_solver_available": (
            solver_available
        ),
        "numerical_gram_candidate_found": (
            certificate_found
        ),
        "numerical_gram_candidate_psd": (
            numerical_psd
        ),
        "rank_one_identity_resolved_on_probes": (
            rank_one_identity_resolved
        ),
        "lower_bound_witness_matches_one_third": (
            witness_matches_one_third
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = (
        audit_pass
        and exact_rational_certificate_recovered
    )

    if theorem_pass:
        verdict = (
            "native_g60_cross_flux_tensor_norm_one_third_certified"
        )
    elif audit_pass:
        verdict = (
            "native_g60_cross_flux_tensor_norm_one_third_"
            "numerical_sos_candidate_found"
        )
    elif not solver_available:
        verdict = (
            "native_g60_cross_flux_tensor_norm_sos_solver_unavailable"
        )
    else:
        verdict = (
            "native_g60_cross_flux_tensor_norm_certificate_not_found"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_tensor_norm_certificate_025"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "target": {
            "inequality": (
                "||T4(u,v)|| <= (1/3)||u||||v||"
            ),
            "target_norm": "1/3",
            "target_norm_squared": (
                TARGET_NORM_SQUARED
            ),
            "known_lower_bound_witness": (
                witness_gain
            ),
        },
        "certificate_model": {
            "rank_one_coordinate": (
                "z = u tensor v in R^36"
            ),
            "base_quadratic_form": (
                "(1/9)I36 - T4^T T4"
            ),
            "rank_one_relations": (
                "z_ab z_cd - z_ad z_cb = 0"
            ),
            "minor_count": len(
                minor_matrices
            ),
            "candidate_form": (
                "Q = base + sum lambda_m M_m"
            ),
            "required_property": (
                "Q positive semidefinite"
            ),
        },
        "base_matrix": {
            "minimum_eigenvalue": float(
                base_eigenvalues[0]
            ),
            "maximum_eigenvalue": float(
                base_eigenvalues[-1]
            ),
            "negative_eigenvalue_count": int(
                np.count_nonzero(
                    base_eigenvalues < -1e-10
                )
            ),
        },
        "solver_search": {
            key: value
            for key, value in search.items()
            if key not in (
                "gram_matrix",
                "multipliers",
                "gram_eigenvalues",
            )
        },
        "probe_summary": (
            probe_summary
        ),
        "checks": checks,
        "certificate_status": {
            "numerical_certificate_pass": (
                numerical_certificate_pass
            ),
            "exact_rational_certificate_recovered": (
                exact_rational_certificate_recovered
            ),
            "sharp_tensor_norm_proved": (
                theorem_pass
            ),
        },
        "boundary": {
            "numerical_sos_candidate_search_completed": (
                solver_available
            ),
            "rank_one_upper_bound_numerically_supported": (
                numerical_certificate_pass
            ),
            "exact_rank_one_upper_bound_proved": (
                theorem_pass
            ),
            "sharp_tensor_norm_proved": (
                theorem_pass
            ),
            "quartic_descent_rate_minus_one_over_180_proved": (
                theorem_pass
            ),
            "physical_claim": False,
        },
        "outputs": {
            "minor_multiplier_csv": str(
                MINOR_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "candidate_npz": str(
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

    with MINOR_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                minor_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            minor_records
        )

    if probe_rows:
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
    else:
        PROBE_CSV_OUT.write_text(
            "probe_id,status\n"
            "0,certificate_not_available\n",
            encoding="utf-8",
        )

    np.savez_compressed(
        NPZ_OUT,
        channel=channel,
        base_matrix=base_matrix,
        base_eigenvalues=base_eigenvalues,
        minor_matrices=np.array(
            minor_matrices,
            dtype=np.float64,
        ),
        multipliers=multiplier_values,
        gram_matrix=gram_matrix,
        target_norm_squared=np.array(
            [TARGET_NORM_SQUARED],
            dtype=np.float64,
        ),
        witness_gain=np.array(
            [witness_gain],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "solver_available:",
        solver_available,
    )
    print(
        "certificate_found:",
        certificate_found,
    )
    print(
        "rank_one_minor_count:",
        len(minor_matrices),
    )
    print(
        "base_minimum_eigenvalue:",
        float(
            base_eigenvalues[0]
        ),
    )

    if certificate_found:
        print(
            "gram_minimum_eigenvalue:",
            minimum_gram_eigenvalue,
        )
        print(
            "objective_margin:",
            search.get(
                "objective_margin"
            ),
        )
        print(
            "probe_summary:",
            probe_summary,
        )

    print(
        "witness_gain:",
        witness_gain,
    )
    print(
        "exact_rational_certificate_recovered:",
        exact_rational_certificate_recovered,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", MINOR_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
