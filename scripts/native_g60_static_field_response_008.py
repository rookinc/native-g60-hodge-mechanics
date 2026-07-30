from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

GAUGE_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_gauge_equivalence_audit_007.json"
)

BASELINE_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_uniform_hodge_baseline_002.json"
)

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
)

DELTA1_PATH = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_Delta1_uniform_002.csv"
)

B1_PATH = (
    SOURCE_ROOT
    / "native_g60_B1_vertex_edge_004.csv"
)

B2_PATH = (
    SOURCE_ROOT
    / "native_g60_B2_edge_face_004.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_static_field_response_008.json"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_static_field_response_probes_008.csv"
)

SOLVER_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_static_field_solver_008.npz"
)

TOLERANCE = 1e-9
PROBE_COUNT = 24
RANDOM_SEED = 46008


def read_matrix_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    return np.array(
        [
            [float(value) for value in row[1:]]
            for row in rows[1:]
        ],
        dtype=np.float64,
    )


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(np.max(np.abs(array)))


def numerical_rank(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> int:
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    return int(
        np.count_nonzero(
            singular_values > tolerance
        )
    )


def spectral_pseudoinverse(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray, dict]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (matrix + matrix.T)
    )

    positive = eigenvalues > tolerance

    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[positive] = (
        1.0 / eigenvalues[positive]
    )

    pseudoinverse = (
        eigenvectors
        @ np.diag(inverse_values)
        @ eigenvectors.T
    )

    pseudoinverse = 0.5 * (
        pseudoinverse + pseudoinverse.T
    )

    return pseudoinverse, {
        "dimension": int(matrix.shape[0]),
        "rank": int(np.count_nonzero(positive)),
        "kernel_dimension": int(
            matrix.shape[0]
            - np.count_nonzero(positive)
        ),
        "smallest_positive_eigenvalue": float(
            eigenvalues[positive][0]
        ),
        "largest_positive_eigenvalue": float(
            eigenvalues[positive][-1]
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    SOLVER_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gauge = json.loads(
        GAUGE_PATH.read_text(encoding="utf-8")
    )

    baseline = json.loads(
        BASELINE_PATH.read_text(encoding="utf-8")
    )

    projector_payload = np.load(PROJECTOR_PATH)

    p_exact = np.array(
        projector_payload["P_exact"],
        dtype=np.float64,
    )
    p_harmonic = np.array(
        projector_payload["P_harmonic"],
        dtype=np.float64,
    )
    p_coexact = np.array(
        projector_payload["P_coexact"],
        dtype=np.float64,
    )

    delta1 = read_matrix_csv(DELTA1_PATH)
    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)

    d1 = b2.T

    static_solver, solver_spectrum = (
        spectral_pseudoinverse(delta1)
    )

    range_projector = delta1 @ static_solver
    kernel_projector = (
        np.eye(120, dtype=np.float64)
        - range_projector
    )

    coexact_solver = (
        p_coexact
        @ static_solver
        @ p_coexact
    )

    operator_residuals = {
        "solver_symmetry_max_abs": max_abs(
            static_solver - static_solver.T
        ),
        "moore_penrose_delta_solver_delta": max_abs(
            delta1 @ static_solver @ delta1
            - delta1
        ),
        "moore_penrose_solver_delta_solver": max_abs(
            static_solver @ delta1 @ static_solver
            - static_solver
        ),
        "range_projector_exact_plus_coexact": max_abs(
            range_projector
            - p_exact
            - p_coexact
        ),
        "kernel_projector_harmonic": max_abs(
            kernel_projector
            - p_harmonic
        ),
        "coexact_solver_stays_coexact": max_abs(
            p_coexact @ coexact_solver
            - coexact_solver
        ),
        "coexact_inverse_left": max_abs(
            delta1 @ coexact_solver
            - p_coexact
        ),
        "coexact_inverse_right": max_abs(
            coexact_solver @ delta1
            - p_coexact
        ),
        "coexact_response_divergence": max_abs(
            b1 @ coexact_solver
        ),
    }

    rng = np.random.default_rng(RANDOM_SEED)

    probe_rows = []

    global_residuals = {
        "coexact_exact_solution": 0.0,
        "coexact_response_stays_coexact": 0.0,
        "coexact_response_divergence": 0.0,
        "harmonic_response_zero": 0.0,
        "harmonic_residual_equals_source": 0.0,
        "mixed_residual_equals_harmonic": 0.0,
        "mixed_response_equals_coexact_response": 0.0,
        "minimum_norm_orthogonal_to_kernel": 0.0,
        "source_decomposition": 0.0,
        "curvature_response_consistency": 0.0,
    }

    for probe_id in range(PROBE_COUNT):
        raw_source = rng.normal(size=120)

        source_exact = p_exact @ raw_source
        source_harmonic = p_harmonic @ raw_source
        source_coexact = p_coexact @ raw_source

        source_gauge_fixed = (
            source_harmonic + source_coexact
        )

        coexact_response = (
            coexact_solver @ source_coexact
        )

        harmonic_response = (
            static_solver @ source_harmonic
        )

        mixed_response = (
            static_solver @ source_gauge_fixed
        )

        coexact_equation_residual = (
            delta1 @ coexact_response
            - source_coexact
        )

        harmonic_equation_residual = (
            delta1 @ harmonic_response
            - source_harmonic
        )

        mixed_equation_residual = (
            delta1 @ mixed_response
            - source_gauge_fixed
        )

        curvature = d1 @ mixed_response

        residuals = {
            "coexact_exact_solution": max_abs(
                coexact_equation_residual
            ),
            "coexact_response_stays_coexact": max_abs(
                p_coexact @ coexact_response
                - coexact_response
            ),
            "coexact_response_divergence": max_abs(
                b1 @ coexact_response
            ),
            "harmonic_response_zero": max_abs(
                harmonic_response
            ),
            "harmonic_residual_equals_source": max_abs(
                harmonic_equation_residual
                + source_harmonic
            ),
            "mixed_residual_equals_harmonic": max_abs(
                mixed_equation_residual
                + source_harmonic
            ),
            "mixed_response_equals_coexact_response": max_abs(
                mixed_response - coexact_response
            ),
            "minimum_norm_orthogonal_to_kernel": max_abs(
                p_harmonic @ mixed_response
            ),
            "source_decomposition": max_abs(
                raw_source
                - source_exact
                - source_harmonic
                - source_coexact
            ),
            "curvature_response_consistency": max_abs(
                d1 @ coexact_response
                - curvature
            ),
        }

        for name, value in residuals.items():
            global_residuals[name] = max(
                global_residuals[name],
                value,
            )

        probe_rows.append(
            {
                "probe_id": probe_id,
                "raw_source_norm": float(
                    np.linalg.norm(raw_source)
                ),
                "exact_source_norm": float(
                    np.linalg.norm(source_exact)
                ),
                "harmonic_source_norm": float(
                    np.linalg.norm(source_harmonic)
                ),
                "coexact_source_norm": float(
                    np.linalg.norm(source_coexact)
                ),
                "coexact_response_norm": float(
                    np.linalg.norm(coexact_response)
                ),
                "mixed_response_norm": float(
                    np.linalg.norm(mixed_response)
                ),
                "mixed_residual_norm": float(
                    np.linalg.norm(
                        mixed_equation_residual
                    )
                ),
                "curvature_norm": float(
                    np.linalg.norm(curvature)
                ),
                **{
                    name + "_max_abs": value
                    for name, value in residuals.items()
                },
                "all_checks_pass": all(
                    value < TOLERANCE
                    for value in residuals.values()
                ),
            }
        )

    all_probes_pass = all(
        row["all_checks_pass"]
        for row in probe_rows
    )

    checks = {
        "input_gauge_audit_pass": (
            gauge.get("audit_pass") is True
        ),
        "input_uniform_hodge_pass": (
            baseline.get("audit_pass") is True
        ),
        "delta1_shape_is_120_by_120": (
            delta1.shape == (120, 120)
        ),
        "delta1_rank_is_78": (
            numerical_rank(delta1) == 78
        ),
        "delta1_kernel_dimension_is_42": (
            120 - numerical_rank(delta1) == 42
        ),
        "static_solver_rank_is_78": (
            numerical_rank(static_solver) == 78
        ),
        "solver_is_symmetric": (
            operator_residuals[
                "solver_symmetry_max_abs"
            ]
            < TOLERANCE
        ),
        "moore_penrose_identities_pass": (
            operator_residuals[
                "moore_penrose_delta_solver_delta"
            ]
            < TOLERANCE
            and operator_residuals[
                "moore_penrose_solver_delta_solver"
            ]
            < TOLERANCE
        ),
        "range_is_exact_plus_coexact": (
            operator_residuals[
                "range_projector_exact_plus_coexact"
            ]
            < TOLERANCE
        ),
        "kernel_is_harmonic": (
            operator_residuals[
                "kernel_projector_harmonic"
            ]
            < TOLERANCE
        ),
        "coexact_solver_is_inverse_on_coexact_sector": (
            operator_residuals[
                "coexact_inverse_left"
            ]
            < TOLERANCE
            and operator_residuals[
                "coexact_inverse_right"
            ]
            < TOLERANCE
        ),
        "coexact_solver_is_divergence_free": (
            operator_residuals[
                "coexact_response_divergence"
            ]
            < TOLERANCE
        ),
        "all_deterministic_probes_pass": (
            all_probes_pass
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_static_field_response_008"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_static_solvability_and_harmonic_obstruction_identified"
            if audit_pass
            else "native_g60_static_field_response_audit_failed"
        ),
        "inputs": {
            "gauge_equivalence": str(
                GAUGE_PATH.relative_to(ROOT)
            ),
            "uniform_hodge_baseline": str(
                BASELINE_PATH.relative_to(ROOT)
            ),
            "delta1": str(
                DELTA1_PATH.relative_to(ROOT)
            ),
            "hodge_projectors": str(
                PROJECTOR_PATH.relative_to(ROOT)
            ),
        },
        "static_problem": {
            "equation": "Delta1 A = J",
            "gauge_fixed_source_space": (
                "Harmonic(42) direct_sum Coexact(19)"
            ),
            "exactly_solvable_gauge_fixed_source_space": (
                "Coexact(19)"
            ),
            "harmonic_obstruction_dimension": 42,
            "minimum_norm_solver": (
                "Moore-Penrose pseudoinverse of Delta1"
            ),
            "coexact_solver": (
                "P_coexact Delta1^+ P_coexact"
            ),
        },
        "solver_spectrum": solver_spectrum,
        "checks": checks,
        "operator_residuals": operator_residuals,
        "probe_audit": {
            "random_seed": RANDOM_SEED,
            "probe_count": PROBE_COUNT,
            "all_probes_pass": all_probes_pass,
            "global_maximum_residuals": (
                global_residuals
            ),
        },
        "earned_interpretation": {
            "coexact_source": (
                "Has a unique minimum-norm gauge-fixed static response."
            ),
            "harmonic_source": (
                "Lies in ker(Delta1), so no static solution exists "
                "unless that harmonic component is removed or supplied "
                "with an additional constitutive constraint."
            ),
            "mixed_source": (
                "The pseudoinverse responds only to the coexact part; "
                "the unsatisfied residual is exactly the harmonic part."
            ),
            "physical_source_claim": False,
        },
        "outputs": {
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(ROOT)
            ),
            "solver_npz": str(
                SOLVER_NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "static_solver_constructed": audit_pass,
            "harmonic_solvability_obstruction_identified": (
                audit_pass
            ),
            "minimum_norm_response_is_mathematical": True,
            "source_law_derived": False,
            "constitutive_law_derived": False,
            "dynamics_defined": False,
            "electromagnetism_claim": False,
            "maxwell_claim": False,
            "physical_energy_claim": False,
            "physical_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False,
        },
    }

    JSON_OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with PROBE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(probe_rows[0]),
        )

        writer.writeheader()
        writer.writerows(probe_rows)

    np.savez_compressed(
        SOLVER_NPZ_OUT,
        Delta1=delta1,
        Delta1_pseudoinverse=static_solver,
        P_range=range_projector,
        P_kernel=kernel_projector,
        coexact_solver=coexact_solver,
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "Delta1_rank/kernel:",
        numerical_rank(delta1),
        120 - numerical_rank(delta1),
    )
    print(
        "static_solver_rank:",
        numerical_rank(static_solver),
    )
    print(
        "range_exact_plus_coexact_max_abs:",
        operator_residuals[
            "range_projector_exact_plus_coexact"
        ],
    )
    print(
        "kernel_harmonic_max_abs:",
        operator_residuals[
            "kernel_projector_harmonic"
        ],
    )
    print(
        "coexact_inverse_left/right:",
        operator_residuals[
            "coexact_inverse_left"
        ],
        operator_residuals[
            "coexact_inverse_right"
        ],
    )
    print(
        "all_deterministic_probes_pass:",
        all_probes_pass,
    )
    print(
        "probe_global_maximum_residuals:",
        global_residuals,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", SOLVER_NPZ_OUT)


if __name__ == "__main__":
    main()
