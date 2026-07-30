from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

DECOMPOSITION_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_hodge_decomposition_003.json"
)

SYMMETRY_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_hodge_symmetry_commutation_005.json"
)

B1_PATH = SOURCE_ROOT / "native_g60_B1_vertex_edge_004.csv"
B2_PATH = SOURCE_ROOT / "native_g60_B2_edge_face_004.csv"

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_gauge_equivalence_audit_007.json"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_gauge_equivalence_probes_007.csv"
)

GAUGE_FIX_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_gauge_fix_operators_007.npz"
)

TOLERANCE = 1e-9
PROBE_COUNT = 32
RANDOM_SEED = 46007


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


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    GAUGE_FIX_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decomposition = json.loads(
        DECOMPOSITION_PATH.read_text(encoding="utf-8")
    )

    symmetry = json.loads(
        SYMMETRY_PATH.read_text(encoding="utf-8")
    )

    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)

    d0 = b1.T
    d1 = b2.T

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

    identity_edge = np.eye(120, dtype=np.float64)

    p_gauge_invariant = (
        p_harmonic + p_coexact
    )

    vertex_laplacian = b1 @ b1.T

    vertex_laplacian_pinv = np.linalg.pinv(
        vertex_laplacian,
        rcond=1e-12,
    )

    # The exact projector can also be constructed by solving the
    # vertex-potential gauge problem.
    p_exact_from_gauge_solver = (
        d0
        @ vertex_laplacian_pinv
        @ b1
    )

    p_gauge_fixed_from_solver = (
        identity_edge
        - p_exact_from_gauge_solver
    )

    exact_projector_solver_residual = max_abs(
        p_exact_from_gauge_solver - p_exact
    )

    gauge_fixed_projector_solver_residual = max_abs(
        p_gauge_fixed_from_solver
        - p_gauge_invariant
    )

    chain_residual = d1 @ d0

    exact_curvature_residual = d1 @ p_exact
    harmonic_curvature_residual = d1 @ p_harmonic

    curvature_coexact_identity_residual = (
        d1 @ p_coexact - d1
    )

    divergence_gauge_fixed_residual = (
        b1 @ p_gauge_invariant
    )

    gauge_fixed_idempotence_residual = (
        p_gauge_invariant
        @ p_gauge_invariant
        - p_gauge_invariant
    )

    quotient_dimension = numerical_rank(
        p_gauge_invariant
    )

    rng = np.random.default_rng(RANDOM_SEED)

    probe_rows = []

    global_residuals = {
        "gauge_curvature": 0.0,
        "exact_shift": 0.0,
        "harmonic_invariance": 0.0,
        "coexact_invariance": 0.0,
        "gauge_fixed_representative": 0.0,
        "gauge_fixed_divergence": 0.0,
        "curvature_from_coexact_only": 0.0,
        "gauge_orbit_reconstruction": 0.0,
        "constant_potential_null": 0.0,
    }

    for probe_id in range(PROBE_COUNT):
        edge_field = rng.normal(size=120)
        potential = rng.normal(size=60)

        gauge_shift = d0 @ potential
        transformed = edge_field + gauge_shift

        exact_before = p_exact @ edge_field
        harmonic_before = p_harmonic @ edge_field
        coexact_before = p_coexact @ edge_field

        exact_after = p_exact @ transformed
        harmonic_after = p_harmonic @ transformed
        coexact_after = p_coexact @ transformed

        gauge_fixed_before = (
            p_gauge_invariant @ edge_field
        )

        gauge_fixed_after = (
            p_gauge_invariant @ transformed
        )

        curvature_before = d1 @ edge_field
        curvature_after = d1 @ transformed

        curvature_from_coexact = (
            d1 @ coexact_before
        )

        reconstructed = (
            exact_before
            + harmonic_before
            + coexact_before
        )

        constant_value = float(rng.normal())
        constant_potential = np.full(
            60,
            constant_value,
            dtype=np.float64,
        )

        constant_shift = d0 @ constant_potential

        residuals = {
            "gauge_curvature": max_abs(
                curvature_after - curvature_before
            ),
            "exact_shift": max_abs(
                exact_after
                - exact_before
                - gauge_shift
            ),
            "harmonic_invariance": max_abs(
                harmonic_after - harmonic_before
            ),
            "coexact_invariance": max_abs(
                coexact_after - coexact_before
            ),
            "gauge_fixed_representative": max_abs(
                gauge_fixed_after
                - gauge_fixed_before
            ),
            "gauge_fixed_divergence": max_abs(
                b1 @ gauge_fixed_before
            ),
            "curvature_from_coexact_only": max_abs(
                curvature_before
                - curvature_from_coexact
            ),
            "gauge_orbit_reconstruction": max_abs(
                reconstructed - edge_field
            ),
            "constant_potential_null": max_abs(
                constant_shift
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
                "edge_field_norm": float(
                    np.linalg.norm(edge_field)
                ),
                "potential_norm": float(
                    np.linalg.norm(potential)
                ),
                "gauge_shift_norm": float(
                    np.linalg.norm(gauge_shift)
                ),
                "exact_norm": float(
                    np.linalg.norm(exact_before)
                ),
                "harmonic_norm": float(
                    np.linalg.norm(harmonic_before)
                ),
                "coexact_norm": float(
                    np.linalg.norm(coexact_before)
                ),
                "curvature_norm": float(
                    np.linalg.norm(curvature_before)
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
        "input_hodge_decomposition_pass": (
            decomposition.get("audit_pass") is True
        ),
        "input_symmetry_commutation_pass": (
            symmetry.get("audit_pass") is True
        ),
        "d0_shape_is_120_by_60": (
            d0.shape == (120, 60)
        ),
        "d1_shape_is_20_by_120": (
            d1.shape == (20, 120)
        ),
        "d1_d0_is_zero": (
            max_abs(chain_residual) < TOLERANCE
        ),
        "exact_projector_matches_gauge_solver": (
            exact_projector_solver_residual
            < TOLERANCE
        ),
        "gauge_fixed_projector_matches_harmonic_plus_coexact": (
            gauge_fixed_projector_solver_residual
            < TOLERANCE
        ),
        "exact_sector_has_zero_curvature": (
            max_abs(exact_curvature_residual)
            < TOLERANCE
        ),
        "harmonic_sector_has_zero_curvature": (
            max_abs(harmonic_curvature_residual)
            < TOLERANCE
        ),
        "curvature_depends_only_on_coexact_sector": (
            max_abs(
                curvature_coexact_identity_residual
            )
            < TOLERANCE
        ),
        "gauge_fixed_sector_is_divergence_free": (
            max_abs(
                divergence_gauge_fixed_residual
            )
            < TOLERANCE
        ),
        "gauge_fixed_projector_is_idempotent": (
            max_abs(
                gauge_fixed_idempotence_residual
            )
            < TOLERANCE
        ),
        "gauge_orbit_dimension_is_59": (
            numerical_rank(p_exact) == 59
        ),
        "gauge_quotient_dimension_is_61": (
            quotient_dimension == 61
        ),
        "gauge_quotient_splits_42_plus_19": (
            numerical_rank(p_harmonic) == 42
            and numerical_rank(p_coexact) == 19
            and 42 + 19 == quotient_dimension
        ),
        "all_deterministic_probes_pass": (
            all_probes_pass
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_gauge_equivalence_audit_007"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_uniform_gauge_equivalence_identified"
            if audit_pass
            else "native_g60_gauge_equivalence_audit_failed"
        ),
        "inputs": {
            "hodge_decomposition": str(
                DECOMPOSITION_PATH.relative_to(ROOT)
            ),
            "symmetry_commutation": str(
                SYMMETRY_PATH.relative_to(ROOT)
            ),
            "b1": str(B1_PATH.relative_to(ROOT)),
            "b2": str(B2_PATH.relative_to(ROOT)),
            "hodge_projectors": str(
                PROJECTOR_PATH.relative_to(ROOT)
            ),
        },
        "gauge_structure": {
            "edge_potential_space_dimension": 120,
            "vertex_parameter_space_dimension": 60,
            "constant_parameter_kernel_dimension": 1,
            "gauge_orbit_dimension": 59,
            "gauge_quotient_dimension": (
                quotient_dimension
            ),
            "transformation": "A -> A + d0 phi",
            "curvature": "F = d1 A",
            "gauge_fixed_representative": (
                "A_perp = (I - P_exact) A"
            ),
            "gauge_fixed_space": (
                "Harmonic(42) direct_sum Coexact(19)"
            ),
        },
        "checks": checks,
        "operator_residuals": {
            "d1_d0_max_abs": max_abs(
                chain_residual
            ),
            "exact_projector_solver_max_abs": (
                exact_projector_solver_residual
            ),
            "gauge_fixed_projector_solver_max_abs": (
                gauge_fixed_projector_solver_residual
            ),
            "exact_curvature_max_abs": max_abs(
                exact_curvature_residual
            ),
            "harmonic_curvature_max_abs": max_abs(
                harmonic_curvature_residual
            ),
            "curvature_coexact_identity_max_abs": (
                max_abs(
                    curvature_coexact_identity_residual
                )
            ),
            "gauge_fixed_divergence_max_abs": (
                max_abs(
                    divergence_gauge_fixed_residual
                )
            ),
            "gauge_fixed_idempotence_max_abs": (
                max_abs(
                    gauge_fixed_idempotence_residual
                )
            ),
        },
        "probe_audit": {
            "random_seed": RANDOM_SEED,
            "probe_count": PROBE_COUNT,
            "all_probes_pass": all_probes_pass,
            "global_maximum_residuals": (
                global_residuals
            ),
        },
        "earned_interpretation": {
            "exact_sector": (
                "Gauge-removable local gradient component."
            ),
            "harmonic_sector": (
                "Gauge-invariant, curvature-free global "
                "circulation component."
            ),
            "coexact_sector": (
                "Gauge-invariant component carrying all "
                "face curvature."
            ),
            "curvature_is_complete_gauge_invariant": False,
            "reason_curvature_is_not_complete": (
                "Distinct harmonic classes have identical "
                "zero face curvature but are not gauge equivalent."
            ),
        },
        "outputs": {
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(ROOT)
            ),
            "gauge_fix_npz": str(
                GAUGE_FIX_NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "gauge_equivalence_audited": audit_pass,
            "uniform_coulomb_like_gauge_constructed": (
                audit_pass
            ),
            "gauge_quotient_identified": audit_pass,
            "harmonic_global_classes_survive_gauge": (
                audit_pass
            ),
            "face_curvature_is_complete_gauge_classifier": (
                False
            ),
            "static_field_equation_defined": False,
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
        GAUGE_FIX_NPZ_OUT,
        vertex_laplacian=vertex_laplacian,
        vertex_laplacian_pseudoinverse=(
            vertex_laplacian_pinv
        ),
        P_exact_from_gauge_solver=(
            p_exact_from_gauge_solver
        ),
        P_gauge_fixed=(
            p_gauge_invariant
        ),
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "gauge_orbit_dimension:",
        numerical_rank(p_exact),
    )
    print(
        "gauge_quotient_dimension:",
        quotient_dimension,
    )
    print(
        "quotient_split_harmonic_coexact:",
        numerical_rank(p_harmonic),
        numerical_rank(p_coexact),
    )
    print(
        "d1_d0_max_abs:",
        payload["operator_residuals"][
            "d1_d0_max_abs"
        ],
    )
    print(
        "exact_projector_solver_max_abs:",
        exact_projector_solver_residual,
    )
    print(
        "gauge_fixed_projector_solver_max_abs:",
        gauge_fixed_projector_solver_residual,
    )
    print(
        "exact/harmonic_curvature_max_abs:",
        max_abs(exact_curvature_residual),
        max_abs(harmonic_curvature_residual),
    )
    print(
        "curvature_coexact_identity_max_abs:",
        max_abs(
            curvature_coexact_identity_residual
        ),
    )
    print(
        "gauge_fixed_divergence_max_abs:",
        max_abs(
            divergence_gauge_fixed_residual
        ),
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
    print("wrote:", GAUGE_FIX_NPZ_OUT)


if __name__ == "__main__":
    main()
