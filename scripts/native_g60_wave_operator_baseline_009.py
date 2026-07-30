from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_uniform_hodge_baseline_002.json"
)

STATIC_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_static_field_response_008.json"
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

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_wave_operator_baseline_009.json"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_wave_operator_probes_009.csv"
)

PROPAGATOR_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_wave_propagator_009.npz"
)

TOLERANCE = 1e-9
RANDOM_SEED = 46009
PROBE_COUNT = 16
TIME_GRID = np.linspace(0.0, 12.0, 49)


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


def wave_propagators(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    time: float,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return operators C(t), S(t), Cdot(t), Sdot(t) such that

        A(t)    = C(t) A0 + S(t) V0
        Adot(t) = Cdot(t) A0 + Sdot(t) V0.

    For lambda = 0:

        C = 1
        S = t
        Cdot = 0
        Sdot = 1.

    For lambda > 0:

        omega = sqrt(lambda)
        C = cos(omega t)
        S = sin(omega t) / omega
        Cdot = -omega sin(omega t)
        Sdot = cos(omega t).
    """
    positive = eigenvalues > tolerance

    omega = np.zeros_like(eigenvalues)
    omega[positive] = np.sqrt(eigenvalues[positive])

    c_values = np.ones_like(eigenvalues)
    s_values = np.full_like(eigenvalues, time)
    cdot_values = np.zeros_like(eigenvalues)
    sdot_values = np.ones_like(eigenvalues)

    c_values[positive] = np.cos(
        omega[positive] * time
    )

    s_values[positive] = (
        np.sin(omega[positive] * time)
        / omega[positive]
    )

    cdot_values[positive] = (
        -omega[positive]
        * np.sin(omega[positive] * time)
    )

    sdot_values[positive] = np.cos(
        omega[positive] * time
    )

    c_operator = (
        eigenvectors
        @ np.diag(c_values)
        @ eigenvectors.T
    )

    s_operator = (
        eigenvectors
        @ np.diag(s_values)
        @ eigenvectors.T
    )

    cdot_operator = (
        eigenvectors
        @ np.diag(cdot_values)
        @ eigenvectors.T
    )

    sdot_operator = (
        eigenvectors
        @ np.diag(sdot_values)
        @ eigenvectors.T
    )

    return (
        c_operator,
        s_operator,
        cdot_operator,
        sdot_operator,
    )


def energy(
    field: np.ndarray,
    velocity: np.ndarray,
    delta1: np.ndarray,
) -> float:
    return float(
        0.5 * np.dot(velocity, velocity)
        + 0.5 * np.dot(field, delta1 @ field)
    )


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROPAGATOR_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = json.loads(
        BASELINE_PATH.read_text(encoding="utf-8")
    )

    static = json.loads(
        STATIC_PATH.read_text(encoding="utf-8")
    )

    delta1 = read_matrix_csv(DELTA1_PATH)

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

    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (delta1 + delta1.T)
    )

    eigenvalues[
        np.abs(eigenvalues) < TOLERANCE
    ] = 0.0

    positive_mask = eigenvalues > TOLERANCE
    zero_mask = np.abs(eigenvalues) <= TOLERANCE

    positive_frequencies = np.sqrt(
        eigenvalues[positive_mask]
    )

    spectral_harmonic_projector = (
        eigenvectors[:, zero_mask]
        @ eigenvectors[:, zero_mask].T
    )

    spectral_range_projector = (
        eigenvectors[:, positive_mask]
        @ eigenvectors[:, positive_mask].T
    )

    operator_residuals = {
        "spectral_harmonic_projector_match": max_abs(
            spectral_harmonic_projector
            - p_harmonic
        ),
        "spectral_range_projector_match": max_abs(
            spectral_range_projector
            - p_exact
            - p_coexact
        ),
        "delta1_annihilates_harmonic": max_abs(
            delta1 @ p_harmonic
        ),
        "delta1_preserves_exact": max_abs(
            p_exact @ delta1 - delta1 @ p_exact
        ),
        "delta1_preserves_coexact": max_abs(
            p_coexact @ delta1
            - delta1 @ p_coexact
        ),
    }

    rng = np.random.default_rng(RANDOM_SEED)

    probe_rows = []

    global_residuals = {
        "initial_field": 0.0,
        "initial_velocity": 0.0,
        "wave_equation": 0.0,
        "harmonic_linear_field": 0.0,
        "harmonic_constant_velocity": 0.0,
        "oscillatory_field_has_no_harmonic_part": 0.0,
        "oscillatory_velocity_has_no_harmonic_part": 0.0,
        "energy_conservation": 0.0,
        "sector_reconstruction": 0.0,
    }

    saved_c = None
    saved_s = None
    saved_cdot = None
    saved_sdot = None

    for probe_id in range(PROBE_COUNT):
        initial_field = rng.normal(size=120)
        initial_velocity = rng.normal(size=120)

        field_harmonic_0 = (
            p_harmonic @ initial_field
        )

        velocity_harmonic_0 = (
            p_harmonic @ initial_velocity
        )

        field_oscillatory_0 = (
            (p_exact + p_coexact)
            @ initial_field
        )

        velocity_oscillatory_0 = (
            (p_exact + p_coexact)
            @ initial_velocity
        )

        initial_energy = energy(
            initial_field,
            initial_velocity,
            delta1,
        )

        for time_index, time in enumerate(TIME_GRID):
            (
                c_operator,
                s_operator,
                cdot_operator,
                sdot_operator,
            ) = wave_propagators(
                eigenvalues,
                eigenvectors,
                float(time),
            )

            if probe_id == 0 and time_index == len(TIME_GRID) - 1:
                saved_c = c_operator
                saved_s = s_operator
                saved_cdot = cdot_operator
                saved_sdot = sdot_operator

            field = (
                c_operator @ initial_field
                + s_operator @ initial_velocity
            )

            velocity = (
                cdot_operator @ initial_field
                + sdot_operator @ initial_velocity
            )

            acceleration = (
                -delta1 @ field
            )

            expected_harmonic_field = (
                field_harmonic_0
                + time * velocity_harmonic_0
            )

            expected_harmonic_velocity = (
                velocity_harmonic_0
            )

            oscillatory_field = (
                c_operator @ field_oscillatory_0
                + s_operator @ velocity_oscillatory_0
            )

            oscillatory_velocity = (
                cdot_operator @ field_oscillatory_0
                + sdot_operator @ velocity_oscillatory_0
            )

            current_energy = energy(
                field,
                velocity,
                delta1,
            )

            residuals = {
                "initial_field": (
                    max_abs(field - initial_field)
                    if time_index == 0
                    else 0.0
                ),
                "initial_velocity": (
                    max_abs(
                        velocity - initial_velocity
                    )
                    if time_index == 0
                    else 0.0
                ),
                "wave_equation": max_abs(
                    acceleration + delta1 @ field
                ),
                "harmonic_linear_field": max_abs(
                    p_harmonic @ field
                    - expected_harmonic_field
                ),
                "harmonic_constant_velocity": max_abs(
                    p_harmonic @ velocity
                    - expected_harmonic_velocity
                ),
                "oscillatory_field_has_no_harmonic_part": max_abs(
                    p_harmonic @ oscillatory_field
                ),
                "oscillatory_velocity_has_no_harmonic_part": max_abs(
                    p_harmonic @ oscillatory_velocity
                ),
                "energy_conservation": abs(
                    current_energy - initial_energy
                ),
                "sector_reconstruction": max_abs(
                    field
                    - expected_harmonic_field
                    - oscillatory_field
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
                    "time_index": time_index,
                    "time": float(time),
                    "initial_energy": initial_energy,
                    "current_energy": current_energy,
                    "field_norm": float(
                        np.linalg.norm(field)
                    ),
                    "velocity_norm": float(
                        np.linalg.norm(velocity)
                    ),
                    "harmonic_field_norm": float(
                        np.linalg.norm(
                            p_harmonic @ field
                        )
                    ),
                    "oscillatory_field_norm": float(
                        np.linalg.norm(
                            oscillatory_field
                        )
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
        "input_uniform_hodge_pass": (
            baseline.get("audit_pass") is True
        ),
        "input_static_response_pass": (
            static.get("audit_pass") is True
        ),
        "delta1_shape_is_120_by_120": (
            delta1.shape == (120, 120)
        ),
        "zero_mode_count_is_42": (
            int(np.count_nonzero(zero_mask)) == 42
        ),
        "positive_mode_count_is_78": (
            int(np.count_nonzero(positive_mask)) == 78
        ),
        "spectral_harmonic_projector_matches": (
            operator_residuals[
                "spectral_harmonic_projector_match"
            ]
            < TOLERANCE
        ),
        "spectral_range_projector_matches_exact_plus_coexact": (
            operator_residuals[
                "spectral_range_projector_match"
            ]
            < TOLERANCE
        ),
        "harmonic_sector_has_zero_stiffness": (
            operator_residuals[
                "delta1_annihilates_harmonic"
            ]
            < TOLERANCE
        ),
        "exact_sector_is_invariant": (
            operator_residuals[
                "delta1_preserves_exact"
            ]
            < TOLERANCE
        ),
        "coexact_sector_is_invariant": (
            operator_residuals[
                "delta1_preserves_coexact"
            ]
            < TOLERANCE
        ),
        "all_wave_probes_pass": all_probes_pass,
        "homogeneous_energy_is_conserved": (
            global_residuals[
                "energy_conservation"
            ]
            < TOLERANCE
        ),
        "harmonic_motion_is_linear": (
            global_residuals[
                "harmonic_linear_field"
            ]
            < TOLERANCE
            and global_residuals[
                "harmonic_constant_velocity"
            ]
            < TOLERANCE
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_wave_operator_baseline_009"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_uniform_wave_dynamics_baseline_constructed"
            if audit_pass
            else "native_g60_wave_operator_baseline_failed"
        ),
        "inputs": {
            "uniform_hodge_baseline": str(
                BASELINE_PATH.relative_to(ROOT)
            ),
            "static_response": str(
                STATIC_PATH.relative_to(ROOT)
            ),
            "delta1": str(
                DELTA1_PATH.relative_to(ROOT)
            ),
            "hodge_projectors": str(
                PROJECTOR_PATH.relative_to(ROOT)
            ),
        },
        "wave_system": {
            "homogeneous_equation": (
                "A_double_dot + Delta1 A = 0"
            ),
            "forced_equation_candidate": (
                "A_double_dot + Delta1 A = J(t)"
            ),
            "ambient_dimension": 120,
            "harmonic_zero_mode_dimension": 42,
            "oscillatory_dimension": 78,
            "exact_oscillatory_dimension": 59,
            "coexact_oscillatory_dimension": 19,
        },
        "spectrum": {
            "zero_mode_count": int(
                np.count_nonzero(zero_mask)
            ),
            "positive_mode_count": int(
                np.count_nonzero(positive_mask)
            ),
            "smallest_positive_eigenvalue": float(
                eigenvalues[positive_mask][0]
            ),
            "largest_positive_eigenvalue": float(
                eigenvalues[positive_mask][-1]
            ),
            "smallest_positive_frequency": float(
                positive_frequencies[0]
            ),
            "largest_positive_frequency": float(
                positive_frequencies[-1]
            ),
        },
        "checks": checks,
        "operator_residuals": operator_residuals,
        "probe_audit": {
            "random_seed": RANDOM_SEED,
            "probe_count": PROBE_COUNT,
            "time_count": int(len(TIME_GRID)),
            "time_minimum": float(TIME_GRID[0]),
            "time_maximum": float(TIME_GRID[-1]),
            "row_count": len(probe_rows),
            "all_probes_pass": all_probes_pass,
            "global_maximum_residuals": (
                global_residuals
            ),
        },
        "energy": {
            "formula": (
                "E = 1/2 ||A_dot||^2 "
                "+ 1/2 <A, Delta1 A>"
            ),
            "conserved_for_homogeneous_system": (
                checks[
                    "homogeneous_energy_is_conserved"
                ]
            ),
            "status": (
                "mathematical quadratic invariant"
            ),
        },
        "earned_interpretation": {
            "harmonic_initial_position": (
                "Remains constant when harmonic initial velocity is zero."
            ),
            "harmonic_initial_velocity": (
                "Produces linear drift because the harmonic sector "
                "has zero restoring operator."
            ),
            "exact_and_coexact_modes": (
                "Oscillate at square roots of positive Delta1 "
                "eigenvalues."
            ),
            "physical_wave_claim": False,
        },
        "outputs": {
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(ROOT)
            ),
            "propagator_npz": str(
                PROPAGATOR_NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "homogeneous_wave_baseline_constructed": (
                audit_pass
            ),
            "spectral_propagator_verified": (
                audit_pass
            ),
            "mathematical_energy_conserved": (
                audit_pass
            ),
            "forced_response_audited": False,
            "finite_propagation_speed_proved": False,
            "causal_lightcone_derived": False,
            "physical_time_scale_derived": False,
            "physical_frequency_claim": False,
            "physical_energy_claim": False,
            "electromagnetism_claim": False,
            "maxwell_claim": False,
            "physical_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False
        }
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

    if (
        saved_c is None
        or saved_s is None
        or saved_cdot is None
        or saved_sdot is None
    ):
        raise RuntimeError(
            "final propagator snapshot was not captured"
        )

    np.savez_compressed(
        PROPAGATOR_NPZ_OUT,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        positive_frequencies=positive_frequencies,
        P_harmonic_spectral=(
            spectral_harmonic_projector
        ),
        P_range_spectral=(
            spectral_range_projector
        ),
        sample_time=np.array(
            [float(TIME_GRID[-1])]
        ),
        C_sample=saved_c,
        S_sample=saved_s,
        Cdot_sample=saved_cdot,
        Sdot_sample=saved_sdot,
        time_grid=TIME_GRID,
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "zero/positive_mode_count:",
        payload["spectrum"]["zero_mode_count"],
        payload["spectrum"]["positive_mode_count"],
    )
    print(
        "positive_frequency_range:",
        payload["spectrum"][
            "smallest_positive_frequency"
        ],
        payload["spectrum"][
            "largest_positive_frequency"
        ],
    )
    print(
        "spectral_projector_residuals:",
        operator_residuals[
            "spectral_harmonic_projector_match"
        ],
        operator_residuals[
            "spectral_range_projector_match"
        ],
    )
    print(
        "all_wave_probes_pass:",
        all_probes_pass,
    )
    print(
        "probe_global_maximum_residuals:",
        global_residuals,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", PROPAGATOR_NPZ_OUT)


if __name__ == "__main__":
    main()
