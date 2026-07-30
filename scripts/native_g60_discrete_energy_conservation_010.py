from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

WAVE_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_wave_operator_baseline_009.json"
)

PROPAGATOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_wave_propagator_009.npz"
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
    / "native_g60_discrete_energy_conservation_010.json"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_discrete_energy_conservation_probes_010.csv"
)

BALANCE_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_discrete_energy_balance_010.npz"
)

TOLERANCE = 1e-9
RANDOM_SEED = 46010
PROBE_COUNT = 24


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


def energy(
    field: np.ndarray,
    velocity: np.ndarray,
    delta1: np.ndarray,
) -> float:
    return float(
        0.5 * np.dot(velocity, velocity)
        + 0.5 * np.dot(
            field,
            delta1 @ field,
        )
    )


def energy_derivative(
    field: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    delta1: np.ndarray,
) -> float:
    return float(
        np.dot(
            velocity,
            acceleration + delta1 @ field,
        )
    )


def velocity_verlet_step(
    field: np.ndarray,
    velocity: np.ndarray,
    source_now: np.ndarray,
    source_next: np.ndarray,
    delta1: np.ndarray,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    acceleration_now = (
        source_now - delta1 @ field
    )

    field_next = (
        field
        + time_step * velocity
        + 0.5
        * time_step
        * time_step
        * acceleration_now
    )

    acceleration_next = (
        source_next
        - delta1 @ field_next
    )

    velocity_next = (
        velocity
        + 0.5
        * time_step
        * (
            acceleration_now
            + acceleration_next
        )
    )

    return field_next, velocity_next


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROBE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    BALANCE_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wave = json.loads(
        WAVE_PATH.read_text(encoding="utf-8")
    )

    propagator = np.load(PROPAGATOR_PATH)

    eigenvalues = np.array(
        propagator["eigenvalues"],
        dtype=np.float64,
    )
    eigenvectors = np.array(
        propagator["eigenvectors"],
        dtype=np.float64,
    )

    delta1 = read_matrix_csv(DELTA1_PATH)

    symmetry_residual = max_abs(
        delta1 - delta1.T
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    algebraic_rows = []

    homogeneous_derivative_max_abs = 0.0
    forced_balance_max_abs = 0.0

    for probe_id in range(PROBE_COUNT):
        field = rng.normal(size=120)
        velocity = rng.normal(size=120)
        source = rng.normal(size=120)

        homogeneous_acceleration = (
            -delta1 @ field
        )

        forced_acceleration = (
            source - delta1 @ field
        )

        homogeneous_derivative = (
            energy_derivative(
                field,
                velocity,
                homogeneous_acceleration,
                delta1,
            )
        )

        forced_derivative = (
            energy_derivative(
                field,
                velocity,
                forced_acceleration,
                delta1,
            )
        )

        source_power = float(
            np.dot(velocity, source)
        )

        forced_balance_residual = (
            forced_derivative
            - source_power
        )

        homogeneous_derivative_max_abs = max(
            homogeneous_derivative_max_abs,
            abs(homogeneous_derivative),
        )

        forced_balance_max_abs = max(
            forced_balance_max_abs,
            abs(forced_balance_residual),
        )

        algebraic_rows.append(
            {
                "probe_id": probe_id,
                "field_norm": float(
                    np.linalg.norm(field)
                ),
                "velocity_norm": float(
                    np.linalg.norm(velocity)
                ),
                "source_norm": float(
                    np.linalg.norm(source)
                ),
                "homogeneous_energy_derivative": (
                    homogeneous_derivative
                ),
                "forced_energy_derivative": (
                    forced_derivative
                ),
                "source_power": source_power,
                "forced_balance_residual": (
                    forced_balance_residual
                ),
                "all_checks_pass": (
                    abs(
                        homogeneous_derivative
                    )
                    < TOLERANCE
                    and abs(
                        forced_balance_residual
                    )
                    < TOLERANCE
                ),
            }
        )

    positive_mask = eigenvalues > TOLERANCE
    zero_mask = np.abs(
        eigenvalues
    ) <= TOLERANCE

    positive_eigenvalues = (
        eigenvalues[positive_mask]
    )

    frequencies = np.sqrt(
        positive_eigenvalues
    )

    spectral_rows = []

    for local_index, eigenvalue in enumerate(
        positive_eigenvalues
    ):
        omega = float(
            frequencies[local_index]
        )

        amplitude = float(
            1.0
            + 0.01 * local_index
        )

        phase = float(
            0.1 * local_index
        )

        times = np.linspace(
            0.0,
            10.0,
            81,
        )

        energies = []

        for time in times:
            coordinate = (
                amplitude
                * np.cos(
                    omega * time
                    + phase
                )
            )

            velocity_coordinate = (
                -amplitude
                * omega
                * np.sin(
                    omega * time
                    + phase
                )
            )

            energies.append(
                0.5
                * velocity_coordinate**2
                + 0.5
                * eigenvalue
                * coordinate**2
            )

        energy_array = np.array(
            energies,
            dtype=np.float64,
        )

        spectral_rows.append(
            {
                "mode_index": local_index,
                "eigenvalue": float(
                    eigenvalue
                ),
                "frequency": omega,
                "energy_minimum": float(
                    np.min(energy_array)
                ),
                "energy_maximum": float(
                    np.max(energy_array)
                ),
                "energy_range": float(
                    np.max(energy_array)
                    - np.min(energy_array)
                ),
            }
        )

    spectral_energy_range_max = max(
        row["energy_range"]
        for row in spectral_rows
    )

    time_step = 0.002
    step_count = 2000

    field = rng.normal(size=120)
    velocity = rng.normal(size=120)

    source_vector = rng.normal(size=120)

    source_amplitude = 0.15
    source_frequency = 0.7

    initial_energy = energy(
        field,
        velocity,
        delta1,
    )

    accumulated_work = 0.0
    forced_rows = []

    maximum_balance_residual = 0.0

    for step in range(step_count):
        time_now = (
            step * time_step
        )
        time_next = (
            (step + 1) * time_step
        )

        source_now = (
            source_amplitude
            * np.sin(
                source_frequency
                * time_now
            )
            * source_vector
        )

        source_next = (
            source_amplitude
            * np.sin(
                source_frequency
                * time_next
            )
            * source_vector
        )

        field_next, velocity_next = (
            velocity_verlet_step(
                field,
                velocity,
                source_now,
                source_next,
                delta1,
                time_step,
            )
        )

        power_now = float(
            np.dot(
                velocity,
                source_now,
            )
        )

        power_next = float(
            np.dot(
                velocity_next,
                source_next,
            )
        )

        accumulated_work += (
            0.5
            * time_step
            * (
                power_now
                + power_next
            )
        )

        current_energy = energy(
            field_next,
            velocity_next,
            delta1,
        )

        balance_residual = (
            current_energy
            - initial_energy
            - accumulated_work
        )

        maximum_balance_residual = max(
            maximum_balance_residual,
            abs(balance_residual),
        )

        if (
            step == 0
            or (step + 1) % 100 == 0
        ):
            forced_rows.append(
                {
                    "step": step + 1,
                    "time": time_next,
                    "energy": current_energy,
                    "accumulated_work": (
                        accumulated_work
                    ),
                    "energy_minus_initial": (
                        current_energy
                        - initial_energy
                    ),
                    "balance_residual": (
                        balance_residual
                    ),
                }
            )

        field = field_next
        velocity = velocity_next

    homogeneous_field = rng.normal(
        size=120
    )
    homogeneous_velocity = rng.normal(
        size=120
    )

    homogeneous_initial_energy = energy(
        homogeneous_field,
        homogeneous_velocity,
        delta1,
    )

    homogeneous_maximum_drift = 0.0

    zero_frequencies = np.zeros_like(
        eigenvalues
    )
    zero_frequencies[positive_mask] = (
        np.sqrt(
            eigenvalues[positive_mask]
        )
    )

    initial_coordinates = (
        eigenvectors.T
        @ homogeneous_field
    )
    initial_velocity_coordinates = (
        eigenvectors.T
        @ homogeneous_velocity
    )

    homogeneous_times = np.linspace(
        0.0,
        16.0,
        65,
    )

    for time in homogeneous_times:
        coordinate = np.empty_like(
            initial_coordinates
        )
        velocity_coordinate = np.empty_like(
            initial_velocity_coordinates
        )

        coordinate[zero_mask] = (
            initial_coordinates[zero_mask]
            + time
            * initial_velocity_coordinates[
                zero_mask
            ]
        )

        velocity_coordinate[zero_mask] = (
            initial_velocity_coordinates[
                zero_mask
            ]
        )

        coordinate[positive_mask] = (
            initial_coordinates[
                positive_mask
            ]
            * np.cos(
                zero_frequencies[
                    positive_mask
                ]
                * time
            )
            + initial_velocity_coordinates[
                positive_mask
            ]
            * np.sin(
                zero_frequencies[
                    positive_mask
                ]
                * time
            )
            / zero_frequencies[
                positive_mask
            ]
        )

        velocity_coordinate[
            positive_mask
        ] = (
            -initial_coordinates[
                positive_mask
            ]
            * zero_frequencies[
                positive_mask
            ]
            * np.sin(
                zero_frequencies[
                    positive_mask
                ]
                * time
            )
            + initial_velocity_coordinates[
                positive_mask
            ]
            * np.cos(
                zero_frequencies[
                    positive_mask
                ]
                * time
            )
        )

        field_time = (
            eigenvectors @ coordinate
        )
        velocity_time = (
            eigenvectors
            @ velocity_coordinate
        )

        current_energy = energy(
            field_time,
            velocity_time,
            delta1,
        )

        homogeneous_maximum_drift = max(
            homogeneous_maximum_drift,
            abs(
                current_energy
                - homogeneous_initial_energy
            ),
        )

    checks = {
        "input_wave_baseline_pass": (
            wave.get("audit_pass") is True
        ),
        "delta1_is_symmetric": (
            symmetry_residual
            < TOLERANCE
        ),
        "algebraic_homogeneous_derivative_is_zero": (
            homogeneous_derivative_max_abs
            < TOLERANCE
        ),
        "algebraic_forced_balance_law_passes": (
            forced_balance_max_abs
            < TOLERANCE
        ),
        "all_positive_mode_energies_constant": (
            spectral_energy_range_max
            < TOLERANCE
        ),
        "spectral_homogeneous_energy_conserved": (
            homogeneous_maximum_drift
            < 1e-9
        ),
        "forced_numerical_energy_work_balance_passes": (
            maximum_balance_residual
            < 2e-3
        ),
        "zero_mode_count_is_42": (
            int(
                np.count_nonzero(
                    zero_mask
                )
            )
            == 42
        ),
        "positive_mode_count_is_78": (
            int(
                np.count_nonzero(
                    positive_mask
                )
            )
            == 78
        ),
    }

    audit_pass = all(
        checks.values()
    )

    payload = {
        "artifact_id": (
            "native_g60_discrete_energy_conservation_010"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_uniform_energy_balance_identified"
            if audit_pass
            else "native_g60_energy_conservation_audit_failed"
        ),
        "inputs": {
            "wave_operator_baseline": str(
                WAVE_PATH.relative_to(ROOT)
            ),
            "wave_propagator": str(
                PROPAGATOR_PATH.relative_to(
                    ROOT
                )
            ),
            "delta1": str(
                DELTA1_PATH.relative_to(ROOT)
            ),
        },
        "energy_law": {
            "energy": (
                "E(A,V) = 1/2 <V,V> "
                "+ 1/2 <A,Delta1 A>"
            ),
            "homogeneous_equation": (
                "A_double_dot "
                "+ Delta1 A = 0"
            ),
            "homogeneous_balance": (
                "dE/dt = 0"
            ),
            "forced_equation": (
                "A_double_dot "
                "+ Delta1 A = J"
            ),
            "forced_balance": (
                "dE/dt = <A_dot,J>"
            ),
            "derivation": (
                "dE/dt = <A_dot,"
                "A_double_dot + Delta1 A>"
            ),
        },
        "checks": checks,
        "algebraic_audit": {
            "probe_count": PROBE_COUNT,
            "random_seed": RANDOM_SEED,
            "homogeneous_derivative_max_abs": (
                homogeneous_derivative_max_abs
            ),
            "forced_balance_max_abs": (
                forced_balance_max_abs
            ),
        },
        "spectral_audit": {
            "zero_mode_count": int(
                np.count_nonzero(
                    zero_mask
                )
            ),
            "positive_mode_count": int(
                np.count_nonzero(
                    positive_mask
                )
            ),
            "positive_mode_energy_range_max": (
                spectral_energy_range_max
            ),
            "homogeneous_total_energy_drift_max": (
                homogeneous_maximum_drift
            ),
        },
        "forced_numerical_audit": {
            "integrator": (
                "velocity Verlet with "
                "trapezoidal work accumulation"
            ),
            "time_step": time_step,
            "step_count": step_count,
            "source_amplitude": (
                source_amplitude
            ),
            "source_frequency": (
                source_frequency
            ),
            "maximum_energy_work_balance_residual": (
                maximum_balance_residual
            ),
        },
        "earned_interpretation": {
            "homogeneous_system": (
                "Conserves the uniform quadratic "
                "mathematical energy exactly "
                "at the differential-equation level."
            ),
            "forced_system": (
                "Changes energy at the rate of "
                "source work <A_dot,J>."
            ),
            "harmonic_motion": (
                "Harmonic velocity contributes "
                "constant kinetic energy despite "
                "linear field drift."
            ),
            "physical_energy_claim": False,
        },
        "outputs": {
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(ROOT)
            ),
            "balance_npz": str(
                BALANCE_NPZ_OUT.relative_to(
                    ROOT
                )
            ),
        },
        "boundary": {
            "algebraic_energy_law_derived": (
                audit_pass
            ),
            "homogeneous_energy_conservation_proved": (
                audit_pass
            ),
            "forced_work_balance_derived": (
                audit_pass
            ),
            "energy_is_mathematical_quadratic_invariant": (
                True
            ),
            "physical_energy_claim": False,
            "physical_source_claim": False,
            "physical_time_scale_derived": False,
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
        fieldnames = [
            "record_type",
            "probe_id",
            "step",
            "time",
            "homogeneous_energy_derivative",
            "forced_energy_derivative",
            "source_power",
            "forced_balance_residual",
            "energy",
            "accumulated_work",
            "energy_minus_initial",
            "balance_residual",
            "all_checks_pass",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in algebraic_rows:
            writer.writerow(
                {
                    "record_type": "algebraic",
                    "probe_id": row[
                        "probe_id"
                    ],
                    "homogeneous_energy_derivative": (
                        row[
                            "homogeneous_energy_derivative"
                        ]
                    ),
                    "forced_energy_derivative": (
                        row[
                            "forced_energy_derivative"
                        ]
                    ),
                    "source_power": row[
                        "source_power"
                    ],
                    "forced_balance_residual": (
                        row[
                            "forced_balance_residual"
                        ]
                    ),
                    "all_checks_pass": row[
                        "all_checks_pass"
                    ],
                }
            )

        for row in forced_rows:
            writer.writerow(
                {
                    "record_type": (
                        "forced_numerical"
                    ),
                    "step": row["step"],
                    "time": row["time"],
                    "energy": row["energy"],
                    "accumulated_work": row[
                        "accumulated_work"
                    ],
                    "energy_minus_initial": (
                        row[
                            "energy_minus_initial"
                        ]
                    ),
                    "balance_residual": row[
                        "balance_residual"
                    ],
                }
            )

    np.savez_compressed(
        BALANCE_NPZ_OUT,
        Delta1=delta1,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        forced_balance_rows=np.array(
            [
                [
                    row["time"],
                    row["energy"],
                    row["accumulated_work"],
                    row["balance_residual"],
                ]
                for row in forced_rows
            ],
            dtype=np.float64,
        ),
        time_step=np.array(
            [time_step]
        ),
        tolerance=np.array(
            [TOLERANCE]
        ),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "homogeneous_derivative_max_abs:",
        homogeneous_derivative_max_abs,
    )
    print(
        "forced_balance_max_abs:",
        forced_balance_max_abs,
    )
    print(
        "positive_mode_energy_range_max:",
        spectral_energy_range_max,
    )
    print(
        "homogeneous_total_energy_drift_max:",
        homogeneous_maximum_drift,
    )
    print(
        "forced_energy_work_balance_max_abs:",
        maximum_balance_residual,
    )
    print(
        "zero/positive_mode_count:",
        int(
            np.count_nonzero(
                zero_mask
            )
        ),
        int(
            np.count_nonzero(
                positive_mask
            )
        ),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", BALANCE_NPZ_OUT)


if __name__ == "__main__":
    main()
