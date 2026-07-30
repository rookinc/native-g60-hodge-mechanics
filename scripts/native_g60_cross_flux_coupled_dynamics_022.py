from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

COUPLING_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_energy_coupling_021.json"
)

COUPLING_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_energy_coupling_021.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_coupled_dynamics_022.json"
)

SCAN_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_coupled_dynamics_scan_022.csv"
)

TRAJECTORY_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_coupled_dynamics_trajectory_022.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_coupled_dynamics_022.npz"
)

RANDOM_SEED = 46022
TIME_STEP = 0.001
STEP_COUNT = 20000
SAMPLE_EVERY = 100

COUPLING_VALUES = (
    0.0,
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.20,
)

ENERGY_RELATIVE_TOLERANCE = 5e-5
GROWTH_THRESHOLD = 20.0


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def bilinear_flux(
    channel: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    return channel @ np.kron(u, v)


def gradient_u(
    channel: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    tensor = channel.reshape(4, 6, 6)

    return np.einsum(
        "rab,b,r->a",
        tensor,
        v,
        f,
    )


def gradient_v(
    channel: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    tensor = channel.reshape(4, 6, 6)

    return np.einsum(
        "rab,a,r->b",
        tensor,
        u,
        f,
    )


def interaction(
    channel: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> float:
    return float(
        np.dot(
            f,
            bilinear_flux(channel, u, v),
        )
    )


def accelerations(
    channel: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acceleration_u = (
        -coupling
        * gradient_u(channel, v, f)
    )

    acceleration_v = (
        -coupling
        * gradient_v(channel, u, f)
    )

    acceleration_f = (
        -stiffness @ f
        - coupling
        * bilinear_flux(channel, u, v)
    )

    return (
        acceleration_u,
        acceleration_v,
        acceleration_f,
    )


def total_energy(
    channel: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    df: np.ndarray,
) -> float:
    kinetic_u = 0.5 * float(np.dot(du, du))
    kinetic_v = 0.5 * float(np.dot(dv, dv))
    kinetic_f = 0.5 * float(np.dot(df, df))

    flux_potential = 0.5 * float(
        np.dot(
            f,
            stiffness @ f,
        )
    )

    interaction_energy = (
        coupling
        * interaction(
            channel,
            u,
            v,
            f,
        )
    )

    return (
        kinetic_u
        + kinetic_v
        + kinetic_f
        + flux_potential
        + interaction_energy
    )


def sector_energies(
    channel: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    df: np.ndarray,
) -> dict[str, float]:
    return {
        "kinetic_u": (
            0.5 * float(np.dot(du, du))
        ),
        "kinetic_v": (
            0.5 * float(np.dot(dv, dv))
        ),
        "kinetic_f": (
            0.5 * float(np.dot(df, df))
        ),
        "flux_potential": (
            0.5
            * float(
                np.dot(
                    f,
                    stiffness @ f,
                )
            )
        ),
        "interaction": (
            coupling
            * interaction(
                channel,
                u,
                v,
                f,
            )
        ),
    }


def verlet_step(
    channel: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    time_step: float,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    df: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    acceleration_u, acceleration_v, acceleration_f = (
        accelerations(
            channel,
            stiffness,
            coupling,
            u,
            v,
            f,
        )
    )

    u_next = (
        u
        + time_step * du
        + 0.5
        * time_step**2
        * acceleration_u
    )

    v_next = (
        v
        + time_step * dv
        + 0.5
        * time_step**2
        * acceleration_v
    )

    f_next = (
        f
        + time_step * df
        + 0.5
        * time_step**2
        * acceleration_f
    )

    (
        acceleration_u_next,
        acceleration_v_next,
        acceleration_f_next,
    ) = accelerations(
        channel,
        stiffness,
        coupling,
        u_next,
        v_next,
        f_next,
    )

    du_next = (
        du
        + 0.5
        * time_step
        * (
            acceleration_u
            + acceleration_u_next
        )
    )

    dv_next = (
        dv
        + 0.5
        * time_step
        * (
            acceleration_v
            + acceleration_v_next
        )
    )

    df_next = (
        df
        + 0.5
        * time_step
        * (
            acceleration_f
            + acceleration_f_next
        )
    )

    return (
        u_next,
        v_next,
        f_next,
        du_next,
        dv_next,
        df_next,
    )


def classify_run(
    coupling: float,
    finite: bool,
    maximum_state_norm: float,
    flux_norm_range: float,
    relative_energy_drift: float,
) -> str:
    if not finite:
        return "nonfinite_growth"

    if maximum_state_norm > GROWTH_THRESHOLD:
        return "large_growth_observed"

    if relative_energy_drift > ENERGY_RELATIVE_TOLERANCE:
        return "numerically_unresolved"

    if coupling == 0.0:
        return "decoupled_baseline"

    if flux_norm_range > 1e-4:
        return "bounded_exchange_observed"

    return "bounded_low_transfer"


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    SCAN_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    TRAJECTORY_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coupling_receipt = json.loads(
        COUPLING_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        COUPLING_NPZ_PATH
    )

    channel = np.array(
        data["unique_four_channel"],
        dtype=np.float64,
    )

    stiffness = np.array(
        data["four_flux_stiffness"],
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    initial_u = rng.normal(size=6)
    initial_v = rng.normal(size=6)

    initial_u *= (
        0.55
        / np.linalg.norm(initial_u)
    )

    initial_v *= (
        0.55
        / np.linalg.norm(initial_v)
    )

    initial_f = np.zeros(
        4,
        dtype=np.float64,
    )

    initial_du = rng.normal(size=6)
    initial_dv = rng.normal(size=6)

    initial_du *= (
        0.08
        / np.linalg.norm(initial_du)
    )

    initial_dv *= (
        0.08
        / np.linalg.norm(initial_dv)
    )

    initial_df = np.zeros(
        4,
        dtype=np.float64,
    )

    scan_rows = []
    trajectory_rows = []

    saved_trajectories = {}

    all_finite = True
    maximum_relative_energy_drift = 0.0

    for coupling in COUPLING_VALUES:
        u = initial_u.copy()
        v = initial_v.copy()
        f = initial_f.copy()

        du = initial_du.copy()
        dv = initial_dv.copy()
        df = initial_df.copy()

        initial_energy = total_energy(
            channel,
            stiffness,
            coupling,
            u,
            v,
            f,
            du,
            dv,
            df,
        )

        energy_values = []
        u_norm_values = []
        v_norm_values = []
        f_norm_values = []
        interaction_values = []

        finite = True

        sampled_state_rows = []

        for step in range(
            STEP_COUNT + 1
        ):
            if step % SAMPLE_EVERY == 0:
                time = (
                    step * TIME_STEP
                )

                current_energy = (
                    total_energy(
                        channel,
                        stiffness,
                        coupling,
                        u,
                        v,
                        f,
                        du,
                        dv,
                        df,
                    )
                )

                sectors = sector_energies(
                    channel,
                    stiffness,
                    coupling,
                    u,
                    v,
                    f,
                    du,
                    dv,
                    df,
                )

                u_norm = float(
                    np.linalg.norm(u)
                )
                v_norm = float(
                    np.linalg.norm(v)
                )
                f_norm = float(
                    np.linalg.norm(f)
                )

                energy_values.append(
                    current_energy
                )
                u_norm_values.append(
                    u_norm
                )
                v_norm_values.append(
                    v_norm
                )
                f_norm_values.append(
                    f_norm
                )
                interaction_values.append(
                    sectors["interaction"]
                )

                row = {
                    "coupling": coupling,
                    "step": step,
                    "time": time,
                    "total_energy": (
                        current_energy
                    ),
                    "u_norm": u_norm,
                    "v_norm": v_norm,
                    "f_norm": f_norm,
                    **sectors,
                }

                sampled_state_rows.append(
                    row
                )

                trajectory_rows.append(
                    row
                )

            if step == STEP_COUNT:
                break

            (
                u,
                v,
                f,
                du,
                dv,
                df,
            ) = verlet_step(
                channel,
                stiffness,
                coupling,
                TIME_STEP,
                u,
                v,
                f,
                du,
                dv,
                df,
            )

            if not all(
                np.all(
                    np.isfinite(value)
                )
                for value in (
                    u,
                    v,
                    f,
                    du,
                    dv,
                    df,
                )
            ):
                finite = False
                break

        all_finite = (
            all_finite and finite
        )

        energy_array = np.array(
            energy_values,
            dtype=np.float64,
        )

        denominator = max(
            abs(initial_energy),
            1e-12,
        )

        relative_energy_drift = (
            float(
                np.max(
                    np.abs(
                        energy_array
                        - initial_energy
                    )
                )
            )
            / denominator
            if len(energy_array)
            else float("inf")
        )

        maximum_relative_energy_drift = max(
            maximum_relative_energy_drift,
            relative_energy_drift,
        )

        maximum_state_norm = max(
            max(u_norm_values),
            max(v_norm_values),
            max(f_norm_values),
        )

        flux_norm_range = (
            max(f_norm_values)
            - min(f_norm_values)
        )

        classification = classify_run(
            coupling,
            finite,
            maximum_state_norm,
            flux_norm_range,
            relative_energy_drift,
        )

        scan_rows.append(
            {
                "coupling": coupling,
                "classification": (
                    classification
                ),
                "finite": finite,
                "sample_count": len(
                    energy_values
                ),
                "initial_energy": (
                    initial_energy
                ),
                "minimum_energy": float(
                    np.min(energy_array)
                ),
                "maximum_energy": float(
                    np.max(energy_array)
                ),
                "relative_energy_drift": (
                    relative_energy_drift
                ),
                "maximum_u_norm": max(
                    u_norm_values
                ),
                "maximum_v_norm": max(
                    v_norm_values
                ),
                "maximum_f_norm": max(
                    f_norm_values
                ),
                "flux_norm_range": (
                    flux_norm_range
                ),
                "minimum_interaction_energy": min(
                    interaction_values
                ),
                "maximum_interaction_energy": max(
                    interaction_values
                ),
                "maximum_state_norm": (
                    maximum_state_norm
                ),
            }
        )

        saved_trajectories[
            str(coupling)
        ] = np.array(
            [
                [
                    row["time"],
                    row["total_energy"],
                    row["u_norm"],
                    row["v_norm"],
                    row["f_norm"],
                    row["kinetic_u"],
                    row["kinetic_v"],
                    row["kinetic_f"],
                    row["flux_potential"],
                    row["interaction"],
                ]
                for row in sampled_state_rows
            ],
            dtype=np.float64,
        )

    bounded_rows = [
        row
        for row in scan_rows
        if row["classification"]
        in (
            "bounded_exchange_observed",
            "bounded_low_transfer",
            "decoupled_baseline",
        )
    ]

    growth_rows = [
        row
        for row in scan_rows
        if row["classification"]
        in (
            "large_growth_observed",
            "nonfinite_growth",
        )
    ]

    checks = {
        "input_021_audit_pass": (
            coupling_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "channel_shape_is_4_by_36": (
            channel.shape == (4, 36)
        ),
        "stiffness_shape_is_4_by_4": (
            stiffness.shape == (4, 4)
        ),
        "stiffness_is_10_identity": (
            np.max(
                np.abs(
                    stiffness
                    - 10.0 * np.eye(4)
                )
            )
            < 1e-9
        ),
        "all_coupling_values_scanned": (
            len(scan_rows)
            == len(COUPLING_VALUES)
        ),
        "all_runs_remain_finite": (
            all_finite
        ),
        "all_runs_meet_energy_tolerance": all(
            row[
                "relative_energy_drift"
            ]
            < ENERGY_RELATIVE_TOLERANCE
            for row in scan_rows
        ),
        "at_least_one_bounded_exchange_run": any(
            row["classification"]
            == "bounded_exchange_observed"
            for row in scan_rows
        ),
    }

    audit_pass = all(
        checks.values()
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_coupled_dynamics_022"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_cross_flux_closed_dynamics_scan_completed"
            if audit_pass
            else "native_g60_cross_flux_coupled_dynamics_scan_incomplete"
        ),
        "system": {
            "dimension": 16,
            "sector_dimensions": {
                "six_a": 6,
                "six_b": 6,
                "four_flux": 4,
            },
            "equations": {
                "u": (
                    "u_double_dot = "
                    "-g grad_u I"
                ),
                "v": (
                    "v_double_dot = "
                    "-g grad_v I"
                ),
                "f": (
                    "f_double_dot + 10 f = "
                    "-g T4(u tensor v)"
                ),
            },
            "interaction": (
                "I(u,v,f) = "
                "<f,T4(u tensor v)>"
            ),
        },
        "scan": {
            "coupling_values": list(
                COUPLING_VALUES
            ),
            "time_step": TIME_STEP,
            "step_count": STEP_COUNT,
            "time_horizon": (
                TIME_STEP
                * STEP_COUNT
            ),
            "sample_every": (
                SAMPLE_EVERY
            ),
            "random_seed": (
                RANDOM_SEED
            ),
            "classification_counts": dict(
                Counter(
                    row["classification"]
                    for row in scan_rows
                )
            ),
            "bounded_run_count": len(
                bounded_rows
            ),
            "growth_run_count": len(
                growth_rows
            ),
            "maximum_relative_energy_drift": (
                maximum_relative_energy_drift
            ),
        },
        "checks": checks,
        "scan_rows": scan_rows,
        "earned_interpretation": {
            "closed_reciprocal_dynamics_evolved": (
                audit_pass
            ),
            "bounded_exchange_observed": any(
                row["classification"]
                == "bounded_exchange_observed"
                for row in scan_rows
            ),
            "periodicity_proved": False,
            "stability_for_all_initial_conditions_proved": (
                False
            ),
            "physical_time_claim": False,
            "physical_energy_claim": False,
            "physical_thread_claim": False,
        },
        "outputs": {
            "scan_csv": str(
                SCAN_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "trajectory_csv": str(
                TRAJECTORY_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "trajectory_npz": str(
                NPZ_OUT.relative_to(
                    ROOT
                )
            ),
        },
        "boundary": {
            "finite_coupled_scan_completed": (
                audit_pass
            ),
            "numerical_energy_conservation_verified": (
                audit_pass
            ),
            "bounded_exchange_regime_observed": any(
                row["classification"]
                == "bounded_exchange_observed"
                for row in scan_rows
            ),
            "global_nonlinear_stability_proved": (
                False
            ),
            "periodic_orbit_proved": False,
            "coupling_strength_derived": False,
            "physical_time_scale_derived": False,
            "physical_energy_claim": False,
            "physical_thread_claim": False,
            "force_claim": False,
            "physical_claim": False,
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

    with SCAN_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                scan_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            scan_rows
        )

    with TRAJECTORY_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                trajectory_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            trajectory_rows
        )

    npz_payload = {
        "channel": channel,
        "stiffness": stiffness,
        "coupling_values": np.array(
            COUPLING_VALUES,
            dtype=np.float64,
        ),
        "initial_u": initial_u,
        "initial_v": initial_v,
        "initial_f": initial_f,
        "initial_du": initial_du,
        "initial_dv": initial_dv,
        "initial_df": initial_df,
        "time_step": np.array(
            [TIME_STEP]
        ),
    }

    for coupling, trajectory in (
        saved_trajectories.items()
    ):
        safe_name = (
            coupling
            .replace(".", "_")
            .replace("-", "minus_")
        )

        npz_payload[
            f"trajectory_g_{safe_name}"
        ] = trajectory

    np.savez_compressed(
        NPZ_OUT,
        **npz_payload,
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "coupling_values:",
        list(COUPLING_VALUES),
    )
    print(
        "classification_counts:",
        payload["scan"][
            "classification_counts"
        ],
    )
    print(
        "maximum_relative_energy_drift:",
        maximum_relative_energy_drift,
    )

    print("\nscan summary:")

    for row in scan_rows:
        print(
            "g=",
            row["coupling"],
            "class=",
            row["classification"],
            "energy_drift=",
            row[
                "relative_energy_drift"
            ],
            "max_u/v/f=",
            row["maximum_u_norm"],
            row["maximum_v_norm"],
            row["maximum_f_norm"],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", SCAN_CSV_OUT)
    print("wrote:", TRAJECTORY_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
