from __future__ import annotations

import csv
import json
import time as time_module
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DYNAMICS_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_coupled_dynamics_022.json"
)

DYNAMICS_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_coupled_dynamics_022.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_nonlinear_threshold_probe_023.json"
)

RUN_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_nonlinear_threshold_runs_023.csv"
)

REFINEMENT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_nonlinear_threshold_refinement_023.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_nonlinear_threshold_probe_023.npz"
)

RANDOM_SEED = 46023

COUPLING_VALUES = (
    0.80,
    1.00,
    1.20,
    1.35,
    1.50,
)

TIME_STEPS = (
    0.002,
    0.001,
)

PERTURBATION_IDS = (
    0,
    1,
    2,
)

TIME_HORIZON = 12.0
SAMPLE_INTERVAL = 0.05

ENERGY_TOLERANCE_COARSE = 2e-4
ENERGY_TOLERANCE_FINE = 5e-5

STATE_NORM_GROWTH_THRESHOLD = 20.0
STATE_ABSOLUTE_SAFETY_CAP = 1e6
HALF_WINDOW_GROWTH_RATIO_THRESHOLD = 2.0
REFINEMENT_RELATIVE_TOLERANCE = 0.08


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
    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
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
            bilinear_flux(
                channel,
                u,
                v,
            ),
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
    return (
        -coupling
        * gradient_u(
            channel,
            v,
            f,
        ),
        -coupling
        * gradient_v(
            channel,
            u,
            f,
        ),
        -stiffness @ f
        - coupling
        * bilinear_flux(
            channel,
            u,
            v,
        ),
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
    return float(
        0.5 * np.dot(du, du)
        + 0.5 * np.dot(dv, dv)
        + 0.5 * np.dot(df, df)
        + 0.5 * np.dot(
            f,
            stiffness @ f,
        )
        + coupling
        * interaction(
            channel,
            u,
            v,
            f,
        )
    )


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
    au, av, af = accelerations(
        channel,
        stiffness,
        coupling,
        u,
        v,
        f,
    )

    u_next = (
        u
        + time_step * du
        + 0.5 * time_step**2 * au
    )

    v_next = (
        v
        + time_step * dv
        + 0.5 * time_step**2 * av
    )

    f_next = (
        f
        + time_step * df
        + 0.5 * time_step**2 * af
    )

    au_next, av_next, af_next = accelerations(
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
            au + au_next
        )
    )

    dv_next = (
        dv
        + 0.5
        * time_step
        * (
            av + av_next
        )
    )

    df_next = (
        df
        + 0.5
        * time_step
        * (
            af + af_next
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


def print_progress(
    prefix: str,
    step: int,
    step_count: int,
    started_at: float,
    width: int = 24,
) -> None:
    fraction = min(
        max(
            step / max(step_count, 1),
            0.0,
        ),
        1.0,
    )

    filled = int(
        round(
            width * fraction
        )
    )

    bar = (
        "#"
        * filled
        + "-"
        * (
            width - filled
        )
    )

    elapsed = (
        time_module.monotonic()
        - started_at
    )

    print(
        "\r"
        + prefix
        + " ["
        + bar
        + "] "
        + f"{100.0 * fraction:6.2f}%"
        + f" elapsed={elapsed:7.1f}s",
        end="",
        flush=True,
    )


def normalized_perturbation(
    rng: np.random.Generator,
    size: int,
    scale: float,
) -> np.ndarray:
    vector = rng.normal(size=size)
    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        vector[0] = 1.0
        norm = 1.0

    return scale * vector / norm


def make_initial_packet(
    base: dict[str, np.ndarray],
    perturbation_id: int,
) -> dict[str, np.ndarray]:
    packet = {
        name: value.copy()
        for name, value in base.items()
    }

    if perturbation_id == 0:
        return packet

    rng = np.random.default_rng(
        RANDOM_SEED + perturbation_id
    )

    packet["u"] += normalized_perturbation(
        rng,
        6,
        0.01,
    )

    packet["v"] += normalized_perturbation(
        rng,
        6,
        0.01,
    )

    packet["du"] += normalized_perturbation(
        rng,
        6,
        0.002,
    )

    packet["dv"] += normalized_perturbation(
        rng,
        6,
        0.002,
    )

    packet["f"] += normalized_perturbation(
        rng,
        4,
        0.001,
    )

    packet["df"] += normalized_perturbation(
        rng,
        4,
        0.001,
    )

    return packet


def classify_run(
    finite: bool,
    maximum_state_norm: float,
    second_half_to_first_half_ratio: float,
    relative_energy_drift: float,
    time_step: float,
) -> str:
    if not finite:
        return "nonfinite_growth"

    if maximum_state_norm > STATE_NORM_GROWTH_THRESHOLD:
        return "large_growth_observed"

    if (
        second_half_to_first_half_ratio
        > HALF_WINDOW_GROWTH_RATIO_THRESHOLD
    ):
        return "finite_horizon_growth_observed"

    tolerance = (
        ENERGY_TOLERANCE_FINE
        if time_step == min(TIME_STEPS)
        else ENERGY_TOLERANCE_COARSE
    )

    if relative_energy_drift > tolerance:
        return "numerically_unresolved"

    return "bounded_on_tested_horizon"


def run_trajectory(
    channel: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    time_step: float,
    perturbation_id: int,
    initial_packet: dict[str, np.ndarray],
    progress_prefix: str,
) -> tuple[dict, np.ndarray]:
    u = initial_packet["u"].copy()
    v = initial_packet["v"].copy()
    f = initial_packet["f"].copy()

    du = initial_packet["du"].copy()
    dv = initial_packet["dv"].copy()
    df = initial_packet["df"].copy()

    step_count = int(
        round(
            TIME_HORIZON
            / time_step
        )
    )

    sample_every = max(
        1,
        int(
            round(
                SAMPLE_INTERVAL
                / time_step
            )
        ),
    )

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

    samples = []
    finite = True
    termination_reason = "completed_horizon"

    started_at = time_module.monotonic()
    progress_every = max(
        1,
        step_count // 100,
    )

    print_progress(
        progress_prefix,
        0,
        step_count,
        started_at,
    )

    for step in range(step_count + 1):
        if (
            step % progress_every == 0
            or step == step_count
        ):
            print_progress(
                progress_prefix,
                step,
                step_count,
                started_at,
            )
        if step % sample_every == 0:
            time = step * time_step

            energy = total_energy(
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

            combined_norm = float(
                np.sqrt(
                    u_norm**2
                    + v_norm**2
                    + f_norm**2
                )
            )

            samples.append(
                [
                    time,
                    energy,
                    u_norm,
                    v_norm,
                    f_norm,
                    combined_norm,
                    interaction(
                        channel,
                        u,
                        v,
                        f,
                    ),
                ]
            )

        if step == step_count:
            break

        current_absolute_maximum = max(
            float(np.max(np.abs(value)))
            for value in (
                u,
                v,
                f,
                du,
                dv,
                df,
            )
        )

        if (
            not np.isfinite(
                current_absolute_maximum
            )
        ):
            finite = False
            termination_reason = "nonfinite_state"
            break

        if (
            current_absolute_maximum
            > STATE_ABSOLUTE_SAFETY_CAP
        ):
            finite = False
            termination_reason = (
                "finite_state_safety_cap_exceeded"
            )
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
            time_step,
            u,
            v,
            f,
            du,
            dv,
            df,
        )

        finite = all(
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
        )

        if not finite:
            termination_reason = "nonfinite_after_step"
            break

    if termination_reason == "completed_horizon":
        print_progress(
            progress_prefix,
            step_count,
            step_count,
            started_at,
        )

    print(
        " status="
        + termination_reason
    )

    trajectory = np.array(
        samples,
        dtype=np.float64,
    )

    energies = trajectory[:, 1]
    combined_norms = trajectory[:, 5]
    f_norms = trajectory[:, 4]

    denominator = max(
        abs(initial_energy),
        1e-12,
    )

    relative_energy_drift = float(
        np.max(
            np.abs(
                energies - initial_energy
            )
        )
        / denominator
    )

    midpoint = max(
        1,
        len(combined_norms) // 2,
    )

    first_half_max = float(
        np.max(
            combined_norms[:midpoint]
        )
    )

    second_half_max = float(
        np.max(
            combined_norms[midpoint:]
        )
    )

    half_ratio = (
        second_half_max
        / max(
            first_half_max,
            1e-12,
        )
    )

    maximum_state_norm = float(
        np.max(combined_norms)
    )

    classification = classify_run(
        finite,
        maximum_state_norm,
        half_ratio,
        relative_energy_drift,
        time_step,
    )

    row = {
        "coupling": coupling,
        "time_step": time_step,
        "perturbation_id": perturbation_id,
        "step_count": step_count,
        "sample_count": len(
            trajectory
        ),
        "finite": finite,
        "termination_reason": termination_reason,
        "classification": classification,
        "initial_energy": initial_energy,
        "minimum_energy": float(
            np.min(energies)
        ),
        "maximum_energy": float(
            np.max(energies)
        ),
        "relative_energy_drift": (
            relative_energy_drift
        ),
        "maximum_u_norm": float(
            np.max(
                trajectory[:, 2]
            )
        ),
        "maximum_v_norm": float(
            np.max(
                trajectory[:, 3]
            )
        ),
        "maximum_f_norm": float(
            np.max(f_norms)
        ),
        "maximum_combined_state_norm": (
            maximum_state_norm
        ),
        "first_half_maximum_combined_norm": (
            first_half_max
        ),
        "second_half_maximum_combined_norm": (
            second_half_max
        ),
        "second_half_to_first_half_ratio": (
            half_ratio
        ),
        "final_combined_state_norm": float(
            combined_norms[-1]
        ),
        "final_f_norm": float(
            f_norms[-1]
        ),
    }

    return row, trajectory


def relative_difference(
    first: float,
    second: float,
) -> float:
    return abs(first - second) / max(
        abs(first),
        abs(second),
        1e-12,
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REFINEMENT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dynamics_receipt = json.loads(
        DYNAMICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        DYNAMICS_NPZ_PATH
    )

    channel = np.array(
        data["channel"],
        dtype=np.float64,
    )

    stiffness = np.array(
        data["stiffness"],
        dtype=np.float64,
    )

    base_packet = {
        "u": np.array(
            data["initial_u"],
            dtype=np.float64,
        ),
        "v": np.array(
            data["initial_v"],
            dtype=np.float64,
        ),
        "f": np.array(
            data["initial_f"],
            dtype=np.float64,
        ),
        "du": np.array(
            data["initial_du"],
            dtype=np.float64,
        ),
        "dv": np.array(
            data["initial_dv"],
            dtype=np.float64,
        ),
        "df": np.array(
            data["initial_df"],
            dtype=np.float64,
        ),
    }

    run_rows = []
    trajectories = {}

    total_run_count = (
        len(COUPLING_VALUES)
        * len(PERTURBATION_IDS)
        * len(TIME_STEPS)
    )

    current_run = 0
    full_scan_started_at = time_module.monotonic()

    for coupling in COUPLING_VALUES:
        for perturbation_id in PERTURBATION_IDS:
            packet = make_initial_packet(
                base_packet,
                perturbation_id,
            )

            for time_step in TIME_STEPS:
                current_run += 1

                progress_prefix = (
                    f"run {current_run:02d}/{total_run_count:02d}"
                    f" g={coupling:.2f}"
                    f" p={perturbation_id}"
                    f" dt={time_step:.4f}"
                )

                row, trajectory = run_trajectory(
                    channel,
                    stiffness,
                    coupling,
                    time_step,
                    perturbation_id,
                    packet,
                    progress_prefix,
                )

                run_rows.append(row)

                key = (
                    f"g_{coupling:.2f}"
                    f"_p_{perturbation_id}"
                    f"_dt_{time_step:.4f}"
                )

                key = key.replace(
                    ".",
                    "_",
                )

                trajectories[key] = trajectory

    indexed_rows = {
        (
            row["coupling"],
            row["perturbation_id"],
            row["time_step"],
        ): row
        for row in run_rows
    }

    coarse_step = max(TIME_STEPS)
    fine_step = min(TIME_STEPS)

    refinement_rows = []

    for coupling in COUPLING_VALUES:
        for perturbation_id in PERTURBATION_IDS:
            coarse = indexed_rows[
                (
                    coupling,
                    perturbation_id,
                    coarse_step,
                )
            ]

            fine = indexed_rows[
                (
                    coupling,
                    perturbation_id,
                    fine_step,
                )
            ]

            metrics = {
                "maximum_combined_state_norm": (
                    relative_difference(
                        coarse[
                            "maximum_combined_state_norm"
                        ],
                        fine[
                            "maximum_combined_state_norm"
                        ],
                    )
                ),
                "maximum_f_norm": (
                    relative_difference(
                        coarse[
                            "maximum_f_norm"
                        ],
                        fine[
                            "maximum_f_norm"
                        ],
                    )
                ),
                "final_combined_state_norm": (
                    relative_difference(
                        coarse[
                            "final_combined_state_norm"
                        ],
                        fine[
                            "final_combined_state_norm"
                        ],
                    )
                ),
                "final_f_norm": (
                    relative_difference(
                        coarse[
                            "final_f_norm"
                        ],
                        fine[
                            "final_f_norm"
                        ],
                    )
                ),
            }

            maximum_metric_difference = max(
                metrics.values()
            )

            classifications_match = (
                coarse["classification"]
                == fine["classification"]
            )

            refinement_pass = (
                classifications_match
                and maximum_metric_difference
                < REFINEMENT_RELATIVE_TOLERANCE
            )

            refinement_rows.append(
                {
                    "coupling": coupling,
                    "perturbation_id": (
                        perturbation_id
                    ),
                    "coarse_time_step": (
                        coarse_step
                    ),
                    "fine_time_step": (
                        fine_step
                    ),
                    "coarse_classification": (
                        coarse[
                            "classification"
                        ]
                    ),
                    "fine_classification": (
                        fine[
                            "classification"
                        ]
                    ),
                    "classifications_match": (
                        classifications_match
                    ),
                    **{
                        name
                        + "_relative_difference": value
                        for name, value in (
                            metrics.items()
                        )
                    },
                    "maximum_metric_relative_difference": (
                        maximum_metric_difference
                    ),
                    "refinement_pass": (
                        refinement_pass
                    ),
                }
            )

    fine_rows = [
        row
        for row in run_rows
        if row["time_step"] == fine_step
    ]

    coupling_fine_profiles = {}

    for coupling in COUPLING_VALUES:
        rows = [
            row
            for row in fine_rows
            if row["coupling"] == coupling
        ]

        coupling_fine_profiles[
            str(coupling)
        ] = {
            "classification_counts": dict(
                Counter(
                    row["classification"]
                    for row in rows
                )
            ),
            "maximum_energy_drift": max(
                row["relative_energy_drift"]
                for row in rows
            ),
            "maximum_combined_state_norm": max(
                row[
                    "maximum_combined_state_norm"
                ]
                for row in rows
            ),
            "maximum_flux_norm": max(
                row["maximum_f_norm"]
                for row in rows
            ),
            "maximum_half_window_growth_ratio": max(
                row[
                    "second_half_to_first_half_ratio"
                ]
                for row in rows
            ),
        }

    fine_classification_counts = dict(
        Counter(
            row["classification"]
            for row in fine_rows
        )
    )

    candidate_growth_couplings = sorted(
        {
            row["coupling"]
            for row in fine_rows
            if row["classification"]
            in (
                "finite_horizon_growth_observed",
                "large_growth_observed",
                "nonfinite_growth",
            )
        }
    )

    all_refinement_pass = all(
        row["refinement_pass"]
        for row in refinement_rows
    )

    all_fine_energy_resolved = all(
        row["relative_energy_drift"]
        < ENERGY_TOLERANCE_FINE
        for row in fine_rows
    )

    all_runs_finite = all(
        row["finite"]
        for row in run_rows
    )

    checks = {
        "input_022_audit_pass": (
            dynamics_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "channel_shape_is_4_by_36": (
            channel.shape == (4, 36)
        ),
        "stiffness_is_10_identity": bool(
            np.max(
                np.abs(
                    stiffness
                    - 10.0 * np.eye(4)
                )
            )
            < 1e-9
        ),
        "all_parameter_combinations_run": (
            len(run_rows)
            == (
                len(COUPLING_VALUES)
                * len(PERTURBATION_IDS)
                * len(TIME_STEPS)
            )
        ),
        "all_runs_remain_finite": (
            all_runs_finite
        ),
        "all_fine_runs_meet_energy_tolerance": (
            all_fine_energy_resolved
        ),
        "all_timestep_refinement_pairs_pass": (
            all_refinement_pass
        ),
    }

    audit_pass = all(
        checks.values()
    )

    if candidate_growth_couplings:
        verdict = (
            "native_g60_cross_flux_resolved_finite_horizon_"
            "growth_candidate_found"
        )
    else:
        verdict = (
            "native_g60_cross_flux_no_resolved_growth_"
            "through_tested_coupling_range"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_nonlinear_threshold_probe_023"
        ),
        "audit_pass": audit_pass,
        "verdict": verdict,
        "scan_design": {
            "coupling_values": list(
                COUPLING_VALUES
            ),
            "time_steps": list(
                TIME_STEPS
            ),
            "perturbation_ids": list(
                PERTURBATION_IDS
            ),
            "time_horizon": (
                TIME_HORIZON
            ),
            "sample_interval": (
                SAMPLE_INTERVAL
            ),
            "run_count": len(
                run_rows
            ),
            "refinement_pair_count": len(
                refinement_rows
            ),
        },
        "checks": checks,
        "fine_step_summary": {
            "time_step": fine_step,
            "classification_counts": (
                fine_classification_counts
            ),
            "candidate_growth_couplings": (
                candidate_growth_couplings
            ),
            "coupling_profiles": (
                coupling_fine_profiles
            ),
        },
        "refinement_summary": {
            "all_pairs_pass": (
                all_refinement_pass
            ),
            "maximum_metric_relative_difference": max(
                row[
                    "maximum_metric_relative_difference"
                ]
                for row in refinement_rows
            ),
            "classification_mismatch_count": sum(
                not row[
                    "classifications_match"
                ]
                for row in refinement_rows
            ),
        },
        "run_rows": run_rows,
        "refinement_rows": (
            refinement_rows
        ),
        "boundary": {
            "longer_horizon_scan_completed": (
                audit_pass
            ),
            "nearby_initial_packets_tested": (
                audit_pass
            ),
            "timestep_refinement_tested": (
                audit_pass
            ),
            "candidate_growth_coupling_identified": (
                bool(
                    candidate_growth_couplings
                )
            ),
            "global_nonlinear_stability_proved": (
                False
            ),
            "critical_coupling_proved": False,
            "asymptotic_unboundedness_proved": (
                False
            ),
            "physical_coupling_selected": (
                False
            ),
            "physical_time_claim": False,
            "physical_energy_claim": False,
            "force_claim": False,
            "physical_claim": False,
        },
        "outputs": {
            "run_csv": str(
                RUN_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "refinement_csv": str(
                REFINEMENT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "trajectory_npz": str(
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

    with RUN_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                run_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(run_rows)

    with REFINEMENT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                refinement_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            refinement_rows
        )

    npz_payload = {
        "channel": channel,
        "stiffness": stiffness,
        "coupling_values": np.array(
            COUPLING_VALUES,
            dtype=np.float64,
        ),
        "time_steps": np.array(
            TIME_STEPS,
            dtype=np.float64,
        ),
        "perturbation_ids": np.array(
            PERTURBATION_IDS,
            dtype=np.int64,
        ),
        "time_horizon": np.array(
            [TIME_HORIZON],
            dtype=np.float64,
        ),
    }

    npz_payload.update(
        trajectories
    )

    np.savez_compressed(
        NPZ_OUT,
        **npz_payload,
    )

    full_scan_elapsed = (
        time_module.monotonic()
        - full_scan_started_at
    )

    print(
        "scan_elapsed_seconds:",
        full_scan_elapsed,
    )
    print("audit_pass:", audit_pass)
    print("verdict:", verdict)
    print(
        "run_count:",
        len(run_rows),
    )
    print(
        "fine_classification_counts:",
        fine_classification_counts,
    )
    print(
        "candidate_growth_couplings:",
        candidate_growth_couplings,
    )
    print(
        "all_fine_energy_resolved:",
        all_fine_energy_resolved,
    )
    print(
        "all_refinement_pairs_pass:",
        all_refinement_pass,
    )
    print(
        "maximum_refinement_difference:",
        payload[
            "refinement_summary"
        ][
            "maximum_metric_relative_difference"
        ],
    )

    print("\nfine-step coupling summary:")

    for coupling in COUPLING_VALUES:
        profile = coupling_fine_profiles[
            str(coupling)
        ]

        print(
            "g=",
            coupling,
            "classes=",
            profile[
                "classification_counts"
            ],
            "max_state=",
            profile[
                "maximum_combined_state_norm"
            ],
            "max_flux=",
            profile[
                "maximum_flux_norm"
            ],
            "half_ratio=",
            profile[
                "maximum_half_window_growth_ratio"
            ],
            "energy_drift=",
            profile[
                "maximum_energy_drift"
            ],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", RUN_CSV_OUT)
    print("wrote:", REFINEMENT_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
