from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

COUPLING_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_energy_coupling_021.json"
)

DYNAMICS_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_coupled_dynamics_022.json"
)

THRESHOLD_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_nonlinear_threshold_probe_023.json"
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
    / "native_g60_cross_flux_coercivity_obstruction_024.json"
)

START_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_extremal_starts_024.csv"
)

RAY_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_coercivity_ray_024.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_coercivity_obstruction_024.npz"
)

RANDOM_SEED = 46024
START_COUNT = 512
MAX_ITERATIONS = 2000
CONVERGENCE_TOLERANCE = 1e-14
ZERO_TOLERANCE = 1e-12
EXTREMAL_STATIONARITY_TOLERANCE = 1e-7

RAY_COUPLINGS = (
    0.25,
    0.50,
    1.00,
    1.50,
)

RAY_AMPLITUDES = (
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
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


def normalized(
    vector: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if norm <= ZERO_TOLERANCE:
        raise RuntimeError(
            "cannot normalize a zero vector"
        )

    result = vector / norm
    pivot = int(
        np.argmax(
            np.abs(result)
        )
    )

    if result[pivot] < 0.0:
        result *= -1.0

    return result


def bilinear_flux(
    tensor: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "rab,a,b->r",
        tensor,
        u,
        v,
    )


def trilinear_value(
    tensor: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> float:
    return float(
        np.einsum(
            "rab,a,b,r->",
            tensor,
            u,
            v,
            f,
        )
    )


def update_u(
    tensor: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "rab,b,r->a",
        tensor,
        v,
        f,
    )


def update_v(
    tensor: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "rab,a,r->b",
        tensor,
        u,
        f,
    )


def canonicalize_triplet(
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = u.copy()
    v = v.copy()
    f = f.copy()

    u_pivot = int(
        np.argmax(
            np.abs(u)
        )
    )

    if u[u_pivot] < 0.0:
        u *= -1.0
        f *= -1.0

    v_pivot = int(
        np.argmax(
            np.abs(v)
        )
    )

    if v[v_pivot] < 0.0:
        v *= -1.0
        f *= -1.0

    return u, v, f


def alternating_extremal_search(
    tensor: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    u = normalized(
        rng.normal(size=6)
    )

    v = normalized(
        rng.normal(size=6)
    )

    flux = bilinear_flux(
        tensor,
        u,
        v,
    )

    if np.linalg.norm(flux) <= ZERO_TOLERANCE:
        f = normalized(
            rng.normal(size=4)
        )
    else:
        f = normalized(flux)

    previous_value = -math.inf
    converged = False

    for iteration in range(
        1,
        MAX_ITERATIONS + 1,
    ):
        u_candidate = update_u(
            tensor,
            v,
            f,
        )

        if (
            np.linalg.norm(u_candidate)
            <= ZERO_TOLERANCE
        ):
            break

        u = normalized(u_candidate)

        v_candidate = update_v(
            tensor,
            u,
            f,
        )

        if (
            np.linalg.norm(v_candidate)
            <= ZERO_TOLERANCE
        ):
            break

        v = normalized(v_candidate)

        flux = bilinear_flux(
            tensor,
            u,
            v,
        )

        flux_norm = float(
            np.linalg.norm(flux)
        )

        if flux_norm <= ZERO_TOLERANCE:
            break

        f = flux / flux_norm

        value = trilinear_value(
            tensor,
            u,
            v,
            f,
        )

        if abs(
            value - previous_value
        ) <= CONVERGENCE_TOLERANCE:
            converged = True
            break

        previous_value = value

    flux = bilinear_flux(
        tensor,
        u,
        v,
    )

    flux_norm = float(
        np.linalg.norm(flux)
    )

    if flux_norm > ZERO_TOLERANCE:
        f = flux / flux_norm

    u, v, f = canonicalize_triplet(
        u,
        v,
        f,
    )

    value = trilinear_value(
        tensor,
        u,
        v,
        f,
    )

    return {
        "u": u,
        "v": v,
        "f": f,
        "value": value,
        "flux_norm": flux_norm,
        "iteration_count": iteration,
        "converged": converged,
    }


def minimized_flux_state(
    tensor: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    amplitude: float,
    u_direction: np.ndarray,
    v_direction: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    u = amplitude * u_direction
    v = amplitude * v_direction

    source = bilinear_flux(
        tensor,
        u,
        v,
    )

    f = -coupling * np.linalg.solve(
        stiffness,
        source,
    )

    return u, v, f


def potential_energy(
    tensor: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> float:
    return float(
        0.5
        * np.dot(
            f,
            stiffness @ f,
        )
        + coupling
        * np.dot(
            f,
            bilinear_flux(
                tensor,
                u,
                v,
            ),
        )
    )


def expected_minimized_potential(
    witness_gain: float,
    coupling: float,
    amplitude: float,
) -> float:
    return float(
        -(
            coupling**2
            * witness_gain**2
            * amplitude**4
        )
        / 20.0
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    START_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAY_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coupling_receipt = json.loads(
        COUPLING_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    dynamics_receipt = json.loads(
        DYNAMICS_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    threshold_receipt = json.loads(
        THRESHOLD_JSON_PATH.read_text(
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

    if channel.shape != (4, 36):
        raise RuntimeError(
            f"unexpected channel shape {channel.shape}"
        )

    tensor = channel.reshape(
        4,
        6,
        6,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    start_rows = []
    search_results = []

    for start_id in range(START_COUNT):
        result = alternating_extremal_search(
            tensor,
            rng,
        )

        search_results.append(result)

        start_rows.append(
            {
                "start_id": start_id,
                "value": result["value"],
                "flux_norm": (
                    result["flux_norm"]
                ),
                "iteration_count": (
                    result["iteration_count"]
                ),
                "converged": (
                    result["converged"]
                ),
            }
        )

    best_index = int(
        np.argmax(
            [
                result["flux_norm"]
                for result in search_results
            ]
        )
    )

    best = search_results[
        best_index
    ]

    witness_u = best["u"]
    witness_v = best["v"]

    witness_flux = bilinear_flux(
        tensor,
        witness_u,
        witness_v,
    )

    witness_gain = float(
        np.linalg.norm(
            witness_flux
        )
    )

    witness_f = (
        witness_flux
        / witness_gain
    )

    flattening = channel
    flattening_singular_values = (
        np.linalg.svd(
            flattening,
            compute_uv=False,
        )
    )

    flattening_upper_bound = float(
        flattening_singular_values[0]
    )

    upper_lower_gap = (
        flattening_upper_bound
        - witness_gain
    )

    relative_upper_lower_gap = (
        upper_lower_gap
        / max(
            flattening_upper_bound,
            ZERO_TOLERANCE,
        )
    )

    stationarity_residual_u = float(
        np.linalg.norm(
            update_u(
                tensor,
                witness_v,
                witness_f,
            )
            - witness_gain
            * witness_u
        )
    )

    stationarity_residual_v = float(
        np.linalg.norm(
            update_v(
                tensor,
                witness_u,
                witness_f,
            )
            - witness_gain
            * witness_v
        )
    )

    stationarity_residual_f = float(
        np.linalg.norm(
            witness_flux
            - witness_gain
            * witness_f
        )
    )


    ray_rows = []
    maximum_ray_formula_residual = 0.0
    negative_ray_value_count = 0

    for coupling in RAY_COUPLINGS:
        for amplitude in RAY_AMPLITUDES:
            u, v, f = minimized_flux_state(
                tensor,
                stiffness,
                coupling,
                amplitude,
                witness_u,
                witness_v,
            )

            observed = potential_energy(
                tensor,
                stiffness,
                coupling,
                u,
                v,
                f,
            )

            expected = (
                expected_minimized_potential(
                    witness_gain,
                    coupling,
                    amplitude,
                )
            )

            residual = abs(
                observed - expected
            )

            maximum_ray_formula_residual = max(
                maximum_ray_formula_residual,
                residual,
            )

            if observed < 0.0:
                negative_ray_value_count += 1

            normalized_quartic_coefficient = (
                observed / amplitude**4
            )

            ray_rows.append(
                {
                    "coupling": coupling,
                    "amplitude": amplitude,
                    "observed_potential": (
                        observed
                    ),
                    "expected_potential": (
                        expected
                    ),
                    "formula_residual": (
                        residual
                    ),
                    "potential_over_amplitude_fourth": (
                        normalized_quartic_coefficient
                    ),
                    "u_norm": float(
                        np.linalg.norm(u)
                    ),
                    "v_norm": float(
                        np.linalg.norm(v)
                    ),
                    "f_norm": float(
                        np.linalg.norm(f)
                    ),
                    "negative_potential": (
                        observed < 0.0
                    ),
                }
            )

    expected_ray_row_count = (
        len(RAY_COUPLINGS)
        * len(RAY_AMPLITUDES)
    )

    converged_count = sum(
        result["converged"]
        for result in search_results
    )

    distinct_rounded_values = sorted(
        {
            round(
                result["flux_norm"],
                12,
            )
            for result in search_results
        }
    )

    checks = {
        "input_021_audit_pass": (
            coupling_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_022_audit_pass": (
            dynamics_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_023_audit_pass": (
            threshold_receipt.get(
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
        "nonzero_bilinear_witness_found": (
            witness_gain
            > ZERO_TOLERANCE
        ),
        "witness_vectors_are_unit": (
            abs(
                np.linalg.norm(
                    witness_u
                )
                - 1.0
            )
            < 1e-10
            and abs(
                np.linalg.norm(
                    witness_v
                )
                - 1.0
            )
            < 1e-10
            and abs(
                np.linalg.norm(
                    witness_f
                )
                - 1.0
            )
            < 1e-10
        ),
        "extremal_candidate_stationarity_residuals_resolved": (
            max(
                stationarity_residual_u,
                stationarity_residual_v,
                stationarity_residual_f,
            )
            < EXTREMAL_STATIONARITY_TOLERANCE
        ),
        "witness_below_flattening_upper_bound": (
            witness_gain
            <= flattening_upper_bound
            + 1e-12
        ),
        "all_ray_potentials_negative": (
            negative_ray_value_count
            == expected_ray_row_count
        ),
        "quartic_ray_formula_verified": (
            maximum_ray_formula_residual
            < 1e-9
        ),
    }

    audit_pass = all(
        checks.values()
    )

    global_extremum_certified = (
        relative_upper_lower_gap
        < 1e-10
    )

    theorem_statement = (
        "Because the cross-six bilinear channel has a nonzero "
        "unit witness with gain sigma_w > 0, the potential restricted "
        "to u=a*u_w, v=a*v_w and "
        "f=-(g/10)T4(u,v) equals "
        "-g^2*sigma_w^2*a^4/20. "
        "For every nonzero g this tends to negative infinity as "
        "a tends to infinity. Therefore the minimal reciprocal "
        "6+6+4 potential is not coercive."
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_coercivity_obstruction_024"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": audit_pass,
        "verdict": (
            "native_g60_cross_flux_potential_noncoercive"
            if audit_pass
            else "native_g60_cross_flux_coercivity_audit_failed"
        ),
        "theorem": {
            "statement": theorem_statement,
            "status": (
                "constructive noncoercivity theorem"
                if audit_pass
                else "not established"
            ),
            "required_fact": (
                "one nonzero bilinear channel witness"
            ),
            "ray": {
                "u": "a u_w",
                "v": "a v_w",
                "f": (
                    "-(g/10) T4(a u_w, a v_w)"
                ),
            },
            "restricted_potential": (
                "-g^2 sigma_w^2 a^4 / 20"
            ),
            "limit": (
                "negative infinity as a tends to infinity "
                "for every nonzero g"
            ),
        },
        "extremal_search": {
            "status": (
                "numerical extremal candidate"
            ),
            "start_count": START_COUNT,
            "converged_start_count": (
                converged_count
            ),
            "maximum_iterations": (
                MAX_ITERATIONS
            ),
            "random_seed": (
                RANDOM_SEED
            ),
            "witness_gain_lower_bound": (
                witness_gain
            ),
            "flattening_upper_bound": (
                flattening_upper_bound
            ),
            "absolute_upper_lower_gap": (
                upper_lower_gap
            ),
            "relative_upper_lower_gap": (
                relative_upper_lower_gap
            ),
            "global_extremum_certified": (
                global_extremum_certified
            ),
            "distinct_rounded_terminal_values": (
                distinct_rounded_values
            ),
        },
        "witness": {
            "u": witness_u.tolist(),
            "v": witness_v.tolist(),
            "unit_flux_direction": (
                witness_f.tolist()
            ),
            "flux": (
                witness_flux.tolist()
            ),
            "gain": witness_gain,
            "quartic_descent_coefficient_at_g_1": (
                -(witness_gain**2)
                / 20.0
            ),
            "stationarity_residuals": {
                "u": (
                    stationarity_residual_u
                ),
                "v": (
                    stationarity_residual_v
                ),
                "f": (
                    stationarity_residual_f
                ),
                "candidate_tolerance": (
                    EXTREMAL_STATIONARITY_TOLERANCE
                ),
                "load_bearing_for_noncoercivity": (
                    False
                ),
            },
        },
        "ray_verification": {
            "couplings": list(
                RAY_COUPLINGS
            ),
            "amplitudes": list(
                RAY_AMPLITUDES
            ),
            "row_count": len(
                ray_rows
            ),
            "negative_value_count": (
                negative_ray_value_count
            ),
            "maximum_formula_residual": (
                maximum_ray_formula_residual
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "minimal_potential_is_noncoercive": (
                audit_pass
            ),
            "energy_conservation_implies_global_boundedness": (
                False
            ),
            "bounded_low_amplitude_trajectories_remain_valid": (
                True
            ),
            "numerical_extremal_is_globally_certified": (
                global_extremum_certified
            ),
            "stabilizing_terms_are_required_for_global_confinement": (
                audit_pass
            ),
        },
        "boundary": {
            "constructive_escape_direction_proved": (
                audit_pass
            ),
            "actual_dynamical_escape_proved": (
                False
            ),
            "finite_time_blowup_proved": False,
            "critical_coupling_proved": False,
            "sharp_tensor_norm_proved": (
                global_extremum_certified
            ),
            "specific_stabilizing_completion_selected": (
                False
            ),
            "physical_energy_claim": False,
            "physical_instability_claim": False,
            "force_claim": False,
            "physical_claim": False,
        },
        "outputs": {
            "start_csv": str(
                START_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "ray_csv": str(
                RAY_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "witness_npz": str(
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

    with START_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                start_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(start_rows)

    with RAY_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                ray_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(ray_rows)

    np.savez_compressed(
        NPZ_OUT,
        channel=channel,
        tensor=tensor,
        stiffness=stiffness,
        witness_u=witness_u,
        witness_v=witness_v,
        witness_flux=witness_flux,
        witness_f=witness_f,
        witness_gain=np.array(
            [witness_gain],
            dtype=np.float64,
        ),
        flattening_upper_bound=np.array(
            [flattening_upper_bound],
            dtype=np.float64,
        ),
        start_terminal_values=np.array(
            [
                result["flux_norm"]
                for result in search_results
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "witness_gain_lower_bound:",
        witness_gain,
    )
    print(
        "flattening_upper_bound:",
        flattening_upper_bound,
    )
    print(
        "relative_upper_lower_gap:",
        relative_upper_lower_gap,
    )
    print(
        "global_extremum_certified:",
        global_extremum_certified,
    )
    print(
        "converged_start_count:",
        converged_count,
        "/",
        START_COUNT,
    )
    print(
        "distinct_rounded_terminal_values:",
        distinct_rounded_values,
    )
    print(
        "stationarity_residuals:",
        {
            "u": stationarity_residual_u,
            "v": stationarity_residual_v,
            "f": stationarity_residual_f,
        },
    )
    print(
        "negative_ray_value_count:",
        negative_ray_value_count,
        "/",
        expected_ray_row_count,
    )
    print(
        "maximum_ray_formula_residual:",
        maximum_ray_formula_residual,
    )
    print(
        "quartic_descent_coefficient_at_g_1:",
        -(witness_gain**2)
        / 20.0,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", START_CSV_OUT)
    print("wrote:", RAY_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
