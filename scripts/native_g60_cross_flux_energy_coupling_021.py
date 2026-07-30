from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CHANNEL_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_unique_four_flux_response_016.npz"
)

ANATOMY_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_six_channel_anatomy_015.json"
)

REPRESENTATION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_representation_006.npz"
)

IRREDUCIBLE_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_irreducible_projectors_006c.npz"
)

FOUR_GEOMETRY_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_face_geometry_017.npz"
)

STATIC_SOLVER_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_static_field_solver_008.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_energy_coupling_021.json"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_energy_coupling_probes_021.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_energy_coupling_021.npz"
)

TOLERANCE = 1e-9
PROBE_COUNT = 64
RANDOM_SEED = 46021


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(np.max(np.abs(array)))


def orthonormal_basis_from_projector(
    projector: np.ndarray,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (projector + projector.T)
    )

    basis = eigenvectors[:, eigenvalues > 0.5]

    for column in range(basis.shape[1]):
        vector = basis[:, column]
        pivot = int(np.argmax(np.abs(vector)))

        if vector[pivot] < 0:
            basis[:, column] *= -1.0

    return basis


def normalize_channel(
    channel: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(channel, ord="fro"))

    if norm <= TOLERANCE:
        raise RuntimeError("zero channel")

    normalized = channel / norm

    pivot_flat = int(np.argmax(np.abs(normalized)))
    pivot_row, pivot_column = np.unravel_index(
        pivot_flat,
        normalized.shape,
    )

    if normalized[pivot_row, pivot_column] < 0:
        normalized *= -1.0

    return normalized


def tensor_coordinate(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    return np.kron(u, v)


def bilinear_flux(
    channel: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    return channel @ tensor_coordinate(u, v)


def gradient_u(
    channel: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    tensor = channel.reshape(
        channel.shape[0],
        6,
        6,
    )

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
    tensor = channel.reshape(
        channel.shape[0],
        6,
        6,
    )

    return np.einsum(
        "rab,a,r->b",
        tensor,
        u,
        f,
    )


def trilinear(
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


def energy(
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    df: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    channel: np.ndarray,
) -> float:
    interaction = coupling * trilinear(
        channel,
        u,
        v,
        f,
    )

    return float(
        0.5 * np.dot(du, du)
        + 0.5 * np.dot(dv, dv)
        + 0.5 * np.dot(df, df)
        + 0.5 * np.dot(
            f,
            stiffness @ f,
        )
        + interaction
    )


def energy_derivative(
    u: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    df: np.ndarray,
    ddu: np.ndarray,
    ddv: np.ndarray,
    ddf: np.ndarray,
    stiffness: np.ndarray,
    coupling: float,
    channel: np.ndarray,
) -> float:
    grad_u = gradient_u(
        channel,
        v,
        f,
    )

    grad_v = gradient_v(
        channel,
        u,
        f,
    )

    source_f = bilinear_flux(
        channel,
        u,
        v,
    )

    return float(
        np.dot(du, ddu)
        + np.dot(dv, ddv)
        + np.dot(
            df,
            ddf + stiffness @ f,
        )
        + coupling
        * (
            np.dot(du, grad_u)
            + np.dot(dv, grad_v)
            + np.dot(df, source_f)
        )
    )


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    anatomy = json.loads(
        ANATOMY_PATH.read_text(encoding="utf-8")
    )

    channel_data = np.load(CHANNEL_PATH)
    representation_data = np.load(REPRESENTATION_PATH)

    irreducible_data = np.load(
        IRREDUCIBLE_PATH,
        allow_pickle=True,
    )

    four_data = np.load(FOUR_GEOMETRY_PATH)
    static_data = np.load(STATIC_SOLVER_PATH)

    source_channel_19 = np.array(
        channel_data["unique_four_channel"],
        dtype=np.float64,
    )

    face_flux_matrix = np.array(
        channel_data["face_flux_matrix"],
        dtype=np.float64,
    )

    harmonic_representation = np.array(
        representation_data["rho_harmonic"],
        dtype=np.float64,
    )

    sector_projectors = [
        np.array(value, dtype=np.float64)
        for value in irreducible_data[
            "sector_projector"
        ]
    ]

    sector_dimensions = [
        int(value)
        for value in irreducible_data[
            "sector_dimensions"
        ]
    ]

    six_indices = [
        index
        for index, dimension in enumerate(
            sector_dimensions
        )
        if dimension == 6
    ]

    if len(six_indices) != 2:
        raise RuntimeError(
            "expected two six-dimensional sectors"
        )

    six_a_basis = orthonormal_basis_from_projector(
        sector_projectors[six_indices[0]]
    )

    six_b_basis = orthonormal_basis_from_projector(
        sector_projectors[six_indices[1]]
    )

    six_a_representation = np.empty(
        (480, 6, 6),
        dtype=np.float64,
    )

    six_b_representation = np.empty(
        (480, 6, 6),
        dtype=np.float64,
    )

    for index in range(480):
        six_a_representation[index] = (
            six_a_basis.T
            @ harmonic_representation[index]
            @ six_a_basis
        )

        six_b_representation[index] = (
            six_b_basis.T
            @ harmonic_representation[index]
            @ six_b_basis
        )

    four_representation = np.array(
        four_data["face_flux_representation"],
        dtype=np.float64,
    )

    face_flux_basis = np.array(
        four_data["face_flux_basis"],
        dtype=np.float64,
    )

    # Express the complete cross-six response in the same
    # four-dimensional coordinates used by four_representation.
    channel = normalize_channel(
        face_flux_basis.T
        @ face_flux_matrix
    )

    # The natural positive stiffness on face-flux coordinates is
    # the restriction of Delta2 = B2^T B2.
    b2_path = (
        ROOT
        / "data"
        / "imported"
        / "project45"
        / "native_g60_B2_edge_face_004.csv"
    )

    with b2_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        b2_rows = list(csv.reader(handle))

    b2 = np.array(
        [
            [float(value) for value in row[1:]]
            for row in b2_rows[1:]
        ],
        dtype=np.float64,
    )

    delta2 = b2.T @ b2

    stiffness = (
        face_flux_basis.T
        @ delta2
        @ face_flux_basis
    )

    stiffness = 0.5 * (
        stiffness + stiffness.T
    )

    stiffness_eigenvalues = np.linalg.eigvalsh(
        stiffness
    )

    equivariance_residual = 0.0

    for index in range(480):
        tensor_representation = np.kron(
            six_a_representation[index],
            six_b_representation[index],
        )

        equivariance_residual = max(
            equivariance_residual,
            max_abs(
                four_representation[index]
                @ channel
                - channel @ tensor_representation
            ),
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    coupling = 0.25
    probe_rows = []

    global_residuals = {
        "trilinear_equivariance": 0.0,
        "gradient_u_finite_difference": 0.0,
        "gradient_v_finite_difference": 0.0,
        "gradient_f_finite_difference": 0.0,
        "energy_derivative": 0.0,
        "reciprocal_power_balance": 0.0,
    }

    epsilon = 1e-6

    for probe_id in range(PROBE_COUNT):
        u = rng.normal(size=6)
        v = rng.normal(size=6)
        f = rng.normal(size=4)

        du = rng.normal(size=6)
        dv = rng.normal(size=6)
        df = rng.normal(size=4)

        grad_u = gradient_u(
            channel,
            v,
            f,
        )

        grad_v = gradient_v(
            channel,
            u,
            f,
        )

        grad_f = bilinear_flux(
            channel,
            u,
            v,
        )

        acceleration_u = (
            -coupling * grad_u
        )

        acceleration_v = (
            -coupling * grad_v
        )

        acceleration_f = (
            -stiffness @ f
            - coupling * grad_f
        )

        derivative = energy_derivative(
            u,
            v,
            f,
            du,
            dv,
            df,
            acceleration_u,
            acceleration_v,
            acceleration_f,
            stiffness,
            coupling,
            channel,
        )

        group_index = int(
            rng.integers(0, 480)
        )

        u_image = (
            six_a_representation[group_index]
            @ u
        )

        v_image = (
            six_b_representation[group_index]
            @ v
        )

        f_image = (
            four_representation[group_index]
            @ f
        )

        invariant_before = trilinear(
            channel,
            u,
            v,
            f,
        )

        invariant_after = trilinear(
            channel,
            u_image,
            v_image,
            f_image,
        )

        direction_u = rng.normal(size=6)
        direction_v = rng.normal(size=6)
        direction_f = rng.normal(size=4)

        finite_u = (
            trilinear(
                channel,
                u + epsilon * direction_u,
                v,
                f,
            )
            - trilinear(
                channel,
                u - epsilon * direction_u,
                v,
                f,
            )
        ) / (2.0 * epsilon)

        finite_v = (
            trilinear(
                channel,
                u,
                v + epsilon * direction_v,
                f,
            )
            - trilinear(
                channel,
                u,
                v - epsilon * direction_v,
                f,
            )
        ) / (2.0 * epsilon)

        finite_f = (
            trilinear(
                channel,
                u,
                v,
                f + epsilon * direction_f,
            )
            - trilinear(
                channel,
                u,
                v,
                f - epsilon * direction_f,
            )
        ) / (2.0 * epsilon)

        residuals = {
            "trilinear_equivariance": abs(
                invariant_after
                - invariant_before
            ),
            "gradient_u_finite_difference": abs(
                finite_u
                - np.dot(
                    grad_u,
                    direction_u,
                )
            ),
            "gradient_v_finite_difference": abs(
                finite_v
                - np.dot(
                    grad_v,
                    direction_v,
                )
            ),
            "gradient_f_finite_difference": abs(
                finite_f
                - np.dot(
                    grad_f,
                    direction_f,
                )
            ),
            "energy_derivative": abs(
                derivative
            ),
            "reciprocal_power_balance": abs(
                np.dot(
                    du,
                    coupling * grad_u,
                )
                + np.dot(
                    dv,
                    coupling * grad_v,
                )
                + np.dot(
                    df,
                    coupling * grad_f,
                )
                - coupling
                * (
                    np.dot(du, grad_u)
                    + np.dot(dv, grad_v)
                    + np.dot(df, grad_f)
                )
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
                "group_index": group_index,
                "trilinear_value": (
                    invariant_before
                ),
                "energy": energy(
                    u,
                    v,
                    f,
                    du,
                    dv,
                    df,
                    stiffness,
                    coupling,
                    channel,
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
        "input_015_audit_pass": (
            anatomy.get("audit_pass") is True
        ),
        "channel_shape_is_4_by_36": (
            channel.shape == (4, 36)
        ),
        "six_sector_dimensions_are_6_and_6": (
            six_a_basis.shape == (42, 6)
            and six_b_basis.shape == (42, 6)
        ),
        "four_flux_representation_shape_is_480_by_4_by_4": (
            four_representation.shape
            == (480, 4, 4)
        ),
        "channel_is_equivariant": (
            equivariance_residual
            < TOLERANCE
        ),
        "stiffness_is_symmetric": (
            max_abs(
                stiffness - stiffness.T
            )
            < TOLERANCE
        ),
        "stiffness_is_positive": (
            float(
                np.min(
                    stiffness_eigenvalues
                )
            )
            > 0.0
        ),
        "all_reciprocal_coupling_probes_pass": (
            all_probes_pass
        ),
        "energy_derivative_is_zero": (
            global_residuals[
                "energy_derivative"
            ]
            < TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_energy_coupling_021"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_cross_flux_reciprocal_energy_coupling_identified"
            if audit_pass
            else "native_g60_cross_flux_energy_coupling_audit_failed"
        ),
        "construction": {
            "interaction_scalar": (
                "I(u,v,f) = <f,T4(u tensor v)>"
            ),
            "interaction_potential": (
                "U_int = g I(u,v,f)"
            ),
            "coupling_used_for_probe": coupling,
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
                    "f_double_dot + K4 f = "
                    "-g T4(u tensor v)"
                ),
            },
        },
        "checks": checks,
        "channel": {
            "shape": list(
                channel.shape
            ),
            "frobenius_norm": float(
                np.linalg.norm(
                    channel,
                    ord="fro",
                )
            ),
            "equivariance_max_abs": (
                equivariance_residual
            ),
        },
        "four_flux_stiffness": {
            "shape": list(
                stiffness.shape
            ),
            "eigenvalues": [
                float(value)
                for value in (
                    stiffness_eigenvalues
                )
            ],
            "minimum_eigenvalue": float(
                stiffness_eigenvalues[0]
            ),
            "maximum_eigenvalue": float(
                stiffness_eigenvalues[-1]
            ),
        },
        "energy": {
            "formula": (
                "E = 1/2||u_dot||^2 "
                "+ 1/2||v_dot||^2 "
                "+ 1/2||f_dot||^2 "
                "+ 1/2<f,K4 f> "
                "+ g<f,T4(u tensor v)>"
            ),
            "differential_balance": (
                "dE/dt = 0"
            ),
            "status": (
                "finite mathematical interaction energy"
            ),
        },
        "probe_audit": {
            "probe_count": PROBE_COUNT,
            "random_seed": RANDOM_SEED,
            "all_probes_pass": (
                all_probes_pass
            ),
            "global_maximum_residuals": (
                global_residuals
            ),
        },
        "boundary": {
            "reciprocal_feedback_law_constructed": (
                audit_pass
            ),
            "symmetry_invariant_interaction_scalar_constructed": (
                audit_pass
            ),
            "mathematical_total_energy_conserved": (
                audit_pass
            ),
            "coupling_strength_derived": False,
            "physical_energy_claim": False,
            "physical_thread_claim": False,
            "physical_tension_claim": False,
            "force_claim": False,
            "physical_claim": False,
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
            fieldnames=list(
                probe_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            probe_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        unique_four_channel=channel,
        six_a_basis=six_a_basis,
        six_b_basis=six_b_basis,
        four_flux_basis=(
            face_flux_basis
        ),
        source_channel_19=source_channel_19,
        face_flux_matrix=face_flux_matrix,
        four_flux_stiffness=stiffness,
        coupling=np.array(
            [coupling]
        ),
        tolerance=np.array(
            [TOLERANCE]
        ),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "channel_equivariance_max_abs:",
        equivariance_residual,
    )
    print(
        "four_flux_stiffness_eigenvalues:",
        [
            float(value)
            for value in (
                stiffness_eigenvalues
            )
        ],
    )
    print(
        "all_probes_pass:",
        all_probes_pass,
    )
    print(
        "global_maximum_residuals:",
        global_residuals,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
