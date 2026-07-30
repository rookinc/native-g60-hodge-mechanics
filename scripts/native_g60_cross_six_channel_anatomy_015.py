from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESOLUTION_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_six_flux_channel_resolution_014.json"
)

RESOLUTION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_six_flux_channel_resolution_014.npz"
)

HARMONIC_REP_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_representation_006.npz"
)

ACTION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_six_channel_anatomy_015.json"
)

CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_six_channel_anatomy_015.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_six_channel_anatomy_015.npz"
)

TOLERANCE = 1e-8


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

    if singular_values.size == 0:
        return 0

    threshold = max(
        tolerance,
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * singular_values[0],
    )

    return int(
        np.count_nonzero(
            singular_values > threshold
        )
    )


def orthonormal_map_basis(
    maps: list[np.ndarray],
) -> list[np.ndarray]:
    if not maps:
        return []

    flattened = np.column_stack(
        [
            matrix.reshape(-1)
            for matrix in maps
        ]
    )

    u, singular_values, _ = np.linalg.svd(
        flattened,
        full_matrices=False,
    )

    if singular_values.size == 0:
        return []

    threshold = max(
        TOLERANCE,
        max(flattened.shape)
        * np.finfo(np.float64).eps
        * singular_values[0],
    )

    rank = int(
        np.count_nonzero(
            singular_values > threshold
        )
    )

    return [
        u[:, index].reshape(maps[0].shape)
        for index in range(rank)
    ]


def tensor_representation(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    result = np.empty(
        (
            left.shape[0],
            left.shape[1] * right.shape[1],
            left.shape[2] * right.shape[2],
        ),
        dtype=np.float64,
    )

    for index in range(left.shape[0]):
        result[index] = np.kron(
            left[index],
            right[index],
        )

    return result


def intertwiner_residual(
    intertwiner: np.ndarray,
    output_representation: np.ndarray,
    input_representation: np.ndarray,
) -> float:
    return max(
        max_abs(
            output_rho @ intertwiner
            - intertwiner @ input_rho
        )
        for output_rho, input_rho in zip(
            output_representation,
            input_representation,
        )
    )


def tensor_flip(
    left_dimension: int,
    right_dimension: int,
) -> np.ndarray:
    """
    Map coordinates in A tensor B to coordinates in B tensor A.

    Source index:
        i * dim(B) + j

    Target index:
        j * dim(A) + i
    """
    flip = np.zeros(
        (
            right_dimension * left_dimension,
            left_dimension * right_dimension,
        ),
        dtype=np.float64,
    )

    for i in range(left_dimension):
        for j in range(right_dimension):
            source = i * right_dimension + j
            target = j * left_dimension + i
            flip[target, source] = 1.0

    return flip


def restricted_representation(
    ambient_representation: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    result = np.empty(
        (
            ambient_representation.shape[0],
            basis.shape[1],
            basis.shape[1],
        ),
        dtype=np.float64,
    )

    for index in range(
        ambient_representation.shape[0]
    ):
        result[index] = (
            basis.T
            @ ambient_representation[index]
            @ basis
        )

    return result


def apply_signed_edge_action(
    basis: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(basis)

    transformed[target, :] = (
        sign[:, None] * basis
    )

    return transformed


def restricted_edge_representation(
    basis: np.ndarray,
    edge_targets: np.ndarray,
    edge_signs: np.ndarray,
) -> np.ndarray:
    result = np.empty(
        (
            edge_targets.shape[0],
            basis.shape[1],
            basis.shape[1],
        ),
        dtype=np.float64,
    )

    for index in range(edge_targets.shape[0]):
        transformed = apply_signed_edge_action(
            basis,
            edge_targets[index],
            edge_signs[index],
        )

        result[index] = basis.T @ transformed

    return result


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolution = json.loads(
        RESOLUTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        RESOLUTION_NPZ_PATH
    )

    harmonic_payload = np.load(
        HARMONIC_REP_PATH
    )

    action_payload = np.load(
        ACTION_PATH
    )

    harmonic_representation = np.array(
        harmonic_payload["rho_harmonic"],
        dtype=np.float64,
    )

    six_a_basis = np.array(
        data[
            "six_a_basis_harmonic_coordinates"
        ],
        dtype=np.float64,
    )

    six_b_basis = np.array(
        data[
            "six_b_basis_harmonic_coordinates"
        ],
        dtype=np.float64,
    )

    coexact_basis = np.array(
        data["coexact_basis_edges"],
        dtype=np.float64,
    )

    target_dimensions = [
        int(value)
        for value in data[
            "target_sector_dimensions"
        ]
    ]

    cross_maps = []

    map_index = 0

    while (
        f"six_a_tensor_six_b_map_{map_index}"
        in data.files
    ):
        cross_maps.append(
            np.array(
                data[
                    f"six_a_tensor_six_b_map_{map_index}"
                ],
                dtype=np.float64,
            )
        )

        map_index += 1

    if len(cross_maps) != 3:
        raise RuntimeError(
            "expected three cross-six maps, found "
            f"{len(cross_maps)}"
        )

    target_bases = []
    target_projectors = []

    for target_id in range(
        len(target_dimensions)
    ):
        target_bases.append(
            np.array(
                data[
                    f"target_sector_{target_id}_basis"
                ],
                dtype=np.float64,
            )
        )

        target_projectors.append(
            np.array(
                data[
                    f"target_sector_{target_id}_projector"
                ],
                dtype=np.float64,
            )
        )

    six_a_representation = (
        restricted_representation(
            harmonic_representation,
            six_a_basis,
        )
    )

    six_b_representation = (
        restricted_representation(
            harmonic_representation,
            six_b_basis,
        )
    )

    edge_targets = np.array(
        action_payload["edge_target"],
        dtype=np.int64,
    )

    edge_signs = np.array(
        action_payload["edge_sign"],
        dtype=np.int8,
    )

    coexact_representation = (
        restricted_edge_representation(
            coexact_basis,
            edge_targets,
            edge_signs,
        )
    )

    tensor_ab = tensor_representation(
        six_a_representation,
        six_b_representation,
    )

    tensor_ba = tensor_representation(
        six_b_representation,
        six_a_representation,
    )

    flip = tensor_flip(6, 6)

    flip_involution_residual = max_abs(
        flip @ flip - np.eye(36)
    )

    tensor_exchange_residual = max(
        max_abs(
            tensor_ba[index] @ flip
            - flip @ tensor_ab[index]
        )
        for index in range(480)
    )

    reverse_maps = [
        matrix @ flip.T
        for matrix in cross_maps
    ]

    reverse_equivariance_residuals = [
        intertwiner_residual(
            matrix,
            coexact_representation,
            tensor_ba,
        )
        for matrix in reverse_maps
    ]

    cross_map_gram = np.array(
        [
            [
                float(np.sum(left * right))
                for right in cross_maps
            ]
            for left in cross_maps
        ],
        dtype=np.float64,
    )

    target_rows = []
    target_channel_bases: dict[
        int,
        list[np.ndarray],
    ] = {}

    for target_id, (
        dimension,
        projector,
    ) in enumerate(
        zip(
            target_dimensions,
            target_projectors,
        )
    ):
        projected_maps = [
            projector @ matrix
            for matrix in cross_maps
        ]

        channel_basis = orthonormal_map_basis(
            projected_maps
        )

        target_channel_bases[
            target_id
        ] = channel_basis

        flattened_projection = np.column_stack(
            [
                matrix.reshape(-1)
                for matrix in projected_maps
            ]
        )

        channel_dimension = numerical_rank(
            flattened_projection
        )

        combined_image_dimension = numerical_rank(
            np.concatenate(
                projected_maps,
                axis=1,
            )
        )

        equivariance_residuals = [
            intertwiner_residual(
                matrix,
                coexact_representation,
                tensor_ab,
            )
            for matrix in channel_basis
        ]

        target_rows.append(
            {
                "target_sector_id": target_id,
                "target_dimension": dimension,
                "cross_channel_dimension": (
                    channel_dimension
                ),
                "cross_combined_image_dimension": (
                    combined_image_dimension
                ),
                "cross_reaches_full_target": (
                    combined_image_dimension
                    == dimension
                ),
                "maximum_equivariance_residual": (
                    max(equivariance_residuals)
                    if equivariance_residuals
                    else 0.0
                ),
            }
        )

    reached_target_rows = [
        row
        for row in target_rows
        if row[
            "cross_combined_image_dimension"
        ]
        > 0
    ]

    total_target_channel_dimension = sum(
        row["cross_channel_dimension"]
        for row in reached_target_rows
    )

    full_cross_hom_dimension = numerical_rank(
        np.column_stack(
            [
                matrix.reshape(-1)
                for matrix in cross_maps
            ]
        )
    )

    target_projection_reconstruction = []

    for matrix in cross_maps:
        reconstructed = sum(
            projector @ matrix
            for projector in target_projectors
        )

        target_projection_reconstruction.append(
            max_abs(
                reconstructed - matrix
            )
        )

    all_cross_maps_reconstruct = max(
        target_projection_reconstruction
    )

    channel_rows = []

    for channel_id, matrix in enumerate(
        cross_maps
    ):
        target_component_ranks = []
        target_component_norms = []

        for target_id, projector in enumerate(
            target_projectors
        ):
            component = projector @ matrix

            target_component_ranks.append(
                numerical_rank(component)
            )

            target_component_norms.append(
                float(
                    np.linalg.norm(
                        component,
                        ord="fro",
                    )
                )
            )

        channel_rows.append(
            {
                "channel_id": channel_id,
                "map_rank": numerical_rank(
                    matrix
                ),
                "equivariance_max_abs": (
                    intertwiner_residual(
                        matrix,
                        coexact_representation,
                        tensor_ab,
                    )
                ),
                "reverse_equivariance_max_abs": (
                    reverse_equivariance_residuals[
                        channel_id
                    ]
                ),
                "target_component_ranks": (
                    json.dumps(
                        target_component_ranks
                    )
                ),
                "target_component_frobenius_norms": (
                    json.dumps(
                        target_component_norms
                    )
                ),
            }
        )

    checks = {
        "input_014_audit_pass": (
            resolution.get("audit_pass") is True
        ),
        "three_cross_maps_loaded": (
            len(cross_maps) == 3
        ),
        "target_dimensions_are_4_4_5_6": (
            target_dimensions == [4, 4, 5, 6]
        ),
        "cross_hom_dimension_is_3": (
            full_cross_hom_dimension == 3
        ),
        "tensor_flip_is_involution": (
            flip_involution_residual
            < TOLERANCE
        ),
        "tensor_flip_intertwines_ab_and_ba": (
            tensor_exchange_residual
            < TOLERANCE
        ),
        "all_reverse_maps_are_equivariant": all(
            residual < TOLERANCE
            for residual in (
                reverse_equivariance_residuals
            )
        ),
        "cross_maps_reconstruct_from_target_sectors": (
            all_cross_maps_reconstruct
            < TOLERANCE
        ),
        "target_channel_dimensions_sum_to_3": (
            total_target_channel_dimension
            == 3
        ),
        "cross_reaches_exactly_two_target_sectors": (
            len(reached_target_rows) == 2
        ),
        "all_target_channel_bases_are_equivariant": all(
            row[
                "maximum_equivariance_residual"
            ]
            < TOLERANCE
            for row in reached_target_rows
        ),
    }

    audit_pass = all(checks.values())

    target_channel_signature = [
        {
            "target_sector_id": row[
                "target_sector_id"
            ],
            "target_dimension": row[
                "target_dimension"
            ],
            "channel_dimension": row[
                "cross_channel_dimension"
            ],
            "image_dimension": row[
                "cross_combined_image_dimension"
            ],
        }
        for row in reached_target_rows
    ]

    payload = {
        "artifact_id": (
            "native_g60_cross_six_channel_anatomy_015"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_cross_six_channel_target_anatomy_resolved"
            if audit_pass
            else "native_g60_cross_six_channel_anatomy_failed"
        ),
        "inputs": {
            "channel_resolution": str(
                RESOLUTION_PATH.relative_to(
                    ROOT
                )
            ),
            "channel_resolution_npz": str(
                RESOLUTION_NPZ_PATH.relative_to(
                    ROOT
                )
            ),
            "harmonic_representation": str(
                HARMONIC_REP_PATH.relative_to(
                    ROOT
                )
            ),
            "signed_edge_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
        },
        "checks": checks,
        "cross_hom_space": {
            "dimension": (
                full_cross_hom_dimension
            ),
            "basis_gram_max_abs": max_abs(
                cross_map_gram
                - np.eye(3)
            ),
            "target_channel_signature": (
                target_channel_signature
            ),
            "target_channel_dimension_sum": (
                total_target_channel_dimension
            ),
        },
        "exchange_transport": {
            "statement": (
                "Tensor flip transports an equivariant map "
                "V6a tensor V6b -> C into an equivariant map "
                "V6b tensor V6a -> C."
            ),
            "flip_shape": [36, 36],
            "flip_involution_max_abs": (
                flip_involution_residual
            ),
            "tensor_representation_intertwining_max_abs": (
                tensor_exchange_residual
            ),
            "reverse_map_equivariance_max_abs": (
                reverse_equivariance_residuals
            ),
            "exchange_parity_defined": False,
            "reason_exchange_parity_not_defined": (
                "V6a and V6b are distinct invariant sectors. "
                "The tensor flip canonically exchanges their order, "
                "but no native identification V6a = V6b has been "
                "derived."
            ),
        },
        "target_rows": target_rows,
        "channel_rows": channel_rows,
        "earned_interpretation": {
            "cross_channel_family_dimension": 3,
            "cross_target_dimension": 9,
            "target_native_channel_split": (
                target_channel_signature
            ),
            "unique_channel_selected": False,
            "exchange_order_transport_resolved": (
                audit_pass
            ),
            "exchange_even_odd_parity_resolved": (
                False
            ),
        },
        "outputs": {
            "channel_csv": str(
                CSV_OUT.relative_to(ROOT)
            ),
            "channel_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "cross_channel_target_anatomy_resolved": (
                audit_pass
            ),
            "target_adapted_channel_basis_exported": (
                audit_pass
            ),
            "canonical_exchange_transport_exported": (
                audit_pass
            ),
            "native_isomorphism_between_sixes_derived": (
                False
            ),
            "exchange_parity_claim": False,
            "unique_constitutive_channel_selected": (
                False
            ),
            "physical_flux_claim": False,
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

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                channel_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(channel_rows)

    npz_payload: dict[str, np.ndarray] = {
        "tensor_flip_ab_to_ba": flip,
        "cross_map_basis": np.stack(
            cross_maps,
            axis=0,
        ),
        "reverse_cross_map_basis": np.stack(
            reverse_maps,
            axis=0,
        ),
        "target_dimensions": np.array(
            target_dimensions,
            dtype=np.int64,
        ),
        "tolerance": np.array(
            [TOLERANCE]
        ),
    }

    for target_id, maps in (
        target_channel_bases.items()
    ):
        if maps:
            npz_payload[
                f"target_{target_id}_channel_basis"
            ] = np.stack(
                maps,
                axis=0,
            )
        else:
            npz_payload[
                f"target_{target_id}_channel_basis"
            ] = np.zeros(
                (0, 19, 36),
                dtype=np.float64,
            )

    np.savez_compressed(
        NPZ_OUT,
        **npz_payload,
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "cross_hom_dimension:",
        full_cross_hom_dimension,
    )
    print(
        "cross_basis_gram_max_abs:",
        payload["cross_hom_space"][
            "basis_gram_max_abs"
        ],
    )
    print(
        "tensor_flip_involution_max_abs:",
        flip_involution_residual,
    )
    print(
        "tensor_exchange_intertwining_max_abs:",
        tensor_exchange_residual,
    )
    print(
        "reverse_map_equivariance_max_abs:",
        reverse_equivariance_residuals,
    )

    print("\ntarget channel anatomy:")

    for row in target_rows:
        print(
            "target",
            row["target_sector_id"],
            "dimension=",
            row["target_dimension"],
            "channel_dimension=",
            row["cross_channel_dimension"],
            "image_dimension=",
            row[
                "cross_combined_image_dimension"
            ],
        )

    print(
        "\ntarget_channel_dimension_sum:",
        total_target_channel_dimension,
    )
    print(
        "target_channel_signature:",
        target_channel_signature,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
