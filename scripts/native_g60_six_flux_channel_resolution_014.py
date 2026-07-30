from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

PERMISSION_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_six_sector_flux_intertwiner_013.json"
)

HARMONIC_REP_PATH = (
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

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
)

ACTION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

ISO_PATH = (
    SOURCE_ROOT
    / "native_g60_fiber_product_isomorphism_044.json"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_six_flux_channel_resolution_014.json"
)

CHANNEL_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_six_flux_channels_014.csv"
)

TARGET_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_coexact_target_sectors_014.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_six_flux_channel_resolution_014.npz"
)

TOLERANCE = 1e-8
SEED_COUNT = 32


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


def orthonormal_column_basis(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(
            (matrix.shape[0], 0),
            dtype=np.float64,
        )

    u, singular_values, _ = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    if singular_values.size == 0:
        return np.zeros(
            (matrix.shape[0], 0),
            dtype=np.float64,
        )

    threshold = max(
        tolerance,
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * singular_values[0],
    )

    rank = int(
        np.count_nonzero(
            singular_values > threshold
        )
    )

    return u[:, :rank]


def orthonormal_basis_from_projector(
    projector: np.ndarray,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (projector + projector.T)
    )

    selected = eigenvalues > 0.5
    basis = eigenvectors[:, selected]

    for column in range(basis.shape[1]):
        vector = basis[:, column]
        pivot = int(np.argmax(np.abs(vector)))

        if vector[pivot] < 0:
            basis[:, column] *= -1.0

    return basis


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


def deterministic_seed(
    output_dimension: int,
    input_dimension: int,
    seed_id: int,
) -> np.ndarray:
    row = np.arange(
        1,
        output_dimension + 1,
        dtype=np.float64,
    )[:, None]

    column = np.arange(
        1,
        input_dimension + 1,
        dtype=np.float64,
    )[None, :]

    family = seed_id % 8
    level = 1 + seed_id // 8

    if family == 0:
        return np.sin(level * row * column)

    if family == 1:
        return np.cos(level * row + column)

    if family == 2:
        return np.sin(row + level * column)

    if family == 3:
        return np.cos(row * column / level)

    if family == 4:
        return (
            ((row + level * column) % 11.0)
            - 5.0
        )

    if family == 5:
        return np.sin(level * row) * np.cos(column)

    if family == 6:
        return np.cos(row) * np.sin(level * column)

    return (
        np.sin(row * column)
        + np.cos(level * row + 2.0 * column)
    )


def reynolds_intertwiner(
    output_representation: np.ndarray,
    input_representation: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(seed)

    for output_rho, input_rho in zip(
        output_representation,
        input_representation,
    ):
        result += (
            output_rho
            @ seed
            @ input_rho.T
        )

    result /= output_representation.shape[0]

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


def construct_hom_basis(
    output_representation: np.ndarray,
    input_representation: np.ndarray,
    expected_dimension: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    averaged_maps = []

    for seed_id in range(SEED_COUNT):
        seed = deterministic_seed(
            output_representation.shape[1],
            input_representation.shape[1],
            seed_id,
        )

        averaged_maps.append(
            reynolds_intertwiner(
                output_representation,
                input_representation,
                seed,
            )
        )

    flattened = np.column_stack(
        [
            matrix.reshape(-1)
            for matrix in averaged_maps
        ]
    )

    u, singular_values, _ = np.linalg.svd(
        flattened,
        full_matrices=False,
    )

    threshold = max(
        TOLERANCE,
        max(flattened.shape)
        * np.finfo(np.float64).eps
        * singular_values[0],
    )

    recovered_dimension = int(
        np.count_nonzero(
            singular_values > threshold
        )
    )

    basis_maps = [
        u[:, index].reshape(
            output_representation.shape[1],
            input_representation.shape[1],
        )
        for index in range(recovered_dimension)
    ]

    gram = np.array(
        [
            [
                float(
                    np.sum(left * right)
                )
                for right in basis_maps
            ]
            for left in basis_maps
        ],
        dtype=np.float64,
    )

    residuals = [
        intertwiner_residual(
            matrix,
            output_representation,
            input_representation,
        )
        for matrix in basis_maps
    ]

    return basis_maps, {
        "seed_count": SEED_COUNT,
        "expected_dimension": expected_dimension,
        "recovered_dimension": recovered_dimension,
        "singular_values": [
            float(value)
            for value in singular_values
        ],
        "hom_basis_gram_max_abs": (
            max_abs(
                gram
                - np.eye(recovered_dimension)
            )
            if recovered_dimension
            else 0.0
        ),
        "equivariance_residuals": residuals,
        "maximum_equivariance_residual": (
            max(residuals)
            if residuals
            else 0.0
        ),
    }


def compose(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        left[right[index]]
        for index in range(len(right))
    )


def inverse(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    result = [0] * len(permutation)

    for source, target in enumerate(permutation):
        result[target] = source

    return tuple(result)


def permutation_order(
    permutation: tuple[int, ...],
) -> int:
    identity = tuple(range(len(permutation)))
    power = identity

    for order in range(1, 10000):
        power = compose(permutation, power)

        if power == identity:
            return order

    raise RuntimeError("order search exceeded bound")


def cycle_type(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    seen = set()
    lengths = []

    for start in range(len(permutation)):
        if start in seen:
            continue

        current = start
        length = 0

        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1

        lengths.append(length)

    return tuple(sorted(lengths, reverse=True))


def conjugacy_classes(
    permutations: list[tuple[int, ...]],
    permutation_to_index: dict[tuple[int, ...], int],
) -> list[list[int]]:
    inverses = [
        inverse(permutation)
        for permutation in permutations
    ]

    unseen = set(range(len(permutations)))
    classes = []

    while unseen:
        representative = min(unseen)
        element = permutations[representative]
        conjugates = set()

        for group_element, group_inverse in zip(
            permutations,
            inverses,
        ):
            conjugate = compose(
                compose(
                    group_element,
                    element,
                ),
                group_inverse,
            )

            conjugates.add(
                permutation_to_index[conjugate]
            )

        class_indices = sorted(conjugates)
        classes.append(class_indices)
        unseen -= conjugates

    classes.sort(
        key=lambda row: (
            len(row),
            row[0],
        )
    )

    return classes


def cluster_eigenvalues(
    values: np.ndarray,
    tolerance: float = TOLERANCE,
) -> list[list[int]]:
    clusters: list[list[int]] = []

    for index, value in enumerate(values):
        if not clusters:
            clusters.append([index])
            continue

        previous = values[clusters[-1][-1]]

        scale = max(
            1.0,
            abs(float(previous)),
            abs(float(value)),
        )

        if abs(float(value - previous)) <= tolerance * scale:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    return clusters


def split_by_central_operators(
    operators: list[np.ndarray],
    dimension: int,
) -> list[np.ndarray]:
    sectors = [
        np.eye(
            dimension,
            dtype=np.float64,
        )
    ]

    for operator in operators:
        next_sectors = []

        for basis in sectors:
            restricted = (
                basis.T
                @ operator
                @ basis
            )

            restricted = 0.5 * (
                restricted + restricted.T
            )

            eigenvalues, eigenvectors = np.linalg.eigh(
                restricted
            )

            clusters = cluster_eigenvalues(
                eigenvalues
            )

            if len(clusters) == 1:
                next_sectors.append(basis)
                continue

            for cluster in clusters:
                next_sectors.append(
                    basis
                    @ eigenvectors[:, cluster]
                )

        sectors = next_sectors

    sectors = [
        orthonormal_column_basis(basis)
        for basis in sectors
        if basis.shape[1] > 0
    ]

    sectors.sort(
        key=lambda basis: basis.shape[1]
    )

    return sectors


def character(
    representation: np.ndarray,
) -> np.ndarray:
    return np.trace(
        representation,
        axis1=1,
        axis2=2,
    )


def character_norm(
    values: np.ndarray,
) -> float:
    return float(
        np.dot(values, values)
        / len(values)
    )


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHANNEL_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    TARGET_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    permission = json.loads(
        PERMISSION_PATH.read_text(
            encoding="utf-8"
        )
    )

    harmonic_payload = np.load(
        HARMONIC_REP_PATH
    )

    harmonic_representation = np.array(
        harmonic_payload["rho_harmonic"],
        dtype=np.float64,
    )

    irreducible_payload = np.load(
        IRREDUCIBLE_PATH,
        allow_pickle=True,
    )

    sector_projectors = [
        np.array(value, dtype=np.float64)
        for value in irreducible_payload[
            "sector_projector"
        ]
    ]

    sector_dimensions = [
        int(value)
        for value in irreducible_payload[
            "sector_dimensions"
        ]
    ]

    projector_payload = np.load(
        PROJECTOR_PATH
    )

    p_coexact = np.array(
        projector_payload["P_coexact"],
        dtype=np.float64,
    )

    action_payload = np.load(
        ACTION_PATH
    )

    edge_targets = np.array(
        action_payload["edge_target"],
        dtype=np.int64,
    )

    edge_signs = np.array(
        action_payload["edge_sign"],
        dtype=np.int8,
    )

    isomorphism = json.loads(
        ISO_PATH.read_text(
            encoding="utf-8"
        )
    )

    permission_dimensions = {
        row["coupling_name"]: int(
            row["hom_dimension_integer"]
        )
        for row in permission[
            "coupling_rows"
        ]
    }

    six_indices = [
        index
        for index, dimension in enumerate(
            sector_dimensions
        )
        if dimension == 6
    ]

    if len(six_indices) != 2:
        raise RuntimeError(
            "expected exactly two six-dimensional sectors"
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

    coexact_basis = orthonormal_basis_from_projector(
        p_coexact
    )

    coexact_representation = restricted_edge_representation(
        coexact_basis,
        edge_targets,
        edge_signs,
    )

    coupling_specs = {
        "six_a_tensor_six_a": (
            tensor_representation(
                six_a_representation,
                six_a_representation,
            )
        ),
        "six_a_tensor_six_b": (
            tensor_representation(
                six_a_representation,
                six_b_representation,
            )
        ),
        "six_b_tensor_six_b": (
            tensor_representation(
                six_b_representation,
                six_b_representation,
            )
        ),
    }

    hom_bases: dict[str, list[np.ndarray]] = {}
    hom_audits: dict[str, dict[str, Any]] = {}

    for name, tensor_rep in coupling_specs.items():
        basis_maps, audit = construct_hom_basis(
            coexact_representation,
            tensor_rep,
            permission_dimensions[name],
        )

        hom_bases[name] = basis_maps
        hom_audits[name] = audit

    mapping_rows = isomorphism["mapping_rows"]

    permutations = [
        tuple(
            int(value)
            for value in row["actual_permutation"]
        )
        for row in mapping_rows
    ]

    permutation_to_index = {
        permutation: index
        for index, permutation in enumerate(
            permutations
        )
    }

    classes = conjugacy_classes(
        permutations,
        permutation_to_index,
    )

    class_operators = []

    for class_indices in classes:
        class_sum = np.sum(
            coexact_representation[class_indices],
            axis=0,
        )

        class_operators.append(
            0.5 * (
                class_sum + class_sum.T
            )
        )

    combined_operator = np.zeros(
        (19, 19),
        dtype=np.float64,
    )

    for class_id, operator in enumerate(
        class_operators
    ):
        combined_operator += (
            (class_id + 1)
            * operator
        )

    target_bases = split_by_central_operators(
        [combined_operator] + class_operators,
        19,
    )

    target_projectors = [
        basis @ basis.T
        for basis in target_bases
    ]

    target_rows = []

    for target_id, (
        basis,
        projector,
    ) in enumerate(
        zip(
            target_bases,
            target_projectors,
        )
    ):
        target_character = np.array(
            [
                float(
                    np.trace(
                        projector
                        @ coexact_representation[index]
                    )
                )
                for index in range(480)
            ],
            dtype=np.float64,
        )

        target_rows.append(
            {
                "target_sector_id": target_id,
                "dimension": int(
                    basis.shape[1]
                ),
                "character_norm": (
                    character_norm(
                        target_character
                    )
                ),
                "group_invariance_max_abs": max(
                    max_abs(
                        projector
                        @ rho
                        - rho
                        @ projector
                    )
                    for rho in coexact_representation
                ),
                "projector_idempotence_max_abs": (
                    max_abs(
                        projector @ projector
                        - projector
                    )
                ),
            }
        )

    channel_rows = []
    image_bases: dict[str, np.ndarray] = {}

    for coupling_name, basis_maps in hom_bases.items():
        concatenated = np.concatenate(
            basis_maps,
            axis=1,
        )

        combined_image_basis = orthonormal_column_basis(
            concatenated
        )

        image_bases[coupling_name] = (
            combined_image_basis
        )

        individual_ranks = [
            numerical_rank(matrix)
            for matrix in basis_maps
        ]

        target_reach = []

        for target_id, projector in enumerate(
            target_projectors
        ):
            projected_maps = [
                projector @ matrix
                for matrix in basis_maps
            ]

            projected_combined = np.concatenate(
                projected_maps,
                axis=1,
            )

            target_reach.append(
                {
                    "target_sector_id": target_id,
                    "target_dimension": (
                        target_bases[
                            target_id
                        ].shape[1]
                    ),
                    "reached_dimension": (
                        numerical_rank(
                            projected_combined
                        )
                    ),
                }
            )

        channel_rows.append(
            {
                "coupling_name": coupling_name,
                "hom_dimension_expected": (
                    permission_dimensions[
                        coupling_name
                    ]
                ),
                "hom_dimension_recovered": (
                    hom_audits[
                        coupling_name
                    ]["recovered_dimension"]
                ),
                "individual_map_ranks": (
                    json.dumps(
                        individual_ranks
                    )
                ),
                "combined_image_dimension": (
                    combined_image_basis.shape[1]
                ),
                "maximum_equivariance_residual": (
                    hom_audits[
                        coupling_name
                    ][
                        "maximum_equivariance_residual"
                    ]
                ),
                "hom_basis_gram_max_abs": (
                    hom_audits[
                        coupling_name
                    ][
                        "hom_basis_gram_max_abs"
                    ]
                ),
                "target_reach": json.dumps(
                    target_reach,
                    sort_keys=True,
                ),
            }
        )

    all_map_matrices = [
        matrix
        for basis_maps in hom_bases.values()
        for matrix in basis_maps
    ]

    all_channel_image_basis = (
        orthonormal_column_basis(
            np.concatenate(
                all_map_matrices,
                axis=1,
            )
        )
    )

    cross_image_dimension = (
        image_bases[
            "six_a_tensor_six_b"
        ].shape[1]
    )

    self_image_basis = orthonormal_column_basis(
        np.concatenate(
            hom_bases[
                "six_a_tensor_six_a"
            ]
            + hom_bases[
                "six_b_tensor_six_b"
            ],
            axis=1,
        )
    )

    all_target_completeness = sum(
        target_projectors
    )

    target_pairwise_max_abs = 0.0

    for left in range(len(target_projectors)):
        for right in range(
            left + 1,
            len(target_projectors),
        ):
            target_pairwise_max_abs = max(
                target_pairwise_max_abs,
                max_abs(
                    target_projectors[left]
                    @ target_projectors[right]
                ),
            )

    checks = {
        "input_permission_audit_pass": (
            permission.get("audit_pass") is True
        ),
        "coexact_dimension_is_19": (
            coexact_basis.shape == (120, 19)
        ),
        "two_six_sectors_found": (
            len(six_indices) == 2
        ),
        "all_expected_hom_dimensions_recovered": all(
            hom_audits[name][
                "recovered_dimension"
            ]
            == permission_dimensions[name]
            for name in coupling_specs
        ),
        "all_hom_basis_maps_are_equivariant": all(
            hom_audits[name][
                "maximum_equivariance_residual"
            ]
            < TOLERANCE
            for name in coupling_specs
        ),
        "all_hom_bases_are_orthonormal": all(
            hom_audits[name][
                "hom_basis_gram_max_abs"
            ]
            < TOLERANCE
            for name in coupling_specs
        ),
        "target_sectors_sum_to_19": (
            sum(
                basis.shape[1]
                for basis in target_bases
            )
            == 19
        ),
        "target_projectors_sum_to_identity": (
            max_abs(
                all_target_completeness
                - np.eye(19)
            )
            < TOLERANCE
        ),
        "target_projectors_are_pairwise_orthogonal": (
            target_pairwise_max_abs
            < TOLERANCE
        ),
        "all_target_sectors_are_group_invariant": all(
            row[
                "group_invariance_max_abs"
            ]
            < TOLERANCE
            for row in target_rows
        ),
        "cross_channel_image_is_nonzero": (
            cross_image_dimension > 0
        ),
        "all_channel_image_is_nonzero": (
            all_channel_image_basis.shape[1]
            > 0
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_six_flux_channel_resolution_014"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_six_flux_channel_family_resolved"
            if audit_pass
            else "native_g60_six_flux_channel_resolution_failed"
        ),
        "inputs": {
            "permission_artifact": str(
                PERMISSION_PATH.relative_to(ROOT)
            ),
            "harmonic_representation": str(
                HARMONIC_REP_PATH.relative_to(ROOT)
            ),
            "harmonic_irreducible_projectors": str(
                IRREDUCIBLE_PATH.relative_to(ROOT)
            ),
            "coexact_projector": str(
                PROJECTOR_PATH.relative_to(ROOT)
            ),
            "signed_edge_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
        },
        "checks": checks,
        "hom_space_resolution": {
            name: audit
            for name, audit in hom_audits.items()
        },
        "coexact_target_decomposition": {
            "sector_count": len(target_rows),
            "sector_dimensions": [
                row["dimension"]
                for row in target_rows
            ],
            "dimension_sum": sum(
                row["dimension"]
                for row in target_rows
            ),
            "sector_rows": target_rows,
        },
        "channel_resolution": {
            "channel_rows": channel_rows,
            "cross_channel_combined_image_dimension": (
                cross_image_dimension
            ),
            "self_channel_combined_image_dimension": (
                self_image_basis.shape[1]
            ),
            "all_channels_combined_image_dimension": (
                all_channel_image_basis.shape[1]
            ),
            "coexact_target_dimension": 19,
        },
        "earned_interpretation": {
            "cross_six_permission_dimension": 3,
            "cross_six_channels_constructed": (
                hom_audits[
                    "six_a_tensor_six_b"
                ]["recovered_dimension"]
            ),
            "cross_six_reached_coexact_dimension": (
                cross_image_dimension
            ),
            "all_six_couplings_reached_coexact_dimension": (
                all_channel_image_basis.shape[1]
            ),
            "normalization_selected": False,
            "constitutive_selector_selected": False,
            "physical_flux_claim": False,
        },
        "outputs": {
            "channel_csv": str(
                CHANNEL_CSV_OUT.relative_to(ROOT)
            ),
            "target_csv": str(
                TARGET_CSV_OUT.relative_to(ROOT)
            ),
            "channel_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "complete_tested_hom_bases_constructed": (
                audit_pass
            ),
            "coexact_target_isotypic_sectors_resolved": (
                audit_pass
            ),
            "cross_six_target_reach_measured": (
                audit_pass
            ),
            "canonical_flux_channel_selected": False,
            "bilinear_normalization_selected": False,
            "source_dynamics_selected": False,
            "physical_flux_claim": False,
            "electromagnetism_claim": False,
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

    with CHANNEL_CSV_OUT.open(
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

    with TARGET_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                target_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(target_rows)

    npz_payload: dict[str, np.ndarray] = {
        "six_a_basis_harmonic_coordinates": six_a_basis,
        "six_b_basis_harmonic_coordinates": six_b_basis,
        "coexact_basis_edges": coexact_basis,
        "cross_image_basis": image_bases[
            "six_a_tensor_six_b"
        ],
        "self_image_basis": self_image_basis,
        "all_channel_image_basis": (
            all_channel_image_basis
        ),
        "target_sector_dimensions": np.array(
            [
                basis.shape[1]
                for basis in target_bases
            ],
            dtype=np.int64,
        ),
        "tolerance": np.array(
            [TOLERANCE]
        ),
    }

    for coupling_name, basis_maps in hom_bases.items():
        for map_index, matrix in enumerate(
            basis_maps
        ):
            npz_payload[
                f"{coupling_name}_map_{map_index}"
            ] = matrix

    for target_id, basis in enumerate(
        target_bases
    ):
        npz_payload[
            f"target_sector_{target_id}_basis"
        ] = basis

        npz_payload[
            f"target_sector_{target_id}_projector"
        ] = target_projectors[target_id]

    np.savez_compressed(
        NPZ_OUT,
        **npz_payload,
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "coexact_target_sector_dimensions:",
        [
            row["dimension"]
            for row in target_rows
        ],
    )

    print("\nHom-space recovery:")

    for name in coupling_specs:
        print(
            name,
            "expected/recovered=",
            permission_dimensions[name],
            hom_audits[name][
                "recovered_dimension"
            ],
            "equivariance_max_abs=",
            hom_audits[name][
                "maximum_equivariance_residual"
            ],
        )

    print("\nchannel images:")

    for row in channel_rows:
        print(
            row["coupling_name"],
            "map_ranks=",
            row["individual_map_ranks"],
            "combined_image_dimension=",
            row[
                "combined_image_dimension"
            ],
            "target_reach=",
            row["target_reach"],
        )

    print(
        "\ncross_channel_combined_image_dimension:",
        cross_image_dimension,
    )
    print(
        "self_channel_combined_image_dimension:",
        self_image_basis.shape[1],
    )
    print(
        "all_channels_combined_image_dimension:",
        all_channel_image_basis.shape[1],
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CHANNEL_CSV_OUT)
    print("wrote:", TARGET_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
