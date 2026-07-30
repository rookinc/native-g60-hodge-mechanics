from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DECOMPOSITION_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_harmonic_irreducible_decomposition_006b.json"
)

REPRESENTATION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_representation_006.npz"
)

ISOTYPIC_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_isotypic_projectors_006b.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_harmonic_multiplicity_split_006c.json"
)

SECTOR_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_harmonic_multiplicity_split_006c.csv"
)

BASIS_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_symmetry_adapted_basis_006c.npz"
)

PROJECTOR_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_irreducible_projectors_006c.npz"
)

TOLERANCE = 1e-8


def max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0

    return float(np.max(np.abs(matrix)))


def orthonormalize(
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

    rank = int(
        np.count_nonzero(
            singular_values > tolerance
        )
    )

    return u[:, :rank]


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


def reynolds_average(
    seed: np.ndarray,
    representation: np.ndarray,
) -> np.ndarray:
    averaged = np.zeros_like(seed)

    for rho in representation:
        averaged += (
            rho
            @ seed
            @ rho.T
        )

    averaged /= representation.shape[0]

    return 0.5 * (
        averaged + averaged.T
    )


def deterministic_seed(
    seed_id: int,
    dimension: int,
) -> np.ndarray:
    indices = np.arange(
        1,
        dimension + 1,
        dtype=np.float64,
    )

    if seed_id == 0:
        return np.diag(indices)

    if seed_id == 1:
        return np.diag(indices * indices)

    if seed_id == 2:
        vector = indices / np.linalg.norm(indices)
        return np.outer(vector, vector)

    if seed_id == 3:
        vector = np.sin(indices)
        return np.outer(vector, vector)

    if seed_id == 4:
        vector_a = np.cos(indices)
        vector_b = np.sin(2.0 * indices)

        return 0.5 * (
            np.outer(vector_a, vector_b)
            + np.outer(vector_b, vector_a)
        )

    if seed_id == 5:
        seed = np.zeros(
            (dimension, dimension),
            dtype=np.float64,
        )

        for index in range(dimension - 1):
            seed[index, index + 1] = 1.0
            seed[index + 1, index] = 1.0

        return seed

    raise ValueError(f"unknown seed_id: {seed_id}")


def projector_rank(
    projector: np.ndarray,
) -> int:
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (
            projector + projector.T
        )
    )

    return int(
        np.count_nonzero(
            eigenvalues > 0.5
        )
    )


def representation_character(
    projector: np.ndarray,
    representation: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            float(
                np.trace(
                    projector @ rho
                )
            )
            for rho in representation
        ],
        dtype=np.float64,
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    SECTOR_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    BASIS_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROJECTOR_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decomposition = json.loads(
        DECOMPOSITION_PATH.read_text(
            encoding="utf-8"
        )
    )

    representation_payload = np.load(
        REPRESENTATION_PATH
    )

    isotypic_payload = np.load(
        ISOTYPIC_PATH,
        allow_pickle=True,
    )

    representation = np.array(
        representation_payload["rho_harmonic"],
        dtype=np.float64,
    )

    sector_bases = [
        np.array(value, dtype=np.float64)
        for value in isotypic_payload["sector_basis"]
    ]

    sector_projectors = [
        np.array(value, dtype=np.float64)
        for value in isotypic_payload[
            "sector_projector"
        ]
    ]

    dimensions = [
        basis.shape[1]
        for basis in sector_bases
    ]

    twenty_indices = [
        index
        for index, dimension in enumerate(dimensions)
        if dimension == 20
    ]

    if len(twenty_indices) != 1:
        raise RuntimeError(
            "expected exactly one 20-dimensional "
            f"isotypic sector, found {twenty_indices}"
        )

    doubled_sector_index = twenty_indices[0]
    doubled_basis = sector_bases[
        doubled_sector_index
    ]
    doubled_projector = sector_projectors[
        doubled_sector_index
    ]

    selected_seed_id = None
    selected_operator = None
    selected_eigenvalues = None
    selected_clusters = None
    seed_audit_rows = []

    for seed_id in range(6):
        seed = deterministic_seed(
            seed_id,
            42,
        )

        commutant_operator = reynolds_average(
            seed,
            representation,
        )

        restricted = (
            doubled_basis.T
            @ commutant_operator
            @ doubled_basis
        )

        restricted = 0.5 * (
            restricted + restricted.T
        )

        eigenvalues = np.linalg.eigvalsh(
            restricted
        )

        clusters = cluster_eigenvalues(
            eigenvalues
        )

        cluster_sizes = [
            len(cluster)
            for cluster in clusters
        ]

        seed_audit_rows.append(
            {
                "seed_id": seed_id,
                "cluster_count": len(clusters),
                "cluster_sizes": cluster_sizes,
                "minimum_eigenvalue": float(
                    eigenvalues[0]
                ),
                "maximum_eigenvalue": float(
                    eigenvalues[-1]
                ),
                "accepted_10_plus_10": (
                    cluster_sizes == [10, 10]
                ),
            }
        )

        if cluster_sizes == [10, 10]:
            selected_seed_id = seed_id
            selected_operator = commutant_operator
            selected_eigenvalues = eigenvalues
            selected_clusters = clusters
            break

    if selected_seed_id is None:
        raise RuntimeError(
            "no deterministic commutant seed produced "
            "a 10+10 split"
        )

    assert selected_operator is not None
    assert selected_eigenvalues is not None
    assert selected_clusters is not None

    restricted = (
        doubled_basis.T
        @ selected_operator
        @ doubled_basis
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

    copy_bases = []

    for cluster in clusters:
        basis = (
            doubled_basis
            @ eigenvectors[:, cluster]
        )

        copy_bases.append(
            orthonormalize(basis)
        )

    if [
        basis.shape[1]
        for basis in copy_bases
    ] != [10, 10]:
        raise RuntimeError(
            "selected split did not reconstruct 10+10"
        )

    copy_projectors = [
        basis @ basis.T
        for basis in copy_bases
    ]

    copy_characters = [
        representation_character(
            projector,
            representation,
        )
        for projector in copy_projectors
    ]

    character_difference_max_abs = max_abs(
        copy_characters[0]
        - copy_characters[1]
    )

    character_norms = [
        float(
            np.sum(character * character)
            / representation.shape[0]
        )
        for character in copy_characters
    ]

    commutant_residual = max(
        max_abs(
            selected_operator @ rho
            - rho @ selected_operator
        )
        for rho in representation
    )

    copy_invariance_residuals = [
        max(
            max_abs(
                projector @ rho
                - rho @ projector
            )
            for rho in representation
        )
        for projector in copy_projectors
    ]

    copy_idempotence_residuals = [
        max_abs(
            projector @ projector
            - projector
        )
        for projector in copy_projectors
    ]

    copy_orthogonality_residual = max_abs(
        copy_projectors[0]
        @ copy_projectors[1]
    )

    copy_completeness_residual = max_abs(
        copy_projectors[0]
        + copy_projectors[1]
        - doubled_projector
    )

    copy_gram_residuals = [
        max_abs(
            basis.T @ basis
            - np.eye(10)
        )
        for basis in copy_bases
    ]

    # Construct a numerical intertwiner by averaging an arbitrary
    # rank-one map from copy 0 to copy 1 over the full group.
    source_vector = copy_bases[0][:, 0]
    target_vector = copy_bases[1][:, 0]

    seed_intertwiner = np.outer(
        target_vector,
        source_vector,
    )

    intertwiner = np.zeros(
        (42, 42),
        dtype=np.float64,
    )

    for rho in representation:
        intertwiner += (
            rho
            @ seed_intertwiner
            @ rho.T
        )

    intertwiner /= representation.shape[0]

    # Restrict and normalize the intertwiner between the two copies.
    restricted_intertwiner = (
        copy_bases[1].T
        @ intertwiner
        @ copy_bases[0]
    )

    intertwiner_singular_values = np.linalg.svd(
        restricted_intertwiner,
        compute_uv=False,
    )

    intertwiner_rank = int(
        np.count_nonzero(
            intertwiner_singular_values > TOLERANCE
        )
    )

    intertwiner_commutation_residual = max(
        max_abs(
            intertwiner @ rho
            - rho @ intertwiner
        )
        for rho in representation
    )

    all_projectors = []

    for index, projector in enumerate(
        sector_projectors
    ):
        if index == doubled_sector_index:
            all_projectors.extend(
                copy_projectors
            )
        else:
            all_projectors.append(
                projector
            )

    all_bases = []

    for index, basis in enumerate(
        sector_bases
    ):
        if index == doubled_sector_index:
            all_bases.extend(copy_bases)
        else:
            all_bases.append(basis)

    final_dimensions = [
        basis.shape[1]
        for basis in all_bases
    ]

    final_completeness = sum(
        all_projectors
    )

    final_pairwise_max_abs = 0.0

    for left in range(len(all_projectors)):
        for right in range(
            left + 1,
            len(all_projectors),
        ):
            final_pairwise_max_abs = max(
                final_pairwise_max_abs,
                max_abs(
                    all_projectors[left]
                    @ all_projectors[right]
                ),
            )

    symmetry_adapted_basis = np.concatenate(
        all_bases,
        axis=1,
    )

    symmetry_adapted_gram_residual = max_abs(
        symmetry_adapted_basis.T
        @ symmetry_adapted_basis
        - np.eye(42)
    )

    checks = {
        "input_006b_audit_pass": (
            decomposition.get("audit_pass") is True
        ),
        "representation_shape_is_480_by_42_by_42": (
            representation.shape == (480, 42, 42)
        ),
        "unique_20_dimensional_isotypic_sector": (
            len(twenty_indices) == 1
        ),
        "deterministic_commutant_seed_found": (
            selected_seed_id is not None
        ),
        "split_dimensions_are_10_and_10": (
            [
                basis.shape[1]
                for basis in copy_bases
            ]
            == [10, 10]
        ),
        "commutant_operator_commutes_with_group": (
            commutant_residual < TOLERANCE
        ),
        "both_copy_projectors_are_invariant": all(
            residual < TOLERANCE
            for residual in copy_invariance_residuals
        ),
        "both_copy_projectors_are_idempotent": all(
            residual < TOLERANCE
            for residual in copy_idempotence_residuals
        ),
        "copy_projectors_are_orthogonal": (
            copy_orthogonality_residual
            < TOLERANCE
        ),
        "copy_projectors_reconstruct_20_sector": (
            copy_completeness_residual
            < TOLERANCE
        ),
        "copy_bases_are_orthonormal": all(
            residual < TOLERANCE
            for residual in copy_gram_residuals
        ),
        "copy_characters_are_equal": (
            character_difference_max_abs
            < TOLERANCE
        ),
        "each_copy_character_norm_is_one": all(
            abs(value - 1.0) < TOLERANCE
            for value in character_norms
        ),
        "nonzero_full_rank_intertwiner_found": (
            intertwiner_rank == 10
        ),
        "intertwiner_commutes_with_group": (
            intertwiner_commutation_residual
            < TOLERANCE
        ),
        "final_irreducible_dimensions_are_2_6_6_8_10_10": (
            sorted(final_dimensions)
            == [2, 6, 6, 8, 10, 10]
        ),
        "final_projectors_sum_to_identity": (
            max_abs(
                final_completeness
                - np.eye(42)
            )
            < TOLERANCE
        ),
        "final_projectors_are_pairwise_orthogonal": (
            final_pairwise_max_abs
            < TOLERANCE
        ),
        "symmetry_adapted_basis_is_orthonormal": (
            symmetry_adapted_gram_residual
            < TOLERANCE
        ),
    }

    audit_pass = all(checks.values())

    sector_rows = []

    for sector_id, (
        basis,
        projector,
    ) in enumerate(
        zip(
            all_bases,
            all_projectors,
        )
    ):
        character = representation_character(
            projector,
            representation,
        )

        sector_rows.append(
            {
                "sector_id": sector_id,
                "dimension": basis.shape[1],
                "character_norm": float(
                    np.sum(character * character)
                    / representation.shape[0]
                ),
                "projector_rank": projector_rank(
                    projector
                ),
                "group_invariance_max_abs": max(
                    max_abs(
                        projector @ rho
                        - rho @ projector
                    )
                    for rho in representation
                ),
                "projector_idempotence_max_abs": (
                    max_abs(
                        projector @ projector
                        - projector
                    )
                ),
            }
        )

    payload = {
        "artifact_id": (
            "native_g60_harmonic_multiplicity_split_006c"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_doubled_10_dimensional_harmonic_sector_split"
            if audit_pass
            else "native_g60_harmonic_multiplicity_split_failed"
        ),
        "inputs": {
            "isotypic_decomposition": str(
                DECOMPOSITION_PATH.relative_to(
                    ROOT
                )
            ),
            "harmonic_representation": str(
                REPRESENTATION_PATH.relative_to(
                    ROOT
                )
            ),
            "isotypic_projectors": str(
                ISOTYPIC_PATH.relative_to(ROOT)
            ),
        },
        "construction": {
            "method": (
                "Reynolds-average deterministic self-adjoint "
                "seed operators into the representation commutant, "
                "then spectrally split the 20-dimensional "
                "multiplicity-two isotypic sector."
            ),
            "selected_seed_id": selected_seed_id,
            "selected_eigenvalue_clusters": [
                {
                    "dimension": len(cluster),
                    "minimum": float(
                        eigenvalues[cluster[0]]
                    ),
                    "maximum": float(
                        eigenvalues[cluster[-1]]
                    ),
                }
                for cluster in clusters
            ],
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "seed_audit_rows": seed_audit_rows,
        "multiplicity_split": {
            "original_sector_dimension": 20,
            "copy_dimensions": [10, 10],
            "copy_character_norms": (
                character_norms
            ),
            "copy_character_difference_max_abs": (
                character_difference_max_abs
            ),
            "copy_invariance_max_abs": (
                copy_invariance_residuals
            ),
            "copy_idempotence_max_abs": (
                copy_idempotence_residuals
            ),
            "copy_orthogonality_max_abs": (
                copy_orthogonality_residual
            ),
            "copy_completeness_max_abs": (
                copy_completeness_residual
            ),
        },
        "intertwiner": {
            "restricted_shape": [10, 10],
            "rank": intertwiner_rank,
            "singular_values": [
                float(value)
                for value in (
                    intertwiner_singular_values
                )
            ],
            "commutation_max_abs": (
                intertwiner_commutation_residual
            ),
        },
        "final_decomposition": {
            "sector_count": len(all_bases),
            "dimensions": final_dimensions,
            "sorted_dimensions": sorted(
                final_dimensions
            ),
            "dimension_sum": sum(
                final_dimensions
            ),
            "projector_completeness_max_abs": (
                max_abs(
                    final_completeness
                    - np.eye(42)
                )
            ),
            "projector_pairwise_max_abs": (
                final_pairwise_max_abs
            ),
            "symmetry_adapted_basis_shape": list(
                symmetry_adapted_basis.shape
            ),
            "symmetry_adapted_basis_gram_max_abs": (
                symmetry_adapted_gram_residual
            ),
        },
        "sector_rows": sector_rows,
        "outputs": {
            "sector_csv": str(
                SECTOR_CSV_OUT.relative_to(ROOT)
            ),
            "symmetry_adapted_basis_npz": str(
                BASIS_NPZ_OUT.relative_to(ROOT)
            ),
            "irreducible_projectors_npz": str(
                PROJECTOR_NPZ_OUT.relative_to(
                    ROOT
                )
            ),
        },
        "boundary": {
            "doubled_isotypic_sector_split": (
                audit_pass
            ),
            "concrete_equivalent_10_dimensional_copies_exported": (
                audit_pass
            ),
            "symmetry_adapted_harmonic_basis_exported": (
                audit_pass
            ),
            "abstract_irreducible_names_assigned": False,
            "physical_mode_interpretation": False,
            "particle_claim": False,
            "frequency_claim": False,
            "energy_claim": False,
            "maxwell_claim": False,
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

    with SECTOR_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                sector_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(sector_rows)

    basis_objects = np.empty(
        len(all_bases),
        dtype=object,
    )
    projector_objects = np.empty(
        len(all_projectors),
        dtype=object,
    )

    for index, basis in enumerate(all_bases):
        basis_objects[index] = basis

    for index, projector in enumerate(
        all_projectors
    ):
        projector_objects[index] = projector

    np.savez_compressed(
        BASIS_NPZ_OUT,
        symmetry_adapted_basis=(
            symmetry_adapted_basis
        ),
        sector_basis=basis_objects,
        sector_dimensions=np.array(
            final_dimensions,
            dtype=np.int64,
        ),
        copy_0_basis=copy_bases[0],
        copy_1_basis=copy_bases[1],
        intertwiner=intertwiner,
        tolerance=np.array([TOLERANCE]),
    )

    np.savez_compressed(
        PROJECTOR_NPZ_OUT,
        sector_projector=projector_objects,
        sector_dimensions=np.array(
            final_dimensions,
            dtype=np.int64,
        ),
        copy_0_projector=copy_projectors[0],
        copy_1_projector=copy_projectors[1],
        doubled_projector=(
            doubled_projector
        ),
        commutant_operator=(
            selected_operator
        ),
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "selected_seed_id:",
        selected_seed_id,
    )
    print(
        "selected_cluster_sizes:",
        [
            len(cluster)
            for cluster in clusters
        ],
    )
    print(
        "commutant_residual:",
        commutant_residual,
    )
    print(
        "copy_dimensions:",
        [
            basis.shape[1]
            for basis in copy_bases
        ],
    )
    print(
        "copy_character_norms:",
        character_norms,
    )
    print(
        "copy_character_difference_max_abs:",
        character_difference_max_abs,
    )
    print(
        "copy_invariance_max_abs:",
        copy_invariance_residuals,
    )
    print(
        "copy_orthogonality_max_abs:",
        copy_orthogonality_residual,
    )
    print(
        "copy_completeness_max_abs:",
        copy_completeness_residual,
    )
    print(
        "intertwiner_rank:",
        intertwiner_rank,
    )
    print(
        "intertwiner_commutation_max_abs:",
        intertwiner_commutation_residual,
    )
    print(
        "final_dimensions:",
        final_dimensions,
    )
    print(
        "final_completeness_max_abs:",
        payload["final_decomposition"][
            "projector_completeness_max_abs"
        ],
    )
    print(
        "symmetry_adapted_basis_gram_max_abs:",
        symmetry_adapted_gram_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SECTOR_CSV_OUT)
    print("wrote:", BASIS_NPZ_OUT)
    print("wrote:", PROJECTOR_NPZ_OUT)


if __name__ == "__main__":
    main()
