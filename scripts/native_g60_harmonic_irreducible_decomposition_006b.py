from __future__ import annotations

import csv
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

PROFILE_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_harmonic_representation_decomposition_006.json"
)

REPRESENTATION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_representation_006.npz"
)

ISO_PATH = (
    SOURCE_ROOT
    / "native_g60_fiber_product_isomorphism_044.json"
)

SURFACE_SYMMETRY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_symmetry_and_kernel_005.json"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_harmonic_irreducible_decomposition_006b.json"
)

SECTOR_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_harmonic_isotypic_sectors_006b.csv"
)

CLASS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_harmonic_conjugacy_classes_006b.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_isotypic_projectors_006b.npz"
)

TOLERANCE = 1e-8


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


def max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0

    return float(np.max(np.abs(matrix)))


def orthonormalize(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((42, 0), dtype=np.float64)

    q, r = np.linalg.qr(matrix)

    diagonal = np.abs(np.diag(r))

    if diagonal.size == 0:
        return np.zeros((matrix.shape[0], 0))

    rank = int(np.count_nonzero(diagonal > tolerance))

    return q[:, :rank]


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

        scale = max(1.0, abs(float(value)), abs(float(previous)))

        if abs(float(value - previous)) <= tolerance * scale:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    return clusters


def conjugacy_classes(
    permutations: list[tuple[int, ...]],
    permutation_to_index: dict[tuple[int, ...], int],
) -> list[list[int]]:
    inverse_rows = [
        inverse(permutation)
        for permutation in permutations
    ]

    unseen = set(range(len(permutations)))
    classes = []

    while unseen:
        representative = min(unseen)
        g = permutations[representative]

        conjugates = set()

        for h, h_inverse in zip(
            permutations,
            inverse_rows,
        ):
            conjugate = compose(
                compose(h, g),
                h_inverse,
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


def split_by_central_operators(
    operators: list[np.ndarray],
) -> list[np.ndarray]:
    sectors = [np.eye(42, dtype=np.float64)]

    for operator in operators:
        next_sectors = []

        for basis in sectors:
            restricted = basis.T @ operator @ basis
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
                sub_basis = (
                    basis
                    @ eigenvectors[:, cluster]
                )

                next_sectors.append(
                    orthonormalize(sub_basis)
                )

        sectors = next_sectors

    sectors = [
        basis
        for basis in sectors
        if basis.shape[1] > 0
    ]

    sectors.sort(
        key=lambda basis: (
            basis.shape[1],
            float(np.trace(
                basis @ basis.T
            )),
        )
    )

    return sectors


def nearest_integer(
    value: float,
    tolerance: float = 1e-7,
) -> int | None:
    candidate = int(round(value))

    if abs(value - candidate) <= tolerance:
        return candidate

    return None


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CLASS_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    profile = json.loads(
        PROFILE_PATH.read_text(encoding="utf-8")
    )

    surface_symmetry = json.loads(
        SURFACE_SYMMETRY_PATH.read_text(
            encoding="utf-8"
        )
    )

    isomorphism = json.loads(
        ISO_PATH.read_text(encoding="utf-8")
    )

    representation_payload = np.load(
        REPRESENTATION_PATH
    )

    representation = np.array(
        representation_payload["rho_harmonic"],
        dtype=np.float64,
    )

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
    class_rows = []

    for class_id, class_indices in enumerate(classes):
        class_sum = np.sum(
            representation[class_indices],
            axis=0,
        )

        # The self-adjoint central part is sufficient for the
        # real isotypic splitting.
        central_operator = 0.5 * (
            class_sum + class_sum.T
        )

        class_operators.append(central_operator)

        representative_index = class_indices[0]
        representative_permutation = permutations[
            representative_index
        ]

        reference = mapping_rows[
            representative_index
        ]["reference_element"]

        class_rows.append(
            {
                "class_id": class_id,
                "class_size": len(class_indices),
                "representative_index": (
                    representative_index
                ),
                "representative_order": (
                    permutation_order(
                        representative_permutation
                    )
                ),
                "representative_cycle_type": (
                    " ".join(
                        str(value)
                        for value in cycle_type(
                            representative_permutation
                        )
                    )
                ),
                "s5_permutation": " ".join(
                    str(value)
                    for value in reference[
                        "s5_permutation"
                    ]
                ),
                "d8_rotation": int(
                    reference["d8_rotation"]
                ),
                "d8_flip": int(
                    reference["d8_flip"]
                ),
                "harmonic_character": float(
                    np.trace(
                        representation[
                            representative_index
                        ]
                    )
                ),
            }
        )

    # Begin with a deterministic weighted combination to split
    # quickly, then refine with every class operator.
    combined = np.zeros((42, 42), dtype=np.float64)

    for class_id, operator in enumerate(
        class_operators
    ):
        combined += (
            (class_id + 1)
            * operator
        )

    operators = [combined] + class_operators

    sectors = split_by_central_operators(
        operators
    )

    projectors = [
        basis @ basis.T
        for basis in sectors
    ]

    identity42 = np.eye(42, dtype=np.float64)

    completeness = (
        sum(projectors)
        if projectors
        else np.zeros((42, 42))
    )

    central_index = int(
        surface_symmetry["face_action_kernel"][
            "nonidentity_index"
        ]
    )

    central_rho = representation[central_index]

    sector_rows = []
    sector_characters = []

    total_character_norm = 0.0

    for sector_id, (basis, projector) in enumerate(
        zip(sectors, projectors)
    ):
        dimension = int(basis.shape[1])

        character = np.array(
            [
                float(
                    np.trace(
                        projector
                        @ representation[index]
                    )
                )
                for index in range(480)
            ],
            dtype=np.float64,
        )

        character_norm = float(
            np.sum(character * character)
            / 480.0
        )

        norm_integer = nearest_integer(
            character_norm
        )

        multiplicity = None
        irreducible_dimension = None

        if norm_integer is not None:
            root = int(round(math.sqrt(norm_integer)))

            if root * root == norm_integer:
                multiplicity = root

                if dimension % multiplicity == 0:
                    irreducible_dimension = (
                        dimension // multiplicity
                    )

        central_trace = float(
            np.trace(
                projector @ central_rho
            )
        )

        central_plus_dimension = nearest_integer(
            (dimension + central_trace) / 2.0
        )

        central_minus_dimension = nearest_integer(
            (dimension - central_trace) / 2.0
        )

        invariance_max_abs = max(
            max_abs(
                representation[index]
                @ projector
                - projector
                @ representation[index]
            )
            for index in range(480)
        )

        idempotence_max_abs = max_abs(
            projector @ projector - projector
        )

        total_character_norm += character_norm
        sector_characters.append(character)

        rounded_profile = Counter(
            nearest_integer(value)
            if nearest_integer(value) is not None
            else round(float(value), 8)
            for value in character
        )

        sector_rows.append(
            {
                "sector_id": sector_id,
                "sector_dimension": dimension,
                "character_norm_raw": character_norm,
                "character_norm_integer": norm_integer,
                "multiplicity_candidate": multiplicity,
                "irreducible_dimension_candidate": (
                    irreducible_dimension
                ),
                "central_trace": central_trace,
                "central_plus_dimension": (
                    central_plus_dimension
                ),
                "central_minus_dimension": (
                    central_minus_dimension
                ),
                "projector_idempotence_max_abs": (
                    idempotence_max_abs
                ),
                "group_invariance_max_abs": (
                    invariance_max_abs
                ),
                "rounded_character_profile": (
                    json.dumps(
                        {
                            str(key): value
                            for key, value in sorted(
                                rounded_profile.items(),
                                key=lambda item: str(
                                    item[0]
                                ),
                            )
                        },
                        sort_keys=True,
                    )
                ),
            }
        )

    pairwise_orthogonality_max_abs = 0.0

    for left in range(len(projectors)):
        for right in range(
            left + 1,
            len(projectors),
        ):
            pairwise_orthogonality_max_abs = max(
                pairwise_orthogonality_max_abs,
                max_abs(
                    projectors[left]
                    @ projectors[right]
                ),
            )

    sector_dimension_sum = sum(
        row["sector_dimension"]
        for row in sector_rows
    )

    sector_character_sum_residual = max_abs(
        np.sum(
            np.array(sector_characters),
            axis=0,
        )
        - np.trace(
            representation,
            axis1=1,
            axis2=2,
        )
    )

    all_norms_integer = all(
        row["character_norm_integer"] is not None
        for row in sector_rows
    )

    all_multiplicities_resolved = all(
        row["multiplicity_candidate"] is not None
        and row[
            "irreducible_dimension_candidate"
        ]
        is not None
        for row in sector_rows
    )

    checks = {
        "input_006_audit_pass": (
            profile.get("audit_pass") is True
        ),
        "representation_shape_is_480_by_42_by_42": (
            representation.shape == (480, 42, 42)
        ),
        "conjugacy_classes_partition_group": (
            sum(len(row) for row in classes) == 480
            and len(
                {
                    index
                    for row in classes
                    for index in row
                }
            )
            == 480
        ),
        "at_least_one_isotypic_sector_found": (
            len(sectors) > 0
        ),
        "sector_dimensions_sum_to_42": (
            sector_dimension_sum == 42
        ),
        "sector_projectors_sum_to_identity": (
            max_abs(
                completeness - identity42
            )
            < TOLERANCE
        ),
        "sector_projectors_pairwise_orthogonal": (
            pairwise_orthogonality_max_abs
            < TOLERANCE
        ),
        "all_sector_projectors_idempotent": all(
            row[
                "projector_idempotence_max_abs"
            ]
            < TOLERANCE
            for row in sector_rows
        ),
        "all_sectors_group_invariant": all(
            row["group_invariance_max_abs"]
            < TOLERANCE
            for row in sector_rows
        ),
        "sector_characters_sum_to_harmonic_character": (
            sector_character_sum_residual
            < TOLERANCE
        ),
        "sector_character_norms_are_integer": (
            all_norms_integer
        ),
        "sector_multiplicity_candidates_resolved": (
            all_multiplicities_resolved
        ),
        "sector_character_norms_sum_to_8": (
            abs(total_character_norm - 8.0)
            < 1e-7
        ),
        "central_sector_dimensions_sum_to_12_30": (
            sum(
                row["central_plus_dimension"]
                for row in sector_rows
                if row[
                    "central_plus_dimension"
                ]
                is not None
            )
            == 12
            and sum(
                row["central_minus_dimension"]
                for row in sector_rows
                if row[
                    "central_minus_dimension"
                ]
                is not None
            )
            == 30
        ),
    }

    audit_pass = all(checks.values())

    decomposition_signature = [
        {
            "sector_dimension": row[
                "sector_dimension"
            ],
            "multiplicity": row[
                "multiplicity_candidate"
            ],
            "irreducible_dimension": row[
                "irreducible_dimension_candidate"
            ],
            "central_plus_dimension": row[
                "central_plus_dimension"
            ],
            "central_minus_dimension": row[
                "central_minus_dimension"
            ],
        }
        for row in sector_rows
    ]

    payload = {
        "artifact_id": (
            "native_g60_harmonic_irreducible_decomposition_006b"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_harmonic_real_isotypic_decomposition_identified"
            if audit_pass
            else "native_g60_harmonic_isotypic_decomposition_incomplete"
        ),
        "inputs": {
            "harmonic_representation_profile": str(
                PROFILE_PATH.relative_to(ROOT)
            ),
            "harmonic_representation": str(
                REPRESENTATION_PATH.relative_to(ROOT)
            ),
            "native_group_coordinates": str(
                ISO_PATH.relative_to(ROOT)
            ),
            "surface_symmetry": str(
                SURFACE_SYMMETRY_PATH.relative_to(
                    ROOT
                )
            ),
        },
        "construction": {
            "group_order": 480,
            "harmonic_dimension": 42,
            "conjugacy_class_count": len(classes),
            "method": (
                "joint spectral splitting by self-adjoint "
                "central conjugacy-class sum operators"
            ),
            "claim_type": (
                "real isotypic decomposition with character-norm "
                "multiplicity candidates"
            ),
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "summary": {
            "sector_count": len(sector_rows),
            "sector_dimension_sum": (
                sector_dimension_sum
            ),
            "total_character_norm": (
                total_character_norm
            ),
            "pairwise_orthogonality_max_abs": (
                pairwise_orthogonality_max_abs
            ),
            "completeness_max_abs": max_abs(
                completeness - identity42
            ),
            "character_sum_max_abs": (
                sector_character_sum_residual
            ),
            "decomposition_signature": (
                decomposition_signature
            ),
        },
        "sector_rows": sector_rows,
        "conjugacy_classes": class_rows,
        "boundary": {
            "real_isotypic_decomposition_computed": (
                audit_pass
            ),
            "abstract_irreducible_names_assigned": False,
            "full_character_table_imported": False,
            "complex_type_vs_real_type_fully_classified": False,
            "symmetry_adapted_basis_exported": False,
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
            fieldnames=list(sector_rows[0]),
        )

        writer.writeheader()
        writer.writerows(sector_rows)

    with CLASS_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(class_rows[0]),
        )

        writer.writeheader()
        writer.writerows(class_rows)

    sector_basis_objects = np.empty(
        len(sectors),
        dtype=object,
    )
    sector_projector_objects = np.empty(
        len(projectors),
        dtype=object,
    )
    sector_character_objects = np.empty(
        len(sector_characters),
        dtype=object,
    )

    for index, value in enumerate(sectors):
        sector_basis_objects[index] = value

    for index, value in enumerate(projectors):
        sector_projector_objects[index] = value

    for index, value in enumerate(sector_characters):
        sector_character_objects[index] = value

    np.savez_compressed(
        NPZ_OUT,
        sector_basis=sector_basis_objects,
        sector_projector=sector_projector_objects,
        sector_character=sector_character_objects,
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print("conjugacy_class_count:", len(classes))
    print("sector_count:", len(sector_rows))
    print(
        "sector_dimension_sum:",
        sector_dimension_sum,
    )
    print(
        "total_character_norm:",
        total_character_norm,
    )
    print(
        "completeness_max_abs:",
        payload["summary"][
            "completeness_max_abs"
        ],
    )
    print(
        "pairwise_orthogonality_max_abs:",
        pairwise_orthogonality_max_abs,
    )
    print(
        "character_sum_max_abs:",
        sector_character_sum_residual,
    )

    print("\nsector decomposition:")

    for row in sector_rows:
        print(
            "sector",
            row["sector_id"],
            "dimension=",
            row["sector_dimension"],
            "norm=",
            row["character_norm_integer"],
            "multiplicity=",
            row["multiplicity_candidate"],
            "irrep_dimension=",
            row[
                "irreducible_dimension_candidate"
            ],
            "central_plus/minus=",
            row["central_plus_dimension"],
            row["central_minus_dimension"],
        )

    print("\nwrote:", JSON_OUT)
    print("wrote:", SECTOR_CSV_OUT)
    print("wrote:", CLASS_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
