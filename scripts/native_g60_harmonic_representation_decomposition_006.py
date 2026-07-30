from __future__ import annotations

import csv
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

HARMONIC_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_1_form_basis_004.npz"
)

ACTION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

SYMMETRY_AUDIT_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_hodge_symmetry_commutation_005.json"
)

SURFACE_SYMMETRY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_symmetry_and_kernel_005.json"
)

ISO_PATH = (
    SOURCE_ROOT
    / "native_g60_fiber_product_isomorphism_044.json"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_harmonic_representation_decomposition_006.json"
)

CHARACTER_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_harmonic_character_006.csv"
)

REPRESENTATION_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_representation_006.npz"
)

SECTOR_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_central_sectors_006.npz"
)

TOLERANCE = 1e-9


def compose(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Return left after right:

        (left o right)(x) = left[right[x]]
    """
    return tuple(
        left[right[index]]
        for index in range(len(right))
    )


def permutation_identity(
    size: int,
) -> tuple[int, ...]:
    return tuple(range(size))


def permutation_order(
    permutation: tuple[int, ...],
) -> int:
    identity = permutation_identity(len(permutation))
    power = identity

    for order in range(1, 10000):
        power = compose(permutation, power)

        if power == identity:
            return order

    raise RuntimeError("permutation order search exceeded bound")


def permutation_cycle_type(
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


def extract_reference_coordinates(
    row: dict[str, Any],
) -> dict[str, Any]:
    reference = row.get("reference_element", {})

    return {
        "s5_permutation": tuple(
            int(value)
            for value in reference["s5_permutation"]
        ),
        "d8_rotation": int(reference["d8_rotation"]),
        "d8_flip": int(reference["d8_flip"]),
    }


def apply_signed_edge_action(
    matrix: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    """
    Apply the signed edge action to every column of matrix.

    Edge basis source maps to:

        sign[source] * edge_basis[target[source]].
    """
    transformed = np.empty_like(matrix)

    transformed[target, :] = (
        sign[:, None] * matrix
    )

    return transformed


def max_abs(
    matrix: np.ndarray,
) -> float:
    if matrix.size == 0:
        return 0.0

    return float(np.max(np.abs(matrix)))


def numerical_rank(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> int:
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    return int(
        np.count_nonzero(
            singular_values > tolerance
        )
    )


def spectral_projector_rank(
    projector: np.ndarray,
    tolerance: float = TOLERANCE,
) -> int:
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (projector + projector.T)
    )

    return int(
        np.count_nonzero(eigenvalues > 0.5)
    )


def closure_generated_by(
    generator_indices: list[int],
    permutations: list[tuple[int, ...]],
    permutation_to_index: dict[tuple[int, ...], int],
    identity_index: int,
) -> set[int]:
    generators = list(generator_indices)

    inverse_indices = []

    for generator_index in generators:
        generator = permutations[generator_index]

        inverse = [0] * len(generator)

        for source, target in enumerate(generator):
            inverse[target] = source

        inverse_indices.append(
            permutation_to_index[tuple(inverse)]
        )

    steps = sorted(set(generators + inverse_indices))

    reached = {identity_index}
    queue = deque([identity_index])

    while queue:
        current_index = queue.popleft()
        current = permutations[current_index]

        for step_index in steps:
            product = compose(
                permutations[step_index],
                current,
            )

            product_index = permutation_to_index[product]

            if product_index in reached:
                continue

            reached.add(product_index)
            queue.append(product_index)

    return reached


def greedy_generator_indices(
    permutations: list[tuple[int, ...]],
    permutation_to_index: dict[tuple[int, ...], int],
    identity_index: int,
) -> list[int]:
    selected: list[int] = []
    generated = {identity_index}

    while len(generated) < len(permutations):
        best_index = None
        best_closure = generated

        for candidate_index in range(len(permutations)):
            if candidate_index in generated:
                continue

            candidate_closure = closure_generated_by(
                selected + [candidate_index],
                permutations,
                permutation_to_index,
                identity_index,
            )

            if len(candidate_closure) > len(best_closure):
                best_index = candidate_index
                best_closure = candidate_closure

        if best_index is None:
            raise RuntimeError(
                "greedy generator search made no progress"
            )

        selected.append(best_index)
        generated = best_closure

    return selected


def rounded_integer(
    value: float,
    tolerance: float = 1e-8,
) -> int | None:
    nearest = int(round(value))

    if abs(value - nearest) <= tolerance:
        return nearest

    return None


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHARACTER_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPRESENTATION_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    SECTOR_NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    symmetry_audit = json.loads(
        SYMMETRY_AUDIT_PATH.read_text(encoding="utf-8")
    )

    surface_symmetry = json.loads(
        SURFACE_SYMMETRY_PATH.read_text(encoding="utf-8")
    )

    isomorphism = json.loads(
        ISO_PATH.read_text(encoding="utf-8")
    )

    harmonic_payload = np.load(HARMONIC_PATH)
    action_payload = np.load(ACTION_PATH)

    harmonic_basis = np.array(
        harmonic_payload["H"],
        dtype=np.float64,
    )

    edge_targets = np.array(
        action_payload["edge_target"],
        dtype=np.int64,
    )

    edge_signs = np.array(
        action_payload["edge_sign"],
        dtype=np.int8,
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
        for index, permutation in enumerate(permutations)
    }

    identity60 = permutation_identity(60)
    identity_index = permutation_to_index[identity60]

    coordinates = [
        extract_reference_coordinates(row)
        for row in mapping_rows
    ]

    representation = np.empty(
        (480, 42, 42),
        dtype=np.float64,
    )

    leakage_maxima = []
    orthogonality_maxima = []
    character_values = []

    harmonic_projector = (
        harmonic_basis @ harmonic_basis.T
    )

    identity42 = np.eye(42, dtype=np.float64)

    for actual_index in range(480):
        transformed_basis = apply_signed_edge_action(
            harmonic_basis,
            edge_targets[actual_index],
            edge_signs[actual_index],
        )

        rho = harmonic_basis.T @ transformed_basis

        representation[actual_index] = rho

        reconstructed = harmonic_basis @ rho

        leakage_maxima.append(
            max_abs(
                transformed_basis - reconstructed
            )
        )

        orthogonality_maxima.append(
            max_abs(
                rho.T @ rho - identity42
            )
        )

        character_values.append(
            float(np.trace(rho))
        )

    character_array = np.array(
        character_values,
        dtype=np.float64,
    )

    representation_kernel_indices = [
        index
        for index in range(480)
        if max_abs(
            representation[index] - identity42
        )
        < TOLERANCE
    ]

    generator_indices = greedy_generator_indices(
        permutations,
        permutation_to_index,
        identity_index,
    )

    representation_law_max_abs = 0.0
    representation_law_test_count = 0

    for generator_index in generator_indices:
        generator = permutations[generator_index]

        for right_index, right in enumerate(permutations):
            product = compose(generator, right)
            product_index = permutation_to_index[product]

            residual = (
                representation[generator_index]
                @ representation[right_index]
                - representation[product_index]
            )

            representation_law_max_abs = max(
                representation_law_max_abs,
                max_abs(residual),
            )

            representation_law_test_count += 1

    character_norm_raw = float(
        np.sum(character_array * character_array)
        / 480.0
    )

    character_norm_integer = rounded_integer(
        character_norm_raw
    )

    group_average = np.mean(
        representation,
        axis=0,
    )

    # Build signs directly by index because preserving_indices is a list
    # of indices rather than one row per group element.
    orientation_sign_by_index = np.full(
        480,
        -1.0,
        dtype=np.float64,
    )

    for index in surface_symmetry["orientation_character"][
        "preserving_indices"
    ]:
        orientation_sign_by_index[int(index)] = 1.0

    orientation_average = np.mean(
        orientation_sign_by_index[:, None, None]
        * representation,
        axis=0,
    )

    trivial_projector_rank = spectral_projector_rank(
        group_average
    )

    orientation_character_projector_rank = (
        spectral_projector_rank(
            orientation_average
        )
    )

    central_index = int(
        surface_symmetry["face_action_kernel"][
            "nonidentity_index"
        ]
    )

    central_rho = representation[central_index]

    central_eigenvalues, central_eigenvectors = np.linalg.eigh(
        0.5 * (central_rho + central_rho.T)
    )

    central_plus_mask = central_eigenvalues > 0.0
    central_minus_mask = central_eigenvalues < 0.0

    central_plus_basis = central_eigenvectors[
        :,
        central_plus_mask,
    ]

    central_minus_basis = central_eigenvectors[
        :,
        central_minus_mask,
    ]

    central_plus_projector = (
        central_plus_basis @ central_plus_basis.T
    )

    central_minus_projector = (
        central_minus_basis @ central_minus_basis.T
    )

    central_plus_dimension = int(
        central_plus_basis.shape[1]
    )

    central_minus_dimension = int(
        central_minus_basis.shape[1]
    )

    central_involution_residual = max_abs(
        central_rho @ central_rho - identity42
    )

    central_commutation_max_abs = max(
        max_abs(
            central_rho @ representation[index]
            - representation[index] @ central_rho
        )
        for index in range(480)
    )

    character_rows = []

    rounded_character_profile = Counter()
    order_character_profile = Counter()
    orientation_character_profile = Counter()

    for actual_index in range(480):
        coordinate = coordinates[actual_index]
        character = character_values[actual_index]
        rounded_character = rounded_integer(character)

        rounded_key = (
            str(rounded_character)
            if rounded_character is not None
            else f"{character:.10f}"
        )

        rounded_character_profile[rounded_key] += 1

        actual_order = permutation_order(
            permutations[actual_index]
        )

        orientation_type = (
            "preserving"
            if orientation_sign_by_index[actual_index] == 1
            else "reversing"
        )

        order_character_profile[
            (actual_order, rounded_key)
        ] += 1

        orientation_character_profile[
            (orientation_type, rounded_key)
        ] += 1

        character_rows.append(
            {
                "actual_index": actual_index,
                "actual_order": actual_order,
                "actual_cycle_type": " ".join(
                    str(value)
                    for value in permutation_cycle_type(
                        permutations[actual_index]
                    )
                ),
                "s5_permutation": " ".join(
                    str(value)
                    for value in coordinate["s5_permutation"]
                ),
                "d8_rotation": coordinate["d8_rotation"],
                "d8_flip": coordinate["d8_flip"],
                "orientation_type": orientation_type,
                "character": character,
                "rounded_character": rounded_character,
                "representation_is_identity": (
                    actual_index
                    in representation_kernel_indices
                ),
            }
        )

    checks = {
        "input_symmetry_commutation_pass": (
            symmetry_audit.get("audit_pass") is True
        ),
        "harmonic_basis_shape_is_120_by_42": (
            harmonic_basis.shape == (120, 42)
        ),
        "edge_action_shape_is_480_by_120": (
            edge_targets.shape == (480, 120)
            and edge_signs.shape == (480, 120)
        ),
        "representation_shape_is_480_by_42_by_42": (
            representation.shape == (480, 42, 42)
        ),
        "identity_representation_is_identity": (
            max_abs(
                representation[identity_index] - identity42
            )
            < TOLERANCE
        ),
        "harmonic_subspace_invariant_for_all_480": (
            max(leakage_maxima) < TOLERANCE
        ),
        "all_representation_matrices_are_orthogonal": (
            max(orthogonality_maxima) < TOLERANCE
        ),
        "generator_representation_law_passes": (
            representation_law_max_abs < TOLERANCE
        ),
        "generator_set_generates_all_480": (
            len(
                closure_generated_by(
                    generator_indices,
                    permutations,
                    permutation_to_index,
                    identity_index,
                )
            )
            == 480
        ),
        "character_identity_value_is_42": (
            abs(character_values[identity_index] - 42.0)
            < TOLERANCE
        ),
        "character_norm_is_integer": (
            character_norm_integer is not None
        ),
        "central_element_is_involution_on_harmonic_space": (
            central_involution_residual < TOLERANCE
        ),
        "central_element_commutes_on_harmonic_space": (
            central_commutation_max_abs < TOLERANCE
        ),
        "central_sector_dimensions_sum_to_42": (
            central_plus_dimension
            + central_minus_dimension
            == 42
        ),
        "central_sector_projectors_sum_to_identity": (
            max_abs(
                central_plus_projector
                + central_minus_projector
                - identity42
            )
            < TOLERANCE
        ),
        "group_average_is_projector": (
            max_abs(
                group_average @ group_average
                - group_average
            )
            < TOLERANCE
        ),
        "orientation_average_is_projector": (
            max_abs(
                orientation_average @ orientation_average
                - orientation_average
            )
            < TOLERANCE
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_harmonic_representation_decomposition_006"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_harmonic_representation_restricted_and_profiled"
            if audit_pass
            else "native_g60_harmonic_representation_audit_failed"
        ),
        "inputs": {
            "harmonic_basis": str(
                HARMONIC_PATH.relative_to(ROOT)
            ),
            "signed_cochain_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
            "hodge_symmetry_commutation": str(
                SYMMETRY_AUDIT_PATH.relative_to(ROOT)
            ),
            "surface_symmetry": str(
                SURFACE_SYMMETRY_PATH.relative_to(ROOT)
            ),
            "native_group_coordinates": str(
                ISO_PATH.relative_to(ROOT)
            ),
        },
        "construction": {
            "formula": "rho_H(g) = H^T R1(g) H",
            "group_order": 480,
            "ambient_edge_dimension": 120,
            "harmonic_dimension": 42,
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "representation": {
            "matrix_count": 480,
            "matrix_shape": [42, 42],
            "kernel_indices": (
                representation_kernel_indices
            ),
            "kernel_order": len(
                representation_kernel_indices
            ),
            "greedy_generator_indices": (
                generator_indices
            ),
            "generator_count": len(
                generator_indices
            ),
            "representation_law_test_count": (
                representation_law_test_count
            ),
            "representation_law_max_abs": (
                representation_law_max_abs
            ),
            "maximum_subspace_leakage": max(
                leakage_maxima
            ),
            "maximum_orthogonality_residual": max(
                orthogonality_maxima
            ),
        },
        "character": {
            "identity_value": (
                character_values[identity_index]
            ),
            "norm_raw": character_norm_raw,
            "norm_integer": character_norm_integer,
            "rounded_character_profile": dict(
                sorted(
                    rounded_character_profile.items(),
                    key=lambda item: item[0],
                )
            ),
            "interpretation": (
                "The integer character norm equals the sum of "
                "squares of irreducible multiplicities over the "
                "real/complex character calculation. It detects "
                "reducibility but does not by itself identify every "
                "irreducible constituent."
            ),
        },
        "one_dimensional_sectors": {
            "trivial_multiplicity": (
                trivial_projector_rank
            ),
            "orientation_character_multiplicity": (
                orientation_character_projector_rank
            ),
        },
        "central_halfturn_split": {
            "central_index": central_index,
            "central_coordinates": {
                "s5_permutation": list(
                    coordinates[central_index][
                        "s5_permutation"
                    ]
                ),
                "d8_rotation": coordinates[
                    central_index
                ]["d8_rotation"],
                "d8_flip": coordinates[
                    central_index
                ]["d8_flip"],
            },
            "plus_dimension": central_plus_dimension,
            "minus_dimension": central_minus_dimension,
            "trace": float(np.trace(central_rho)),
            "involution_residual": (
                central_involution_residual
            ),
            "commutation_max_abs": (
                central_commutation_max_abs
            ),
        },
        "profiles": {
            "order_character_profile": [
                {
                    "actual_order": key[0],
                    "rounded_character": key[1],
                    "count": count,
                }
                for key, count in sorted(
                    order_character_profile.items()
                )
            ],
            "orientation_character_profile": [
                {
                    "orientation_type": key[0],
                    "rounded_character": key[1],
                    "count": count,
                }
                for key, count in sorted(
                    orientation_character_profile.items()
                )
            ],
        },
        "outputs": {
            "character_csv": str(
                CHARACTER_CSV_OUT.relative_to(ROOT)
            ),
            "representation_npz": str(
                REPRESENTATION_NPZ_OUT.relative_to(ROOT)
            ),
            "central_sectors_npz": str(
                SECTOR_NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "harmonic_representation_constructed": (
                audit_pass
            ),
            "representation_law_verified_on_generators_and_all_elements": (
                audit_pass
            ),
            "character_profile_computed": audit_pass,
            "central_halfturn_split_computed": (
                audit_pass
            ),
            "full_irreducible_decomposition_computed": False,
            "symmetry_adapted_harmonic_basis_exported": False,
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

    with CHARACTER_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(character_rows[0]),
        )

        writer.writeheader()
        writer.writerows(character_rows)

    np.savez_compressed(
        REPRESENTATION_NPZ_OUT,
        rho_harmonic=representation,
        character=character_array,
        generator_indices=np.array(
            generator_indices,
            dtype=np.int64,
        ),
        representation_kernel_indices=np.array(
            representation_kernel_indices,
            dtype=np.int64,
        ),
        tolerance=np.array([TOLERANCE]),
    )

    np.savez_compressed(
        SECTOR_NPZ_OUT,
        central_rho=central_rho,
        central_plus_basis=central_plus_basis,
        central_minus_basis=central_minus_basis,
        central_plus_projector=central_plus_projector,
        central_minus_projector=central_minus_projector,
        trivial_projector=group_average,
        orientation_character_projector=(
            orientation_average
        ),
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print("representation_shape:", list(representation.shape))
    print(
        "representation_kernel_indices:",
        representation_kernel_indices,
    )
    print(
        "representation_kernel_order:",
        len(representation_kernel_indices),
    )
    print(
        "greedy_generator_indices:",
        generator_indices,
    )
    print(
        "representation_law_tests/max_abs:",
        representation_law_test_count,
        representation_law_max_abs,
    )
    print(
        "maximum_subspace_leakage:",
        max(leakage_maxima),
    )
    print(
        "maximum_orthogonality_residual:",
        max(orthogonality_maxima),
    )
    print(
        "character_identity_value:",
        character_values[identity_index],
    )
    print(
        "character_norm_raw/integer:",
        character_norm_raw,
        character_norm_integer,
    )
    print(
        "rounded_character_profile:",
        dict(
            sorted(
                rounded_character_profile.items(),
                key=lambda item: item[0],
            )
        ),
    )
    print(
        "trivial/orientation_multiplicity:",
        trivial_projector_rank,
        orientation_character_projector_rank,
    )
    print(
        "central_halfturn_plus_minus:",
        central_plus_dimension,
        central_minus_dimension,
    )
    print(
        "central_halfturn_trace:",
        float(np.trace(central_rho)),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CHARACTER_CSV_OUT)
    print("wrote:", REPRESENTATION_NPZ_OUT)
    print("wrote:", SECTOR_NPZ_OUT)


if __name__ == "__main__":
    main()
