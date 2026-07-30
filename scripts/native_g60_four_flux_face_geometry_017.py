from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

FLUX_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_unique_four_flux_response_016.json"
)

FLUX_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_unique_four_flux_response_016.npz"
)

ACTION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

SURFACE_SYMMETRY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_symmetry_and_kernel_005.json"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_four_flux_face_geometry_017.json"
)

MODE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_four_flux_face_modes_017.csv"
)

ACTION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_four_flux_representation_017.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_face_geometry_017.npz"
)

TOLERANCE = 1e-9


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


def deterministic_sign_basis(
    basis: np.ndarray,
) -> np.ndarray:
    result = basis.copy()

    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))

        if vector[pivot] < 0:
            result[:, column] *= -1.0

    return result


def apply_signed_face_action(
    basis: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(basis)

    transformed[target, :] = (
        sign[:, None] * basis
    )

    return transformed


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
    MODE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    ACTION_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    flux = json.loads(
        FLUX_PATH.read_text(encoding="utf-8")
    )

    flux_data = np.load(FLUX_NPZ_PATH)
    action_data = np.load(ACTION_PATH)

    surface_symmetry = json.loads(
        SURFACE_SYMMETRY_PATH.read_text(encoding="utf-8")
    )

    face_flux_matrix = np.array(
        flux_data["face_flux_matrix"],
        dtype=np.float64,
    )

    face_targets = np.array(
        action_data["face_target"],
        dtype=np.int64,
    )

    face_signs = np.array(
        action_data["face_sign"],
        dtype=np.int8,
    )

    if face_flux_matrix.shape != (20, 36):
        raise RuntimeError(
            f"unexpected face flux matrix shape: {face_flux_matrix.shape}"
        )

    # Extract an orthonormal basis of the four-dimensional flux image.
    u, singular_values, _ = np.linalg.svd(
        face_flux_matrix,
        full_matrices=False,
    )

    rank = numerical_rank(face_flux_matrix)

    if rank != 4:
        raise RuntimeError(
            f"expected face-flux rank 4, found {rank}"
        )

    flux_basis = deterministic_sign_basis(
        u[:, :rank]
    )

    flux_projector = flux_basis @ flux_basis.T

    representation = np.empty(
        (480, 4, 4),
        dtype=np.float64,
    )

    leakage_maxima = []
    orthogonality_maxima = []
    character_values = []

    identity4 = np.eye(4, dtype=np.float64)

    for index in range(480):
        transformed = apply_signed_face_action(
            flux_basis,
            face_targets[index],
            face_signs[index],
        )

        rho = flux_basis.T @ transformed

        representation[index] = rho

        leakage_maxima.append(
            max_abs(
                transformed
                - flux_basis @ rho
            )
        )

        orthogonality_maxima.append(
            max_abs(
                rho.T @ rho
                - identity4
            )
        )

        character_values.append(
            float(np.trace(rho))
        )

    character_array = np.array(
        character_values,
        dtype=np.float64,
    )

    character_norm_raw = float(
        np.dot(
            character_array,
            character_array,
        )
        / 480.0
    )

    character_norm_integer = nearest_integer(
        character_norm_raw
    )

    representation_kernel_indices = [
        index
        for index in range(480)
        if max_abs(
            representation[index] - identity4
        )
        < TOLERANCE
    ]

    central_index = int(
        surface_symmetry["face_action_kernel"][
            "nonidentity_index"
        ]
    )

    central_rho = representation[central_index]

    orientation_preserving = set(
        int(value)
        for value in surface_symmetry[
            "orientation_character"
        ]["preserving_indices"]
    )

    mode_rows = []

    for mode_id in range(4):
        mode = flux_basis[:, mode_id]

        positive = np.flatnonzero(
            mode > TOLERANCE
        )
        negative = np.flatnonzero(
            mode < -TOLERANCE
        )
        near_zero = np.flatnonzero(
            np.abs(mode) <= TOLERANCE
        )

        maximum_face = int(np.argmax(mode))
        minimum_face = int(np.argmin(mode))

        mode_rows.append(
            {
                "mode_id": mode_id,
                "norm": float(np.linalg.norm(mode)),
                "sum": float(np.sum(mode)),
                "positive_face_count": int(len(positive)),
                "negative_face_count": int(len(negative)),
                "near_zero_face_count": int(len(near_zero)),
                "maximum_face_id": maximum_face,
                "maximum_face_value": float(mode[maximum_face]),
                "minimum_face_id": minimum_face,
                "minimum_face_value": float(mode[minimum_face]),
                "support_above_1e_12": int(
                    np.count_nonzero(
                        np.abs(mode) > 1e-12
                    )
                ),
            }
        )

    stabilizer_rows = []

    for mode_id in range(4):
        coordinate = np.zeros(4, dtype=np.float64)
        coordinate[mode_id] = 1.0

        exact_stabilizer = []
        line_stabilizer = []
        sign_reversers = []

        for index in range(480):
            image = representation[index] @ coordinate

            if max_abs(image - coordinate) < TOLERANCE:
                exact_stabilizer.append(index)

            if (
                max_abs(image - coordinate) < TOLERANCE
                or max_abs(image + coordinate) < TOLERANCE
            ):
                line_stabilizer.append(index)

            if max_abs(image + coordinate) < TOLERANCE:
                sign_reversers.append(index)

        stabilizer_rows.append(
            {
                "mode_id": mode_id,
                "exact_stabilizer_order": len(
                    exact_stabilizer
                ),
                "line_stabilizer_order": len(
                    line_stabilizer
                ),
                "sign_reverser_count": len(
                    sign_reversers
                ),
                "orientation_preserving_exact_count": sum(
                    1
                    for index in exact_stabilizer
                    if index in orientation_preserving
                ),
                "orientation_reversing_exact_count": sum(
                    1
                    for index in exact_stabilizer
                    if index not in orientation_preserving
                ),
            }
        )

    rounded_character_profile = Counter()

    for value in character_values:
        integer = nearest_integer(value)

        key = (
            str(integer)
            if integer is not None
            else f"{value:.10f}"
        )

        rounded_character_profile[key] += 1

    pair_inner_products = (
        flux_basis.T @ flux_basis
    )

    face_covariance = (
        flux_basis @ flux_basis.T
    )

    face_diagonal_profile = Counter(
        round(float(value), 12)
        for value in np.diag(face_covariance)
    )

    checks = {
        "input_016_audit_pass": (
            flux.get("audit_pass") is True
        ),
        "face_flux_matrix_shape_is_20_by_36": (
            face_flux_matrix.shape == (20, 36)
        ),
        "face_flux_rank_is_4": (
            rank == 4
        ),
        "flux_basis_shape_is_20_by_4": (
            flux_basis.shape == (20, 4)
        ),
        "flux_basis_is_orthonormal": (
            max_abs(
                pair_inner_products - identity4
            )
            < TOLERANCE
        ),
        "all_flux_modes_have_zero_total": all(
            abs(row["sum"]) < TOLERANCE
            for row in mode_rows
        ),
        "flux_subspace_is_invariant_for_all_480": (
            max(leakage_maxima) < TOLERANCE
        ),
        "all_restricted_actions_are_orthogonal": (
            max(orthogonality_maxima)
            < TOLERANCE
        ),
        "character_identity_value_is_4": (
            abs(character_values[0] - 4.0)
            < TOLERANCE
        ),
        "character_norm_is_one": (
            character_norm_integer == 1
        ),
        "central_face_kernel_element_acts_trivially": (
            max_abs(
                central_rho - identity4
            )
            < TOLERANCE
        ),
        "all_faces_have_equal_projector_diagonal": (
            len(face_diagonal_profile) == 1
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_four_flux_face_geometry_017"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_unique_four_flux_face_representation_identified"
            if audit_pass
            else "native_g60_four_flux_face_geometry_audit_failed"
        ),
        "inputs": {
            "unique_four_flux_response": str(
                FLUX_PATH.relative_to(ROOT)
            ),
            "unique_four_flux_npz": str(
                FLUX_NPZ_PATH.relative_to(ROOT)
            ),
            "signed_cochain_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
            "surface_symmetry": str(
                SURFACE_SYMMETRY_PATH.relative_to(
                    ROOT
                )
            ),
        },
        "flux_face_space": {
            "ambient_face_dimension": 20,
            "flux_dimension": 4,
            "orthogonal_to_constant_face_mode": True,
            "projector_diagonal_profile": {
                str(key): value
                for key, value in sorted(
                    face_diagonal_profile.items()
                )
            },
            "equal_face_weight": (
                len(face_diagonal_profile) == 1
            ),
        },
        "representation": {
            "group_order": 480,
            "dimension": 4,
            "kernel_indices": (
                representation_kernel_indices
            ),
            "kernel_order": len(
                representation_kernel_indices
            ),
            "character_identity_value": (
                character_values[0]
            ),
            "character_norm_raw": (
                character_norm_raw
            ),
            "character_norm_integer": (
                character_norm_integer
            ),
            "rounded_character_profile": {
                str(key): value
                for key, value in sorted(
                    rounded_character_profile.items()
                )
            },
            "maximum_subspace_leakage": max(
                leakage_maxima
            ),
            "maximum_orthogonality_residual": max(
                orthogonality_maxima
            ),
            "central_halfturn_action": (
                "identity"
                if max_abs(
                    central_rho - identity4
                )
                < TOLERANCE
                else "nontrivial"
            ),
        },
        "mode_rows": mode_rows,
        "stabilizer_rows": stabilizer_rows,
        "checks": checks,
        "earned_interpretation": {
            "four_flux_space_is_irreducible": (
                character_norm_integer == 1
            ),
            "all_twenty_faces_participate_equally_at_projector_level": (
                len(face_diagonal_profile) == 1
            ),
            "central_halfturn_is_invisible_to_four_flux": (
                max_abs(
                    central_rho - identity4
                )
                < TOLERANCE
            ),
            "canonical_coordinate_basis_found": False,
            "physical_flux_claim": False,
        },
        "outputs": {
            "mode_csv": str(
                MODE_CSV_OUT.relative_to(ROOT)
            ),
            "representation_csv": str(
                ACTION_CSV_OUT.relative_to(ROOT)
            ),
            "geometry_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "four_flux_face_subspace_identified": (
                audit_pass
            ),
            "four_flux_group_representation_identified": (
                audit_pass
            ),
            "irreducibility_established": (
                character_norm_integer == 1
            ),
            "canonical_mode_basis_selected": False,
            "localization_claim": False,
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

    with MODE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(mode_rows[0]),
        )

        writer.writeheader()
        writer.writerows(mode_rows)

    action_rows = []

    for index in range(480):
        action_rows.append(
            {
                "actual_index": index,
                "orientation_type": (
                    "preserving"
                    if index in orientation_preserving
                    else "reversing"
                ),
                "character": character_values[index],
                "rounded_character": nearest_integer(
                    character_values[index]
                ),
                "representation_is_identity": (
                    index
                    in representation_kernel_indices
                ),
                "orthogonality_max_abs": (
                    orthogonality_maxima[index]
                ),
                "subspace_leakage_max_abs": (
                    leakage_maxima[index]
                ),
            }
        )

    with ACTION_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(action_rows[0]),
        )

        writer.writeheader()
        writer.writerows(action_rows)

    np.savez_compressed(
        NPZ_OUT,
        face_flux_basis=flux_basis,
        face_flux_projector=flux_projector,
        face_flux_representation=representation,
        face_flux_character=character_array,
        singular_values=singular_values,
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print("face_flux_rank:", rank)
    print(
        "basis_orthonormality_max_abs:",
        max_abs(
            pair_inner_products - identity4
        ),
    )
    print(
        "representation_kernel_order:",
        len(representation_kernel_indices),
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
                rounded_character_profile.items()
            )
        ),
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
        "central_halfturn_action:",
        payload["representation"][
            "central_halfturn_action"
        ],
    )
    print(
        "face_projector_diagonal_profile:",
        dict(
            sorted(
                face_diagonal_profile.items()
            )
        ),
    )

    print("\nmode stabilizers:")

    for row in stabilizer_rows:
        print(
            "mode",
            row["mode_id"],
            "exact=",
            row["exact_stabilizer_order"],
            "line=",
            row["line_stabilizer_order"],
            "sign_reversers=",
            row["sign_reverser_count"],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", MODE_CSV_OUT)
    print("wrote:", ACTION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
