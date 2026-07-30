from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

GEOMETRY_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_four_flux_face_geometry_017.json"
)

GEOMETRY_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_face_geometry_017.npz"
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
    / "native_g60_four_flux_axis_register_018.json"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_four_flux_axis_register_018.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_four_flux_axis_gram_018.csv"
)

CHARACTER_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_axis_vs_face_character_018.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_axis_register_018.npz"
)

TOLERANCE = 1e-9
ROUND_DIGITS = 10


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(np.max(np.abs(array)))


def vector_key(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(
        round(float(value), ROUND_DIGITS)
        for value in vector
    )


def line_key(vector: np.ndarray) -> tuple[float, ...]:
    return min(
        vector_key(vector),
        vector_key(-vector),
    )


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


def orbit_index_lookup(
    orbit_vectors: list[np.ndarray],
) -> dict[tuple[float, ...], int]:
    return {
        vector_key(vector): index
        for index, vector in enumerate(orbit_vectors)
    }


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    AXIS_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    GRAM_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHARACTER_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(
        GEOMETRY_PATH.read_text(encoding="utf-8")
    )

    geometry_data = np.load(
        GEOMETRY_NPZ_PATH
    )

    action_data = np.load(
        ACTION_PATH
    )

    isomorphism = json.loads(
        ISO_PATH.read_text(encoding="utf-8")
    )

    representation = np.array(
        geometry_data["face_flux_representation"],
        dtype=np.float64,
    )

    face_targets = np.array(
        action_data["face_target"],
        dtype=np.int64,
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

    stabilizer_rows = geometry["stabilizer_rows"]

    distinguished_modes = [
        int(row["mode_id"])
        for row in stabilizer_rows
        if int(row["exact_stabilizer_order"]) == 24
    ]

    if distinguished_modes != [2]:
        raise RuntimeError(
            "expected distinguished mode [2], found "
            f"{distinguished_modes}"
        )

    base_mode_id = 2

    base_vector = np.zeros(
        4,
        dtype=np.float64,
    )
    base_vector[base_mode_id] = 1.0

    axis_stabilizer = {
        index
        for index in range(480)
        if max_abs(
            representation[index]
            @ base_vector
            - base_vector
        )
        < TOLERANCE
    }

    axis_line_stabilizer = {
        index
        for index in range(480)
        if (
            max_abs(
                representation[index]
                @ base_vector
                - base_vector
            )
            < TOLERANCE
            or max_abs(
                representation[index]
                @ base_vector
                + base_vector
            )
            < TOLERANCE
        )
    }

    signed_orbit_dict: dict[
        tuple[float, ...],
        np.ndarray,
    ] = {}

    for index in range(480):
        image = representation[index] @ base_vector
        signed_orbit_dict.setdefault(
            vector_key(image),
            image,
        )

    signed_axes = list(
        signed_orbit_dict.values()
    )

    signed_axes.sort(
        key=vector_key
    )

    line_keys = {
        line_key(vector)
        for vector in signed_axes
    }

    axis_lookup = orbit_index_lookup(
        signed_axes
    )

    axis_action = np.empty(
        (480, len(signed_axes)),
        dtype=np.int64,
    )

    for group_index in range(480):
        for axis_index, axis in enumerate(
            signed_axes
        ):
            image = (
                representation[group_index]
                @ axis
            )

            key = vector_key(image)

            if key not in axis_lookup:
                raise RuntimeError(
                    "group image left signed-axis orbit"
                )

            axis_action[
                group_index,
                axis_index,
            ] = axis_lookup[key]

    axis_character = np.array(
        [
            sum(
                1
                for source, target in enumerate(
                    action
                )
                if source == int(target)
            )
            for action in axis_action
        ],
        dtype=np.int64,
    )

    face_character = np.array(
        [
            sum(
                1
                for source, target in enumerate(
                    action
                )
                if source == int(target)
            )
            for action in face_targets
        ],
        dtype=np.int64,
    )

    character_difference = (
        axis_character - face_character
    )

    character_profiles_equal = np.array_equal(
        axis_character,
        face_character,
    )

    axis_matrix = np.column_stack(
        signed_axes
    )

    gram = axis_matrix.T @ axis_matrix

    vector_norms = np.linalg.norm(
        axis_matrix,
        axis=0,
    )

    signed_sum = np.sum(
        axis_matrix,
        axis=1,
    )

    frame_operator = (
        axis_matrix @ axis_matrix.T
    )

    frame_constant = float(
        np.sum(vector_norms**2) / 4.0
    )

    tight_frame_residual = max_abs(
        frame_operator
        - frame_constant * np.eye(4)
    )

    gram_polynomial_residual = max_abs(
        gram @ gram
        - frame_constant * gram
    )

    off_diagonal_profile = Counter(
        round(
            float(gram[left, right]),
            ROUND_DIGITS,
        )
        for left in range(len(signed_axes))
        for right in range(
            left + 1,
            len(signed_axes),
        )
    )

    antipodal_map: dict[int, int] = {}

    for axis_index, axis in enumerate(
        signed_axes
    ):
        candidates = [
            other_index
            for other_index, other in enumerate(
                signed_axes
            )
            if max_abs(other + axis) < TOLERANCE
        ]

        if len(candidates) == 1:
            antipodal_map[
                axis_index
            ] = candidates[0]

    face_stabilizers = {
        face: {
            index
            for index in range(480)
            if int(
                face_targets[index, face]
            )
            == face
        }
        for face in range(20)
    }

    direct_stabilizer_matches = [
        face
        for face, stabilizer in (
            face_stabilizers.items()
        )
        if stabilizer == axis_stabilizer
    ]

    conjugate_stabilizer_matches = []

    inverse_rows = [
        inverse(permutation)
        for permutation in permutations
    ]

    for face, face_stabilizer in (
        face_stabilizers.items()
    ):
        matched = False

        for conjugator_index, (
            conjugator,
            conjugator_inverse,
        ) in enumerate(
            zip(
                permutations,
                inverse_rows,
            )
        ):
            conjugated = {
                permutation_to_index[
                    compose(
                        compose(
                            conjugator,
                            permutations[element],
                        ),
                        conjugator_inverse,
                    )
                ]
                for element in axis_stabilizer
            }

            if conjugated == face_stabilizer:
                conjugate_stabilizer_matches.append(
                    {
                        "face": face,
                        "conjugator_index": (
                            conjugator_index
                        ),
                    }
                )
                matched = True
                break

        if matched:
            continue

    intersection_profile = Counter(
        len(
            axis_stabilizer
            & face_stabilizer
        )
        for face_stabilizer in (
            face_stabilizers.values()
        )
    )

    axis_rows = []

    for axis_index, axis in enumerate(
        signed_axes
    ):
        exact_stabilizer_order = sum(
            1
            for index in range(480)
            if max_abs(
                representation[index]
                @ axis
                - axis
            )
            < TOLERANCE
        )

        line_stabilizer_order = sum(
            1
            for index in range(480)
            if (
                max_abs(
                    representation[index]
                    @ axis
                    - axis
                )
                < TOLERANCE
                or max_abs(
                    representation[index]
                    @ axis
                    + axis
                )
                < TOLERANCE
            )
        )

        axis_rows.append(
            {
                "axis_id": axis_index,
                "coordinate_0": float(axis[0]),
                "coordinate_1": float(axis[1]),
                "coordinate_2": float(axis[2]),
                "coordinate_3": float(axis[3]),
                "norm": float(
                    np.linalg.norm(axis)
                ),
                "antipodal_axis_id": (
                    antipodal_map.get(
                        axis_index
                    )
                ),
                "exact_stabilizer_order": (
                    exact_stabilizer_order
                ),
                "line_stabilizer_order": (
                    line_stabilizer_order
                ),
            }
        )

    character_rows = [
        {
            "actual_index": index,
            "axis_fixed_points": int(
                axis_character[index]
            ),
            "face_fixed_points": int(
                face_character[index]
            ),
            "difference": int(
                character_difference[index]
            ),
        }
        for index in range(480)
    ]

    checks = {
        "input_017_audit_pass": (
            geometry.get("audit_pass") is True
        ),
        "distinguished_mode_is_2": (
            distinguished_modes == [2]
        ),
        "axis_stabilizer_order_is_24": (
            len(axis_stabilizer) == 24
        ),
        "axis_line_stabilizer_order_is_48": (
            len(axis_line_stabilizer) == 48
        ),
        "signed_axis_orbit_size_is_20": (
            len(signed_axes) == 20
        ),
        "unoriented_line_orbit_size_is_10": (
            len(line_keys) == 10
        ),
        "axis_action_shape_is_480_by_20": (
            axis_action.shape == (480, 20)
        ),
        "all_axis_actions_are_permutations": all(
            sorted(row.tolist())
            == list(range(20))
            for row in axis_action
        ),
        "all_axes_have_unit_norm": (
            max_abs(
                vector_norms - 1.0
            )
            < TOLERANCE
        ),
        "axis_sum_is_zero": (
            max_abs(signed_sum)
            < TOLERANCE
        ),
        "axis_frame_is_tight": (
            tight_frame_residual
            < TOLERANCE
        ),
        "frame_constant_is_5": (
            abs(frame_constant - 5.0)
            < TOLERANCE
        ),
        "gram_polynomial_identity_passes": (
            gram_polynomial_residual
            < TOLERANCE
        ),
        "each_axis_has_unique_antipode": (
            len(antipodal_map) == 20
        ),
        "no_direct_face_stabilizer_match": (
            direct_stabilizer_matches == []
        ),
        "no_conjugate_face_stabilizer_match": (
            conjugate_stabilizer_matches == []
        ),
        "axis_and_face_actions_are_inequivalent": (
            not character_profiles_equal
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_four_flux_axis_register_018"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_four_flux_axis_register_is_distinct_from_face_register"
            if audit_pass
            else "native_g60_four_flux_axis_register_audit_failed"
        ),
        "inputs": {
            "four_flux_geometry": str(
                GEOMETRY_PATH.relative_to(
                    ROOT
                )
            ),
            "four_flux_geometry_npz": str(
                GEOMETRY_NPZ_PATH.relative_to(
                    ROOT
                )
            ),
            "signed_cochain_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
            "native_group": str(
                ISO_PATH.relative_to(ROOT)
            ),
        },
        "axis_register": {
            "signed_axis_count": len(
                signed_axes
            ),
            "unoriented_line_count": len(
                line_keys
            ),
            "axis_stabilizer_order": len(
                axis_stabilizer
            ),
            "line_stabilizer_order": len(
                axis_line_stabilizer
            ),
            "antipodal_pair_count": (
                len(antipodal_map) // 2
            ),
        },
        "tight_frame": {
            "ambient_dimension": 4,
            "vector_count": 20,
            "frame_constant": (
                frame_constant
            ),
            "axis_sum_max_abs": (
                max_abs(signed_sum)
            ),
            "tight_frame_max_abs": (
                tight_frame_residual
            ),
            "gram_polynomial_max_abs": (
                gram_polynomial_residual
            ),
            "off_diagonal_profile": {
                str(key): value
                for key, value in sorted(
                    off_diagonal_profile.items()
                )
            },
        },
        "face_register_comparison": {
            "axis_character_equals_face_character": (
                character_profiles_equal
            ),
            "character_difference_nonzero_count": int(
                np.count_nonzero(
                    character_difference
                )
            ),
            "direct_stabilizer_matches": (
                direct_stabilizer_matches
            ),
            "conjugate_stabilizer_matches": (
                conjugate_stabilizer_matches
            ),
            "axis_face_stabilizer_intersection_profile": {
                str(key): value
                for key, value in sorted(
                    intersection_profile.items()
                )
            },
            "equivariant_register_isomorphism": False,
        },
        "checks": checks,
        "earned_interpretation": {
            "twenty_signed_flux_axes_exist": (
                audit_pass
            ),
            "axes_form_ten_antipodal_lines": (
                audit_pass
            ),
            "axes_form_unit_norm_tight_frame": (
                audit_pass
            ),
            "axis_register_is_not_face_register": (
                audit_pass
            ),
            "axis_register_is_new_native_twenty_object_register": (
                audit_pass
            ),
            "physical_direction_claim": False,
            "physical_flux_claim": False,
        },
        "outputs": {
            "axis_csv": str(
                AXIS_CSV_OUT.relative_to(ROOT)
            ),
            "gram_csv": str(
                GRAM_CSV_OUT.relative_to(ROOT)
            ),
            "character_csv": str(
                CHARACTER_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "axis_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "axis_register_identified": (
                audit_pass
            ),
            "axis_face_register_equivalence_refuted": (
                audit_pass
            ),
            "axis_register_native_origin_named": False,
            "axis_register_matched_to_known_quotient": False,
            "metric_geometry_derived": False,
            "physical_direction_claim": False,
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

    with AXIS_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                axis_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(axis_rows)

    with GRAM_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["axis_id"]
            + [
                f"axis_{index:02d}"
                for index in range(20)
            ]
        )

        for index in range(20):
            writer.writerow(
                [index]
                + [
                    f"{float(value):.17g}"
                    for value in gram[index]
                ]
            )

    with CHARACTER_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                character_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            character_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        axis_matrix=axis_matrix,
        gram_matrix=gram,
        frame_operator=frame_operator,
        axis_action=axis_action,
        axis_character=axis_character,
        face_character=face_character,
        character_difference=character_difference,
        tolerance=np.array(
            [TOLERANCE]
        ),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "signed_axis/unoriented_line_count:",
        len(signed_axes),
        len(line_keys),
    )
    print(
        "axis_stabilizer/line_stabilizer:",
        len(axis_stabilizer),
        len(axis_line_stabilizer),
    )
    print(
        "antipodal_pair_count:",
        len(antipodal_map) // 2,
    )
    print(
        "axis_sum_max_abs:",
        max_abs(signed_sum),
    )
    print(
        "frame_constant:",
        frame_constant,
    )
    print(
        "tight_frame_residual:",
        tight_frame_residual,
    )
    print(
        "gram_polynomial_residual:",
        gram_polynomial_residual,
    )
    print(
        "axis_character_equals_face_character:",
        character_profiles_equal,
    )
    print(
        "character_difference_nonzero_count:",
        int(
            np.count_nonzero(
                character_difference
            )
        ),
    )
    print(
        "direct_stabilizer_matches:",
        direct_stabilizer_matches,
    )
    print(
        "conjugate_stabilizer_matches:",
        conjugate_stabilizer_matches,
    )
    print(
        "axis_face_intersection_profile:",
        dict(
            sorted(
                intersection_profile.items()
            )
        ),
    )
    print(
        "off_diagonal_inner_product_profile:",
        dict(
            sorted(
                off_diagonal_profile.items()
            )
        ),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", CHARACTER_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
