from __future__ import annotations

import csv
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

AXIS_REGISTER_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_four_flux_axis_register_018.json"
)

AXIS_REGISTER_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_axis_register_018.npz"
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
    / "native_g60_flux_axis_to_face_relation_019.json"
)

FACE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_flux_axis_to_face_relation_019.csv"
)

ORBIT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_flux_axis_face_orbits_019.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_flux_axis_to_face_relation_019.npz"
)

TOLERANCE = 1e-9
ROUND_DIGITS = 12


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(np.max(np.abs(array)))


def rounded(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def stabilizer_face_orbits(
    stabilizer_indices: set[int],
    face_targets: np.ndarray,
) -> list[list[int]]:
    adjacency = {
        face: set()
        for face in range(20)
    }

    for group_index in stabilizer_indices:
        action = face_targets[group_index]

        for source, target in enumerate(action):
            target_int = int(target)
            adjacency[source].add(target_int)
            adjacency[target_int].add(source)

    unseen = set(range(20))
    orbits = []

    while unseen:
        start = min(unseen)
        reached = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor in adjacency[current]:
                if neighbor in reached:
                    continue

                reached.add(neighbor)
                queue.append(neighbor)

        unseen -= reached
        orbits.append(sorted(reached))

    orbits.sort(
        key=lambda orbit: (
            -len(orbit),
            orbit[0],
        )
    )

    return orbits


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    FACE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    ORBIT_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    axis_register = json.loads(
        AXIS_REGISTER_PATH.read_text(
            encoding="utf-8"
        )
    )

    axis_data = np.load(
        AXIS_REGISTER_NPZ_PATH
    )

    geometry_data = np.load(
        GEOMETRY_NPZ_PATH
    )

    action_data = np.load(
        ACTION_PATH
    )

    isomorphism = json.loads(
        ISO_PATH.read_text(
            encoding="utf-8"
        )
    )

    axis_matrix = np.array(
        axis_data["axis_matrix"],
        dtype=np.float64,
    )

    flux_basis = np.array(
        geometry_data["face_flux_basis"],
        dtype=np.float64,
    )

    flux_representation = np.array(
        geometry_data[
            "face_flux_representation"
        ],
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

    mapping_rows = isomorphism["mapping_rows"]

    if axis_matrix.shape != (4, 20):
        raise RuntimeError(
            f"unexpected axis matrix shape: {axis_matrix.shape}"
        )

    # Recover the same distinguished axis used in artifacts 017 and 018.
    base_coordinate = np.zeros(
        4,
        dtype=np.float64,
    )
    base_coordinate[2] = 1.0

    base_pattern = (
        flux_basis @ base_coordinate
    )

    axis_stabilizer = {
        index
        for index in range(480)
        if max_abs(
            flux_representation[index]
            @ base_coordinate
            - base_coordinate
        )
        < TOLERANCE
    }

    if len(axis_stabilizer) != 24:
        raise RuntimeError(
            "expected axis stabilizer order 24, found "
            f"{len(axis_stabilizer)}"
        )

    face_orbits = stabilizer_face_orbits(
        axis_stabilizer,
        face_targets,
    )

    orbit_sizes = [
        len(orbit)
        for orbit in face_orbits
    ]

    # Verify the complete signed face pattern is fixed by the stabilizer.
    signed_pattern_residuals = []

    for group_index in sorted(
        axis_stabilizer
    ):
        transformed = np.empty_like(
            base_pattern
        )

        transformed[
            face_targets[group_index]
        ] = (
            face_signs[group_index]
            * base_pattern
        )

        signed_pattern_residuals.append(
            max_abs(
                transformed - base_pattern
            )
        )

    signed_pattern_invariance_max_abs = max(
        signed_pattern_residuals
    )

    face_to_orbit = {}

    for orbit_id, orbit in enumerate(
        face_orbits
    ):
        for face in orbit:
            face_to_orbit[face] = orbit_id

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

    face_rows = []

    for face in range(20):
        intersection_order = len(
            axis_stabilizer
            & face_stabilizers[face]
        )

        face_rows.append(
            {
                "face_id": face,
                "orbit_id": face_to_orbit[face],
                "orbit_size": len(
                    face_orbits[
                        face_to_orbit[face]
                    ]
                ),
                "axis_face_stabilizer_intersection_order": (
                    intersection_order
                ),
                "flux_value": float(
                    base_pattern[face]
                ),
                "absolute_flux_value": float(
                    abs(base_pattern[face])
                ),
                "flux_sign": (
                    1
                    if base_pattern[face] > TOLERANCE
                    else -1
                    if base_pattern[face] < -TOLERANCE
                    else 0
                ),
            }
        )

    orbit_rows = []

    for orbit_id, orbit in enumerate(
        face_orbits
    ):
        values = np.array(
            [
                base_pattern[face]
                for face in orbit
            ],
            dtype=np.float64,
        )

        absolute_values = np.abs(values)

        intersection_profile = Counter(
            len(
                axis_stabilizer
                & face_stabilizers[face]
            )
            for face in orbit
        )

        signed_profile = Counter(
            rounded(value)
            for value in values
        )

        absolute_profile = Counter(
            rounded(value)
            for value in absolute_values
        )

        sign_profile = Counter(
            (
                1
                if value > TOLERANCE
                else -1
                if value < -TOLERANCE
                else 0
            )
            for value in values
        )

        orbit_rows.append(
            {
                "orbit_id": orbit_id,
                "orbit_size": len(orbit),
                "faces": json.dumps(orbit),
                "intersection_order_profile": json.dumps(
                    {
                        str(key): count
                        for key, count in sorted(
                            intersection_profile.items()
                        )
                    },
                    sort_keys=True,
                ),
                "signed_flux_profile": json.dumps(
                    {
                        str(key): count
                        for key, count in sorted(
                            signed_profile.items()
                        )
                    },
                    sort_keys=True,
                ),
                "absolute_flux_profile": json.dumps(
                    {
                        str(key): count
                        for key, count in sorted(
                            absolute_profile.items()
                        )
                    },
                    sort_keys=True,
                ),
                "sign_profile": json.dumps(
                    {
                        str(key): count
                        for key, count in sorted(
                            sign_profile.items()
                        )
                    },
                    sort_keys=True,
                ),
                "flux_sum": float(
                    np.sum(values)
                ),
                "flux_mean": float(
                    np.mean(values)
                ),
                "flux_norm": float(
                    np.linalg.norm(values)
                ),
                "constant_signed_value": (
                    len(signed_profile) == 1
                ),
                "constant_absolute_value": (
                    len(absolute_profile) == 1
                ),
            }
        )

    intersection_size_classes = Counter(
        row[
            "axis_face_stabilizer_intersection_order"
        ]
        for row in face_rows
    )

    orbit_size_profile = Counter(
        orbit_sizes
    )

    orbit_intersection_order_sets = [
        sorted(
            {
                row[
                    "axis_face_stabilizer_intersection_order"
                ]
                for row in face_rows
                if row["orbit_id"] == orbit_id
            }
        )
        for orbit_id in range(
            len(face_orbits)
        )
    ]

    orbit_partition_matches_intersection_classes = (
        all(
            len(order_set) == 1
            for order_set in (
                orbit_intersection_order_sets
            )
        )
        and sorted(orbit_sizes)
        == sorted(
            intersection_size_classes.values()
        )
    )

    positive_faces = [
        row["face_id"]
        for row in face_rows
        if row["flux_sign"] == 1
    ]

    negative_faces = [
        row["face_id"]
        for row in face_rows
        if row["flux_sign"] == -1
    ]

    zero_faces = [
        row["face_id"]
        for row in face_rows
        if row["flux_sign"] == 0
    ]

    checks = {
        "input_018_audit_pass": (
            axis_register.get("audit_pass") is True
        ),
        "axis_stabilizer_order_is_24": (
            len(axis_stabilizer) == 24
        ),
        "stabilizer_face_orbits_partition_all_20_faces": (
            sorted(
                face
                for orbit in face_orbits
                for face in orbit
            )
            == list(range(20))
        ),
        "face_orbit_sizes_are_12_6_2": (
            sorted(orbit_sizes)
            == [2, 6, 12]
        ),
        "orbit_count_is_3": (
            len(face_orbits) == 3
        ),
        "signed_face_pattern_is_axis_stabilizer_invariant": (
            signed_pattern_invariance_max_abs
            < TOLERANCE
        ),
        "stabilizer_orbits_match_intersection_order_classes": (
            orbit_partition_matches_intersection_classes
        ),
        "base_pattern_has_zero_total_flux": (
            abs(
                float(
                    np.sum(base_pattern)
                )
            )
            < TOLERANCE
        ),
        "every_face_has_orbit_label": (
            len(face_to_orbit) == 20
        ),
    }

    audit_pass = all(
        checks.values()
    )

    payload = {
        "artifact_id": (
            "native_g60_flux_axis_to_face_relation_019"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_flux_axis_induces_exact_12_6_2_face_relation"
            if audit_pass
            else "native_g60_flux_axis_to_face_relation_audit_failed"
        ),
        "inputs": {
            "axis_register": str(
                AXIS_REGISTER_PATH.relative_to(
                    ROOT
                )
            ),
            "axis_register_npz": str(
                AXIS_REGISTER_NPZ_PATH.relative_to(
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
        "axis": {
            "coordinate": [
                float(value)
                for value in base_coordinate
            ],
            "stabilizer_order": len(
                axis_stabilizer
            ),
            "face_pattern_sum": float(
                np.sum(base_pattern)
            ),
            "signed_pattern_invariance_max_abs": (
                signed_pattern_invariance_max_abs
            ),
        },
        "face_relation": {
            "orbit_count": len(
                face_orbits
            ),
            "orbit_sizes": orbit_sizes,
            "sorted_orbit_sizes": sorted(
                orbit_sizes
            ),
            "orbit_size_profile": {
                str(key): value
                for key, value in sorted(
                    orbit_size_profile.items()
                )
            },
            "intersection_size_classes": {
                str(key): value
                for key, value in sorted(
                    intersection_size_classes.items()
                )
            },
            "orbit_intersection_order_sets": (
                orbit_intersection_order_sets
            ),
            "orbit_partition_matches_intersection_classes": (
                orbit_partition_matches_intersection_classes
            ),
        },
        "flux_sign_summary": {
            "positive_face_count": len(
                positive_faces
            ),
            "negative_face_count": len(
                negative_faces
            ),
            "zero_face_count": len(
                zero_faces
            ),
            "positive_faces": positive_faces,
            "negative_faces": negative_faces,
            "zero_faces": zero_faces,
        },
        "orbit_rows": orbit_rows,
        "face_rows": face_rows,
        "checks": checks,
        "earned_interpretation": {
            "one_flux_axis_induces_three_face_relation_classes": (
                audit_pass
            ),
            "relation_class_sizes": (
                sorted(orbit_sizes)
            ),
            "relation_classes_are_not_face_attachment": True,
            "flux_values_are_stabilizer_constrained": (
                audit_pass
            ),
            "physical_flux_claim": False,
        },
        "outputs": {
            "face_csv": str(
                FACE_CSV_OUT.relative_to(ROOT)
            ),
            "orbit_csv": str(
                ORBIT_CSV_OUT.relative_to(ROOT)
            ),
            "relation_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "axis_to_face_orbit_relation_resolved": (
                audit_pass
            ),
            "twelve_six_two_partition_verified": (
                sorted(orbit_sizes)
                == [2, 6, 12]
            ),
            "physical_near_far_interpretation": False,
            "metric_distance_derived": False,
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

    with FACE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                face_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(face_rows)

    with ORBIT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                orbit_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(orbit_rows)

    np.savez_compressed(
        NPZ_OUT,
        base_axis_coordinate=base_coordinate,
        base_face_pattern=base_pattern,
        face_orbit_id=np.array(
            [
                face_to_orbit[face]
                for face in range(20)
            ],
            dtype=np.int64,
        ),
        face_flux_values=base_pattern,
        face_intersection_orders=np.array(
            [
                row[
                    "axis_face_stabilizer_intersection_order"
                ]
                for row in face_rows
            ],
            dtype=np.int64,
        ),
        axis_stabilizer_indices=np.array(
            sorted(axis_stabilizer),
            dtype=np.int64,
        ),
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "axis_stabilizer_order:",
        len(axis_stabilizer),
    )
    print(
        "face_orbit_sizes:",
        orbit_sizes,
    )
    print(
        "sorted_face_orbit_sizes:",
        sorted(orbit_sizes),
    )
    print(
        "intersection_size_classes:",
        dict(
            sorted(
                intersection_size_classes.items()
            )
        ),
    )
    print(
        "orbit_intersection_order_sets:",
        orbit_intersection_order_sets,
    )
    print(
        "orbit_partition_matches_intersection_classes:",
        orbit_partition_matches_intersection_classes,
    )
    print(
        "signed_pattern_invariance_max_abs:",
        signed_pattern_invariance_max_abs,
    )
    print(
        "positive/negative/zero_faces:",
        len(positive_faces),
        len(negative_faces),
        len(zero_faces),
    )

    print("\nface orbit anatomy:")

    for row in orbit_rows:
        print(
            "orbit",
            row["orbit_id"],
            "size=",
            row["orbit_size"],
            "intersection=",
            row[
                "intersection_order_profile"
            ],
            "signed_flux=",
            row[
                "signed_flux_profile"
            ],
            "absolute_flux=",
            row[
                "absolute_flux_profile"
            ],
            "signs=",
            row["sign_profile"],
            "sum=",
            row["flux_sum"],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", FACE_CSV_OUT)
    print("wrote:", ORBIT_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
