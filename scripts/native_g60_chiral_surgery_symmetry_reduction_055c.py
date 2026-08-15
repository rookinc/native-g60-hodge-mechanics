#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BRIDGE_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_chart_dyad_bridge_search_055a.v1.json"
)

SURGERY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_attractor_dyad_surgery_audit_055b.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_surgery_symmetry_reduction_055c.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_chiral_surgery_symmetry_reduction_055c.csv"
)


def edge(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((left, right)))


def permutation_inverse(
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
    current = identity

    for exponent in range(1, 100):
        current = tuple(
            permutation[current[index]]
            for index in range(len(permutation))
        )

        if current == identity:
            return exponent

    raise RuntimeError("permutation order bound exceeded")


def transform_edge(
    value: tuple[int, int],
    permutation: tuple[int, ...],
) -> tuple[int, int]:
    return edge(
        permutation[value[0]],
        permutation[value[1]],
    )


def canonical_surgery(
    circulating: frozenset[int],
    matching: frozenset[tuple[int, int]],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
]:
    return (
        tuple(sorted(circulating)),
        tuple(sorted(matching)),
    )


bridge = json.loads(BRIDGE_JSON.read_text())
surgery_source = json.loads(SURGERY_JSON.read_text())

feature_names = tuple(
    sorted(bridge["feature_names"])
)

pair_features: dict[
    tuple[int, int],
    tuple[int, ...],
] = {}

for row in bridge["chart_pairs"]:
    pair = edge(
        int(row["left"]),
        int(row["right"]),
    )

    pair_features[pair] = tuple(
        int(row[name])
        for name in feature_names
    )

if len(pair_features) != 15:
    raise RuntimeError(
        f"expected 15 chart pairs, found {len(pair_features)}"
    )

source_edges = frozenset(
    edge(*value)
    for value in surgery_source["source_edges"]
)

left_side = frozenset(
    surgery_source["source_bipartition"]["left"]
)

right_side = frozenset(
    surgery_source["source_bipartition"]["right"]
)

vertices = tuple(range(6))

undirected_automorphisms = []
full_feature_automorphisms = []

for permutation in itertools.permutations(vertices):
    permutation = tuple(permutation)

    transformed_source = frozenset(
        transform_edge(value, permutation)
        for value in source_edges
    )

    if transformed_source != source_edges:
        continue

    undirected_automorphisms.append(permutation)

    preserves_all_features = all(
        pair_features[value]
        == pair_features[
            transform_edge(value, permutation)
        ]
        for value in pair_features
    )

    if preserves_all_features:
        full_feature_automorphisms.append(
            permutation
        )

surgeries = []
surgery_index_by_key = {}

for index, row in enumerate(
    surgery_source["surgeries"]
):
    circulating = frozenset(
        int(value)
        for value in row["circulating_vertices"]
    )

    rooted = frozenset(
        int(value)
        for value in row["rooted_vertices"]
    )

    matching = frozenset(
        edge(*value)
        for value in row["removed_matching"]
    )

    key = canonical_surgery(
        circulating,
        matching,
    )

    surgery_record = {
        "surgery_index": index,
        "circulating_side": row["circulating_side"],
        "circulating": circulating,
        "rooted": rooted,
        "matching": matching,
        "key": key,
    }

    surgeries.append(surgery_record)
    surgery_index_by_key[key] = index

if len(surgery_index_by_key) != 12:
    raise RuntimeError(
        "expected twelve distinct surgery keys"
    )


def transform_surgery(
    surgery: dict[str, Any],
    permutation: tuple[int, ...],
) -> int:
    circulating = frozenset(
        permutation[value]
        for value in surgery["circulating"]
    )

    matching = frozenset(
        transform_edge(value, permutation)
        for value in surgery["matching"]
    )

    key = canonical_surgery(
        circulating,
        matching,
    )

    if key not in surgery_index_by_key:
        raise RuntimeError(
            "feature automorphism left surgery space"
        )

    return surgery_index_by_key[key]


def action_orbits(
    automorphisms: list[tuple[int, ...]],
) -> list[list[int]]:
    unseen = set(range(len(surgeries)))
    orbits = []

    while unseen:
        start = min(unseen)
        orbit = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for permutation in automorphisms:
                target = transform_surgery(
                    surgeries[current],
                    permutation,
                )

                if target in orbit:
                    continue

                orbit.add(target)
                queue.append(target)

        unseen -= orbit
        orbits.append(sorted(orbit))

    return sorted(
        orbits,
        key=lambda value: (
            len(value),
            value,
        ),
    )


undirected_orbits = action_orbits(
    undirected_automorphisms
)

full_feature_orbits = action_orbits(
    full_feature_automorphisms
)

automorphism_order_profile = Counter(
    permutation_order(permutation)
    for permutation in full_feature_automorphisms
)

side_behavior_profile = Counter()

for permutation in full_feature_automorphisms:
    image_left = frozenset(
        permutation[value]
        for value in left_side
    )

    image_right = frozenset(
        permutation[value]
        for value in right_side
    )

    if (
        image_left == left_side
        and image_right == right_side
    ):
        behavior = "preserves_both_sides"
    elif (
        image_left == right_side
        and image_right == left_side
    ):
        behavior = "swaps_sides"
    else:
        behavior = "invalid_side_behavior"

    side_behavior_profile[behavior] += 1

surgery_rows = []

for surgery in surgeries:
    feature_orbit_id = next(
        index
        for index, orbit in enumerate(
            full_feature_orbits,
            start=1,
        )
        if surgery["surgery_index"] in orbit
    )

    undirected_orbit_id = next(
        index
        for index, orbit in enumerate(
            undirected_orbits,
            start=1,
        )
        if surgery["surgery_index"] in orbit
    )

    surgery_rows.append(
        {
            "surgery_index": (
                surgery["surgery_index"]
            ),
            "circulating_side": (
                surgery["circulating_side"]
            ),
            "circulating_vertices": sorted(
                surgery["circulating"]
            ),
            "rooted_vertices": sorted(
                surgery["rooted"]
            ),
            "removed_matching": [
                list(value)
                for value in sorted(
                    surgery["matching"]
                )
            ],
            "undirected_orbit_id": (
                undirected_orbit_id
            ),
            "full_feature_orbit_id": (
                feature_orbit_id
            ),
        }
    )

checks = {
    "source_graph_is_K33": (
        surgery_source["source_is_K33"] is True
    ),
    "surgery_count_is_12": (
        len(surgeries) == 12
    ),
    "undirected_K33_automorphism_count_is_72": (
        len(undirected_automorphisms) == 72
    ),
    "full_feature_automorphism_group_nonempty": (
        len(full_feature_automorphisms) > 0
    ),
    "identity_is_full_feature_automorphism": (
        tuple(range(6))
        in full_feature_automorphisms
    ),
    "all_full_feature_automorphisms_preserve_K33": all(
        permutation in undirected_automorphisms
        for permutation in full_feature_automorphisms
    ),
    "full_feature_orbits_partition_12": (
        sum(
            len(orbit)
            for orbit in full_feature_orbits
        )
        == 12
    ),
    "no_invalid_side_behavior": (
        side_behavior_profile[
            "invalid_side_behavior"
        ]
        == 0
    ),
}

audit_pass = all(checks.values())

if len(full_feature_orbits) == 1:
    verdict = (
        "directed_chart_overlap_symmetry_leaves_"
        "all_twelve_polarizations_equivalent"
    )
elif (
    len(full_feature_orbits) == 2
    and sorted(
        len(orbit)
        for orbit in full_feature_orbits
    )
    == [6, 6]
):
    verdict = (
        "directed_chart_overlap_symmetry_reduces_"
        "twelve_polarizations_to_two_six_state_classes"
    )
else:
    verdict = (
        "directed_chart_overlap_symmetry_produces_"
        "nontrivial_surgery_orbit_partition"
    )

record = {
    "artifact_id": (
        "native_g60_chiral_surgery_symmetry_reduction_055c"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "feature_names": list(feature_names),
    "undirected_automorphism_count": (
        len(undirected_automorphisms)
    ),
    "full_feature_automorphism_count": (
        len(full_feature_automorphisms)
    ),
    "full_feature_automorphism_order_profile": {
        str(key): value
        for key, value in sorted(
            automorphism_order_profile.items()
        )
    },
    "side_behavior_profile": dict(
        sorted(side_behavior_profile.items())
    ),
    "undirected_surgery_orbit_count": (
        len(undirected_orbits)
    ),
    "undirected_surgery_orbits": (
        undirected_orbits
    ),
    "full_feature_surgery_orbit_count": (
        len(full_feature_orbits)
    ),
    "full_feature_surgery_orbits": (
        full_feature_orbits
    ),
    "full_feature_surgery_orbit_sizes": [
        len(orbit)
        for orbit in full_feature_orbits
    ],
    "surgeries": surgery_rows,
    "checks": checks,
    "boundary": {
        "undirected_K33_symmetry_classified": True,
        "directed_overlap_symmetry_classified": (
            audit_pass
        ),
        "preferred_surgery_selected": (
            len(full_feature_orbits) == 12
            and all(
                len(orbit) == 1
                for orbit in full_feature_orbits
            )
        ),
        "attractor_selects_surgery_proved": False,
        "order_four_face_maps_used_directly": False,
        "signed_axis_overlap_used": False,
        "physical_chirality_proved": False,
        "physical_attraction_proved": False,
    },
}

JSON_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

JSON_OUT.write_text(
    json.dumps(
        record,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

with CSV_OUT.open("w") as handle:
    handle.write(
        "surgery_index,circulating_side,"
        "circulating_vertices,rooted_vertices,"
        "removed_matching,undirected_orbit_id,"
        "full_feature_orbit_id\n"
    )

    for row in surgery_rows:
        handle.write(
            ",".join(
                [
                    str(row["surgery_index"]),
                    row["circulating_side"],
                    json.dumps(
                        row["circulating_vertices"],
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        row["rooted_vertices"],
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        row["removed_matching"],
                        separators=(",", ":"),
                    ),
                    str(row["undirected_orbit_id"]),
                    str(row["full_feature_orbit_id"]),
                ]
            )
            + "\n"
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print(
    "undirected_automorphism_count:",
    record["undirected_automorphism_count"],
)
print(
    "full_feature_automorphism_count:",
    record["full_feature_automorphism_count"],
)
print(
    "full_feature_automorphism_order_profile:",
    record[
        "full_feature_automorphism_order_profile"
    ],
)
print(
    "side_behavior_profile:",
    record["side_behavior_profile"],
)
print(
    "undirected_surgery_orbit_count:",
    record["undirected_surgery_orbit_count"],
)
print(
    "undirected_surgery_orbits:",
    record["undirected_surgery_orbits"],
)
print(
    "full_feature_surgery_orbit_count:",
    record["full_feature_surgery_orbit_count"],
)
print(
    "full_feature_surgery_orbit_sizes:",
    record["full_feature_surgery_orbit_sizes"],
)
print(
    "full_feature_surgery_orbits:",
    record["full_feature_surgery_orbits"],
)
print("checks:", record["checks"])
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
