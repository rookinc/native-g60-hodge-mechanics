#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CHART_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chart_specific_chiral_map_derivation_055e.v1.json"
)

MATCHING_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_matching_chiral_compatibility_census_055f.v1.json"
)

SYMMETRY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_surgery_symmetry_reduction_055c.v1.json"
)

SURGERY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_attractor_dyad_surgery_audit_055b.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_polarization_selector_audit_055g.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_chiral_polarization_selector_audit_055g.csv"
)


def normalized_edge(
    left: int,
    right: int,
) -> tuple[int, int]:
    return tuple(sorted((left, right)))


def inverse(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    result = [0] * len(permutation)

    for source, target in enumerate(permutation):
        result[target] = source

    return tuple(result)


def triangle_edges(
    vertices: list[int],
) -> list[tuple[int, int]]:
    return sorted(
        normalized_edge(left, right)
        for left, right in itertools.combinations(
            sorted(vertices),
            2,
        )
    )


chart_source = json.loads(
    CHART_JSON.read_text()
)

matching_source = json.loads(
    MATCHING_JSON.read_text()
)

symmetry_source = json.loads(
    SYMMETRY_JSON.read_text()
)

surgery_source = json.loads(
    SURGERY_JSON.read_text()
)

charts: dict[int, dict[str, Any]] = {}

for row in chart_source["chart_records"]:
    permutation = tuple(
        int(value)
        for value in row["derived_permutation"]
    )

    charts[int(row["chart_index"])] = {
        "dyad_side": row["dyad_side"],
        "chirality": row["chirality"],
        "map_index": int(row["map_index"]),
        "inverse_pair_id": int(row["inverse_pair_id"]),
        "permutation": permutation,
        "inverse": inverse(permutation),
    }

matching_records = {
    int(row["matching_id"]): row
    for row in matching_source["matching_records"]
}

surgery_rows = {
    index: row
    for index, row in enumerate(
        surgery_source["surgeries"]
    )
}

directed_overlap_orbits = [
    sorted(int(value) for value in orbit)
    for orbit in symmetry_source[
        "full_feature_surgery_orbits"
    ]
]

size_two_orbits = sorted(
    orbit
    for orbit in directed_overlap_orbits
    if len(orbit) == 2
)

distinguished_matching_ids = sorted(
    {
        int(
            next(
                row["matching_id"]
                for row in matching_records.values()
                if surgery_index
                in row["surgery_indices"]
            )
        )
        for orbit in size_two_orbits
        for surgery_index in orbit
    }
)

matching_stage_records = []

for matching_id in distinguished_matching_ids:
    row = matching_records[matching_id]

    matching_stage_records.append(
        {
            "matching_id": matching_id,
            "matching": row["matching"],
            "surgery_indices": sorted(
                int(value)
                for value in row["surgery_indices"]
            ),
            "exact_inverse_edge_count": int(
                row["exact_inverse_edge_count"]
            ),
            "same_inverse_pair_edge_count": int(
                row["same_inverse_pair_edge_count"]
            ),
            "same_map_edge_count": int(
                row["same_map_edge_count"]
            ),
        }
    )

maximum_removed_inverse_count = max(
    row["exact_inverse_edge_count"]
    for row in matching_stage_records
)

matching_stage_survivors = [
    row
    for row in matching_stage_records
    if row["exact_inverse_edge_count"]
    == maximum_removed_inverse_count
]

matching_stage_unique = (
    len(matching_stage_survivors) == 1
)

selected_matching = (
    matching_stage_survivors[0]
    if matching_stage_unique
    else None
)

candidate_surgery_indices = (
    selected_matching["surgery_indices"]
    if selected_matching is not None
    else []
)

surgery_stage_records = []

for surgery_index in candidate_surgery_indices:
    surgery = surgery_rows[surgery_index]

    circulating_vertices = sorted(
        int(value)
        for value in surgery[
            "circulating_vertices"
        ]
    )

    added_edges = triangle_edges(
        circulating_vertices
    )

    edge_records = []

    for left, right in added_edges:
        left_chart = charts[left]
        right_chart = charts[right]

        exact_inverse = (
            left_chart["inverse"]
            == right_chart["permutation"]
            and right_chart["inverse"]
            == left_chart["permutation"]
        )

        same_map = (
            left_chart["permutation"]
            == right_chart["permutation"]
        )

        same_inverse_pair = (
            left_chart["inverse_pair_id"]
            == right_chart["inverse_pair_id"]
        )

        opposite_chirality = (
            left_chart["chirality"]
            != right_chart["chirality"]
        )

        edge_records.append(
            {
                "edge": [left, right],
                "left_chirality": (
                    left_chart["chirality"]
                ),
                "right_chirality": (
                    right_chart["chirality"]
                ),
                "left_map_index": (
                    left_chart["map_index"]
                ),
                "right_map_index": (
                    right_chart["map_index"]
                ),
                "exact_inverse": exact_inverse,
                "same_map": same_map,
                "same_inverse_pair": (
                    same_inverse_pair
                ),
                "opposite_chirality": (
                    opposite_chirality
                ),
            }
        )

    chirality_profile = Counter(
        charts[vertex]["chirality"]
        for vertex in circulating_vertices
    )

    surgery_stage_records.append(
        {
            "surgery_index": surgery_index,
            "circulating_side": surgery[
                "circulating_side"
            ],
            "circulating_vertices": (
                circulating_vertices
            ),
            "rooted_vertices": sorted(
                int(value)
                for value in surgery[
                    "rooted_vertices"
                ]
            ),
            "removed_matching": surgery[
                "removed_matching"
            ],
            "added_triangle_edges": [
                list(edge)
                for edge in added_edges
            ],
            "circulating_chirality_profile": dict(
                sorted(chirality_profile.items())
            ),
            "exact_inverse_edge_count": sum(
                row["exact_inverse"]
                for row in edge_records
            ),
            "same_map_edge_count": sum(
                row["same_map"]
                for row in edge_records
            ),
            "same_inverse_pair_edge_count": sum(
                row["same_inverse_pair"]
                for row in edge_records
            ),
            "opposite_chirality_edge_count": sum(
                row["opposite_chirality"]
                for row in edge_records
            ),
            "edge_records": edge_records,
        }
    )

if surgery_stage_records:
    maximum_triangle_inverse_count = max(
        row["exact_inverse_edge_count"]
        for row in surgery_stage_records
    )
else:
    maximum_triangle_inverse_count = 0

surgery_stage_survivors = [
    row
    for row in surgery_stage_records
    if row["exact_inverse_edge_count"]
    == maximum_triangle_inverse_count
]

surgery_stage_unique = (
    len(surgery_stage_survivors) == 1
)

selected_surgery = (
    surgery_stage_survivors[0]
    if surgery_stage_unique
    else None
)

all_surgery_triangle_profiles = []

for surgery_index, surgery in sorted(
    surgery_rows.items()
):
    circulating_vertices = sorted(
        int(value)
        for value in surgery[
            "circulating_vertices"
        ]
    )

    added_edges = triangle_edges(
        circulating_vertices
    )

    inverse_count = 0

    for left, right in added_edges:
        if (
            charts[left]["inverse"]
            == charts[right]["permutation"]
            and charts[right]["inverse"]
            == charts[left]["permutation"]
        ):
            inverse_count += 1

    all_surgery_triangle_profiles.append(
        {
            "surgery_index": surgery_index,
            "circulating_side": surgery[
                "circulating_side"
            ],
            "exact_inverse_edge_count": (
                inverse_count
            ),
        }
    )

side_inverse_profile = Counter()

for row in all_surgery_triangle_profiles:
    side_inverse_profile[
        (
            row["circulating_side"],
            row["exact_inverse_edge_count"],
        )
    ] += 1

checks = {
    "chart_count_is_six": (
        len(charts) == 6
    ),
    "directed_overlap_has_two_size_two_orbits": (
        size_two_orbits == [[2, 8], [4, 9]]
    ),
    "distinguished_matching_count_is_two": (
        len(distinguished_matching_ids) == 2
    ),
    "matching_stage_is_unique": (
        matching_stage_unique
    ),
    "selected_matching_has_one_exact_inverse_edge": (
        selected_matching is not None
        and selected_matching[
            "exact_inverse_edge_count"
        ]
        == 1
    ),
    "residual_surgery_count_is_two": (
        len(surgery_stage_records) == 2
    ),
    "surgery_stage_is_unique": (
        surgery_stage_unique
    ),
    "selected_circulation_triangle_has_inverse_edge": (
        selected_surgery is not None
        and selected_surgery[
            "exact_inverse_edge_count"
        ]
        == 1
    ),
    "other_circulation_triangle_has_no_inverse_edge": (
        len(surgery_stage_records) == 2
        and sorted(
            row["exact_inverse_edge_count"]
            for row in surgery_stage_records
        )
        == [0, 1]
    ),
}

audit_pass = all(checks.values())

if audit_pass:
    verdict = (
        "layered_native_chiral_selector_uniquely_"
        "selects_polarized_surgery"
    )
else:
    verdict = (
        "layered_native_chiral_selector_does_not_"
        "uniquely_select_polarized_surgery"
    )

record = {
    "artifact_id": (
        "native_g60_chiral_polarization_selector_audit_055g"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "directed_overlap_size_two_orbits": (
        size_two_orbits
    ),
    "distinguished_matching_ids": (
        distinguished_matching_ids
    ),
    "matching_stage_records": (
        matching_stage_records
    ),
    "matching_stage_maximum_exact_inverse_count": (
        maximum_removed_inverse_count
    ),
    "matching_stage_survivor_count": (
        len(matching_stage_survivors)
    ),
    "selected_matching": selected_matching,
    "surgery_stage_records": (
        surgery_stage_records
    ),
    "surgery_stage_maximum_exact_inverse_count": (
        maximum_triangle_inverse_count
    ),
    "surgery_stage_survivor_count": (
        len(surgery_stage_survivors)
    ),
    "selected_surgery": selected_surgery,
    "all_surgery_triangle_profiles": (
        all_surgery_triangle_profiles
    ),
    "side_inverse_profile": [
        {
            "circulating_side": side,
            "exact_inverse_edge_count": count,
            "surgery_count": multiplicity,
        }
        for (
            side,
            count,
        ), multiplicity in sorted(
            side_inverse_profile.items()
        )
    ],
    "checks": checks,
    "interpretation": (
        "Directed chart-overlap symmetry first isolates two "
        "individually distinguished perfect matchings. The "
        "chart-specific chiral maps select the matching containing "
        "an exact inverse edge. Among its two side orientations, "
        "the same native chiral relation selects the circulating "
        "triangle containing an exact inverse edge."
    ),
    "boundary": {
        "layered_finite_selector_proved": audit_pass,
        "preferred_matching_selected_inside_directed_frontier": (
            matching_stage_unique
        ),
        "preferred_circulating_side_selected_inside_matching_frontier": (
            surgery_stage_unique
        ),
        "preferred_surgery_selected": audit_pass,
        "selector_rule_is_exact_inverse_chiral_compatibility": True,
        "attractor_dynamically_executes_selector_proved": False,
        "attractor_to_specific_G900_source_map_proved": False,
        "physical_chirality_proved": False,
        "physical_attraction_proved": False,
        "force_claim": False,
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
        "circulating_vertices,removed_matching,"
        "exact_inverse_edge_count,"
        "same_map_edge_count,"
        "same_inverse_pair_edge_count,"
        "opposite_chirality_edge_count\n"
    )

    for row in surgery_stage_records:
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
                        row["removed_matching"],
                        separators=(",", ":"),
                    ),
                    str(
                        row[
                            "exact_inverse_edge_count"
                        ]
                    ),
                    str(
                        row[
                            "same_map_edge_count"
                        ]
                    ),
                    str(
                        row[
                            "same_inverse_pair_edge_count"
                        ]
                    ),
                    str(
                        row[
                            "opposite_chirality_edge_count"
                        ]
                    ),
                ]
            )
            + "\n"
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print(
    "directed_overlap_size_two_orbits:",
    record[
        "directed_overlap_size_two_orbits"
    ],
)
print(
    "distinguished_matching_ids:",
    record["distinguished_matching_ids"],
)
print("matching_stage_records:")
for row in record["matching_stage_records"]:
    print(row)
print(
    "matching_stage_survivor_count:",
    record[
        "matching_stage_survivor_count"
    ],
)
print(
    "selected_matching:",
    record["selected_matching"],
)
print("surgery_stage_records:")
for row in record["surgery_stage_records"]:
    print(
        {
            key: value
            for key, value in row.items()
            if key != "edge_records"
        }
    )
print(
    "surgery_stage_survivor_count:",
    record[
        "surgery_stage_survivor_count"
    ],
)
print(
    "selected_surgery:",
    {
        key: value
        for key, value in (
            record["selected_surgery"] or {}
        ).items()
        if key != "edge_records"
    },
)
print(
    "side_inverse_profile:",
    record["side_inverse_profile"],
)
print("checks:", record["checks"])
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
