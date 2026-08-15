#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

ATLAS_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_fold_swap_atlas_054c.v1.json"
)

QUOTIENT_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g900_attractor_nine_cell_carrier_quotient_054j.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_chart_dyad_bridge_search_055a.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_petersen_chart_dyad_bridge_search_055a.csv"
)


def normalize_cycle(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, list):
        return None

    if len(value) != 5:
        return None

    try:
        if len(set(value)) != 5:
            return None
    except TypeError:
        return None

    return tuple(value)


def edge_set(
    cycle: tuple[Any, ...],
) -> frozenset[tuple[Any, Any]]:
    edges = []

    for index in range(5):
        left = cycle[index]
        right = cycle[(index + 1) % 5]

        edges.append(
            tuple(
                sorted(
                    (left, right),
                    key=str,
                )
            )
        )

    return frozenset(edges)


def directed_edges(
    cycle: tuple[Any, ...],
) -> frozenset[tuple[Any, Any]]:
    return frozenset(
        (
            cycle[index],
            cycle[(index + 1) % 5],
        )
        for index in range(5)
    )


def extract_chart_cycles(
    chart: dict[str, Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    first_candidates = [
        "first_cycle",
        "cycle_1",
        "cycle_a",
        "face_1",
        "first_face",
    ]

    second_candidates = [
        "matched_second_order",
        "second_cycle",
        "cycle_2",
        "cycle_b",
        "face_2",
        "second_face",
    ]

    first = None
    second = None

    for key in first_candidates:
        candidate = normalize_cycle(chart.get(key))

        if candidate is not None:
            first = candidate
            break

    for key in second_candidates:
        candidate = normalize_cycle(chart.get(key))

        if candidate is not None:
            second = candidate
            break

    if first is None or second is None:
        available = sorted(chart)

        raise RuntimeError(
            "could not extract both chart cycles; "
            f"available keys: {available}"
        )

    return first, second


def graph_fingerprint(
    vertex_count: int,
    edges: set[tuple[int, int]],
) -> dict[str, Any]:
    adjacency = {
        vertex: set()
        for vertex in range(vertex_count)
    }

    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    degree_sequence = sorted(
        len(adjacency[vertex])
        for vertex in adjacency
    )

    triangles = []

    for triple in itertools.combinations(
        range(vertex_count),
        3,
    ):
        left, middle, right = triple

        if (
            middle in adjacency[left]
            and right in adjacency[left]
            and right in adjacency[middle]
        ):
            triangles.append(list(triple))

    degree_two = sorted(
        vertex
        for vertex in adjacency
        if len(adjacency[vertex]) == 2
    )

    degree_four = sorted(
        vertex
        for vertex in adjacency
        if len(adjacency[vertex]) == 4
    )

    degree_two_internal_edges = sum(
        1
        for left, right in edges
        if left in degree_two
        and right in degree_two
    )

    degree_four_internal_edges = sum(
        1
        for left, right in edges
        if left in degree_four
        and right in degree_four
    )

    cross_edges = sum(
        1
        for left, right in edges
        if (
            left in degree_two
            and right in degree_four
        )
        or (
            right in degree_two
            and left in degree_four
        )
    )

    return {
        "vertex_count": vertex_count,
        "edge_count": len(edges),
        "degree_sequence": degree_sequence,
        "triangle_count": len(triangles),
        "triangles": triangles,
        "degree_two_vertices": degree_two,
        "degree_four_vertices": degree_four,
        "degree_two_internal_edge_count": (
            degree_two_internal_edges
        ),
        "degree_four_internal_edge_count": (
            degree_four_internal_edges
        ),
        "cross_edge_count": cross_edges,
    }


def is_target_dyad(
    fingerprint: dict[str, Any],
) -> bool:
    return (
        fingerprint["vertex_count"] == 6
        and fingerprint["edge_count"] == 9
        and fingerprint["degree_sequence"]
        == [2, 2, 2, 4, 4, 4]
        and fingerprint["triangle_count"] == 4
        and fingerprint[
            "degree_two_internal_edge_count"
        ]
        == 0
        and fingerprint[
            "degree_four_internal_edge_count"
        ]
        == 3
        and fingerprint["cross_edge_count"] == 6
    )


atlas = json.loads(
    ATLAS_JSON.read_text()
)

quotient = json.loads(
    QUOTIENT_JSON.read_text()
)

charts_raw = (
    atlas
    .get("native_graph", {})
    .get("complementary_pairs", [])
)

if len(charts_raw) != 6:
    raise RuntimeError(
        "expected six complementary Petersen charts, "
        f"found {len(charts_raw)}"
    )

charts = []

for chart_index, chart in enumerate(charts_raw):
    first_cycle, second_cycle = extract_chart_cycles(chart)

    charts.append(
        {
            "chart_index": chart_index,
            "first_cycle": first_cycle,
            "second_cycle": second_cycle,
            "first_edges": edge_set(first_cycle),
            "second_edges": edge_set(second_cycle),
            "first_directed": directed_edges(first_cycle),
            "second_directed": directed_edges(second_cycle),
        }
    )

pair_records = []

for left_index, right_index in itertools.combinations(
    range(6),
    2,
):
    left = charts[left_index]
    right = charts[right_index]

    features = {
        "ff_edge_overlap": len(
            left["first_edges"]
            & right["first_edges"]
        ),
        "fs_edge_overlap": len(
            left["first_edges"]
            & right["second_edges"]
        ),
        "sf_edge_overlap": len(
            left["second_edges"]
            & right["first_edges"]
        ),
        "ss_edge_overlap": len(
            left["second_edges"]
            & right["second_edges"]
        ),
        "ff_directed_agree": len(
            left["first_directed"]
            & right["first_directed"]
        ),
        "fs_directed_agree": len(
            left["first_directed"]
            & right["second_directed"]
        ),
        "sf_directed_agree": len(
            left["second_directed"]
            & right["first_directed"]
        ),
        "ss_directed_agree": len(
            left["second_directed"]
            & right["second_directed"]
        ),
        "ff_directed_reverse": len(
            left["first_directed"]
            & frozenset(
                (right_endpoint, left_endpoint)
                for left_endpoint, right_endpoint
                in right["first_directed"]
            )
        ),
        "ss_directed_reverse": len(
            left["second_directed"]
            & frozenset(
                (right_endpoint, left_endpoint)
                for left_endpoint, right_endpoint
                in right["second_directed"]
            )
        ),
    }

    pair_records.append(
        {
            "left": left_index,
            "right": right_index,
            "features": features,
        }
    )

feature_values = {}

for feature_name in sorted(
    pair_records[0]["features"]
):
    feature_values[feature_name] = sorted(
        {
            row["features"][feature_name]
            for row in pair_records
        }
    )

atomic_predicates = []

for feature_name, values in feature_values.items():
    for value in values:
        atomic_predicates.append(
            {
                "name": f"{feature_name}=={value}",
                "tests": [(feature_name, value)],
            }
        )

candidate_predicates = list(atomic_predicates)

for first_index, second_index in itertools.combinations(
    range(len(atomic_predicates)),
    2,
):
    first = atomic_predicates[first_index]
    second = atomic_predicates[second_index]

    if (
        first["tests"][0][0]
        == second["tests"][0][0]
    ):
        continue

    candidate_predicates.append(
        {
            "name": (
                first["name"]
                + " AND "
                + second["name"]
            ),
            "tests": (
                first["tests"]
                + second["tests"]
            ),
        }
    )

search_records = []
exact_hits = []

for predicate in candidate_predicates:
    edges = set()

    for pair in pair_records:
        passes = all(
            pair["features"][feature_name] == value
            for feature_name, value
            in predicate["tests"]
        )

        if passes:
            edges.add(
                (
                    pair["left"],
                    pair["right"],
                )
            )

    fingerprint = graph_fingerprint(
        6,
        edges,
    )

    hit = is_target_dyad(fingerprint)

    row = {
        "predicate": predicate["name"],
        "edge_count": fingerprint["edge_count"],
        "degree_sequence": (
            fingerprint["degree_sequence"]
        ),
        "triangle_count": (
            fingerprint["triangle_count"]
        ),
        "degree_two_vertices": (
            fingerprint["degree_two_vertices"]
        ),
        "degree_four_vertices": (
            fingerprint["degree_four_vertices"]
        ),
        "cross_edge_count": (
            fingerprint["cross_edge_count"]
        ),
        "exact_target_hit": hit,
        "edges": [
            list(edge)
            for edge in sorted(edges)
        ],
    }

    search_records.append(row)

    if hit:
        exact_hits.append(row)

pair_feature_profile = {
    feature_name: dict(
        sorted(
            Counter(
                row["features"][feature_name]
                for row in pair_records
            ).items()
        )
    )
    for feature_name in sorted(feature_values)
}

checks = {
    "chart_count_is_six": len(charts) == 6,
    "chart_pair_count_is_fifteen": (
        len(pair_records) == 15
    ),
    "target_quotient_has_six_doubletons": (
        sum(
            1
            for cell in quotient.get("cells", [])
            if cell.get("orbit_size") == 2
        )
        == 6
    ),
    "candidate_predicate_count_positive": (
        len(candidate_predicates) > 0
    ),
}

audit_pass = all(checks.values())

if exact_hits:
    verdict = (
        "petersen_chart_overlap_family_contains_"
        "exact_rooted_circulating_dyad_candidate"
    )
else:
    verdict = (
        "petersen_chart_overlap_family_does_not_"
        "recover_attractor_dyad_in_tested_family"
    )

record = {
    "artifact_id": (
        "native_g60_petersen_chart_dyad_bridge_search_055a"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "chart_count": len(charts),
    "chart_pair_count": len(pair_records),
    "feature_names": sorted(feature_values),
    "pair_feature_profile": pair_feature_profile,
    "candidate_predicate_count": (
        len(candidate_predicates)
    ),
    "exact_hit_count": len(exact_hits),
    "exact_hits": exact_hits,
    "best_near_hits": sorted(
        search_records,
        key=lambda row: (
            abs(row["edge_count"] - 9),
            abs(
                row["triangle_count"] - 1
            ),
            sum(
                abs(left - right)
                for left, right in zip(
                    row["degree_sequence"],
                    [2, 2, 2, 4, 4, 4],
                )
            )
            if len(row["degree_sequence"]) == 6
            else 999,
            row["predicate"],
        ),
    )[:20],
    "chart_pairs": [
        {
            "left": row["left"],
            "right": row["right"],
            **row["features"],
        }
        for row in pair_records
    ],
    "checks": checks,
    "boundary": {
        "six_chart_count_matches_six_doubletons": True,
        "cardinality_match_is_not_bridge": True,
        "tested_overlap_family_only": True,
        "native_chart_to_attractor_cell_bijection_proved": (
            len(exact_hits) > 0
        ),
        "signed_overlap_bridge_tested": False,
        "face_selection_proved": False,
        "chirality_selection_proved": False,
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
        "left,right,"
        + ",".join(sorted(feature_values))
        + "\n"
    )

    for row in pair_records:
        handle.write(
            ",".join(
                [
                    str(row["left"]),
                    str(row["right"]),
                    *[
                        str(
                            row["features"][
                                feature_name
                            ]
                        )
                        for feature_name
                        in sorted(feature_values)
                    ],
                ]
            )
            + "\n"
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("chart_count:", record["chart_count"])
print(
    "chart_pair_count:",
    record["chart_pair_count"],
)
print(
    "candidate_predicate_count:",
    record["candidate_predicate_count"],
)
print(
    "exact_hit_count:",
    record["exact_hit_count"],
)
print("pair_feature_profile:")
for key, value in record["pair_feature_profile"].items():
    print(key, value)

print("exact_hits:")
for row in record["exact_hits"]:
    print(row)

print("best_near_hits:")
for row in record["best_near_hits"][:10]:
    print(row)

print("json:", JSON_OUT)
print("csv:", CSV_OUT)
