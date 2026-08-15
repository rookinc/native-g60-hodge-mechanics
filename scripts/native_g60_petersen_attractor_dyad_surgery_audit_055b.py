#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BRIDGE_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_chart_dyad_bridge_search_055a.v1.json"
)

QUOTIENT_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g900_attractor_nine_cell_carrier_quotient_054j.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_attractor_dyad_surgery_audit_055b.v1.json"
)


def normalized_edge(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((left, right)))


def adjacency(
    vertices: set[int],
    edges: set[tuple[int, int]],
) -> dict[int, set[int]]:
    result = {vertex: set() for vertex in vertices}

    for left, right in edges:
        result[left].add(right)
        result[right].add(left)

    return result


def bipartition(
    vertices: set[int],
    edges: set[tuple[int, int]],
) -> tuple[set[int], set[int]]:
    graph = adjacency(vertices, edges)
    color: dict[int, int] = {}

    for start in sorted(vertices):
        if start in color:
            continue

        color[start] = 0
        queue = deque([start])

        while queue:
            vertex = queue.popleft()

            for neighbor in graph[vertex]:
                expected = 1 - color[vertex]

                if neighbor in color:
                    if color[neighbor] != expected:
                        raise RuntimeError("graph is not bipartite")
                    continue

                color[neighbor] = expected
                queue.append(neighbor)

    left = {
        vertex
        for vertex, value in color.items()
        if value == 0
    }

    right = vertices - left
    return left, right


def degree_sequence(
    vertices: set[int],
    edges: set[tuple[int, int]],
) -> list[int]:
    graph = adjacency(vertices, edges)

    return sorted(
        len(graph[vertex])
        for vertex in vertices
    )


def triangle_count(
    vertices: set[int],
    edges: set[tuple[int, int]],
) -> int:
    graph = adjacency(vertices, edges)
    count = 0

    for triple in itertools.combinations(sorted(vertices), 3):
        left, middle, right = triple

        if (
            middle in graph[left]
            and right in graph[left]
            and right in graph[middle]
        ):
            count += 1

    return count


def is_target_shape(
    vertices: set[int],
    edges: set[tuple[int, int]],
) -> bool:
    return (
        len(edges) == 9
        and degree_sequence(vertices, edges)
        == [2, 2, 2, 4, 4, 4]
        and triangle_count(vertices, edges) == 4
    )


bridge = json.loads(BRIDGE_JSON.read_text())
quotient = json.loads(QUOTIENT_JSON.read_text())

near_hits = bridge["best_near_hits"]

source_hit = next(
    row
    for row in near_hits
    if row["predicate"] == "ff_edge_overlap==2"
)

source_edges = {
    normalized_edge(*edge)
    for edge in source_hit["edges"]
}

vertices = set(range(6))
left, right = bipartition(vertices, source_edges)

if len(left) != 3 or len(right) != 3:
    raise RuntimeError("expected a 3+3 bipartition")

complete_cross_edges = {
    normalized_edge(left_vertex, right_vertex)
    for left_vertex in left
    for right_vertex in right
}

source_is_k33 = source_edges == complete_cross_edges

surgeries = []

for circulating_side_name, circulating, rooted in (
    ("left", left, right),
    ("right", right, left),
):
    circulating_order = sorted(circulating)
    rooted_order = sorted(rooted)

    for permutation in itertools.permutations(circulating_order):
        matching = {
            normalized_edge(
                rooted_vertex,
                circulating_vertex,
            )
            for rooted_vertex, circulating_vertex
            in zip(rooted_order, permutation)
        }

        triangle = {
            normalized_edge(left_vertex, right_vertex)
            for left_vertex, right_vertex
            in itertools.combinations(circulating_order, 2)
        }

        result_edges = (
            source_edges
            - matching
        ) | triangle

        surgeries.append(
            {
                "circulating_side": circulating_side_name,
                "rooted_vertices": rooted_order,
                "circulating_vertices": circulating_order,
                "removed_matching": [
                    list(edge)
                    for edge in sorted(matching)
                ],
                "added_triangle": [
                    list(edge)
                    for edge in sorted(triangle)
                ],
                "result_edges": [
                    list(edge)
                    for edge in sorted(result_edges)
                ],
                "target_shape": is_target_shape(
                    vertices,
                    result_edges,
                ),
                "degree_sequence": degree_sequence(
                    vertices,
                    result_edges,
                ),
                "triangle_count": triangle_count(
                    vertices,
                    result_edges,
                ),
            }
        )

target_shape_count = sum(
    1
    for row in surgeries
    if row["target_shape"]
)

distinct_result_graphs = {
    tuple(
        tuple(edge)
        for edge in row["result_edges"]
    )
    for row in surgeries
}

checks = {
    "source_has_six_vertices": len(vertices) == 6,
    "source_has_nine_edges": len(source_edges) == 9,
    "source_degree_sequence_is_cubic": (
        degree_sequence(vertices, source_edges)
        == [3, 3, 3, 3, 3, 3]
    ),
    "source_is_triangle_free": (
        triangle_count(vertices, source_edges) == 0
    ),
    "source_is_K33": source_is_k33,
    "bipartition_is_three_plus_three": (
        len(left) == 3 and len(right) == 3
    ),
    "surgery_candidate_count_is_12": (
        len(surgeries) == 12
    ),
    "all_surgeries_have_target_shape": (
        target_shape_count == 12
    ),
    "all_surgeries_are_distinct": (
        len(distinct_result_graphs) == 12
    ),
}

audit_pass = all(checks.values())

record = {
    "artifact_id": (
        "native_g60_petersen_attractor_dyad_surgery_audit_055b"
    ),
    "audit_pass": audit_pass,
    "verdict": (
        "petersen_chart_K33_is_exact_unpolarized_"
        "precursor_of_attractor_root_cycle_dyad"
        if audit_pass
        else
        "petersen_to_attractor_dyad_surgery_not_established"
    ),
    "source_predicate": source_hit["predicate"],
    "source_edges": [
        list(edge)
        for edge in sorted(source_edges)
    ],
    "source_bipartition": {
        "left": sorted(left),
        "right": sorted(right),
    },
    "source_is_K33": source_is_k33,
    "surgery_candidate_count": len(surgeries),
    "target_shape_count": target_shape_count,
    "distinct_result_graph_count": len(distinct_result_graphs),
    "surgeries": surgeries,
    "checks": checks,
    "interpretation": (
        "The undirected Petersen chart-overlap graph is K3,3. "
        "Each rooted/circulating attractor dyad shape is obtained "
        "by choosing one bipartition side as circulating, removing "
        "one perfect matching, and inserting the triangle on the "
        "circulating side."
    ),
    "boundary": {
        "unpolarized_dyad_precursor_proved": audit_pass,
        "unique_circulating_side_selected": False,
        "unique_removed_matching_selected": False,
        "chirality_selection_proved": False,
        "chart_to_attractor_cell_bijection_proved": False,
        "physical_attraction_proved": False,
        "force_claim": False,
    },
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)

JSON_OUT.write_text(
    json.dumps(
        record,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("source_is_K33:", record["source_is_K33"])
print("source_bipartition:", record["source_bipartition"])
print(
    "surgery_candidate_count:",
    record["surgery_candidate_count"],
)
print(
    "target_shape_count:",
    record["target_shape_count"],
)
print(
    "distinct_result_graph_count:",
    record["distinct_result_graph_count"],
)
print("checks:", record["checks"])
print("json:", JSON_OUT)
