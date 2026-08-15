#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PAIR_CSV = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_axis_pair_candidates_037.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_fold_swap_atlas_054c.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_petersen_fold_swap_atlas_054c.csv"
)

NOTE_OUT = (
    ROOT
    / "notes"
    / "native_g60_petersen_fold_swap_atlas_054c.md"
)


def edge(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def build_adjacency(
    vertices: list[str],
    edges: set[tuple[str, str]],
) -> dict[str, set[str]]:
    adjacency = {
        vertex: set()
        for vertex in vertices
    }

    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    return adjacency


def canonical_cycle(
    cycle: list[str],
) -> tuple[str, ...]:
    variants = []

    for sequence in (
        cycle,
        list(reversed(cycle)),
    ):
        for shift in range(len(sequence)):
            variants.append(
                tuple(
                    sequence[shift:]
                    + sequence[:shift]
                )
            )

    return min(variants)


def find_five_cycles(
    vertices: list[str],
    adjacency: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()

    for start in vertices:
        stack = [
            (start, [start])
        ]

        while stack:
            current, path = stack.pop()

            if len(path) == 5:
                if start in adjacency[current]:
                    found.add(
                        canonical_cycle(path)
                    )
                continue

            for nxt in sorted(adjacency[current]):
                if nxt not in path:
                    stack.append(
                        (
                            nxt,
                            path + [nxt],
                        )
                    )

    return sorted(found)


def cycle_edges(
    cycle: tuple[str, ...],
) -> set[tuple[str, str]]:
    return {
        edge(
            cycle[index],
            cycle[(index + 1) % len(cycle)],
        )
        for index in range(len(cycle))
    }


def is_cycle_on_vertex_set(
    vertex_set: set[str],
    adjacency: dict[str, set[str]],
) -> bool:
    if len(vertex_set) != 5:
        return False

    induced_degrees = []

    for vertex in vertex_set:
        induced_degrees.append(
            len(
                adjacency[vertex]
                & vertex_set
            )
        )

    return induced_degrees == [2] * 5


def connected_components(
    vertices: list[str],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    unseen = set(vertices)
    components = []

    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()

        while stack:
            vertex = stack.pop()

            if vertex in component:
                continue

            component.add(vertex)
            unseen.discard(vertex)

            stack.extend(
                adjacency[vertex]
                - component
            )

        components.append(
            sorted(component)
        )

    return sorted(components)


def triangle_count(
    vertices: list[str],
    adjacency: dict[str, set[str]],
) -> int:
    count = 0

    for left_index, left in enumerate(vertices):
        for middle_index in range(
            left_index + 1,
            len(vertices),
        ):
            middle = vertices[middle_index]

            if middle not in adjacency[left]:
                continue

            for right_index in range(
                middle_index + 1,
                len(vertices),
            ):
                right = vertices[right_index]

                if (
                    right in adjacency[left]
                    and right in adjacency[middle]
                ):
                    count += 1

    return count


def constant_step_for_edges(
    size: int,
    indexed_edges: set[tuple[int, int]],
) -> int | None:
    for step in range(1, size):
        expected = {
            tuple(
                sorted(
                    (
                        index,
                        (index + step) % size,
                    )
                )
            )
            for index in range(size)
        }

        if expected == indexed_edges:
            return min(step, size - step)

    return None


def analyze_graph(
    vertices: list[str],
    edges: set[tuple[str, str]],
) -> dict[str, Any]:
    adjacency = build_adjacency(
        vertices,
        edges,
    )

    cycles = find_five_cycles(
        vertices,
        adjacency,
    )

    cycle_set = set(cycles)
    vertex_universe = set(vertices)

    complementary_pairs = []
    seen_pairs = set()

    for cycle in cycles:
        complement_vertices = (
            vertex_universe
            - set(cycle)
        )

        complement_cycles = [
            candidate
            for candidate in cycles
            if set(candidate)
            == complement_vertices
        ]

        if len(complement_cycles) != 1:
            continue

        complement_cycle = complement_cycles[0]

        pair_key = tuple(
            sorted(
                (
                    canonical_cycle(
                        list(cycle)
                    ),
                    canonical_cycle(
                        list(complement_cycle)
                    ),
                )
            )
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        first_cycle = pair_key[0]
        second_cycle = pair_key[1]

        first_set = set(first_cycle)
        second_set = set(second_cycle)

        first_internal_edges = (
            cycle_edges(first_cycle)
        )
        second_internal_edges = (
            cycle_edges(second_cycle)
        )

        cross_edges = {
            candidate
            for candidate in edges
            if (
                candidate[0] in first_set
                and candidate[1] in second_set
            )
            or (
                candidate[1] in first_set
                and candidate[0] in second_set
            )
        }

        first_cross_degrees = {
            vertex: 0
            for vertex in first_cycle
        }

        second_cross_degrees = {
            vertex: 0
            for vertex in second_cycle
        }

        matching_map = {}

        for left, right in cross_edges:
            if left in first_set:
                first_vertex = left
                second_vertex = right
            else:
                first_vertex = right
                second_vertex = left

            first_cross_degrees[
                first_vertex
            ] += 1

            second_cross_degrees[
                second_vertex
            ] += 1

            matching_map[
                first_vertex
            ] = second_vertex

        perfect_matching = (
            len(cross_edges) == 5
            and all(
                degree == 1
                for degree in (
                    first_cross_degrees.values()
                )
            )
            and all(
                degree == 1
                for degree in (
                    second_cross_degrees.values()
                )
            )
        )

        matched_second_order = tuple(
            matching_map[vertex]
            for vertex in first_cycle
        ) if perfect_matching else tuple()

        matched_second_index = {
            vertex: index
            for index, vertex in enumerate(
                matched_second_order
            )
        }

        second_edges_in_matched_indices = set()

        if perfect_matching:
            for left, right in second_internal_edges:
                second_edges_in_matched_indices.add(
                    tuple(
                        sorted(
                            (
                                matched_second_index[left],
                                matched_second_index[right],
                            )
                        )
                    )
                )

        induced_step = (
            constant_step_for_edges(
                5,
                second_edges_in_matched_indices,
            )
            if perfect_matching
            else None
        )

        first_edges_in_first_indices = {
            tuple(
                sorted(
                    (
                        index,
                        (index + 1) % 5,
                    )
                )
            )
            for index in range(5)
        }

        first_step = constant_step_for_edges(
            5,
            first_edges_in_first_indices,
        )

        pentagram_matched_face = (
            induced_step == 2
        )

        complementary_pairs.append(
            {
                "first_cycle": list(first_cycle),
                "second_cycle": list(second_cycle),
                "first_internal_edge_count": len(
                    first_internal_edges
                ),
                "second_internal_edge_count": len(
                    second_internal_edges
                ),
                "cross_edge_count": len(
                    cross_edges
                ),
                "cross_edges": [
                    list(candidate)
                    for candidate in sorted(
                        cross_edges
                    )
                ],
                "perfect_matching": (
                    perfect_matching
                ),
                "matching_map": {
                    key: value
                    for key, value in sorted(
                        matching_map.items()
                    )
                },
                "matched_second_order": list(
                    matched_second_order
                ),
                "first_face_step": first_step,
                "matched_second_face_step": (
                    induced_step
                ),
                "matched_second_face_is_pentagram": (
                    pentagram_matched_face
                ),
                "folded_dimension": (
                    5 if perfect_matching else None
                ),
                "swap_dimension": (
                    5 if perfect_matching else None
                ),
                "joint_dimension": (
                    10 if perfect_matching else None
                ),
            }
        )

    degree_profile = dict(
        sorted(
            Counter(
                len(adjacency[vertex])
                for vertex in vertices
            ).items()
        )
    )

    components = connected_components(
        vertices,
        adjacency,
    )

    all_pairs_perfect_matching = all(
        row["perfect_matching"]
        for row in complementary_pairs
    )

    all_pairs_pentagram_matched = all(
        row[
            "matched_second_face_is_pentagram"
        ]
        for row in complementary_pairs
    )

    full_atlas = (
        len(vertices) == 10
        and len(edges) == 15
        and degree_profile == {3: 10}
        and len(components) == 1
        and triangle_count(
            vertices,
            adjacency,
        ) == 0
        and len(cycles) == 12
        and len(complementary_pairs) == 6
        and all_pairs_perfect_matching
        and all_pairs_pentagram_matched
    )

    return {
        "vertex_count": len(vertices),
        "edge_count": len(edges),
        "degree_profile": degree_profile,
        "components": components,
        "triangle_count": triangle_count(
            vertices,
            adjacency,
        ),
        "five_cycle_count": len(cycles),
        "five_cycles": [
            list(cycle)
            for cycle in cycles
        ],
        "complementary_pair_count": len(
            complementary_pairs
        ),
        "all_complementary_pairs_perfect_matching": (
            all_pairs_perfect_matching
        ),
        "all_matched_second_faces_are_pentagrams": (
            all_pairs_pentagram_matched
        ),
        "complementary_pairs": (
            complementary_pairs
        ),
        "full_six_chart_fold_swap_atlas": (
            full_atlas
        ),
    }


with PAIR_CSV.open() as handle:
    pair_rows = list(
        csv.DictReader(handle)
    )

strong_rows = [
    row
    for row in pair_rows
    if row.get("overlap_class")
    == "strong_2_over_3"
]

vertices = sorted(
    {
        row["first_axis"]
        for row in strong_rows
    }
    | {
        row["second_axis"]
        for row in strong_rows
    }
)

strong_edges = {
    edge(
        row["first_axis"],
        row["second_axis"],
    )
    for row in strong_rows
}

native = analyze_graph(
    vertices,
    strong_edges,
)

deletion_controls = []

for removed_edge in sorted(
    strong_edges
):
    damaged_edges = (
        strong_edges
        - {removed_edge}
    )

    damaged = analyze_graph(
        vertices,
        damaged_edges,
    )

    deletion_controls.append(
        {
            "removed_edge": list(
                removed_edge
            ),
            "five_cycle_count": (
                damaged[
                    "five_cycle_count"
                ]
            ),
            "complementary_pair_count": (
                damaged[
                    "complementary_pair_count"
                ]
            ),
            "full_six_chart_fold_swap_atlas": (
                damaged[
                    "full_six_chart_fold_swap_atlas"
                ]
            ),
        }
    )

full_atlas_survives_deletion_count = sum(
    1
    for row in deletion_controls
    if row[
        "full_six_chart_fold_swap_atlas"
    ]
)

checks = {
    "native_vertex_count_is_10": (
        native["vertex_count"] == 10
    ),
    "native_edge_count_is_15": (
        native["edge_count"] == 15
    ),
    "native_degree_profile_is_cubic": (
        native["degree_profile"]
        == {3: 10}
    ),
    "native_graph_is_connected": (
        len(native["components"]) == 1
    ),
    "native_graph_is_triangle_free": (
        native["triangle_count"] == 0
    ),
    "native_five_cycle_count_is_12": (
        native["five_cycle_count"] == 12
    ),
    "native_complementary_pair_count_is_6": (
        native[
            "complementary_pair_count"
        ]
        == 6
    ),
    "all_pairs_have_perfect_matching": (
        native[
            "all_complementary_pairs_perfect_matching"
        ]
    ),
    "all_matched_faces_are_pentagram_step_2": (
        native[
            "all_matched_second_faces_are_pentagrams"
        ]
    ),
    "native_full_six_chart_atlas_passes": (
        native[
            "full_six_chart_fold_swap_atlas"
        ]
    ),
    "no_single_edge_deletion_preserves_full_atlas": (
        full_atlas_survives_deletion_count
        == 0
    ),
}

failed_checks = [
    key
    for key, value in checks.items()
    if not value
]

audit_pass = len(failed_checks) == 0

if audit_pass:
    verdict = (
        "native_strong_overlap_graph_is_six_chart_"
        "petersen_fold_swap_atlas"
    )
else:
    verdict = (
        "native_strong_overlap_graph_does_not_realize_"
        "proposed_fold_swap_atlas"
    )

record = {
    "artifact_id": (
        "native_g60_petersen_fold_swap_atlas_054c"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "source": str(PAIR_CSV),
    "native_graph": native,
    "deletion_controls": {
        "control_count": len(
            deletion_controls
        ),
        "full_atlas_survives_deletion_count": (
            full_atlas_survives_deletion_count
        ),
        "rows": deletion_controls,
    },
    "checks": checks,
    "failed_checks": failed_checks,
    "interpretation": (
        "The native ten-axis strong-overlap relation "
        "contains twelve five-cycles arranged into six "
        "complementary pairs. Every pair is joined by a "
        "five-edge perfect matching. When one cycle is "
        "used as the sector order, the matched ordering "
        "on its complementary face carries step 2, the "
        "pentagram traversal. Each chart therefore gives "
        "five folded directions and five swap directions."
    ),
    "boundary": {
        "native_strong_overlap_geometry_used": True,
        "local_fold_swap_atlas_proved": audit_pass,
        "e5_kernel_identification_proved": False,
        "global_g60_closure_proved": False,
        "physical_electron_claim": False,
        "requires_053_for_e5_bridge": True,
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

NOTE_OUT.parent.mkdir(
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

csv_lines = [
    (
        "chart_index,first_cycle,second_cycle,"
        "cross_edge_count,perfect_matching,"
        "first_face_step,matched_second_face_step,"
        "matched_second_face_is_pentagram,"
        "folded_dimension,swap_dimension,joint_dimension"
    )
]

for index, row in enumerate(
    native["complementary_pairs"],
    start=1,
):
    csv_lines.append(
        ",".join(
            [
                str(index),
                "-".join(
                    row["first_cycle"]
                ),
                "-".join(
                    row["second_cycle"]
                ),
                str(row["cross_edge_count"]),
                str(
                    row["perfect_matching"]
                ).lower(),
                str(
                    row["first_face_step"]
                ),
                str(
                    row[
                        "matched_second_face_step"
                    ]
                ),
                str(
                    row[
                        "matched_second_face_is_pentagram"
                    ]
                ).lower(),
                str(
                    row["folded_dimension"]
                ),
                str(
                    row["swap_dimension"]
                ),
                str(
                    row["joint_dimension"]
                ),
            ]
        )
    )

CSV_OUT.write_text(
    "\n".join(csv_lines)
    + "\n"
)

note_lines = [
    "# Native G60 Petersen fold-swap atlas 054c",
    "",
    f"- audit_pass: `{str(audit_pass).lower()}`",
    f"- verdict: `{verdict}`",
    (
        "- vertex count: "
        f"`{native['vertex_count']}`"
    ),
    (
        "- edge count: "
        f"`{native['edge_count']}`"
    ),
    (
        "- five-cycle count: "
        f"`{native['five_cycle_count']}`"
    ),
    (
        "- complementary pair count: "
        f"`{native['complementary_pair_count']}`"
    ),
    (
        "- all cross relations are perfect matchings: "
        f"`{str(native['all_complementary_pairs_perfect_matching']).lower()}`"
    ),
    (
        "- all matched second faces use pentagram step 2: "
        f"`{str(native['all_matched_second_faces_are_pentagrams']).lower()}`"
    ),
    (
        "- single-edge deletions preserving full atlas: "
        f"`{full_atlas_survives_deletion_count}`"
    ),
    "",
    "## Interpretation",
    "",
    (
        "The native strong-overlap graph supplies six "
        "complementary five-cycle charts. Each chart has "
        "two five-sector faces joined by a perfect matching, "
        "and the matched second face is a pentagram traversal "
        "relative to the first."
    ),
    "",
    "## Boundary",
    "",
    (
        "This proves a native local fold-swap atlas inside "
        "the ten-axis strong-overlap geometry. It does not "
        "yet identify the E5 kernel with this atlas or prove "
        "global G60 closure."
    ),
    "",
]

NOTE_OUT.write_text(
    "\n".join(note_lines)
)

print(
    "artifact_id:",
    record["artifact_id"],
)
print(
    "audit_pass:",
    record["audit_pass"],
)
print(
    "verdict:",
    record["verdict"],
)
print(
    "vertex_count:",
    native["vertex_count"],
)
print(
    "edge_count:",
    native["edge_count"],
)
print(
    "degree_profile:",
    native["degree_profile"],
)
print(
    "triangle_count:",
    native["triangle_count"],
)
print(
    "five_cycle_count:",
    native["five_cycle_count"],
)
print(
    "complementary_pair_count:",
    native["complementary_pair_count"],
)
print(
    "all_pairs_perfect_matching:",
    native[
        "all_complementary_pairs_perfect_matching"
    ],
)
print(
    "all_matched_faces_pentagram:",
    native[
        "all_matched_second_faces_are_pentagrams"
    ],
)
print(
    "single_edge_deletion_full_atlas_survivors:",
    full_atlas_survives_deletion_count,
)
print(
    "failed_check_count:",
    len(failed_checks),
)
print(
    "json:",
    JSON_OUT,
)
print(
    "csv:",
    CSV_OUT,
)
print(
    "note:",
    NOTE_OUT,
)
