#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SELECTOR_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_polarization_selector_audit_055g.v1.json"
)

SURGERY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_attractor_dyad_surgery_audit_055b.v1.json"
)

CHART_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chart_specific_chiral_map_derivation_055e.v1.json"
)

OVERLAP_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_chart_dyad_bridge_search_055a.v1.json"
)

G900_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g900_attractor_nine_cell_carrier_quotient_054j.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_g900_polarized_dyad_bridge_census_055h.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_g900_polarized_dyad_bridge_census_055h.csv"
)


def edge(left: int, right: int) -> tuple[int, int]:
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
) -> set[tuple[int, int]]:
    return {
        edge(left, right)
        for left, right in itertools.combinations(
            sorted(vertices),
            2,
        )
    }


selector = json.loads(SELECTOR_JSON.read_text())
surgery_source = json.loads(SURGERY_JSON.read_text())
chart_source = json.loads(CHART_JSON.read_text())
overlap_source = json.loads(OVERLAP_JSON.read_text())
g900_source = json.loads(G900_JSON.read_text())

selected = selector["selected_surgery"]

if selected is None:
    raise RuntimeError("055g has no selected surgery")

selected_index = int(selected["surgery_index"])
selected_surgery = surgery_source["surgeries"][selected_index]

g60_circulating = sorted(
    int(value)
    for value in selected_surgery["circulating_vertices"]
)

g60_rooted = sorted(
    int(value)
    for value in selected_surgery["rooted_vertices"]
)

g60_edges = {
    edge(*value)
    for value in selected_surgery["result_edges"]
}

g60_removed_matching = {
    edge(*value)
    for value in selected_surgery["removed_matching"]
}

g60_added_triangle = triangle_edges(g60_circulating)

charts: dict[int, dict[str, Any]] = {}

for row in chart_source["chart_records"]:
    permutation = tuple(
        int(value)
        for value in row["derived_permutation"]
    )

    charts[int(row["chart_index"])] = {
        "chirality": row["chirality"],
        "map_index": int(row["map_index"]),
        "inverse_pair_id": int(row["inverse_pair_id"]),
        "permutation": permutation,
        "inverse": inverse(permutation),
        "dyad_side": row["dyad_side"],
    }

overlap_features = {}

for row in overlap_source["chart_pairs"]:
    pair = edge(
        int(row["left"]),
        int(row["right"]),
    )

    overlap_features[pair] = {
        key: int(value)
        for key, value in row.items()
        if key not in {"left", "right"}
    }

doubleton_cells = [
    row
    for row in g900_source["cells"]
    if int(row["orbit_size"]) == 2
]

g900_cells = sorted(
    int(row["cell_id"])
    for row in doubleton_cells
)

g900_edges = set()
g900_edge_law = {}

for row in g900_source["quotient_edges"]:
    left = int(row["cell_u"])
    right = int(row["cell_v"])

    if left not in g900_cells or right not in g900_cells:
        continue

    if left == right:
        continue

    pair = edge(left, right)
    g900_edges.add(pair)

    identity_weight = int(row["identity_weight"])
    half_flip_weight = int(row["half_flip_weight"])

    if identity_weight > 0 and half_flip_weight == 0:
        law = "identity"
    elif half_flip_weight > 0 and identity_weight == 0:
        law = "half_flip"
    else:
        law = "mixed"

    g900_edge_law[pair] = law

g900_loop_cells = {
    int(row["cell_u"])
    for row in g900_source["quotient_edges"]
    if (
        int(row["cell_u"]) == int(row["cell_v"])
        and int(row["cell_u"]) in g900_cells
    )
}

g900_circulating = sorted(
    cell
    for cell in g900_cells
    if all(
        edge(cell, other) in g900_edges
        for other in g900_cells
        if other != cell
        and other not in g900_loop_cells
    )
)

# Derive the classes structurally rather than trusting names:
# rooted cells are exactly the looped doubleton cells.
g900_rooted = sorted(g900_loop_cells)

# The remaining three cells form the circulating triangle.
g900_circulating = sorted(
    set(g900_cells) - set(g900_rooted)
)

if len(g900_rooted) != 3 or len(g900_circulating) != 3:
    raise RuntimeError(
        "could not derive the G900 rooted/circulating 3+3 split"
    )


def g60_edge_features(
    left: int,
    right: int,
) -> dict[str, int]:
    left_chart = charts[left]
    right_chart = charts[right]

    pair = edge(left, right)
    overlap = overlap_features[pair]

    exact_inverse = int(
        left_chart["inverse"]
        == right_chart["permutation"]
        and right_chart["inverse"]
        == left_chart["permutation"]
    )

    same_map = int(
        left_chart["permutation"]
        == right_chart["permutation"]
    )

    same_inverse_pair = int(
        left_chart["inverse_pair_id"]
        == right_chart["inverse_pair_id"]
    )

    opposite_chirality = int(
        left_chart["chirality"]
        != right_chart["chirality"]
    )

    same_chirality = 1 - opposite_chirality

    return {
        **overlap,
        "exact_inverse": exact_inverse,
        "same_map": same_map,
        "same_inverse_pair": same_inverse_pair,
        "opposite_chirality": opposite_chirality,
        "same_chirality": same_chirality,
        "is_added_triangle_edge": int(
            pair in g60_added_triangle
        ),
        "is_surviving_cross_edge": int(
            pair not in g60_added_triangle
        ),
    }


g60_feature_table = {
    pair: g60_edge_features(*pair)
    for pair in sorted(g60_edges)
}

class_preserving_bijections = []
graph_isomorphisms = []

for circulating_image in itertools.permutations(
    g900_circulating
):
    for rooted_image in itertools.permutations(
        g900_rooted
    ):
        mapping = {
            source: target
            for source, target in zip(
                g60_circulating,
                circulating_image,
            )
        }

        mapping.update(
            {
                source: target
                for source, target in zip(
                    g60_rooted,
                    rooted_image,
                )
            }
        )

        class_preserving_bijections.append(mapping)

        mapped_edges = {
            edge(mapping[left], mapping[right])
            for left, right in g60_edges
        }

        if mapped_edges == g900_edges:
            graph_isomorphisms.append(mapping)

if not graph_isomorphisms:
    raise RuntimeError(
        "selected G60 polarization is not class-preserving "
        "isomorphic to the G900 six-cell quotient"
    )

feature_names = sorted(
    next(iter(g60_feature_table.values()))
)

feature_value_support = {
    name: sorted(
        {
            features[name]
            for features in g60_feature_table.values()
        }
    )
    for name in feature_names
}

atomic_predicates = []

for feature_name, values in feature_value_support.items():
    for value in values:
        atomic_predicates.append(
            {
                "name": f"{feature_name}=={value}",
                "tests": [(feature_name, value)],
            }
        )

candidate_predicates = list(atomic_predicates)

for left_index, right_index in itertools.combinations(
    range(len(atomic_predicates)),
    2,
):
    left = atomic_predicates[left_index]
    right = atomic_predicates[right_index]

    if left["tests"][0][0] == right["tests"][0][0]:
        continue

    candidate_predicates.append(
        {
            "name": (
                left["name"]
                + " AND "
                + right["name"]
            ),
            "tests": left["tests"] + right["tests"],
        }
    )

bridge_records = []
exact_law_hits = []

for isomorphism_index, mapping in enumerate(
    graph_isomorphisms,
    start=1,
):
    inverse_mapping = {
        target: source
        for source, target in mapping.items()
    }

    edge_rows = []

    for g900_pair in sorted(g900_edges):
        g60_pair = edge(
            inverse_mapping[g900_pair[0]],
            inverse_mapping[g900_pair[1]],
        )

        edge_rows.append(
            {
                "g900_edge": list(g900_pair),
                "g900_law": g900_edge_law[g900_pair],
                "g60_edge": list(g60_pair),
                "g60_features": g60_feature_table[g60_pair],
            }
        )

    predicate_hits = []

    for predicate in candidate_predicates:
        for true_law in ("identity", "half_flip"):
            predicted = {}

            for row in edge_rows:
                passes = all(
                    row["g60_features"][feature_name] == value
                    for feature_name, value
                    in predicate["tests"]
                )

                predicted[tuple(row["g900_edge"])] = (
                    true_law
                    if passes
                    else (
                        "half_flip"
                        if true_law == "identity"
                        else "identity"
                    )
                )

            exact = all(
                predicted[tuple(row["g900_edge"])]
                == row["g900_law"]
                for row in edge_rows
            )

            if exact:
                hit = {
                    "isomorphism_index": isomorphism_index,
                    "predicate": predicate["name"],
                    "predicate_true_law": true_law,
                }

                predicate_hits.append(hit)
                exact_law_hits.append(hit)

    bridge_records.append(
        {
            "isomorphism_index": isomorphism_index,
            "mapping": {
                str(key): value
                for key, value in sorted(mapping.items())
            },
            "edge_rows": edge_rows,
            "exact_law_hit_count": len(predicate_hits),
            "exact_law_hits": predicate_hits,
        }
    )

isomorphism_law_hit_profile = Counter(
    row["exact_law_hit_count"]
    for row in bridge_records
)

law_compatible_isomorphisms = [
    row
    for row in bridge_records
    if row["exact_law_hit_count"] > 0
]

checks = {
    "G60_selected_surgery_is_2": (
        selected_index == 2
    ),
    "G60_rooted_count_is_3": (
        len(g60_rooted) == 3
    ),
    "G60_circulating_count_is_3": (
        len(g60_circulating) == 3
    ),
    "G900_doubleton_cell_count_is_6": (
        len(g900_cells) == 6
    ),
    "G900_rooted_count_is_3": (
        len(g900_rooted) == 3
    ),
    "G900_circulating_count_is_3": (
        len(g900_circulating) == 3
    ),
    "both_six_cell_graphs_have_9_edges": (
        len(g60_edges) == 9
        and len(g900_edges) == 9
    ),
    "class_preserving_bijection_count_is_36": (
        len(class_preserving_bijections) == 36
    ),
    "class_preserving_isomorphism_count_is_6": (
        len(graph_isomorphisms) == 6
    ),
    "all_G900_edges_have_pure_binary_law": all(
        law in {"identity", "half_flip"}
        for law in g900_edge_law.values()
    ),
}

audit_pass = all(checks.values())

if exact_law_hits:
    verdict = (
        "selected_G60_polarization_is_G900_six_cell_"
        "quotient_shape_with_candidate_law_bridge"
    )
else:
    verdict = (
        "selected_G60_polarization_is_G900_six_cell_"
        "quotient_shape_but_simple_law_bridge_not_found"
    )

record = {
    "artifact_id": (
        "native_g60_g900_polarized_dyad_bridge_census_055h"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "selected_G60_surgery_index": selected_index,
    "G60_rooted": g60_rooted,
    "G60_circulating": g60_circulating,
    "G60_removed_matching": [
        list(value)
        for value in sorted(g60_removed_matching)
    ],
    "G60_edge_count": len(g60_edges),
    "G900_rooted": g900_rooted,
    "G900_circulating": g900_circulating,
    "G900_edge_count": len(g900_edges),
    "G900_edge_law_profile": dict(
        sorted(Counter(g900_edge_law.values()).items())
    ),
    "class_preserving_bijection_count": (
        len(class_preserving_bijections)
    ),
    "class_preserving_isomorphism_count": (
        len(graph_isomorphisms)
    ),
    "bridge_records": bridge_records,
    "candidate_predicate_count": (
        len(candidate_predicates)
    ),
    "exact_law_hit_count": len(exact_law_hits),
    "exact_law_hits": exact_law_hits,
    "law_compatible_isomorphism_count": (
        len(law_compatible_isomorphisms)
    ),
    "isomorphism_law_hit_profile": {
        str(key): value
        for key, value in sorted(
            isomorphism_law_hit_profile.items()
        )
    },
    "checks": checks,
    "boundary": {
        "class_preserving_six_cell_isomorphism_proved": (
            audit_pass
        ),
        "unique_unlabeled_bijection_proved": False,
        "simple_G60_feature_to_G900_law_bridge_found": (
            len(exact_law_hits) > 0
        ),
        "unique_labeled_bridge_proved": (
            len(law_compatible_isomorphisms) == 1
        ),
        "attractor_dynamically_executes_selector_proved": False,
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
        "isomorphism_index,G60_chart,G900_cell,"
        "exact_law_hit_count\n"
    )

    for row in bridge_records:
        for chart, cell in sorted(
            (
                int(key),
                int(value),
            )
            for key, value in row["mapping"].items()
        ):
            handle.write(
                f"{row['isomorphism_index']},"
                f"{chart},"
                f"{cell},"
                f"{row['exact_law_hit_count']}\n"
            )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print(
    "selected_G60_surgery_index:",
    record["selected_G60_surgery_index"],
)
print("G60_rooted:", record["G60_rooted"])
print("G60_circulating:", record["G60_circulating"])
print("G900_rooted:", record["G900_rooted"])
print(
    "G900_circulating:",
    record["G900_circulating"],
)
print(
    "G900_edge_law_profile:",
    record["G900_edge_law_profile"],
)
print(
    "class_preserving_bijection_count:",
    record["class_preserving_bijection_count"],
)
print(
    "class_preserving_isomorphism_count:",
    record["class_preserving_isomorphism_count"],
)
print(
    "candidate_predicate_count:",
    record["candidate_predicate_count"],
)
print(
    "exact_law_hit_count:",
    record["exact_law_hit_count"],
)
print(
    "law_compatible_isomorphism_count:",
    record["law_compatible_isomorphism_count"],
)
print(
    "isomorphism_law_hit_profile:",
    record["isomorphism_law_hit_profile"],
)
print("exact_law_hits:")
for row in record["exact_law_hits"]:
    print(row)
print("isomorphisms:")
for row in record["bridge_records"]:
    print(
        {
            "isomorphism_index": row["isomorphism_index"],
            "mapping": row["mapping"],
            "exact_law_hit_count": (
                row["exact_law_hit_count"]
            ),
        }
    )
print("checks:", record["checks"])
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
