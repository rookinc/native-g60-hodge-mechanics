#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EDGE_SOURCE = (
    Path.home()
    / "dev/cori/research/mathematics/thalean-graph-theory"
    / "16-fifteen-thalion-ring/support/generated"
    / "g900_twisted_ring_candidate_edges.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g900_attractor_nine_cell_carrier_quotient_054j.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g900_attractor_nine_cell_carrier_quotient_054j.csv"
)

VERTEX_PATTERN = re.compile(r"^T(\d+)_S(\d+)$")

TAU = {
    0: 1,
    1: 0,
    2: 5,
    5: 2,
    3: 10,
    10: 3,
    4: 7,
    7: 4,
    8: 14,
    14: 8,
    9: 11,
    11: 9,
    6: 6,
    12: 12,
    13: 13,
}

T_CLASS = {
    0: "0_4",
    1: "0_4",
    2: "0_4",
    3: "2_2",
    4: "2_2",
    5: "0_4",
    6: "2_2",
    7: "2_2",
    8: "2_2",
    9: "4_0",
    10: "2_2",
    11: "4_0",
    12: "4_0",
    13: "4_0",
    14: "2_2",
}


def parse_vertex(label: str) -> tuple[int, int]:
    match = VERTEX_PATTERN.match(label)

    if match is None:
        raise ValueError(f"bad vertex label: {label}")

    return int(match.group(1)), int(match.group(2))


def orbit(t_value: int) -> tuple[int, ...]:
    partner = TAU[t_value]

    if partner == t_value:
        return (t_value,)

    return tuple(sorted((t_value, partner)))


ORBITS = sorted(
    {orbit(t_value) for t_value in range(15)},
    key=lambda item: (len(item), item),
)

ORBIT_ID = {
    item: index
    for index, item in enumerate(ORBITS, start=1)
}

with EDGE_SOURCE.open() as handle:
    rows = list(csv.DictReader(handle))

external_rows = [
    row
    for row in rows
    if row["kind"] == "external_signed_carrier"
]

t_edge_lifts = Counter()
t_edge_law_lifts = Counter()
quotient_edge_lifts = Counter()
quotient_edge_law_lifts = Counter()

for row in external_rows:
    t_u, s_u = parse_vertex(row["u"])
    t_v, s_v = parse_vertex(row["v"])

    t_edge = tuple(sorted((t_u, t_v)))
    law = row["law"]

    cell_u = ORBIT_ID[orbit(t_u)]
    cell_v = ORBIT_ID[orbit(t_v)]
    quotient_edge = tuple(sorted((cell_u, cell_v)))

    t_edge_lifts[t_edge] += 1
    t_edge_law_lifts[(t_edge, law)] += 1

    quotient_edge_lifts[quotient_edge] += 1
    quotient_edge_law_lifts[(quotient_edge, law)] += 1

t_edge_records = []

for t_edge, lift_count in sorted(t_edge_lifts.items()):
    law_counts = {
        law: t_edge_law_lifts[(t_edge, law)]
        for law in ("identity", "half_flip_x_plus_30")
    }

    t_edge_records.append(
        {
            "T_edge": list(t_edge),
            "lift_count": lift_count,
            "identity_count": law_counts["identity"],
            "half_flip_count": law_counts["half_flip_x_plus_30"],
        }
    )

quotient_records = []

for edge, lift_count in sorted(quotient_edge_lifts.items()):
    left, right = edge
    left_orbit = ORBITS[left - 1]
    right_orbit = ORBITS[right - 1]

    identity_count = quotient_edge_law_lifts[
        (edge, "identity")
    ]

    half_flip_count = quotient_edge_law_lifts[
        (edge, "half_flip_x_plus_30")
    ]

    quotient_records.append(
        {
            "cell_u": left,
            "cell_v": right,
            "orbit_u": list(left_orbit),
            "orbit_v": list(right_orbit),
            "is_loop": left == right,
            "lift_count": lift_count,
            "normalized_weight": lift_count // 60,
            "identity_count": identity_count,
            "half_flip_count": half_flip_count,
            "identity_weight": identity_count // 60,
            "half_flip_weight": half_flip_count // 60,
        }
    )

simple_adjacency = defaultdict(set)
weighted_degree = Counter()
loop_count = 0

for row in quotient_records:
    u = row["cell_u"]
    v = row["cell_v"]
    weight = row["normalized_weight"]

    if u == v:
        loop_count += 1
        weighted_degree[u] += 2 * weight
    else:
        simple_adjacency[u].add(v)
        simple_adjacency[v].add(u)
        weighted_degree[u] += weight
        weighted_degree[v] += weight

unseen = set(range(1, 10))
components = []

while unseen:
    start = min(unseen)
    queue = deque([start])
    component = set()

    while queue:
        vertex = queue.popleft()

        if vertex in component:
            continue

        component.add(vertex)
        unseen.discard(vertex)
        queue.extend(simple_adjacency[vertex] - component)

    components.append(sorted(component))

cell_records = []

for cell_id, cell_orbit in enumerate(ORBITS, start=1):
    classes = sorted({T_CLASS[t] for t in cell_orbit})

    cell_records.append(
        {
            "cell_id": cell_id,
            "T_orbit": list(cell_orbit),
            "orbit_size": len(cell_orbit),
            "carrier_class": classes[0],
            "simple_degree": len(simple_adjacency[cell_id]),
            "weighted_degree": weighted_degree[cell_id],
        }
    )

t_lift_profile = Counter(t_edge_lifts.values())
quotient_weight_profile = Counter(
    row["normalized_weight"]
    for row in quotient_records
)

checks = {
    "external_edge_count_is_1800": len(external_rows) == 1800,
    "T_edge_count_is_30": len(t_edge_records) == 30,
    "every_T_edge_has_60_lifts": (
        t_lift_profile == Counter({60: 30})
    ),
    "quotient_cell_count_is_9": len(ORBITS) == 9,
    "all_quotient_weights_integral_over_60": all(
        row["lift_count"] % 60 == 0
        and row["identity_count"] % 60 == 0
        and row["half_flip_count"] % 60 == 0
        for row in quotient_records
    ),
    "quotient_weight_accounts_for_30_T_edges": (
        sum(
            row["normalized_weight"]
            for row in quotient_records
        )
        == 30
    ),
}

audit_pass = all(checks.values())

record = {
    "artifact_id": (
        "native_g900_attractor_nine_cell_carrier_quotient_054j"
    ),
    "audit_pass": audit_pass,
    "verdict": (
        "native_g900_external_carrier_quotiented_to_"
        "nine_attractor_response_cells"
        if audit_pass
        else
        "native_g900_nine_cell_carrier_quotient_failed"
    ),
    "external_edge_count": len(external_rows),
    "T_edge_count": len(t_edge_records),
    "T_edge_lift_profile": {
        str(key): value
        for key, value in sorted(t_lift_profile.items())
    },
    "quotient_cell_count": len(ORBITS),
    "quotient_edge_record_count": len(quotient_records),
    "quotient_loop_count": loop_count,
    "quotient_weight_profile": {
        str(key): value
        for key, value in sorted(
            quotient_weight_profile.items()
        )
    },
    "component_count": len(components),
    "components": components,
    "cells": cell_records,
    "quotient_edges": quotient_records,
    "T_edges": t_edge_records,
    "checks": checks,
    "boundary": {
        "finite_carrier_quotient_derived": audit_pass,
        "six_doubletons_identified_with_Petersen_charts": False,
        "face_selection_proved": False,
        "chirality_selection_proved": False,
        "physical_attraction_proved": False,
        "force_claim": False,
    },
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

JSON_OUT.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)

with CSV_OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "cell_u",
            "cell_v",
            "orbit_u",
            "orbit_v",
            "is_loop",
            "lift_count",
            "normalized_weight",
            "identity_count",
            "half_flip_count",
            "identity_weight",
            "half_flip_weight",
        ],
    )

    writer.writeheader()

    for row in quotient_records:
        output = dict(row)
        output["orbit_u"] = json.dumps(
            output["orbit_u"],
            separators=(",", ":"),
        )
        output["orbit_v"] = json.dumps(
            output["orbit_v"],
            separators=(",", ":"),
        )
        writer.writerow(output)

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("external_edge_count:", record["external_edge_count"])
print("T_edge_count:", record["T_edge_count"])
print("T_edge_lift_profile:", record["T_edge_lift_profile"])
print("quotient_cell_count:", record["quotient_cell_count"])
print(
    "quotient_edge_record_count:",
    record["quotient_edge_record_count"],
)
print("quotient_loop_count:", record["quotient_loop_count"])
print(
    "quotient_weight_profile:",
    record["quotient_weight_profile"],
)
print("component_count:", record["component_count"])
print("components:", record["components"])
print("cells:")
for row in record["cells"]:
    print(row)
print("quotient_edges:")
for row in record["quotient_edges"]:
    print(row)
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
