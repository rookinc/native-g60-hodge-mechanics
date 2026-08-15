#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CSV = (
    ROOT
    / "artifacts/csv"
    / "native_g900_attractor_shell_flux_census_054f.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g900_attractor_T_collision_law_audit_054h.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g900_attractor_T_collision_law_audit_054h.csv"
)

VERTEX_PATTERN = re.compile(r"^T(\d+)_S(\d+)$")


def parse_vertex(label: str) -> tuple[int, int]:
    match = VERTEX_PATTERN.match(label)

    if match is None:
        raise ValueError(f"bad vertex label: {label}")

    return int(match.group(1)), int(match.group(2))


with SOURCE_CSV.open() as handle:
    rows = list(csv.DictReader(handle))

groups: dict[int, list[dict[str, str]]] = defaultdict(list)

for row in rows:
    groups[int(row["signature_id"])].append(row)

pair_rows = []
directed_delta_profile = Counter()
undirected_delta_profile = Counter()
class_delta_profile = Counter()
s_delta_profile = Counter()
t_pair_profile = Counter()
t_incidence = Counter()

for signature_id, members in sorted(groups.items()):
    if len(members) != 2:
        continue

    first, second = sorted(
        members,
        key=lambda row: row["attractor"],
    )

    t1, s1 = parse_vertex(first["attractor"])
    t2, s2 = parse_vertex(second["attractor"])

    directed_delta = (t2 - t1) % 15
    reverse_delta = (-directed_delta) % 15
    undirected_delta = min(directed_delta, reverse_delta)
    s_delta = (s2 - s1) % 60

    carrier_class = first["carrier_class"]

    if second["carrier_class"] != carrier_class:
        raise RuntimeError("cross-class pair found")

    directed_delta_profile[directed_delta] += 1
    undirected_delta_profile[undirected_delta] += 1
    class_delta_profile[
        (carrier_class, undirected_delta)
    ] += 1
    s_delta_profile[s_delta] += 1
    t_pair_profile[tuple(sorted((t1, t2)))] += 1
    t_incidence[t1] += 1
    t_incidence[t2] += 1

    pair_rows.append(
        {
            "signature_id": signature_id,
            "carrier_class": carrier_class,
            "vertex_1": first["attractor"],
            "vertex_2": second["attractor"],
            "T_1": t1,
            "T_2": t2,
            "S": s1,
            "S_delta_mod60": s_delta,
            "directed_T_delta_mod15": directed_delta,
            "undirected_T_delta_mod15": undirected_delta,
        }
    )

checks = {
    "pair_signature_count_is_360": len(pair_rows) == 360,
    "all_pairs_preserve_S": (
        s_delta_profile == Counter({0: 360})
    ),
    "all_pairs_are_same_carrier_class": all(
        row["carrier_class"]
        in {"0_4", "2_2", "4_0"}
        for row in pair_rows
    ),
    "all_T_pairs_are_distinct": all(
        row["T_1"] != row["T_2"]
        for row in pair_rows
    ),
}

record = {
    "artifact_id": (
        "native_g900_attractor_T_collision_law_audit_054h"
    ),
    "audit_pass": all(checks.values()),
    "verdict": (
        "native_g900_attractor_T_collision_law_recorded"
    ),
    "pair_signature_count": len(pair_rows),
    "paired_vertex_count": 2 * len(pair_rows),
    "directed_T_delta_profile": {
        str(key): value
        for key, value in sorted(
            directed_delta_profile.items()
        )
    },
    "undirected_T_delta_profile": {
        str(key): value
        for key, value in sorted(
            undirected_delta_profile.items()
        )
    },
    "class_T_delta_profile": [
        {
            "carrier_class": carrier_class,
            "undirected_T_delta_mod15": delta,
            "pair_count": count,
        }
        for (
            carrier_class,
            delta,
        ), count in sorted(
            class_delta_profile.items()
        )
    ],
    "T_pair_profile": [
        {
            "T_pair": list(pair),
            "S_fiber_count": count,
        }
        for pair, count in sorted(
            t_pair_profile.items()
        )
    ],
    "T_incidence_profile": {
        str(key): value
        for key, value in sorted(
            t_incidence.items()
        )
    },
    "checks": checks,
    "boundary": {
        "G60_S_coordinate_recovery": True,
        "exact_T_recovery_for_all_vertices": False,
        "T_collision_law_classified": True,
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

with CSV_OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(pair_rows[0]),
    )
    writer.writeheader()
    writer.writerows(pair_rows)

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("pair_signature_count:", record["pair_signature_count"])
print(
    "directed_T_delta_profile:",
    record["directed_T_delta_profile"],
)
print(
    "undirected_T_delta_profile:",
    record["undirected_T_delta_profile"],
)
print("class_T_delta_profile:")
for row in record["class_T_delta_profile"]:
    print(row)
print("T_pair_profile:")
for row in record["T_pair_profile"]:
    print(row)
print(
    "T_incidence_profile:",
    record["T_incidence_profile"],
)
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
