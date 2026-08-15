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
    / "native_g900_attractor_signature_collision_audit_054g.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g900_attractor_signature_collision_audit_054g.csv"
)

pattern = re.compile(r"^T(\d+)_S(\d+)$")


def parse_vertex(vertex: str) -> tuple[int, int]:
    match = pattern.match(vertex)

    if not match:
        raise RuntimeError(
            f"unrecognized vertex label: {vertex}"
        )

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


with SOURCE_CSV.open() as handle:
    rows = list(csv.DictReader(handle))

groups: dict[int, list[dict[str, str]]] = defaultdict(list)

for row in rows:
    groups[int(row["signature_id"])].append(row)

collision_records = []

group_size_profile = Counter()
class_group_size_profile = Counter()
relation_profile = Counter()

for signature_id, members in sorted(groups.items()):
    group_size = len(members)
    carrier_classes = sorted(
        {
            row["carrier_class"]
            for row in members
        }
    )

    if len(carrier_classes) != 1:
        raise RuntimeError(
            "cross-class signature collision found"
        )

    carrier_class = carrier_classes[0]
    group_size_profile[group_size] += 1
    class_group_size_profile[
        (
            carrier_class,
            group_size,
        )
    ] += 1

    parsed = [
        (
            row["attractor"],
            *parse_vertex(row["attractor"]),
        )
        for row in members
    ]

    same_t = (
        len({t for _, t, _ in parsed}) == 1
    )

    same_s = (
        len({s for _, _, s in parsed}) == 1
    )

    half_flip_closed = all(
        any(
            other_t == t
            and other_s == (s + 30) % 60
            for _, other_t, other_s in parsed
        )
        for _, t, s in parsed
    )

    t_values = sorted(
        {t for _, t, _ in parsed}
    )

    s_values = sorted(
        {s for _, _, s in parsed}
    )

    pair_deltas = Counter()

    for left_index in range(len(parsed)):
        _, t1, s1 = parsed[left_index]

        for right_index in range(
            left_index + 1,
            len(parsed),
        ):
            _, t2, s2 = parsed[right_index]

            pair_deltas[
                (
                    (t2 - t1) % 15,
                    (s2 - s1) % 60,
                )
            ] += 1

    if group_size == 1:
        relation = "singleton"
    elif half_flip_closed:
        relation = "same_T_half_flip_closed"
    elif same_t:
        relation = "same_T_other_S_relation"
    elif same_s:
        relation = "same_S_multiple_T"
    else:
        relation = "mixed_T_and_S"

    relation_profile[
        (
            carrier_class,
            relation,
        )
    ] += 1

    collision_records.append(
        {
            "signature_id": signature_id,
            "carrier_class": carrier_class,
            "group_size": group_size,
            "relation": relation,
            "same_T": same_t,
            "same_S": same_s,
            "half_flip_closed": (
                half_flip_closed
            ),
            "T_values": t_values,
            "S_values": s_values,
            "vertices": [
                vertex
                for vertex, _, _ in parsed
            ],
            "pair_delta_profile": [
                {
                    "T_delta_mod15": delta[0],
                    "S_delta_mod60": delta[1],
                    "count": count,
                }
                for delta, count in sorted(
                    pair_deltas.items()
                )
            ],
        }
    )

singleton_signature_count = sum(
    1
    for row in collision_records
    if row["group_size"] == 1
)

ambiguous_signature_count = (
    len(collision_records)
    - singleton_signature_count
)

exactly_recovered_vertex_count = sum(
    row["group_size"]
    for row in collision_records
    if row["group_size"] == 1
)

half_flip_collision_group_count = sum(
    1
    for row in collision_records
    if row["group_size"] > 1
    and row["half_flip_closed"]
)

checks = {
    "source_vertex_count_is_900": (
        len(rows) == 900
    ),
    "signature_count_is_540": (
        len(groups) == 540
    ),
    "all_signatures_class_pure": all(
        len(
            {
                row["carrier_class"]
                for row in members
            }
        )
        == 1
        for members in groups.values()
    ),
    "group_sizes_account_for_all_vertices": (
        sum(
            size * count
            for size, count
            in group_size_profile.items()
        )
        == 900
    ),
}

record = {
    "artifact_id": (
        "native_g900_attractor_signature_collision_audit_054g"
    ),
    "audit_pass": all(checks.values()),
    "verdict": (
        "native_g900_attractor_signature_collision_"
        "structure_recorded"
    ),
    "vertex_count": len(rows),
    "signature_count": len(groups),
    "group_size_profile": {
        str(size): count
        for size, count in sorted(
            group_size_profile.items()
        )
    },
    "class_group_size_profile": [
        {
            "carrier_class": carrier_class,
            "group_size": group_size,
            "signature_count": count,
        }
        for (
            carrier_class,
            group_size,
        ), count in sorted(
            class_group_size_profile.items()
        )
    ],
    "relation_profile": [
        {
            "carrier_class": carrier_class,
            "relation": relation,
            "signature_count": count,
        }
        for (
            carrier_class,
            relation,
        ), count in sorted(
            relation_profile.items()
        )
    ],
    "singleton_signature_count": (
        singleton_signature_count
    ),
    "ambiguous_signature_count": (
        ambiguous_signature_count
    ),
    "exactly_recovered_vertex_count": (
        exactly_recovered_vertex_count
    ),
    "half_flip_collision_group_count": (
        half_flip_collision_group_count
    ),
    "checks": checks,
    "boundary": {
        "carrier_class_recovery_proved": True,
        "exact_attractor_recovery_proved": (
            ambiguous_signature_count == 0
        ),
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
        fieldnames=[
            "signature_id",
            "carrier_class",
            "group_size",
            "relation",
            "same_T",
            "same_S",
            "half_flip_closed",
            "T_values",
            "S_values",
            "vertices",
        ],
    )

    writer.writeheader()

    for row in collision_records:
        writer.writerow(
            {
                "signature_id": (
                    row["signature_id"]
                ),
                "carrier_class": (
                    row["carrier_class"]
                ),
                "group_size": row["group_size"],
                "relation": row["relation"],
                "same_T": row["same_T"],
                "same_S": row["same_S"],
                "half_flip_closed": (
                    row["half_flip_closed"]
                ),
                "T_values": json.dumps(
                    row["T_values"],
                    separators=(",", ":"),
                ),
                "S_values": json.dumps(
                    row["S_values"],
                    separators=(",", ":"),
                ),
                "vertices": json.dumps(
                    row["vertices"],
                    separators=(",", ":"),
                ),
            }
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print(
    "group_size_profile:",
    record["group_size_profile"],
)
print(
    "singleton_signature_count:",
    record["singleton_signature_count"],
)
print(
    "ambiguous_signature_count:",
    record["ambiguous_signature_count"],
)
print(
    "exactly_recovered_vertex_count:",
    record["exactly_recovered_vertex_count"],
)
print(
    "half_flip_collision_group_count:",
    record["half_flip_collision_group_count"],
)
print("class_group_size_profile:")
for row in record["class_group_size_profile"]:
    print(row)
print("relation_profile:")
for row in record["relation_profile"]:
    print(row)
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
