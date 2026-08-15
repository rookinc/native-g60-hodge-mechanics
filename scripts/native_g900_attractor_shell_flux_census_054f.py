#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    Path.home()
    / "dev/cori/research/mathematics/thalean-graph-theory"
    / "16-fifteen-thalion-ring/support/generated"
    / "g900_twisted_ring_candidate_edges.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g900_attractor_shell_flux_census_054f.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g900_attractor_shell_flux_census_054f.csv"
)


def bfs_distances(
    start: str,
    adjacency: dict[str, set[str]],
) -> dict[str, int]:
    distance = {start: 0}
    queue = deque([start])

    while queue:
        vertex = queue.popleft()

        for neighbor in adjacency[vertex]:
            if neighbor in distance:
                continue

            distance[neighbor] = distance[vertex] + 1
            queue.append(neighbor)

    return distance


with SOURCE.open() as handle:
    rows = list(csv.DictReader(handle))

vertices = sorted(
    {row["u"] for row in rows}
    | {row["v"] for row in rows}
)

adjacency: dict[str, set[str]] = defaultdict(set)
external_degrees: dict[str, Counter[str]] = defaultdict(Counter)

for row in rows:
    u = row["u"]
    v = row["v"]

    adjacency[u].add(v)
    adjacency[v].add(u)

    if row["kind"] == "external_signed_carrier":
        external_degrees[u][row["law"]] += 1
        external_degrees[v][row["law"]] += 1


def carrier_class(vertex: str) -> str:
    identity = external_degrees[vertex]["identity"]
    half_flip = external_degrees[vertex]["half_flip_x_plus_30"]

    return f"{identity}_{half_flip}"


attractor_records = []

for index, attractor in enumerate(vertices, start=1):
    distance = bfs_distances(
        attractor,
        adjacency,
    )

    shell_sizes = Counter(distance.values())
    edge_profile = Counter()

    for row in rows:
        du = distance[row["u"]]
        dv = distance[row["v"]]

        lower = min(du, dv)
        higher = max(du, dv)

        if du == dv:
            relation = "tangent"
        else:
            relation = "shell_crossing"

        edge_profile[
            (
                row["kind"],
                row["law"],
                relation,
                lower,
                higher,
            )
        ] += 1

    signature_payload = {
        "shell_sizes": [
            [shell, shell_sizes[shell]]
            for shell in sorted(shell_sizes)
        ],
        "edge_profile": [
            [
                kind,
                law,
                relation,
                lower,
                higher,
                count,
            ]
            for (
                kind,
                law,
                relation,
                lower,
                higher,
            ), count in sorted(edge_profile.items())
        ],
    }

    signature_text = json.dumps(
        signature_payload,
        separators=(",", ":"),
        sort_keys=True,
    )

    attractor_records.append(
        {
            "attractor": attractor,
            "carrier_class": carrier_class(attractor),
            "eccentricity": max(distance.values()),
            "shell_count": len(shell_sizes),
            "signature": signature_text,
        }
    )

    if index % 100 == 0:
        print(
            f"progress: {index}/{len(vertices)}",
            flush=True,
        )

signature_ids = {
    signature: number
    for number, signature in enumerate(
        sorted(
            {
                row["signature"]
                for row in attractor_records
            }
        ),
        start=1,
    )
}

for row in attractor_records:
    row["signature_id"] = signature_ids[row["signature"]]

joint_profile = Counter(
    (
        row["carrier_class"],
        row["signature_id"],
    )
    for row in attractor_records
)

signature_class_support = defaultdict(set)

for row in attractor_records:
    signature_class_support[row["signature_id"]].add(
        row["carrier_class"]
    )

class_signature_counts = defaultdict(set)

for row in attractor_records:
    class_signature_counts[row["carrier_class"]].add(
        row["signature_id"]
    )

pure_signature_count = sum(
    1
    for classes in signature_class_support.values()
    if len(classes) == 1
)

checks = {
    "vertex_count_is_900": len(attractor_records) == 900,
    "carrier_class_profile_is_240_420_240": (
        Counter(
            row["carrier_class"]
            for row in attractor_records
        )
        == Counter(
            {
                "0_4": 240,
                "2_2": 420,
                "4_0": 240,
            }
        )
    ),
    "all_attractors_reach_full_graph": all(
        row["shell_count"] > 1
        for row in attractor_records
    ),
}

record = {
    "artifact_id": (
        "native_g900_attractor_shell_flux_census_054f"
    ),
    "audit_pass": all(checks.values()),
    "verdict": (
        "native_g900_attractor_shell_flux_census_recorded"
    ),
    "source": str(SOURCE),
    "vertex_count": len(attractor_records),
    "carrier_class_counts": dict(
        sorted(
            Counter(
                row["carrier_class"]
                for row in attractor_records
            ).items()
        )
    ),
    "global_signature_count": len(signature_ids),
    "pure_signature_count": pure_signature_count,
    "class_signature_counts": {
        key: len(value)
        for key, value in sorted(
            class_signature_counts.items()
        )
    },
    "joint_carrier_class_signature_profile": [
        {
            "carrier_class": carrier_class_name,
            "signature_id": signature_id,
            "count": count,
        }
        for (
            carrier_class_name,
            signature_id,
        ), count in sorted(joint_profile.items())
    ],
    "checks": checks,
    "boundary": {
        "attractor_is_graph_distance_source": True,
        "face_selection_proved": False,
        "chirality_selection_proved": False,
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

with CSV_OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "attractor",
            "carrier_class",
            "signature_id",
            "eccentricity",
            "shell_count",
        ],
    )

    writer.writeheader()

    for row in attractor_records:
        writer.writerow(
            {
                key: row[key]
                for key in writer.fieldnames
            }
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("vertex_count:", record["vertex_count"])
print("carrier_class_counts:", record["carrier_class_counts"])
print(
    "global_signature_count:",
    record["global_signature_count"],
)
print(
    "pure_signature_count:",
    record["pure_signature_count"],
)
print(
    "class_signature_counts:",
    record["class_signature_counts"],
)
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
