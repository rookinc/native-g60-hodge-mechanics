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
    / "native_g900_attractor_nine_cell_quotient_audit_054i.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g900_attractor_nine_cell_quotient_audit_054i.csv"
)

PATTERN = re.compile(r"^T(\d+)_S(\d+)$")

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


def parse_vertex(label: str) -> tuple[int, int]:
    match = PATTERN.match(label)

    if match is None:
        raise ValueError(f"bad vertex label: {label}")

    return int(match.group(1)), int(match.group(2))


def orbit_key(t_value: int) -> tuple[int, ...]:
    partner = TAU[t_value]

    if partner == t_value:
        return (t_value,)

    return tuple(sorted((t_value, partner)))


with SOURCE_CSV.open() as handle:
    rows = list(csv.DictReader(handle))

records = []

for row in rows:
    t_value, s_value = parse_vertex(row["attractor"])

    records.append(
        {
            "vertex": row["attractor"],
            "T": t_value,
            "S": s_value,
            "carrier_class": row["carrier_class"],
            "signature_id": int(row["signature_id"]),
            "orbit": orbit_key(t_value),
        }
    )

records.sort(
    key=lambda row: (
        row["T"],
        row["S"],
    )
)

t_class_support = defaultdict(set)
t_signature_count = Counter()
orbit_class_support = defaultdict(set)
orbit_vertex_count = Counter()
orbit_signature_count = defaultdict(set)

for row in records:
    t_class_support[row["T"]].add(
        row["carrier_class"]
    )

    t_signature_count[row["T"]] += 1

    orbit_class_support[row["orbit"]].add(
        row["carrier_class"]
    )

    orbit_vertex_count[row["orbit"]] += 1

    orbit_signature_count[row["orbit"]].add(
        row["signature_id"]
    )

pair_count = 0
predicted_equal_count = 0
observed_equal_count = 0
false_positive_count = 0
false_negative_count = 0
counterexamples = []

for left_index, left in enumerate(records):
    for right in records[left_index + 1:]:
        pair_count += 1

        predicted_equal = (
            left["S"] == right["S"]
            and right["T"] in {
                left["T"],
                TAU[left["T"]],
            }
        )

        observed_equal = (
            left["signature_id"]
            == right["signature_id"]
        )

        if predicted_equal:
            predicted_equal_count += 1

        if observed_equal:
            observed_equal_count += 1

        if predicted_equal and not observed_equal:
            false_negative_count += 1

            if len(counterexamples) < 20:
                counterexamples.append(
                    {
                        "kind": "false_negative",
                        "left": left["vertex"],
                        "right": right["vertex"],
                    }
                )

        if observed_equal and not predicted_equal:
            false_positive_count += 1

            if len(counterexamples) < 20:
                counterexamples.append(
                    {
                        "kind": "false_positive",
                        "left": left["vertex"],
                        "right": right["vertex"],
                    }
                )

quotient_rows = []

for quotient_index, orbit in enumerate(
    sorted(
        orbit_vertex_count,
        key=lambda item: (
            len(item),
            item,
        )
    ),
    start=1,
):
    classes = sorted(
        orbit_class_support[orbit]
    )

    quotient_rows.append(
        {
            "quotient_cell": quotient_index,
            "T_orbit": list(orbit),
            "orbit_size": len(orbit),
            "carrier_classes": classes,
            "G900_vertex_count": orbit_vertex_count[orbit],
            "signature_count": len(
                orbit_signature_count[orbit]
            ),
        }
    )

fixed_points = sorted(
    t_value
    for t_value, partner in TAU.items()
    if t_value == partner
)

doubleton_orbits = sorted(
    {
        orbit_key(t_value)
        for t_value in TAU
        if len(orbit_key(t_value)) == 2
    }
)

fixed_point_classes = {
    str(t_value): sorted(
        t_class_support[t_value]
    )
    for t_value in fixed_points
}

t_carrier_class = {
    str(t_value): sorted(
        t_class_support[t_value]
    )
    for t_value in sorted(t_class_support)
}

checks = {
    "vertex_count_is_900": (
        len(records) == 900
    ),
    "tau_defined_on_all_15_T_states": (
        set(TAU) == set(range(15))
    ),
    "tau_is_involution": all(
        TAU[TAU[t_value]] == t_value
        for t_value in TAU
    ),
    "tau_has_six_doubletons": (
        len(doubleton_orbits) == 6
    ),
    "tau_has_three_fixed_points": (
        fixed_points == [6, 12, 13]
    ),
    "carrier_class_depends_only_on_T": all(
        len(classes) == 1
        for classes in t_class_support.values()
    ),
    "each_orbit_has_one_carrier_class": all(
        len(classes) == 1
        for classes in orbit_class_support.values()
    ),
    "quotient_cell_count_is_9": (
        len(quotient_rows) == 9
    ),
    "signature_equality_biconditional_passes": (
        false_positive_count == 0
        and false_negative_count == 0
    ),
    "observed_equal_pair_count_is_360": (
        observed_equal_count == 360
    ),
    "predicted_equal_pair_count_is_360": (
        predicted_equal_count == 360
    ),
}

audit_pass = all(checks.values())

verdict = (
    "native_g900_attractor_response_is_exact_"
    "G60_times_nine_cell_T_quotient"
    if audit_pass
    else
    "native_g900_attractor_nine_cell_quotient_not_established"
)

record = {
    "artifact_id": (
        "native_g900_attractor_nine_cell_quotient_audit_054i"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "vertex_count": len(records),
    "pair_count_tested": pair_count,
    "tau_cycles": [
        list(orbit)
        for orbit in doubleton_orbits
    ],
    "tau_fixed_points": fixed_points,
    "fixed_point_classes": fixed_point_classes,
    "T_carrier_class": t_carrier_class,
    "quotient_cell_count": len(quotient_rows),
    "quotient_cells": quotient_rows,
    "predicted_equal_pair_count": predicted_equal_count,
    "observed_equal_pair_count": observed_equal_count,
    "false_positive_count": false_positive_count,
    "false_negative_count": false_negative_count,
    "counterexamples": counterexamples,
    "checks": checks,
    "boundary": {
        "finite_attractor_quotient_proved": audit_pass,
        "G60_coordinate_preserved_exactly": audit_pass,
        "T_coordinate_quotiented_by_tau": audit_pass,
        "six_tau_pairs_identified_with_six_Petersen_charts": False,
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
            "quotient_cell",
            "T_orbit",
            "orbit_size",
            "carrier_classes",
            "G900_vertex_count",
            "signature_count",
        ],
    )

    writer.writeheader()

    for row in quotient_rows:
        writer.writerow(
            {
                "quotient_cell": row["quotient_cell"],
                "T_orbit": json.dumps(
                    row["T_orbit"],
                    separators=(",", ":"),
                ),
                "orbit_size": row["orbit_size"],
                "carrier_classes": json.dumps(
                    row["carrier_classes"],
                    separators=(",", ":"),
                ),
                "G900_vertex_count": row["G900_vertex_count"],
                "signature_count": row["signature_count"],
            }
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("pair_count_tested:", record["pair_count_tested"])
print("tau_cycles:", record["tau_cycles"])
print("tau_fixed_points:", record["tau_fixed_points"])
print("fixed_point_classes:", record["fixed_point_classes"])
print("T_carrier_class:")
for key, value in record["T_carrier_class"].items():
    print(key, value)
print(
    "quotient_cell_count:",
    record["quotient_cell_count"],
)
print(
    "predicted_equal_pair_count:",
    record["predicted_equal_pair_count"],
)
print(
    "observed_equal_pair_count:",
    record["observed_equal_pair_count"],
)
print(
    "false_positive_count:",
    record["false_positive_count"],
)
print(
    "false_negative_count:",
    record["false_negative_count"],
)
print("quotient_cells:")
for row in record["quotient_cells"]:
    print(row)
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
