#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHART_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chart_specific_chiral_map_derivation_055e.v1.json"
)

SURGERY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_attractor_dyad_surgery_audit_055b.v1.json"
)

SYMMETRY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_surgery_symmetry_reduction_055c.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_matching_chiral_compatibility_census_055f.v1.json"
)


def edge_key(value: list[int]) -> tuple[int, int]:
    return tuple(sorted((int(value[0]), int(value[1]))))


chart_data = json.loads(CHART_JSON.read_text())
surgery_data = json.loads(SURGERY_JSON.read_text())
symmetry_data = json.loads(SYMMETRY_JSON.read_text())

charts = {
    int(row["chart_index"]): {
        "chirality": row["chirality"],
        "map_index": int(row["map_index"]),
        "inverse_pair_id": int(row["inverse_pair_id"]),
        "permutation": tuple(
            int(value)
            for value in row["derived_permutation"]
        ),
        "inverse_permutation": tuple(
            int(value)
            for value in row["inverse_permutation"]
        ),
        "dyad_side": row["dyad_side"],
    }
    for row in chart_data["chart_records"]
}

unique_matchings = {}

for surgery_index, row in enumerate(surgery_data["surgeries"]):
    matching = tuple(
        sorted(
            edge_key(value)
            for value in row["removed_matching"]
        )
    )

    unique_matchings.setdefault(
        matching,
        {
            "matching": matching,
            "surgery_indices": [],
        },
    )

    unique_matchings[matching][
        "surgery_indices"
    ].append(surgery_index)

distinguished_orbits = {
    tuple(sorted(orbit)): orbit_index
    for orbit_index, orbit in enumerate(
        symmetry_data["full_feature_surgery_orbits"],
        start=1,
    )
}

matching_records = []

for matching_id, matching in enumerate(
    sorted(unique_matchings),
    start=1,
):
    surgery_indices = sorted(
        unique_matchings[matching]["surgery_indices"]
    )

    edge_records = []

    for left, right in matching:
        left_chart = charts[left]
        right_chart = charts[right]

        opposite_chirality = (
            left_chart["chirality"]
            != right_chart["chirality"]
        )

        same_chirality = (
            left_chart["chirality"]
            == right_chart["chirality"]
        )

        same_map = (
            left_chart["permutation"]
            == right_chart["permutation"]
        )

        exact_inverse = (
            left_chart["inverse_permutation"]
            == right_chart["permutation"]
            and right_chart["inverse_permutation"]
            == left_chart["permutation"]
        )

        same_inverse_pair = (
            left_chart["inverse_pair_id"]
            == right_chart["inverse_pair_id"]
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
                "left_inverse_pair_id": (
                    left_chart["inverse_pair_id"]
                ),
                "right_inverse_pair_id": (
                    right_chart["inverse_pair_id"]
                ),
                "same_chirality": same_chirality,
                "opposite_chirality": opposite_chirality,
                "same_map": same_map,
                "same_inverse_pair": same_inverse_pair,
                "exact_inverse": exact_inverse,
            }
        )

    orbit_id = next(
        (
            orbit_index
            for orbit, orbit_index
            in distinguished_orbits.items()
            if any(
                surgery_index in orbit
                for surgery_index in surgery_indices
            )
        ),
        None,
    )

    matching_records.append(
        {
            "matching_id": matching_id,
            "matching": [
                list(value)
                for value in matching
            ],
            "surgery_indices": surgery_indices,
            "directed_overlap_orbit_id": orbit_id,
            "same_chirality_edge_count": sum(
                row["same_chirality"]
                for row in edge_records
            ),
            "opposite_chirality_edge_count": sum(
                row["opposite_chirality"]
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
            "exact_inverse_edge_count": sum(
                row["exact_inverse"]
                for row in edge_records
            ),
            "edge_records": edge_records,
        }
    )

signature_profile = Counter(
    (
        row["same_chirality_edge_count"],
        row["opposite_chirality_edge_count"],
        row["same_map_edge_count"],
        row["same_inverse_pair_edge_count"],
        row["exact_inverse_edge_count"],
    )
    for row in matching_records
)

for row in matching_records:
    signature = (
        row["same_chirality_edge_count"],
        row["opposite_chirality_edge_count"],
        row["same_map_edge_count"],
        row["same_inverse_pair_edge_count"],
        row["exact_inverse_edge_count"],
    )

    row["signature_multiplicity"] = (
        signature_profile[signature]
    )

unique_chiral_signature_matchings = [
    row
    for row in matching_records
    if row["signature_multiplicity"] == 1
]

exact_inverse_matchings = [
    row
    for row in matching_records
    if row["exact_inverse_edge_count"] > 0
]

m1 = next(
    row
    for row in matching_records
    if row["matching"]
    == [[0, 3], [1, 2], [4, 5]]
)

m2 = next(
    row
    for row in matching_records
    if row["matching"]
    == [[0, 3], [1, 5], [2, 4]]
)

checks = {
    "chart_count_is_six": len(charts) == 6,
    "perfect_matching_count_is_six": (
        len(matching_records) == 6
    ),
    "each_matching_has_two_surgeries": all(
        len(row["surgery_indices"]) == 2
        for row in matching_records
    ),
    "all_matching_edges_cross_dyad": all(
        charts[left]["dyad_side"]
        != charts[right]["dyad_side"]
        for row in matching_records
        for left, right in (
            tuple(edge)
            for edge in row["matching"]
        )
    ),
    "M1_and_M2_found": (
        m1 is not None and m2 is not None
    ),
}

audit_pass = all(checks.values())

record = {
    "artifact_id": (
        "native_g60_matching_chiral_compatibility_census_055f"
    ),
    "audit_pass": audit_pass,
    "verdict": (
        "native_chart_specific_chiral_matching_census_recorded"
    ),
    "matching_count": len(matching_records),
    "matching_records": matching_records,
    "chiral_signature_profile": [
        {
            "same_chirality_edge_count": signature[0],
            "opposite_chirality_edge_count": signature[1],
            "same_map_edge_count": signature[2],
            "same_inverse_pair_edge_count": signature[3],
            "exact_inverse_edge_count": signature[4],
            "matching_count": count,
        }
        for signature, count in sorted(
            signature_profile.items()
        )
    ],
    "unique_chiral_signature_matching_ids": [
        row["matching_id"]
        for row in unique_chiral_signature_matchings
    ],
    "exact_inverse_matching_ids": [
        row["matching_id"]
        for row in exact_inverse_matchings
    ],
    "M1": m1,
    "M2": m2,
    "checks": checks,
    "boundary": {
        "matching_chiral_signatures_classified": (
            audit_pass
        ),
        "preferred_matching_selected": False,
        "exact_inverse_edge_as_selector_proved": False,
        "preferred_circulating_side_selected": False,
        "preferred_surgery_selected": False,
        "attractor_to_chart_map_proved": False,
        "physical_chirality_proved": False,
    },
}

JSON_OUT.parent.mkdir(
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

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("matching_count:", record["matching_count"])
print(
    "unique_chiral_signature_matching_ids:",
    record["unique_chiral_signature_matching_ids"],
)
print(
    "exact_inverse_matching_ids:",
    record["exact_inverse_matching_ids"],
)

print("matching_records:")
for row in record["matching_records"]:
    print(
        {
            "matching_id": row["matching_id"],
            "matching": row["matching"],
            "surgery_indices": row["surgery_indices"],
            "directed_overlap_orbit_id": (
                row["directed_overlap_orbit_id"]
            ),
            "same_chirality": (
                row["same_chirality_edge_count"]
            ),
            "opposite_chirality": (
                row["opposite_chirality_edge_count"]
            ),
            "same_map": (
                row["same_map_edge_count"]
            ),
            "same_inverse_pair": (
                row["same_inverse_pair_edge_count"]
            ),
            "exact_inverse": (
                row["exact_inverse_edge_count"]
            ),
            "signature_multiplicity": (
                row["signature_multiplicity"]
            ),
        }
    )

print("M1:")
print(
    {
        key: value
        for key, value in record["M1"].items()
        if key != "edge_records"
    }
)

print("M2:")
print(
    {
        key: value
        for key, value in record["M2"].items()
        if key != "edge_records"
    }
)

print("checks:", record["checks"])
print("json:", JSON_OUT)
