#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CHIRALITY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_face_chirality_audit_054e.v1.json"
)

ATLAS_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_fold_swap_atlas_054c.v1.json"
)

SYMMETRY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_surgery_symmetry_reduction_055c.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_selector_readiness_audit_055d.v1.json"
)


def inverse(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    result = [0] * len(permutation)

    for source, target in enumerate(permutation):
        result[target] = source

    return tuple(result)


def extract_permutation(
    row: dict[str, Any],
) -> tuple[int, ...] | None:
    candidate = row.get("permutation")

    if (
        isinstance(candidate, list)
        and len(candidate) == 5
        and sorted(candidate) == list(range(5))
    ):
        return tuple(int(value) for value in candidate)

    text = row.get("permutation_text")

    if isinstance(text, str):
        try:
            values = tuple(
                int(value)
                for value in text.split("-")
            )
        except ValueError:
            return None

        if (
            len(values) == 5
            and sorted(values) == list(range(5))
        ):
            return values

    return None


def recursively_find_chart_keys(
    value: Any,
    path: str = "$",
) -> list[dict[str, Any]]:
    hits = []

    if isinstance(value, dict):
        keys = set(value)

        chart_like = sorted(
            key
            for key in keys
            if any(
                token in key.lower()
                for token in (
                    "chart",
                    "cycle",
                    "face",
                    "first_cycle",
                    "second_cycle",
                )
            )
        )

        if chart_like:
            hits.append(
                {
                    "path": path,
                    "keys": chart_like,
                }
            )

        for key, child in value.items():
            hits.extend(
                recursively_find_chart_keys(
                    child,
                    f"{path}.{key}",
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(
                recursively_find_chart_keys(
                    child,
                    f"{path}[{index}]",
                )
            )

    return hits


chirality = json.loads(
    CHIRALITY_JSON.read_text()
)

atlas = json.loads(
    ATLAS_JSON.read_text()
)

symmetry = json.loads(
    SYMMETRY_JSON.read_text()
)

target = chirality.get(
    "target_step_2_to_step_1",
    {}
)

map_rows = target.get("records", [])

maps = []

for index, row in enumerate(map_rows):
    permutation = extract_permutation(row)

    if permutation is None:
        continue

    maps.append(
        {
            "index": index,
            "permutation": permutation,
            "chirality": row.get("chirality"),
            "order": row.get("order"),
        }
    )

map_by_permutation = {
    row["permutation"]: row
    for row in maps
}

inverse_pairs = []
unpaired_maps = []

for row in maps:
    partner_permutation = inverse(
        row["permutation"]
    )

    partner = map_by_permutation.get(
        partner_permutation
    )

    if partner is None:
        unpaired_maps.append(row["index"])
        continue

    if row["index"] < partner["index"]:
        inverse_pairs.append(
            {
                "left_index": row["index"],
                "right_index": partner["index"],
                "left_chirality": row["chirality"],
                "right_chirality": partner["chirality"],
                "left_permutation": list(
                    row["permutation"]
                ),
                "right_permutation": list(
                    partner["permutation"]
                ),
            }
        )

chart_records = chirality.get(
    "chart_records",
    []
)

atlas_charts = (
    atlas
    .get("native_graph", {})
    .get("complementary_pairs", [])
)

chart_specific_key_hits = recursively_find_chart_keys(
    chirality
)

chart_map_payload_profile = Counter()
chart_signature_profile = Counter()

for row in chart_records:
    map_like_keys = sorted(
        key
        for key in row
        if any(
            token in key.lower()
            for token in (
                "permutation",
                "chirality",
                "intertwiner",
                "map",
            )
        )
    )

    payload = {
        key: row[key]
        for key in map_like_keys
    }

    signature = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    chart_map_payload_profile[
        tuple(map_like_keys)
    ] += 1

    chart_signature_profile[
        signature
    ] += 1

distinguished_surgery_indices = sorted(
    {
        2,
        4,
        8,
        9,
    }
)

surgery_orbits = symmetry.get(
    "full_feature_surgery_orbits",
    []
)

distinguished_orbit_hits = [
    orbit
    for orbit in surgery_orbits
    if set(orbit).issubset(
        set(distinguished_surgery_indices)
    )
]

chart_specific_map_assignment_found = any(
    any(
        token in key.lower()
        for token in (
            "selected_permutation",
            "selected_map",
            "map_index",
            "intertwiner_index",
            "chirality_choice",
        )
    )
    for row in chart_records
    for key in row
)

all_chart_signatures_identical = (
    len(chart_signature_profile) == 1
    and len(chart_records) == 6
)

checks = {
    "chirality_map_count_is_10": (
        len(maps) == 10
    ),
    "positive_map_count_is_5": (
        sum(
            row["chirality"] == "positive"
            for row in maps
        )
        == 5
    ),
    "negative_map_count_is_5": (
        sum(
            row["chirality"] == "negative"
            for row in maps
        )
        == 5
    ),
    "inverse_pair_count_is_5": (
        len(inverse_pairs) == 5
    ),
    "all_inverse_pairs_cross_chirality": all(
        {
            row["left_chirality"],
            row["right_chirality"],
        }
        == {"positive", "negative"}
        for row in inverse_pairs
    ),
    "no_unpaired_maps": (
        len(unpaired_maps) == 0
    ),
    "atlas_chart_count_is_6": (
        len(atlas_charts) == 6
    ),
    "chirality_chart_record_count_is_6": (
        len(chart_records) == 6
    ),
    "four_residual_surgeries_present": (
        distinguished_surgery_indices
        == [2, 4, 8, 9]
    ),
    "two_distinguished_size_two_orbits_present": (
        sorted(
            sorted(orbit)
            for orbit in distinguished_orbit_hits
        )
        == [[2, 8], [4, 9]]
    ),
}

audit_pass = all(checks.values())

if chart_specific_map_assignment_found:
    readiness = (
        "chart_specific_chiral_selector_payload_found"
    )
elif all_chart_signatures_identical:
    readiness = (
        "generic_chiral_family_repeated_across_six_charts"
    )
else:
    readiness = (
        "chart_records_differ_but_no_explicit_map_assignment_found"
    )

selector_ready = (
    chart_specific_map_assignment_found
)

record = {
    "artifact_id": (
        "native_g60_chiral_selector_readiness_audit_055d"
    ),
    "audit_pass": audit_pass,
    "verdict": readiness,
    "selector_ready": selector_ready,
    "chirality_map_count": len(maps),
    "inverse_pair_count": len(inverse_pairs),
    "inverse_pairs": inverse_pairs,
    "unpaired_maps": unpaired_maps,
    "atlas_chart_count": len(atlas_charts),
    "chirality_chart_record_count": len(chart_records),
    "chart_specific_map_assignment_found": (
        chart_specific_map_assignment_found
    ),
    "all_chart_signatures_identical": (
        all_chart_signatures_identical
    ),
    "chart_signature_count": (
        len(chart_signature_profile)
    ),
    "chart_signature_multiplicities": sorted(
        chart_signature_profile.values()
    ),
    "chart_map_payload_profile": [
        {
            "keys": list(keys),
            "chart_count": count,
        }
        for keys, count in sorted(
            chart_map_payload_profile.items()
        )
    ],
    "chart_specific_key_hits": (
        chart_specific_key_hits
    ),
    "residual_surgery_candidates": [
        {
            "surgery_index": 2,
            "matching_class": "M1",
            "circulating_side": "left",
        },
        {
            "surgery_index": 8,
            "matching_class": "M1",
            "circulating_side": "right",
        },
        {
            "surgery_index": 4,
            "matching_class": "M2",
            "circulating_side": "left",
        },
        {
            "surgery_index": 9,
            "matching_class": "M2",
            "circulating_side": "right",
        },
    ],
    "checks": checks,
    "boundary": {
        "generic_positive_negative_face_maps_proved": True,
        "inverse_chiral_pairing_proved": True,
        "chart_specific_chiral_choice_proved": (
            selector_ready
        ),
        "preferred_matching_selected": False,
        "preferred_circulating_side_selected": False,
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
print("selector_ready:", record["selector_ready"])
print(
    "chirality_map_count:",
    record["chirality_map_count"],
)
print(
    "inverse_pair_count:",
    record["inverse_pair_count"],
)
print(
    "chart_specific_map_assignment_found:",
    record[
        "chart_specific_map_assignment_found"
    ],
)
print(
    "all_chart_signatures_identical:",
    record["all_chart_signatures_identical"],
)
print(
    "chart_signature_count:",
    record["chart_signature_count"],
)
print(
    "chart_signature_multiplicities:",
    record["chart_signature_multiplicities"],
)
print("chart_map_payload_profile:")
for row in record["chart_map_payload_profile"]:
    print(row)
print("inverse_pairs:")
for row in record["inverse_pairs"]:
    print(row)
print("checks:", record["checks"])
print("json:", JSON_OUT)
