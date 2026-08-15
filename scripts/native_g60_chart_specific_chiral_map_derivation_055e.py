#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

ATLAS_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_fold_swap_atlas_054c.v1.json"
)

CHIRALITY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_face_chirality_audit_054e.v1.json"
)

SYMMETRY_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_chiral_surgery_symmetry_reduction_055c.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_chart_specific_chiral_map_derivation_055e.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_chart_specific_chiral_map_derivation_055e.csv"
)


def inverse(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    result = [0] * len(permutation)

    for source, target in enumerate(permutation):
        result[target] = source

    return tuple(result)


def derive_chart_permutation(
    second_cycle: list[str],
    matched_second_order: list[str],
) -> tuple[int, ...]:
    if len(second_cycle) != 5:
        raise RuntimeError("second cycle does not have length five")

    if len(matched_second_order) != 5:
        raise RuntimeError(
            "matched second order does not have length five"
        )

    if set(second_cycle) != set(matched_second_order):
        raise RuntimeError(
            "second cycle and matched order have different support"
        )

    position = {
        vertex: index
        for index, vertex in enumerate(second_cycle)
    }

    return tuple(
        position[vertex]
        for vertex in matched_second_order
    )


atlas = json.loads(ATLAS_JSON.read_text())
chirality = json.loads(CHIRALITY_JSON.read_text())
symmetry = json.loads(SYMMETRY_JSON.read_text())

charts = (
    atlas
    .get("native_graph", {})
    .get("complementary_pairs", [])
)

map_rows = (
    chirality
    .get("target_step_2_to_step_1", {})
    .get("records", [])
)

map_lookup: dict[tuple[int, ...], dict[str, Any]] = {}

for map_index, row in enumerate(map_rows):
    permutation = tuple(
        int(value)
        for value in row["permutation"]
    )

    map_lookup[permutation] = {
        "map_index": map_index,
        "chirality": row["chirality"],
        "order": int(row["order"]),
        "inverse_permutation": inverse(permutation),
    }

inverse_pair_id = {}
seen = set()
pair_number = 0

for permutation in sorted(map_lookup):
    if permutation in seen:
        continue

    partner = inverse(permutation)

    if partner not in map_lookup:
        raise RuntimeError(
            f"missing inverse map for {permutation}"
        )

    pair_number += 1

    inverse_pair_id[permutation] = pair_number
    inverse_pair_id[partner] = pair_number

    seen.add(permutation)
    seen.add(partner)

source_bipartition = None

for row in symmetry.get("checks", {}):
    pass

surgery_json = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_attractor_dyad_surgery_audit_055b.v1.json"
)

surgery = json.loads(surgery_json.read_text())

left_side = set(
    surgery["source_bipartition"]["left"]
)

right_side = set(
    surgery["source_bipartition"]["right"]
)

chart_records = []

for chart_index, chart in enumerate(charts):
    second_cycle = chart["second_cycle"]
    matched_second_order = chart["matched_second_order"]

    permutation = derive_chart_permutation(
        second_cycle,
        matched_second_order,
    )

    map_record = map_lookup.get(permutation)

    if map_record is None:
        raise RuntimeError(
            "derived chart permutation is absent from 054e: "
            f"chart={chart_index} permutation={permutation}"
        )

    if chart_index in left_side:
        dyad_side = "left"
    elif chart_index in right_side:
        dyad_side = "right"
    else:
        raise RuntimeError(
            f"chart {chart_index} is outside the dyad"
        )

    chart_records.append(
        {
            "chart_index": chart_index,
            "dyad_side": dyad_side,
            "first_cycle": chart["first_cycle"],
            "second_cycle": second_cycle,
            "matched_second_order": matched_second_order,
            "derived_permutation": list(permutation),
            "map_index": map_record["map_index"],
            "chirality": map_record["chirality"],
            "order": map_record["order"],
            "inverse_pair_id": inverse_pair_id[permutation],
            "inverse_permutation": list(
                map_record["inverse_permutation"]
            ),
        }
    )

side_chirality_profile = defaultdict(Counter)
side_map_profile = defaultdict(Counter)
side_inverse_pair_profile = defaultdict(Counter)

for row in chart_records:
    side = row["dyad_side"]

    side_chirality_profile[side][
        row["chirality"]
    ] += 1

    side_map_profile[side][
        row["map_index"]
    ] += 1

    side_inverse_pair_profile[side][
        row["inverse_pair_id"]
    ] += 1

uniform_sides = {}

for side in ("left", "right"):
    support = set(side_chirality_profile[side])

    uniform_sides[side] = (
        len(support) == 1
    )

uniform_side_names = sorted(
    side
    for side, is_uniform in uniform_sides.items()
    if is_uniform
)

mixed_side_names = sorted(
    side
    for side, is_uniform in uniform_sides.items()
    if not is_uniform
)

checks = {
    "chart_count_is_six": len(chart_records) == 6,
    "all_derived_maps_are_in_054e": (
        len(chart_records) == 6
    ),
    "all_derived_maps_have_order_four": all(
        row["order"] == 4
        for row in chart_records
    ),
    "left_side_is_three_charts": (
        sum(
            row["dyad_side"] == "left"
            for row in chart_records
        )
        == 3
    ),
    "right_side_is_three_charts": (
        sum(
            row["dyad_side"] == "right"
            for row in chart_records
        )
        == 3
    ),
    "exactly_one_side_is_chirality_uniform": (
        len(uniform_side_names) == 1
    ),
    "exactly_one_side_is_chirality_mixed": (
        len(mixed_side_names) == 1
    ),
    "right_side_is_uniform_negative": (
        side_chirality_profile["right"]
        == Counter({"negative": 3})
    ),
    "left_side_is_one_negative_two_positive": (
        side_chirality_profile["left"]
        == Counter(
            {
                "positive": 2,
                "negative": 1,
            }
        )
    ),
}

audit_pass = all(checks.values())

if audit_pass:
    verdict = (
        "native_chart_orders_derive_chart_specific_"
        "chiral_maps_and_polarize_K33_sides"
    )
else:
    verdict = (
        "native_chart_specific_chiral_polarization_"
        "not_established"
    )

record = {
    "artifact_id": (
        "native_g60_chart_specific_chiral_map_derivation_055e"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "chart_count": len(chart_records),
    "chart_records": chart_records,
    "left_side": sorted(left_side),
    "right_side": sorted(right_side),
    "side_chirality_profile": {
        side: dict(
            sorted(profile.items())
        )
        for side, profile in sorted(
            side_chirality_profile.items()
        )
    },
    "side_map_profile": {
        side: {
            str(key): value
            for key, value in sorted(profile.items())
        }
        for side, profile in sorted(
            side_map_profile.items()
        )
    },
    "side_inverse_pair_profile": {
        side: {
            str(key): value
            for key, value in sorted(profile.items())
        }
        for side, profile in sorted(
            side_inverse_pair_profile.items()
        )
    },
    "uniform_side_names": uniform_side_names,
    "mixed_side_names": mixed_side_names,
    "checks": checks,
    "interpretation": (
        "The native matched-second-face ordering in each Petersen "
        "chart selects one of the ten generic order-four chiral "
        "intertwiners. The K3,3 chart bipartition is thereby "
        "polarized: the right triad is uniformly negative, while "
        "the left triad contains two positive and one negative map."
    ),
    "boundary": {
        "chart_specific_chiral_maps_derived": audit_pass,
        "K33_side_chirality_polarization_derived": audit_pass,
        "uniform_side_selected_as_circulating": False,
        "mixed_side_selected_as_circulating": False,
        "preferred_matching_selected": False,
        "preferred_surgery_selected": False,
        "attractor_to_chart_map_proved": False,
        "physical_chirality_proved": False,
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
        "chart_index,dyad_side,map_index,chirality,"
        "order,inverse_pair_id,derived_permutation\n"
    )

    for row in chart_records:
        handle.write(
            ",".join(
                [
                    str(row["chart_index"]),
                    row["dyad_side"],
                    str(row["map_index"]),
                    row["chirality"],
                    str(row["order"]),
                    str(row["inverse_pair_id"]),
                    json.dumps(
                        row["derived_permutation"],
                        separators=(",", ":"),
                    ),
                ]
            )
            + "\n"
        )

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print("left_side:", record["left_side"])
print("right_side:", record["right_side"])
print(
    "side_chirality_profile:",
    record["side_chirality_profile"],
)
print(
    "side_map_profile:",
    record["side_map_profile"],
)
print(
    "side_inverse_pair_profile:",
    record["side_inverse_pair_profile"],
)
print(
    "uniform_side_names:",
    record["uniform_side_names"],
)
print(
    "mixed_side_names:",
    record["mixed_side_names"],
)
print("chart_records:")
for row in record["chart_records"]:
    print(
        {
            "chart_index": row["chart_index"],
            "dyad_side": row["dyad_side"],
            "derived_permutation": (
                row["derived_permutation"]
            ),
            "map_index": row["map_index"],
            "chirality": row["chirality"],
            "inverse_pair_id": (
                row["inverse_pair_id"]
            ),
        }
    )
print("checks:", record["checks"])
print("json:", JSON_OUT)
print("csv:", CSV_OUT)
