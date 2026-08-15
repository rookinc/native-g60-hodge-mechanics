#!/usr/bin/env python3

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

ATLAS_JSON = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_fold_swap_atlas_054c.v1.json"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_face_chirality_audit_054e.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_petersen_face_chirality_audit_054e.csv"
)

NOTE_OUT = (
    ROOT
    / "notes"
    / "native_g60_petersen_face_chirality_audit_054e.md"
)


def compose(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        left[right[index]]
        for index in range(len(left))
    )


def inverse(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    result = [0] * len(permutation)

    for source, target in enumerate(permutation):
        result[target] = source

    return tuple(result)


def permutation_order(
    permutation: tuple[int, ...],
) -> int:
    identity = tuple(range(len(permutation)))
    current = identity

    for exponent in range(1, 100):
        current = compose(
            permutation,
            current,
        )

        if current == identity:
            return exponent

    raise RuntimeError(
        "permutation order exceeded search bound"
    )


def shift(
    size: int,
    step: int,
) -> tuple[int, ...]:
    return tuple(
        (index + step) % size
        for index in range(size)
    )


def conjugate(
    left: tuple[int, ...],
    middle: tuple[int, ...],
) -> tuple[int, ...]:
    return compose(
        compose(
            left,
            middle,
        ),
        inverse(left),
    )


def undirected_edges_from_shift(
    size: int,
    step: int,
) -> set[tuple[int, int]]:
    return {
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


def map_edges(
    edges: set[tuple[int, int]],
    permutation: tuple[int, ...],
) -> set[tuple[int, int]]:
    return {
        tuple(
            sorted(
                (
                    permutation[left],
                    permutation[right],
                )
            )
        )
        for left, right in edges
    }


def classify_intertwiners(
    size: int,
    source_step: int,
    target_step: int,
) -> dict[str, Any]:
    source_rotation = shift(
        size,
        source_step,
    )

    target_rotation = shift(
        size,
        target_step,
    )

    target_rotation_inverse = inverse(
        target_rotation
    )

    source_edges = (
        undirected_edges_from_shift(
            size,
            source_step,
        )
    )

    target_edges = (
        undirected_edges_from_shift(
            size,
            target_step,
        )
    )

    records = []

    for permutation in itertools.permutations(
        range(size)
    ):
        permutation = tuple(permutation)

        if map_edges(
            source_edges,
            permutation,
        ) != target_edges:
            continue

        transported_rotation = conjugate(
            permutation,
            source_rotation,
        )

        if transported_rotation == target_rotation:
            chirality = "positive"
        elif (
            transported_rotation
            == target_rotation_inverse
        ):
            chirality = "negative"
        else:
            chirality = "unclassified"

        order = permutation_order(
            permutation
        )

        permutation_inverse = inverse(
            permutation
        )

        is_involution = (
            order in (1, 2)
        )

        is_symmetric_permutation = (
            permutation
            == permutation_inverse
        )

        records.append(
            {
                "permutation": list(
                    permutation
                ),
                "permutation_text": "-".join(
                    str(value)
                    for value in permutation
                ),
                "chirality": chirality,
                "order": order,
                "is_involution": (
                    is_involution
                ),
                "is_symmetric_permutation": (
                    is_symmetric_permutation
                ),
            }
        )

    chirality_profile = dict(
        sorted(
            Counter(
                row["chirality"]
                for row in records
            ).items()
        )
    )

    order_profile = {
        str(key): value
        for key, value in sorted(
            Counter(
                row["order"]
                for row in records
            ).items()
        )
    }

    involution_count = sum(
        1
        for row in records
        if row["is_involution"]
    )

    symmetric_count = sum(
        1
        for row in records
        if row[
            "is_symmetric_permutation"
        ]
    )

    return {
        "size": size,
        "source_step": source_step,
        "target_step": target_step,
        "candidate_count": (
            factorial_small(size)
        ),
        "intertwiner_count": len(
            records
        ),
        "chirality_profile": (
            chirality_profile
        ),
        "order_profile": order_profile,
        "involution_count": (
            involution_count
        ),
        "symmetric_permutation_count": (
            symmetric_count
        ),
        "records": records,
    }


def factorial_small(value: int) -> int:
    result = 1

    for factor in range(2, value + 1):
        result *= factor

    return result


atlas = json.loads(
    ATLAS_JSON.read_text()
)

charts = (
    atlas[
        "native_graph"
    ][
        "complementary_pairs"
    ]
)

target_profile = classify_intertwiners(
    size=5,
    source_step=2,
    target_step=1,
)

reverse_profile = classify_intertwiners(
    size=5,
    source_step=3,
    target_step=1,
)

same_face_control = (
    classify_intertwiners(
        size=5,
        source_step=1,
        target_step=1,
    )
)


chart_records = []

for chart_index, chart in enumerate(
    charts,
    start=1,
):
    chart_records.append(
        {
            "chart_index": chart_index,
            "first_cycle": (
                chart["first_cycle"]
            ),
            "matched_second_order": (
                chart[
                    "matched_second_order"
                ]
            ),
            "source_step": (
                chart[
                    "matched_second_face_step"
                ]
            ),
            "target_step": (
                chart[
                    "first_face_step"
                ]
            ),
            "intertwiner_count": (
                target_profile[
                    "intertwiner_count"
                ]
            ),
            "positive_chirality_count": (
                target_profile[
                    "chirality_profile"
                ].get(
                    "positive",
                    0,
                )
            ),
            "negative_chirality_count": (
                target_profile[
                    "chirality_profile"
                ].get(
                    "negative",
                    0,
                )
            ),
            "involution_count": (
                target_profile[
                    "involution_count"
                ]
            ),
            "order_profile": (
                target_profile[
                    "order_profile"
                ]
            ),
        }
    )


checks = {
    "native_chart_count_is_six": (
        len(chart_records) == 6
    ),
    "target_candidate_count_is_120": (
        target_profile[
            "candidate_count"
        ]
        == 120
    ),
    "target_intertwiner_count_is_10": (
        target_profile[
            "intertwiner_count"
        ]
        == 10
    ),
    "target_positive_chirality_count_is_5": (
        target_profile[
            "chirality_profile"
        ].get(
            "positive",
            0,
        )
        == 5
    ),
    "target_negative_chirality_count_is_5": (
        target_profile[
            "chirality_profile"
        ].get(
            "negative",
            0,
        )
        == 5
    ),
    "target_has_no_unclassified_intertwiners": (
        target_profile[
            "chirality_profile"
        ].get(
            "unclassified",
            0,
        )
        == 0
    ),
    "target_has_no_involutive_face_map": (
        target_profile[
            "involution_count"
        ]
        == 0
    ),
    "target_all_face_maps_have_order_4": (
        target_profile[
            "order_profile"
        ]
        == {"4": 10}
    ),
    "target_has_no_symmetric_face_map": (
        target_profile[
            "symmetric_permutation_count"
        ]
        == 0
    ),
    "same_face_control_has_dihedral_orders": (
        same_face_control[
            "order_profile"
        ]
        == {
            "1": 1,
            "2": 5,
            "5": 4,
        }
    ),
    "same_face_control_has_five_reflections": (
        same_face_control[
            "involution_count"
        ]
        == 6
    ),
    "reverse_step_has_opposite_balanced_chirality": (
        reverse_profile[
            "chirality_profile"
        ].get(
            "positive",
            0,
        )
        == 5
        and reverse_profile[
            "chirality_profile"
        ].get(
            "negative",
            0,
        )
        == 5
    ),
}

failed_checks = [
    key
    for key, value in checks.items()
    if not value
]

audit_pass = len(
    failed_checks
) == 0

if audit_pass:
    verdict = (
        "native_petersen_face_relation_is_"
        "balanced_chiral_order_4_intertwiner_space_"
        "with_no_static_mirror"
    )
else:
    verdict = (
        "native_petersen_face_chirality_"
        "signature_not_established"
    )

record = {
    "artifact_id": (
        "native_g60_petersen_face_chirality_audit_054e"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "target_step_2_to_step_1": (
        target_profile
    ),
    "reverse_step_3_to_step_1": (
        reverse_profile
    ),
    "same_face_step_1_control": (
        same_face_control
    ),
    "chart_records": (
        chart_records
    ),
    "checks": checks,
    "failed_checks": failed_checks,
    "interpretation": (
        "The native map from a matched step-2 face "
        "to its step-1 partner is not an involutive "
        "mirror. Exactly ten face intertwiners exist. "
        "Five preserve directed circulation and five "
        "reverse it. Every valid face map has permutation "
        "order four. The corresponding same-face cycle "
        "control instead has the ordinary dihedral order "
        "profile with reflections."
    ),
    "boundary": {
        "native_face_chirality_established": (
            audit_pass
        ),
        "irreversible_time_direction_proved": False,
        "spinor_double_cover_proved": False,
        "physical_particle_chirality_proved": False,
        "schrodinger_dynamics_derived": False,
        "e5_kernel_identification_proved": False,
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
        "permutation,chirality,order,"
        "is_involution,is_symmetric_permutation"
    )
]

for row in target_profile[
    "records"
]:
    csv_lines.append(
        ",".join(
            [
                row[
                    "permutation_text"
                ],
                row["chirality"],
                str(row["order"]),
                str(
                    row["is_involution"]
                ).lower(),
                str(
                    row[
                        "is_symmetric_permutation"
                    ]
                ).lower(),
            ]
        )
    )

CSV_OUT.write_text(
    "\n".join(csv_lines)
    + "\n"
)

note_lines = [
    "# Native G60 Petersen face chirality audit 054e",
    "",
    f"- audit_pass: `{str(audit_pass).lower()}`",
    f"- verdict: `{verdict}`",
    (
        "- target intertwiner count: "
        f"`{target_profile['intertwiner_count']}`"
    ),
    (
        "- target chirality profile: "
        f"`{target_profile['chirality_profile']}`"
    ),
    (
        "- target order profile: "
        f"`{target_profile['order_profile']}`"
    ),
    (
        "- target involution count: "
        f"`{target_profile['involution_count']}`"
    ),
    (
        "- same-face control order profile: "
        f"`{same_face_control['order_profile']}`"
    ),
    "",
    "## Interpretation",
    "",
    (
        "The matched Petersen faces are connected by "
        "a balanced pair of directed chirality classes, "
        "not by a static order-two mirror."
    ),
    "",
    "## Boundary",
    "",
    (
        "This is finite face chirality in the native "
        "Petersen atlas. It is not yet physical chirality, "
        "a spinor representation, or a time arrow."
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
    "chart_count:",
    len(chart_records),
)
print(
    "target_intertwiner_count:",
    target_profile[
        "intertwiner_count"
    ],
)
print(
    "target_chirality_profile:",
    target_profile[
        "chirality_profile"
    ],
)
print(
    "target_order_profile:",
    target_profile[
        "order_profile"
    ],
)
print(
    "target_involution_count:",
    target_profile[
        "involution_count"
    ],
)
print(
    "same_face_control_order_profile:",
    same_face_control[
        "order_profile"
    ],
)
print(
    "same_face_control_involution_count:",
    same_face_control[
        "involution_count"
    ],
)
print(
    "failed_checks:",
    failed_checks,
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
