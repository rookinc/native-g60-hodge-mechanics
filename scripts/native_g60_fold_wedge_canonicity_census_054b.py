#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from math import factorial, gcd
from pathlib import Path
from typing import Any

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_fold_wedge_canonicity_census_054b.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_fold_wedge_canonicity_census_054b.csv"
)

NOTE_OUT = (
    ROOT
    / "notes"
    / "native_g60_fold_wedge_canonicity_census_054b.md"
)


def permutation_matrix(mapping: tuple[int, ...]) -> sp.Matrix:
    size = len(mapping)
    matrix = sp.zeros(size, size)

    for source, target in enumerate(mapping):
        matrix[target, source] = 1

    return matrix


def shift_permutation(
    size: int,
    step: int,
) -> tuple[int, ...]:
    return tuple(
        (index + step) % size
        for index in range(size)
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


def is_rotation(
    permutation: tuple[int, ...],
) -> tuple[bool, int | None]:
    size = len(permutation)
    offset = permutation[0] % size

    expected = tuple(
        (index + offset) % size
        for index in range(size)
    )

    return (
        permutation == expected,
        offset if permutation == expected else None,
    )


def build_sheet_operators(
    size: int,
    step: int,
    pairing: tuple[int, ...],
) -> dict[str, Any]:
    dimension = 2 * size

    def plus(index: int) -> int:
        return index

    def minus(index: int) -> int:
        return size + index

    # Plus face advances by +step.
    # Minus face advances by -step.
    transport_map = []

    for index in range(size):
        transport_map.append(
            plus((index + step) % size)
        )

    for index in range(size):
        transport_map.append(
            minus((index - step) % size)
        )

    transport = permutation_matrix(
        tuple(transport_map)
    )

    pairing_inverse = inverse(pairing)

    # Mirror maps plus i to minus pairing[i].
    # Its reverse maps minus j to plus pairing^-1[j].
    mirror_map = []

    for index in range(size):
        mirror_map.append(
            minus(pairing[index])
        )

    for index in range(size):
        mirror_map.append(
            plus(pairing_inverse[index])
        )

    mirror = permutation_matrix(
        tuple(mirror_map)
    )

    identity = sp.eye(dimension)

    # Fold coordinate i receives plus i and minus pairing[i].
    fold = sp.zeros(size, dimension)

    for index in range(size):
        fold[index, plus(index)] = 1
        fold[index, minus(pairing[index])] = 1

    folded_basis = sp.zeros(dimension, size)
    swap_basis = sp.zeros(dimension, size)

    for index in range(size):
        folded_basis[plus(index), index] = 1
        folded_basis[minus(pairing[index]), index] = 1

        swap_basis[plus(index), index] = 1
        swap_basis[minus(pairing[index]), index] = -1

    inverse_transport = transport.inv()

    mirror_involution = (
        mirror**2 == identity
    )

    mirror_reverses_transport = (
        mirror * transport * mirror
        == inverse_transport
    )

    transport_period = (
        transport**size == identity
    )

    fold_rank = fold.rank()
    folded_rank = folded_basis.rank()
    swap_rank = swap_basis.rank()
    joint_rank = (
        folded_basis.row_join(
            swap_basis
        ).rank()
    )

    fold_annihilates_swap = (
        fold * swap_basis
        == sp.zeros(size, size)
    )

    mirror_fixes_folded = (
        mirror * folded_basis
        == folded_basis
    )

    mirror_negates_swap = (
        mirror * swap_basis
        == -swap_basis
    )

    rotation_flag, rotation_offset = (
        is_rotation(pairing)
    )

    anchor_fixed = (
        pairing[0] == 0
    )

    full_pass = all(
        [
            mirror_involution,
            mirror_reverses_transport,
            transport_period,
            fold_rank == size,
            folded_rank == size,
            swap_rank == size,
            joint_rank == dimension,
            fold_annihilates_swap,
            mirror_fixes_folded,
            mirror_negates_swap,
        ]
    )

    return {
        "size": size,
        "step": step,
        "pairing": list(pairing),
        "pairing_text": ",".join(
            str(value)
            for value in pairing
        ),
        "anchor_fixed": anchor_fixed,
        "is_rotation_pairing": rotation_flag,
        "rotation_offset": rotation_offset,
        "mirror_involution": mirror_involution,
        "mirror_reverses_transport": (
            mirror_reverses_transport
        ),
        "transport_period": transport_period,
        "fold_rank": fold_rank,
        "folded_rank": folded_rank,
        "swap_rank": swap_rank,
        "joint_rank": joint_rank,
        "fold_annihilates_swap": (
            fold_annihilates_swap
        ),
        "mirror_fixes_folded": (
            mirror_fixes_folded
        ),
        "mirror_negates_swap": (
            mirror_negates_swap
        ),
        "full_pass": full_pass,
        "anchored_full_pass": (
            full_pass
            and anchor_fixed
        ),
    }


def census(
    size: int,
    step: int,
) -> dict[str, Any]:
    rows = []

    for pairing in itertools.permutations(
        range(size)
    ):
        rows.append(
            build_sheet_operators(
                size=size,
                step=step,
                pairing=pairing,
            )
        )

    passing = [
        row
        for row in rows
        if row["full_pass"]
    ]

    anchored_passing = [
        row
        for row in rows
        if row["anchored_full_pass"]
    ]

    rotation_passing = [
        row
        for row in passing
        if row["is_rotation_pairing"]
    ]

    passing_offsets = sorted(
        row["rotation_offset"]
        for row in rotation_passing
        if row["rotation_offset"] is not None
    )

    expected_translation_count = (
        size
        if gcd(size, step) == 1
        else None
    )

    return {
        "size": size,
        "step": step,
        "gcd_size_step": gcd(size, step),
        "candidate_count": len(rows),
        "factorial_candidate_count": factorial(size),
        "pass_count": len(passing),
        "anchored_pass_count": len(
            anchored_passing
        ),
        "rotation_pass_count": len(
            rotation_passing
        ),
        "all_passes_are_rotations": (
            len(passing)
            == len(rotation_passing)
        ),
        "passing_rotation_offsets": (
            passing_offsets
        ),
        "expected_translation_count_when_coprime": (
            expected_translation_count
        ),
        "passing_pairings": [
            row["pairing"]
            for row in passing
        ],
        "anchored_passing_pairings": [
            row["pairing"]
            for row in anchored_passing
        ],
        "rows": rows,
    }


target = census(
    size=5,
    step=2,
)

pentagon_control = census(
    size=5,
    step=1,
)

reverse_pentagram_control = census(
    size=5,
    step=3,
)

four_sector_control = census(
    size=4,
    step=2,
)

six_sector_control = census(
    size=6,
    step=2,
)


target_checks = {
    "target_candidate_count_is_120": (
        target["candidate_count"] == 120
    ),
    "target_pass_count_is_five": (
        target["pass_count"] == 5
    ),
    "target_all_passes_are_rotations": (
        target[
            "all_passes_are_rotations"
        ]
    ),
    "target_offsets_are_all_five_rotations": (
        target[
            "passing_rotation_offsets"
        ]
        == [0, 1, 2, 3, 4]
    ),
    "target_anchor_selects_unique_pairing": (
        target[
            "anchored_pass_count"
        ]
        == 1
    ),
    "target_unique_anchored_pairing_is_identity": (
        target[
            "anchored_passing_pairings"
        ]
        == [[0, 1, 2, 3, 4]]
    ),
    "four_sector_control_not_same_profile": (
        (
            four_sector_control[
                "pass_count"
            ],
            four_sector_control[
                "anchored_pass_count"
            ],
        )
        != (5, 1)
    ),
    "six_sector_control_not_same_profile": (
        (
            six_sector_control[
                "pass_count"
            ],
            six_sector_control[
                "anchored_pass_count"
            ],
        )
        != (5, 1)
    ),
}

failed_checks = [
    name
    for name, value in target_checks.items()
    if not value
]

audit_pass = len(failed_checks) == 0

if audit_pass:
    verdict = (
        "fivefold_wedge_selected_up_to_rotation_"
        "and_unique_after_anchor"
    )
else:
    verdict = (
        "fivefold_wedge_not_selected_by_tested_laws"
    )

record = {
    "artifact_id": (
        "native_g60_fold_wedge_canonicity_census_054b"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "target": {
        key: value
        for key, value in target.items()
        if key != "rows"
    },
    "controls": {
        "five_sector_pentagon_step_1": {
            key: value
            for key, value in (
                pentagon_control.items()
            )
            if key != "rows"
        },
        "five_sector_reverse_pentagram_step_3": {
            key: value
            for key, value in (
                reverse_pentagram_control.items()
            )
            if key != "rows"
        },
        "four_sector_step_2": {
            key: value
            for key, value in (
                four_sector_control.items()
            )
            if key != "rows"
        },
        "six_sector_step_2": {
            key: value
            for key, value in (
                six_sector_control.items()
            )
            if key != "rows"
        },
    },
    "checks": target_checks,
    "failed_checks": failed_checks,
    "interpretation": (
        "Among all 120 cross-face pairings for the "
        "five-sector step-2 sheet, exactly the five "
        "rotational translations preserve the full "
        "mirror, fold, swap, and counter-transport laws. "
        "Choosing the depicted overlap as sector 0 removes "
        "the residual rotational gauge and selects the "
        "identity pairing uniquely."
    ),
    "boundary": {
        "tests_local_fold_wedge_canonicity": True,
        "tests_all_five_sector_bijections": True,
        "proves_e5_kernel_identification": False,
        "proves_global_g60_closure": False,
        "proves_native_physical_interpretation": False,
        "electron_claim": False,
        "requires_053_for_e5_bridge": True,
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

csv_header = [
    "size",
    "step",
    "pairing",
    "anchor_fixed",
    "is_rotation_pairing",
    "rotation_offset",
    "mirror_involution",
    "mirror_reverses_transport",
    "transport_period",
    "fold_rank",
    "folded_rank",
    "swap_rank",
    "joint_rank",
    "fold_annihilates_swap",
    "mirror_fixes_folded",
    "mirror_negates_swap",
    "full_pass",
    "anchored_full_pass",
]

csv_lines = [
    ",".join(csv_header)
]

for row in target["rows"]:
    csv_lines.append(
        ",".join(
            [
                str(row["size"]),
                str(row["step"]),
                row["pairing_text"],
                str(row["anchor_fixed"]).lower(),
                str(
                    row[
                        "is_rotation_pairing"
                    ]
                ).lower(),
                str(row["rotation_offset"]),
                str(
                    row["mirror_involution"]
                ).lower(),
                str(
                    row[
                        "mirror_reverses_transport"
                    ]
                ).lower(),
                str(
                    row["transport_period"]
                ).lower(),
                str(row["fold_rank"]),
                str(row["folded_rank"]),
                str(row["swap_rank"]),
                str(row["joint_rank"]),
                str(
                    row[
                        "fold_annihilates_swap"
                    ]
                ).lower(),
                str(
                    row[
                        "mirror_fixes_folded"
                    ]
                ).lower(),
                str(
                    row[
                        "mirror_negates_swap"
                    ]
                ).lower(),
                str(row["full_pass"]).lower(),
                str(
                    row["anchored_full_pass"]
                ).lower(),
            ]
        )
    )

CSV_OUT.write_text(
    "\n".join(csv_lines)
    + "\n"
)

note_lines = [
    "# Native G60 fold-wedge canonicity census 054b",
    "",
    f"- audit_pass: `{str(audit_pass).lower()}`",
    f"- verdict: `{verdict}`",
    (
        "- target candidate count: "
        f"`{target['candidate_count']}`"
    ),
    (
        "- target pass count: "
        f"`{target['pass_count']}`"
    ),
    (
        "- target anchored pass count: "
        f"`{target['anchored_pass_count']}`"
    ),
    (
        "- passing rotation offsets: "
        f"`{target['passing_rotation_offsets']}`"
    ),
    "",
    "## Result",
    "",
    (
        "Exactly five pairings preserve all tested "
        "five-sector fold-swap laws. They are the five "
        "rotational translations. Fixing the depicted "
        "overlap as sector 0 selects one pairing."
    ),
    "",
    "## Boundary",
    "",
    (
        "This is a local exact canonicity result for the "
        "specified sheet model. It does not identify the "
        "E5 Gram kernel with this module or prove global "
        "G60 closure."
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
    "target_candidate_count:",
    target["candidate_count"],
)
print(
    "target_pass_count:",
    target["pass_count"],
)
print(
    "target_anchored_pass_count:",
    target["anchored_pass_count"],
)
print(
    "target_passing_offsets:",
    target["passing_rotation_offsets"],
)
print(
    "target_anchored_pairings:",
    target["anchored_passing_pairings"],
)
print(
    "pentagon_control_pass_count:",
    pentagon_control["pass_count"],
)
print(
    "reverse_pentagram_control_pass_count:",
    reverse_pentagram_control[
        "pass_count"
    ],
)
print(
    "four_sector_control_profile:",
    (
        four_sector_control["pass_count"],
        four_sector_control[
            "anchored_pass_count"
        ],
    ),
)
print(
    "six_sector_control_profile:",
    (
        six_sector_control["pass_count"],
        six_sector_control[
            "anchored_pass_count"
        ],
    ),
)
print(
    "failed_check_count:",
    len(failed_checks),
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
