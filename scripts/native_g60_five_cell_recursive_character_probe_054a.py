#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_five_cell_recursive_character_probe_054a.v1.json"
)

NOTE_OUT = (
    ROOT
    / "notes"
    / "native_g60_five_cell_recursive_character_probe_054a.md"
)


# ---------------------------------------------------------------------
# V4 as Z2 x Z2
# ---------------------------------------------------------------------

ONE = (0, 0)
A = (1, 0)
B = (0, 1)
AB = (1, 1)

NAMES = {
    ONE: "1",
    A: "A",
    B: "B",
    AB: "AB",
}


def mul(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return (
        left[0] ^ right[0],
        left[1] ^ right[1],
    )


def name(value: tuple[int, int]) -> str:
    return NAMES[value]


LOCAL_REGISTER = [ONE, A, B, AB]
MIRROR_REGISTER = [AB, B, A, ONE]

# Mirror index j maps to the matching element in LOCAL_REGISTER.
MIRROR_TO_LOCAL = [
    LOCAL_REGISTER.index(value)
    for value in MIRROR_REGISTER
]


# ---------------------------------------------------------------------
# Small disjoint-set implementation for the five-cell quotient
# ---------------------------------------------------------------------

class DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left

        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def node(cell: int, local_index: int) -> int:
    return 4 * cell + local_index


def build_five_cell_quotient(
    damaged_shift: int = 0,
) -> dict[str, Any]:
    cell_count = 5
    dsu = DSU(cell_count * 4)

    hinge_rows = []

    for cell in range(cell_count):
        next_cell = (cell + 1) % cell_count

        for mirror_index, mirror_value in enumerate(MIRROR_REGISTER):
            correct_local_index = MIRROR_TO_LOCAL[mirror_index]

            target_local_index = (
                correct_local_index + damaged_shift
            ) % 4

            dsu.union(
                node(cell, correct_local_index),
                node(next_cell, target_local_index),
            )

            hinge_rows.append(
                {
                    "from_cell": cell,
                    "to_cell": next_cell,
                    "mirror_index": mirror_index,
                    "mirror_value": name(mirror_value),
                    "source_local_index": correct_local_index,
                    "target_local_index": target_local_index,
                    "target_value": name(
                        LOCAL_REGISTER[target_local_index]
                    ),
                    "label_match": (
                        mirror_value
                        == LOCAL_REGISTER[target_local_index]
                    ),
                }
            )

    classes: dict[int, list[dict[str, Any]]] = {}

    for cell in range(cell_count):
        for local_index, value in enumerate(LOCAL_REGISTER):
            root = dsu.find(node(cell, local_index))

            classes.setdefault(root, []).append(
                {
                    "cell": cell,
                    "local_index": local_index,
                    "value": name(value),
                }
            )

    class_rows = []

    for members in classes.values():
        values = sorted(
            {
                member["value"]
                for member in members
            }
        )

        class_rows.append(
            {
                "member_count": len(members),
                "values": values,
                "value_pure": len(values) == 1,
                "members": members,
            }
        )

    quotient_class_count = len(class_rows)
    all_classes_value_pure = all(
        row["value_pure"]
        for row in class_rows
    )

    observed_values = sorted(
        row["values"][0]
        for row in class_rows
        if row["value_pure"]
    )

    expected_values = sorted(
        name(value)
        for value in LOCAL_REGISTER
    )

    same_four_state_character = (
        quotient_class_count == 4
        and all_classes_value_pure
        and observed_values == expected_values
    )

    return {
        "damaged_shift": damaged_shift,
        "hinge_count": len(hinge_rows),
        "all_hinge_labels_match": all(
            row["label_match"]
            for row in hinge_rows
        ),
        "quotient_class_count": quotient_class_count,
        "all_classes_value_pure": all_classes_value_pure,
        "same_four_state_character": same_four_state_character,
        "classes": class_rows,
        "hinges": hinge_rows,
    }


# ---------------------------------------------------------------------
# Exact ten-position folded sheet
#
# Basis order:
#   e_0+, e_1+, e_2+, e_3+, e_4+,
#   e_0-, e_1-, e_2-, e_3-, e_4-
# ---------------------------------------------------------------------

def basis_index(sector: int, side: int) -> int:
    # side 0 is plus, side 1 is minus
    return sector + 5 * side


def permutation_matrix(mapping: list[int]) -> sp.Matrix:
    size = len(mapping)
    matrix = sp.zeros(size, size)

    for source, target in enumerate(mapping):
        matrix[target, source] = 1

    return matrix


def matrix_is_zero(matrix: sp.Matrix) -> bool:
    return matrix == sp.zeros(*matrix.shape)


I10 = sp.eye(10)

# R rotates plus forward and minus backward.
R_MAP = []

for side in (0, 1):
    for sector in range(5):
        if side == 0:
            target_sector = (sector + 1) % 5
        else:
            target_sector = (sector - 1) % 5

        R_MAP.append(
            basis_index(target_sector, side)
        )

R = permutation_matrix(R_MAP)
P = R**2

# J swaps the two faces at the same sector.
J_MAP = []

for side in (0, 1):
    for sector in range(5):
        J_MAP.append(
            basis_index(sector, 1 - side)
        )

J = permutation_matrix(J_MAP)

# Fold map F sends both faces to the same five-sector register.
F = sp.zeros(5, 10)

for sector in range(5):
    F[sector, basis_index(sector, 0)] = 1
    F[sector, basis_index(sector, 1)] = 1

# Symmetric and antisymmetric basis columns.
U = sp.zeros(10, 5)
V = sp.zeros(10, 5)

for sector in range(5):
    U[basis_index(sector, 0), sector] = 1
    U[basis_index(sector, 1), sector] = 1

    V[basis_index(sector, 0), sector] = 1
    V[basis_index(sector, 1), sector] = -1


# ---------------------------------------------------------------------
# Local closure spine
# ---------------------------------------------------------------------

closure_words = {
    "1": ONE,
    "A_squared": mul(A, A),
    "B_squared": mul(B, B),
    "AB_times_BA": mul(
        mul(A, B),
        mul(B, A),
    ),
}

closure_names = {
    key: name(value)
    for key, value in closure_words.items()
}

closure_all_identity = all(
    value == ONE
    for value in closure_words.values()
)


# ---------------------------------------------------------------------
# Exact checks
# ---------------------------------------------------------------------

correct_quotient = build_five_cell_quotient(
    damaged_shift=0,
)

damaged_quotient = build_five_cell_quotient(
    damaged_shift=1,
)

checks = {
    "local_register_has_four_states": (
        len(set(LOCAL_REGISTER)) == 4
    ),
    "mirror_is_reverse_register": (
        MIRROR_REGISTER
        == list(reversed(LOCAL_REGISTER))
    ),
    "closure_spine_all_identity": (
        closure_all_identity
    ),
    "five_cell_hinges_all_match": (
        correct_quotient[
            "all_hinge_labels_match"
        ]
    ),
    "five_cell_quotient_has_four_classes": (
        correct_quotient[
            "quotient_class_count"
        ]
        == 4
    ),
    "five_cell_quotient_is_value_pure": (
        correct_quotient[
            "all_classes_value_pure"
        ]
    ),
    "five_cells_recover_same_v4_character": (
        correct_quotient[
            "same_four_state_character"
        ]
    ),
    "mirror_involution": (
        J**2 == I10
    ),
    "counter_rotation_relation": (
        J * R * J == R.inv()
    ),
    "pentagon_period_five": (
        R**5 == I10
    ),
    "pentagram_period_five": (
        P**5 == I10
    ),
    "pentagram_is_rotation_squared": (
        P == R**2
    ),
    "mirror_reverses_pentagram": (
        J * P * J == P.inv()
    ),
    "fold_rank_is_five": (
        F.rank() == 5
    ),
    "swap_space_rank_is_five": (
        V.rank() == 5
    ),
    "folded_space_rank_is_five": (
        U.rank() == 5
    ),
    "fold_annihilates_swap_space": (
        matrix_is_zero(F * V)
    ),
    "mirror_fixes_folded_space": (
        J * U == U
    ),
    "mirror_negates_swap_space": (
        J * V == -V
    ),
    "fold_and_swap_span_ten_dimensions": (
        U.row_join(V).rank() == 10
    ),
    "damaged_hinge_control_fails_label_match": (
        not damaged_quotient[
            "all_hinge_labels_match"
        ]
    ),
    "damaged_hinge_control_fails_character": (
        not damaged_quotient[
            "same_four_state_character"
        ]
    ),
}

failed_checks = [
    key
    for key, value in checks.items()
    if not value
]

audit_pass = len(failed_checks) == 0

if audit_pass:
    verdict = (
        "five_mirrored_v4_cells_close_to_one_"
        "four_state_character_with_exact_5_plus_5_"
        "fold_swap_module"
    )
else:
    verdict = (
        "five_cell_recursive_character_candidate_failed"
    )

record = {
    "artifact_id": (
        "native_g60_five_cell_recursive_character_probe_054a"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "local_register": [
        name(value)
        for value in LOCAL_REGISTER
    ],
    "mirror_register": [
        name(value)
        for value in MIRROR_REGISTER
    ],
    "closure_spine": closure_names,
    "five_cell_quotient": correct_quotient,
    "damaged_hinge_control": {
        "damaged_shift": 1,
        "all_hinge_labels_match": (
            damaged_quotient[
                "all_hinge_labels_match"
            ]
        ),
        "quotient_class_count": (
            damaged_quotient[
                "quotient_class_count"
            ]
        ),
        "all_classes_value_pure": (
            damaged_quotient[
                "all_classes_value_pure"
            ]
        ),
        "same_four_state_character": (
            damaged_quotient[
                "same_four_state_character"
            ]
        ),
    },
    "folded_sheet": {
        "ambient_dimension": 10,
        "folded_dimension": U.rank(),
        "swap_dimension": V.rank(),
        "fold_rank": F.rank(),
        "joint_dimension": U.row_join(V).rank(),
        "rotation_order": 5,
        "pentagram_step": 2,
        "relations": {
            "J_squared_is_identity": (
                J**2 == I10
            ),
            "R_fifth_is_identity": (
                R**5 == I10
            ),
            "P_fifth_is_identity": (
                P**5 == I10
            ),
            "J_R_J_is_R_inverse": (
                J * R * J == R.inv()
            ),
            "J_P_J_is_P_inverse": (
                J * P * J == P.inv()
            ),
        },
    },
    "checks": checks,
    "failed_checks": failed_checks,
    "interpretation": (
        "Five mirrored V4 cells admit an exact cyclic "
        "hinge quotient with the same four-state V4 "
        "character. Their ten face coordinates split "
        "exactly into five folded modes and five swap "
        "modes. This establishes the proposed local "
        "mechanism conditionally, but does not yet prove "
        "that the E5 Gram kernel realizes this module."
    ),
    "boundary": {
        "e5_kernel_identification_proved": False,
        "g60_global_closure_proved": False,
        "electron_identification_claimed": False,
        "physical_particle_composition_claimed": False,
        "native_uniqueness_proved": False,
        "requires_053_for_e5_bridge": True,
    },
}

JSON_OUT.parent.mkdir(
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

note_lines = [
    "# Native G60 five-cell recursive character probe 054a",
    "",
    f"- audit_pass: `{str(audit_pass).lower()}`",
    f"- verdict: `{verdict}`",
    "- local register: `1, A, B, AB`",
    "- mirror register: `AB, B, A, 1`",
    (
        "- closure spine: "
        "`1 = A^2 = B^2 = (AB)(BA)`"
    ),
    (
        "- five-cell quotient class count: "
        f"`{correct_quotient['quotient_class_count']}`"
    ),
    (
        "- folded dimension: "
        f"`{U.rank()}`"
    ),
    (
        "- swap dimension: "
        f"`{V.rank()}`"
    ),
    (
        "- folded plus swap dimension: "
        f"`{U.row_join(V).rank()}`"
    ),
    "",
    "## Interpretation",
    "",
    (
        "Five mirrored V4 cells close conditionally through "
        "the proposed hinges to a quotient with the same "
        "four-state character. The ten two-sided sector "
        "coordinates split exactly into five folded modes "
        "and five swap modes."
    ),
    "",
    "## Boundary",
    "",
    (
        "This is a local exact mechanism probe. It does not "
        "identify the E5 kernel with this module, prove "
        "global G60 closure, or make a physical electron "
        "composition claim. Artifact 053 is still required "
        "for the E5 bridge."
    ),
    "",
]

NOTE_OUT.write_text(
    "\n".join(note_lines)
)

print("artifact_id:", record["artifact_id"])
print("audit_pass:", record["audit_pass"])
print("verdict:", record["verdict"])
print(
    "closure_spine:",
    record["closure_spine"],
)
print(
    "five_cell_quotient_class_count:",
    correct_quotient["quotient_class_count"],
)
print(
    "same_four_state_character:",
    correct_quotient["same_four_state_character"],
)
print(
    "folded_dimension:",
    U.rank(),
)
print(
    "swap_dimension:",
    V.rank(),
)
print(
    "joint_dimension:",
    U.row_join(V).rank(),
)
print(
    "damaged_control_same_character:",
    damaged_quotient["same_four_state_character"],
)
print("failed_check_count:", len(failed_checks))
print("json:", JSON_OUT)
print("note:", NOTE_OUT)
