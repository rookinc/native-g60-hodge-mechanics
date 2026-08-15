#!/usr/bin/env python3

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PAIR_CSV = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_axis_pair_candidates_037.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_petersen_schrodinger_fold_swap_audit_054d.v1.json"
)

CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_petersen_schrodinger_fold_swap_audit_054d.csv"
)

NOTE_OUT = (
    ROOT
    / "notes"
    / "native_g60_petersen_schrodinger_fold_swap_audit_054d.md"
)

TOLERANCE = 1e-10
EVOLUTION_TIMES = [0.0, 0.25, 0.5, 1.0, 2.0]


def edge(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def adjacency_from_edges(
    order: tuple[str, ...],
    edges: set[tuple[str, str]],
) -> np.ndarray:
    index = {
        vertex: position
        for position, vertex in enumerate(order)
    }

    matrix = np.zeros(
        (len(order), len(order)),
        dtype=np.float64,
    )

    for left, right in edges:
        if left not in index or right not in index:
            continue

        i = index[left]
        j = index[right]

        matrix[i, j] = 1.0
        matrix[j, i] = 1.0

    return matrix


def canonical_cycle(
    cycle: list[str],
) -> tuple[str, ...]:
    variants = []

    for sequence in (
        cycle,
        list(reversed(cycle)),
    ):
        for shift in range(len(sequence)):
            variants.append(
                tuple(
                    sequence[shift:]
                    + sequence[:shift]
                )
            )

    return min(variants)


def find_five_cycles(
    vertices: list[str],
    adjacency: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()

    for start in vertices:
        stack = [(start, [start])]

        while stack:
            current, path = stack.pop()

            if len(path) == 5:
                if start in adjacency[current]:
                    found.add(canonical_cycle(path))
                continue

            for nxt in sorted(adjacency[current]):
                if nxt not in path:
                    stack.append(
                        (
                            nxt,
                            path + [nxt],
                        )
                    )

    return sorted(found)


def cycle_edges(
    cycle: tuple[str, ...],
) -> set[tuple[str, str]]:
    return {
        edge(
            cycle[index],
            cycle[(index + 1) % len(cycle)],
        )
        for index in range(len(cycle))
    }


def permutation_matrix(
    permutation: tuple[int, ...],
) -> np.ndarray:
    size = len(permutation)

    matrix = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    for source, target in enumerate(permutation):
        matrix[target, source] = 1.0

    return matrix


def matrix_rank(
    matrix: np.ndarray,
) -> int:
    return int(
        np.linalg.matrix_rank(
            matrix,
            tol=TOLERANCE,
        )
    )


def maximum_abs(
    matrix: np.ndarray,
) -> float:
    if matrix.size == 0:
        return 0.0

    return float(
        np.max(np.abs(matrix))
    )


def unitary_from_hermitian(
    hamiltonian: np.ndarray,
    time_value: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian
    )

    phases = np.exp(
        -1j * time_value * eigenvalues
    )

    return (
        eigenvectors
        @ np.diag(phases)
        @ eigenvectors.conj().T
    )


with PAIR_CSV.open() as handle:
    pair_rows = list(
        csv.DictReader(handle)
    )

strong_rows = [
    row
    for row in pair_rows
    if row.get("overlap_class")
    == "strong_2_over_3"
]

vertices = sorted(
    {
        row["first_axis"]
        for row in strong_rows
    }
    | {
        row["second_axis"]
        for row in strong_rows
    }
)

strong_edges = {
    edge(
        row["first_axis"],
        row["second_axis"],
    )
    for row in strong_rows
}

adjacency = {
    vertex: set()
    for vertex in vertices
}

for left, right in strong_edges:
    adjacency[left].add(right)
    adjacency[right].add(left)

cycles = find_five_cycles(
    vertices,
    adjacency,
)

cycle_set = set(cycles)
vertex_set = set(vertices)

chart_pairs = []
seen = set()

for first_cycle in cycles:
    complement_set = (
        vertex_set
        - set(first_cycle)
    )

    complement_candidates = [
        candidate
        for candidate in cycles
        if set(candidate) == complement_set
    ]

    if len(complement_candidates) != 1:
        continue

    second_cycle = complement_candidates[0]

    pair_key = tuple(
        sorted(
            (
                canonical_cycle(
                    list(first_cycle)
                ),
                canonical_cycle(
                    list(second_cycle)
                ),
            )
        )
    )

    if pair_key in seen:
        continue

    seen.add(pair_key)
    chart_pairs.append(pair_key)


chart_records: list[dict[str, Any]] = []
reference_spectrum: np.ndarray | None = None

for chart_index, pair in enumerate(
    sorted(chart_pairs),
    start=1,
):
    first_cycle = pair[0]
    second_cycle = pair[1]

    first_set = set(first_cycle)
    second_set = set(second_cycle)

    first_internal = {
        candidate
        for candidate in strong_edges
        if (
            candidate[0] in first_set
            and candidate[1] in first_set
        )
    }

    second_internal = {
        candidate
        for candidate in strong_edges
        if (
            candidate[0] in second_set
            and candidate[1] in second_set
        )
    }

    cross_edges = {
        candidate
        for candidate in strong_edges
        if (
            candidate[0] in first_set
            and candidate[1] in second_set
        )
        or (
            candidate[1] in first_set
            and candidate[0] in second_set
        )
    }

    matching_map: dict[str, str] = {}

    for left, right in cross_edges:
        if left in first_set:
            matching_map[left] = right
        else:
            matching_map[right] = left

    matched_second_order = tuple(
        matching_map[vertex]
        for vertex in first_cycle
    )

    a_plus = adjacency_from_edges(
        first_cycle,
        first_internal,
    )

    a_minus = adjacency_from_edges(
        matched_second_order,
        second_internal,
    )

    matching = np.eye(
        5,
        dtype=np.float64,
    )

    hamiltonian = np.block(
        [
            [a_plus, matching],
            [matching.T, a_minus],
        ]
    )

    hermitian_residual = maximum_abs(
        hamiltonian
        - hamiltonian.conj().T
    )

    spectrum = np.linalg.eigvalsh(
        hamiltonian
    )

    if reference_spectrum is None:
        reference_spectrum = spectrum.copy()

    spectrum_residual = maximum_abs(
        spectrum - reference_spectrum
    )

    mirror_candidates = []

    for permutation in itertools.permutations(
        range(5)
    ):
        q = permutation_matrix(
            permutation
        )

        q_is_involution = (
            maximum_abs(
                q @ q - np.eye(5)
            )
            < TOLERANCE
        )

        if not q_is_involution:
            continue

        conjugacy_residual = maximum_abs(
            q @ a_minus @ q.T
            - a_plus
        )

        j = np.block(
            [
                [
                    np.zeros((5, 5)),
                    q,
                ],
                [
                    q.T,
                    np.zeros((5, 5)),
                ],
            ]
        )

        involution_residual = maximum_abs(
            j @ j - np.eye(10)
        )

        commutator_residual = maximum_abs(
            hamiltonian @ j
            - j @ hamiltonian
        )

        if (
            conjugacy_residual < TOLERANCE
            and involution_residual < TOLERANCE
            and commutator_residual < TOLERANCE
        ):
            mirror_candidates.append(
                {
                    "permutation": list(
                        permutation
                    ),
                    "q": q,
                    "j": j,
                    "conjugacy_residual": (
                        conjugacy_residual
                    ),
                    "involution_residual": (
                        involution_residual
                    ),
                    "commutator_residual": (
                        commutator_residual
                    ),
                }
            )

    selected_mirror = (
        mirror_candidates[0]
        if mirror_candidates
        else None
    )

    plus_dimension = None
    minus_dimension = None
    fold_swap_off_block_residual = None
    fold_swap_block_diagonal = False

    if selected_mirror is not None:
        j = selected_mirror["j"]

        j_eigenvalues, j_eigenvectors = (
            np.linalg.eigh(j)
        )

        plus_mask = (
            j_eigenvalues > 0.0
        )
        minus_mask = (
            j_eigenvalues < 0.0
        )

        plus_basis = j_eigenvectors[
            :,
            plus_mask,
        ]

        minus_basis = j_eigenvectors[
            :,
            minus_mask,
        ]

        plus_dimension = int(
            plus_basis.shape[1]
        )

        minus_dimension = int(
            minus_basis.shape[1]
        )

        off_block = (
            plus_basis.conj().T
            @ hamiltonian
            @ minus_basis
        )

        fold_swap_off_block_residual = (
            maximum_abs(off_block)
        )

        fold_swap_block_diagonal = (
            fold_swap_off_block_residual
            < TOLERANCE
        )

    initial_state = np.zeros(
        10,
        dtype=np.complex128,
    )
    initial_state[0] = 1.0

    unitary_records = []

    for time_value in EVOLUTION_TIMES:
        evolution = unitary_from_hermitian(
            hamiltonian,
            time_value,
        )

        unitary_residual = maximum_abs(
            evolution.conj().T
            @ evolution
            - np.eye(10)
        )

        evolved_state = (
            evolution
            @ initial_state
        )

        norm_error = abs(
            float(
                np.vdot(
                    evolved_state,
                    evolved_state,
                ).real
            )
            - 1.0
        )

        unitary_records.append(
            {
                "time": time_value,
                "unitary_residual": (
                    unitary_residual
                ),
                "norm_error": (
                    norm_error
                ),
            }
        )

    maximum_unitary_residual = max(
        row["unitary_residual"]
        for row in unitary_records
    )

    maximum_norm_error = max(
        row["norm_error"]
        for row in unitary_records
    )

    chart_pass = all(
        [
            hermitian_residual < TOLERANCE,
            spectrum_residual < TOLERANCE,
            selected_mirror is not None,
            plus_dimension == 5,
            minus_dimension == 5,
            fold_swap_block_diagonal,
            maximum_unitary_residual
            < TOLERANCE,
            maximum_norm_error
            < TOLERANCE,
        ]
    )

    chart_records.append(
        {
            "chart_index": chart_index,
            "first_cycle": list(
                first_cycle
            ),
            "matched_second_order": list(
                matched_second_order
            ),
            "hamiltonian_dimension": 10,
            "hermitian_residual": (
                hermitian_residual
            ),
            "spectrum": [
                float(value)
                for value in spectrum
            ],
            "spectrum_residual_from_chart_1": (
                spectrum_residual
            ),
            "mirror_candidate_count": len(
                mirror_candidates
            ),
            "selected_mirror_permutation": (
                selected_mirror[
                    "permutation"
                ]
                if selected_mirror is not None
                else None
            ),
            "plus_dimension": (
                plus_dimension
            ),
            "minus_dimension": (
                minus_dimension
            ),
            "fold_swap_off_block_residual": (
                fold_swap_off_block_residual
            ),
            "fold_swap_block_diagonal": (
                fold_swap_block_diagonal
            ),
            "maximum_unitary_residual": (
                maximum_unitary_residual
            ),
            "maximum_norm_error": (
                maximum_norm_error
            ),
            "unitary_samples": (
                unitary_records
            ),
            "chart_pass": chart_pass,
        }
    )


all_hermitian = all(
    row["hermitian_residual"]
    < TOLERANCE
    for row in chart_records
)

all_spectra_match = all(
    row[
        "spectrum_residual_from_chart_1"
    ]
    < TOLERANCE
    for row in chart_records
)

all_have_native_mirror = all(
    row["mirror_candidate_count"] > 0
    for row in chart_records
)

all_have_five_plus_five = all(
    row["plus_dimension"] == 5
    and row["minus_dimension"] == 5
    for row in chart_records
)

all_block_diagonal = all(
    row["fold_swap_block_diagonal"]
    for row in chart_records
)

all_unitary = all(
    row["maximum_unitary_residual"]
    < TOLERANCE
    for row in chart_records
)

all_norm_preserving = all(
    row["maximum_norm_error"]
    < TOLERANCE
    for row in chart_records
)

checks = {
    "chart_count_is_six": (
        len(chart_records) == 6
    ),
    "all_hamiltonians_are_hermitian": (
        all_hermitian
    ),
    "all_chart_spectra_match": (
        all_spectra_match
    ),
    "all_charts_have_commuting_mirror": (
        all_have_native_mirror
    ),
    "all_mirror_splits_are_five_plus_five": (
        all_have_five_plus_five
    ),
    "all_hamiltonians_block_diagonalize_in_fold_swap_basis": (
        all_block_diagonal
    ),
    "all_sampled_evolutions_are_unitary": (
        all_unitary
    ),
    "all_sampled_state_norms_are_preserved": (
        all_norm_preserving
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
        "native_petersen_atlas_supports_"
        "six_chart_unitary_fold_swap_schrodinger_dynamics"
    )
else:
    verdict = (
        "native_petersen_adjacency_dynamics_is_unitary_"
        "but_fold_swap_separation_not_fully_established"
    )

record = {
    "artifact_id": (
        "native_g60_petersen_schrodinger_fold_swap_audit_054d"
    ),
    "audit_pass": audit_pass,
    "verdict": verdict,
    "hamiltonian_definition": {
        "alpha": 1,
        "beta": 1,
        "form": "[[A_plus, M], [M_transpose, A_minus]]",
        "units": "dimensionless_probe_units",
    },
    "chart_records": chart_records,
    "checks": checks,
    "failed_checks": failed_checks,
    "interpretation": (
        "The probe constructs a dimensionless adjacency "
        "Hamiltonian from each native complementary "
        "Petersen chart. Hermiticity and sampled unitarity "
        "test whether a discrete Schrodinger evolution is "
        "mathematically available. A commuting mirror with "
        "five positive and five negative modes tests whether "
        "fold and swap are invariant dynamical sectors."
    ),
    "boundary": {
        "discrete_schrodinger_model_constructed": True,
        "physical_time_derived": False,
        "hbar_derived": False,
        "physical_hamiltonian_uniqueness_proved": False,
        "electron_wavefunction_identified": False,
        "born_rule_derived": False,
        "continuum_quantum_mechanics_recovered": False,
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
        "chart_index,hermitian_residual,"
        "spectrum_residual_from_chart_1,"
        "mirror_candidate_count,"
        "plus_dimension,minus_dimension,"
        "fold_swap_off_block_residual,"
        "fold_swap_block_diagonal,"
        "maximum_unitary_residual,"
        "maximum_norm_error,chart_pass"
    )
]

for row in chart_records:
    csv_lines.append(
        ",".join(
            [
                str(row["chart_index"]),
                repr(
                    row[
                        "hermitian_residual"
                    ]
                ),
                repr(
                    row[
                        "spectrum_residual_from_chart_1"
                    ]
                ),
                str(
                    row[
                        "mirror_candidate_count"
                    ]
                ),
                str(row["plus_dimension"]),
                str(row["minus_dimension"]),
                repr(
                    row[
                        "fold_swap_off_block_residual"
                    ]
                ),
                str(
                    row[
                        "fold_swap_block_diagonal"
                    ]
                ).lower(),
                repr(
                    row[
                        "maximum_unitary_residual"
                    ]
                ),
                repr(
                    row[
                        "maximum_norm_error"
                    ]
                ),
                str(
                    row["chart_pass"]
                ).lower(),
            ]
        )
    )

CSV_OUT.write_text(
    "\n".join(csv_lines)
    + "\n"
)

note_lines = [
    "# Native G60 Petersen Schrodinger fold-swap audit 054d",
    "",
    f"- audit_pass: `{str(audit_pass).lower()}`",
    f"- verdict: `{verdict}`",
    f"- chart count: `{len(chart_records)}`",
    f"- all Hermitian: `{str(all_hermitian).lower()}`",
    f"- all spectra match: `{str(all_spectra_match).lower()}`",
    f"- all have commuting mirror: `{str(all_have_native_mirror).lower()}`",
    f"- all have 5+5 mirror split: `{str(all_have_five_plus_five).lower()}`",
    f"- all fold/swap block diagonal: `{str(all_block_diagonal).lower()}`",
    f"- all sampled evolutions unitary: `{str(all_unitary).lower()}`",
    f"- all sampled norms preserved: `{str(all_norm_preserving).lower()}`",
    "",
    "## Boundary",
    "",
    (
        "This is a dimensionless finite adjacency "
        "Hamiltonian probe. It does not derive physical "
        "time, hbar, the physical electron Hamiltonian, "
        "the Born rule, or continuum quantum mechanics."
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
    "all_hamiltonians_hermitian:",
    all_hermitian,
)
print(
    "all_chart_spectra_match:",
    all_spectra_match,
)
print(
    "all_charts_have_commuting_mirror:",
    all_have_native_mirror,
)
print(
    "all_mirror_splits_are_5_plus_5:",
    all_have_five_plus_five,
)
print(
    "all_fold_swap_block_diagonal:",
    all_block_diagonal,
)
print(
    "all_sampled_evolutions_unitary:",
    all_unitary,
)
print(
    "all_sampled_norms_preserved:",
    all_norm_preserving,
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
