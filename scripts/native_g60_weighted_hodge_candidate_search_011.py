from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

ENERGY_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_discrete_energy_conservation_010.json"
)

ACTION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

B1_PATH = SOURCE_ROOT / "native_g60_B1_vertex_edge_004.csv"
B2_PATH = SOURCE_ROOT / "native_g60_B2_edge_face_004.csv"

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_weighted_hodge_candidate_search_011.json"
)

CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_weighted_hodge_candidate_search_011.csv"
)

TOLERANCE = 1e-9

WEIGHT_VALUES = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
)


def read_matrix_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    return np.array(
        [
            [float(value) for value in row[1:]]
            for row in rows[1:]
        ],
        dtype=np.float64,
    )


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(np.max(np.abs(array)))


def numerical_rank(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> int:
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    if singular_values.size == 0:
        return 0

    threshold = (
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * singular_values[0]
    )

    threshold = max(threshold, tolerance)

    return int(
        np.count_nonzero(
            singular_values > threshold
        )
    )


def action_orbits(
    action_targets: np.ndarray,
    size: int,
) -> list[list[int]]:
    adjacency = {
        index: set()
        for index in range(size)
    }

    for action in action_targets:
        for source, target in enumerate(action):
            target_int = int(target)

            adjacency[source].add(target_int)
            adjacency[target_int].add(source)

    unseen = set(range(size))
    orbits = []

    while unseen:
        start = min(unseen)
        reached = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor in adjacency[current]:
                if neighbor in reached:
                    continue

                reached.add(neighbor)
                queue.append(neighbor)

        unseen -= reached
        orbits.append(sorted(reached))

    orbits.sort(
        key=lambda orbit: (
            len(orbit),
            orbit[0],
        )
    )

    return orbits


def diagonal_invariance_residual(
    weights: np.ndarray,
    targets: np.ndarray,
) -> float:
    maximum = 0.0

    for target in targets:
        maximum = max(
            maximum,
            max_abs(
                weights[target] - weights
            ),
        )

    return maximum


def weighted_operators(
    b1: np.ndarray,
    b2: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> dict[str, np.ndarray]:
    x = b / a
    y = c / b

    exact_term = (
        x * (b1.T @ b1)
    )

    coexact_term = (
        y * (b2 @ b2.T)
    )

    delta0 = (
        x * (b1 @ b1.T)
    )

    delta1 = (
        exact_term + coexact_term
    )

    delta2 = (
        y * (b2.T @ b2)
    )

    return {
        "Delta0": delta0,
        "Delta1": delta1,
        "Delta2": delta2,
        "exact_term": exact_term,
        "coexact_term": coexact_term,
    }


def spectrum_summary(
    matrix: np.ndarray,
) -> dict:
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (matrix + matrix.T)
    )

    eigenvalues[
        np.abs(eigenvalues) < TOLERANCE
    ] = 0.0

    positive = eigenvalues[
        eigenvalues > TOLERANCE
    ]

    return {
        "rank": numerical_rank(matrix),
        "kernel_dimension": (
            matrix.shape[0]
            - numerical_rank(matrix)
        ),
        "negative_count": int(
            np.count_nonzero(
                eigenvalues < -TOLERANCE
            )
        ),
        "smallest_positive": (
            float(positive[0])
            if len(positive)
            else None
        ),
        "largest_positive": (
            float(positive[-1])
            if len(positive)
            else None
        ),
        "trace": float(
            np.sum(eigenvalues)
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    energy = json.loads(
        ENERGY_PATH.read_text(
            encoding="utf-8"
        )
    )

    actions = np.load(
        ACTION_PATH
    )

    vertex_targets = np.array(
        actions["vertex_target"],
        dtype=np.int64,
    )

    edge_targets = np.array(
        actions["edge_target"],
        dtype=np.int64,
    )

    face_targets = np.array(
        actions["face_target"],
        dtype=np.int64,
    )

    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)

    vertex_orbits = action_orbits(
        vertex_targets,
        60,
    )

    edge_orbits = action_orbits(
        edge_targets,
        120,
    )

    face_orbits = action_orbits(
        face_targets,
        20,
    )

    rng = np.random.default_rng(
        46011
    )

    random_vertex_weights = rng.uniform(
        0.5,
        2.0,
        size=60,
    )

    random_edge_weights = rng.uniform(
        0.5,
        2.0,
        size=120,
    )

    random_face_weights = rng.uniform(
        0.5,
        2.0,
        size=20,
    )

    random_invariance = {
        "vertex": diagonal_invariance_residual(
            random_vertex_weights,
            vertex_targets,
        ),
        "edge": diagonal_invariance_residual(
            random_edge_weights,
            edge_targets,
        ),
        "face": diagonal_invariance_residual(
            random_face_weights,
            face_targets,
        ),
    }

    uniform_invariance = {
        "vertex": diagonal_invariance_residual(
            np.ones(60),
            vertex_targets,
        ),
        "edge": diagonal_invariance_residual(
            np.ones(120),
            edge_targets,
        ),
        "face": diagonal_invariance_residual(
            np.ones(20),
            face_targets,
        ),
    }

    candidate_rows = []

    for a, b, c in itertools.product(
        WEIGHT_VALUES,
        repeat=3,
    ):
        operators = weighted_operators(
            b1,
            b2,
            a,
            b,
            c,
        )

        delta0_summary = spectrum_summary(
            operators["Delta0"]
        )

        delta1_summary = spectrum_summary(
            operators["Delta1"]
        )

        delta2_summary = spectrum_summary(
            operators["Delta2"]
        )

        exact_coexact_left = (
            operators["exact_term"]
            @ operators["coexact_term"]
        )

        exact_coexact_right = (
            operators["coexact_term"]
            @ operators["exact_term"]
        )

        x = b / a
        y = c / b

        candidate_rows.append(
            {
                "a_star0": a,
                "b_star1": b,
                "c_star2": c,
                "x_b_over_a": x,
                "y_c_over_b": y,
                "Delta0_rank": (
                    delta0_summary["rank"]
                ),
                "Delta0_kernel": (
                    delta0_summary[
                        "kernel_dimension"
                    ]
                ),
                "Delta1_rank": (
                    delta1_summary["rank"]
                ),
                "Delta1_kernel": (
                    delta1_summary[
                        "kernel_dimension"
                    ]
                ),
                "Delta2_rank": (
                    delta2_summary["rank"]
                ),
                "Delta2_kernel": (
                    delta2_summary[
                        "kernel_dimension"
                    ]
                ),
                "Delta1_smallest_positive": (
                    delta1_summary[
                        "smallest_positive"
                    ]
                ),
                "Delta1_largest_positive": (
                    delta1_summary[
                        "largest_positive"
                    ]
                ),
                "Delta1_negative_count": (
                    delta1_summary[
                        "negative_count"
                    ]
                ),
                "exact_coexact_left_max_abs": (
                    max_abs(
                        exact_coexact_left
                    )
                ),
                "exact_coexact_right_max_abs": (
                    max_abs(
                        exact_coexact_right
                    )
                ),
                "dimension_partition_pass": (
                    numerical_rank(
                        operators["exact_term"]
                    )
                    == 59
                    and delta1_summary[
                        "kernel_dimension"
                    ]
                    == 42
                    and numerical_rank(
                        operators["coexact_term"]
                    )
                    == 19
                ),
                "all_checks_pass": (
                    delta0_summary["rank"] == 59
                    and delta0_summary[
                        "kernel_dimension"
                    ]
                    == 1
                    and delta1_summary["rank"] == 78
                    and delta1_summary[
                        "kernel_dimension"
                    ]
                    == 42
                    and delta2_summary["rank"] == 19
                    and delta2_summary[
                        "kernel_dimension"
                    ]
                    == 1
                    and delta1_summary[
                        "negative_count"
                    ]
                    == 0
                    and max_abs(
                        exact_coexact_left
                    )
                    < TOLERANCE
                    and max_abs(
                        exact_coexact_right
                    )
                    < TOLERANCE
                ),
            }
        )

    all_candidates_pass = all(
        row["all_checks_pass"]
        for row in candidate_rows
    )

    ratio_pairs = sorted(
        {
            (
                row["x_b_over_a"],
                row["y_c_over_b"],
            )
            for row in candidate_rows
        }
    )

    same_ratio_groups = {}

    for row in candidate_rows:
        key = (
            row["x_b_over_a"],
            row["y_c_over_b"],
        )

        same_ratio_groups.setdefault(
            key,
            [],
        ).append(row)

    same_ratio_spectrum_pass = True
    same_ratio_max_difference = 0.0

    for rows in same_ratio_groups.values():
        reference_smallest = rows[0][
            "Delta1_smallest_positive"
        ]
        reference_largest = rows[0][
            "Delta1_largest_positive"
        ]

        for row in rows[1:]:
            difference = max(
                abs(
                    row[
                        "Delta1_smallest_positive"
                    ]
                    - reference_smallest
                ),
                abs(
                    row[
                        "Delta1_largest_positive"
                    ]
                    - reference_largest
                ),
            )

            same_ratio_max_difference = max(
                same_ratio_max_difference,
                difference,
            )

            if difference >= TOLERANCE:
                same_ratio_spectrum_pass = False

    checks = {
        "input_energy_audit_pass": (
            energy.get("audit_pass") is True
        ),
        "vertex_action_is_transitive": (
            [len(orbit) for orbit in vertex_orbits]
            == [60]
        ),
        "edge_action_is_transitive": (
            [len(orbit) for orbit in edge_orbits]
            == [120]
        ),
        "face_action_is_transitive": (
            [len(orbit) for orbit in face_orbits]
            == [20]
        ),
        "uniform_diagonal_weights_are_invariant": (
            max(uniform_invariance.values())
            < TOLERANCE
        ),
        "generic_nonuniform_weights_fail_invariance": (
            min(random_invariance.values())
            > TOLERANCE
        ),
        "diagonal_invariant_parameter_count_is_3": (
            len(vertex_orbits)
            + len(edge_orbits)
            + len(face_orbits)
            == 3
        ),
        "sampled_positive_candidate_count_is_125": (
            len(candidate_rows) == 125
        ),
        "all_sampled_positive_candidates_pass": (
            all_candidates_pass
        ),
        "all_sampled_harmonic_dimensions_are_42": (
            {
                row["Delta1_kernel"]
                for row in candidate_rows
            }
            == {42}
        ),
        "all_sampled_exact_dimensions_are_59": (
            {
                row["Delta0_rank"]
                for row in candidate_rows
            }
            == {59}
        ),
        "all_sampled_coexact_dimensions_are_19": (
            {
                row["Delta2_rank"]
                for row in candidate_rows
            }
            == {19}
        ),
        "same_ratio_triples_give_same_Delta1_extrema": (
            same_ratio_spectrum_pass
        ),
    }

    audit_pass = all(
        checks.values()
    )

    payload = {
        "artifact_id": (
            "native_g60_weighted_hodge_candidate_search_011"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "full_symmetry_forces_three_scalar_positive_hodge_family"
            if audit_pass
            else "native_g60_weighted_hodge_candidate_search_failed"
        ),
        "inputs": {
            "energy_conservation": str(
                ENERGY_PATH.relative_to(ROOT)
            ),
            "signed_cochain_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
            "b1": str(
                B1_PATH.relative_to(ROOT)
            ),
            "b2": str(
                B2_PATH.relative_to(ROOT)
            ),
        },
        "cell_action_orbits": {
            "vertex_orbits": vertex_orbits,
            "edge_orbits": edge_orbits,
            "face_orbits": face_orbits,
            "vertex_orbit_sizes": [
                len(orbit)
                for orbit in vertex_orbits
            ],
            "edge_orbit_sizes": [
                len(orbit)
                for orbit in edge_orbits
            ],
            "face_orbit_sizes": [
                len(orbit)
                for orbit in face_orbits
            ],
        },
        "exact_diagonal_classification": {
            "star0": "a I60",
            "star1": "b I120",
            "star2": "c I20",
            "positivity_conditions": (
                "a > 0, b > 0, c > 0"
            ),
            "raw_parameter_count": 3,
            "operator_ratio_parameters": {
                "x": "b/a",
                "y": "c/b",
            },
            "effective_operator_parameter_count": 2,
            "reason": (
                "A diagonal operator commuting with a transitive "
                "permutation action is constant on that action orbit."
            ),
        },
        "weighted_operators": {
            "delta1_vertex_to_edge_adjoint": (
                "(b/a) B1"
            ),
            "delta2_face_to_edge_adjoint": (
                "(c/b) B2"
            ),
            "Delta0": (
                "(b/a) B1 B1^T"
            ),
            "Delta1": (
                "(b/a) B1^T B1 "
                "+ (c/b) B2 B2^T"
            ),
            "Delta2": (
                "(c/b) B2^T B2"
            ),
        },
        "checks": checks,
        "invariance_controls": {
            "uniform_weight_residuals": (
                uniform_invariance
            ),
            "generic_nonuniform_weight_residuals": (
                random_invariance
            ),
        },
        "sampled_family": {
            "weight_values": list(
                WEIGHT_VALUES
            ),
            "triple_count": len(
                candidate_rows
            ),
            "distinct_ratio_pair_count": len(
                ratio_pairs
            ),
            "all_candidates_pass": (
                all_candidates_pass
            ),
            "same_ratio_spectrum_max_difference": (
                same_ratio_max_difference
            ),
        },
        "candidate_rows": candidate_rows,
        "earned_interpretation": {
            "uniform_identity_weights": (
                "The point a=b=c=1 inside a larger "
                "symmetry-compatible positive family."
            ),
            "topological_sector_dimensions": (
                "Remain 59 exact, 42 harmonic, and 19 coexact "
                "for every positive a,b,c."
            ),
            "spectral_effect": (
                "The exact spectrum scales with x=b/a, "
                "and the coexact spectrum scales with y=c/b."
            ),
            "native_symmetry_selects_unique_numeric_weights": False,
            "native_symmetry_selects_weight_form": True,
        },
        "outputs": {
            "candidate_csv": str(
                CSV_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "full_symmetry_compatible_diagonal_family_classified": (
                audit_pass
            ),
            "positive_sample_family_audited": (
                audit_pass
            ),
            "identity_weight_baseline_unique": False,
            "three_scalar_form_is_symmetry_forced": (
                audit_pass
            ),
            "two_operator_ratios_remain_free": True,
            "non_diagonal_commutant_weights_classified": False,
            "native_metric_derived": False,
            "constitutive_ratios_derived": False,
            "physical_units_derived": False,
            "physical_constants_derived": False,
            "electromagnetism_claim": False,
            "maxwell_claim": False,
            "physical_energy_claim": False,
            "physical_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False
        }
    }

    JSON_OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                candidate_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            candidate_rows
        )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "vertex/edge/face_orbit_sizes:",
        payload["cell_action_orbits"][
            "vertex_orbit_sizes"
        ],
        payload["cell_action_orbits"][
            "edge_orbit_sizes"
        ],
        payload["cell_action_orbits"][
            "face_orbit_sizes"
        ],
    )
    print(
        "diagonal_invariant_parameter_count:",
        3,
    )
    print(
        "effective_operator_ratio_count:",
        2,
    )
    print(
        "generic_nonuniform_invariance_residuals:",
        random_invariance,
    )
    print(
        "sampled_candidate_count:",
        len(candidate_rows),
    )
    print(
        "distinct_ratio_pair_count:",
        len(ratio_pairs),
    )
    print(
        "all_sampled_candidates_pass:",
        all_candidates_pass,
    )
    print(
        "sampled_Delta1_kernel_dimensions:",
        sorted(
            {
                row["Delta1_kernel"]
                for row in candidate_rows
            }
        ),
    )
    print(
        "same_ratio_spectrum_max_difference:",
        same_ratio_max_difference,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CSV_OUT)


if __name__ == "__main__":
    main()
