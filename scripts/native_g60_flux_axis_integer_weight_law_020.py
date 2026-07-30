from __future__ import annotations

import csv
import json
import math
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

AXIS_REGISTER_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_four_flux_axis_register_018.json"
)

AXIS_REGISTER_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_axis_register_018.npz"
)

RELATION_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_flux_axis_to_face_relation_019.json"
)

GEOMETRY_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_face_geometry_017.npz"
)

ACTION_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_flux_axis_integer_weight_law_020.json"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_flux_axis_integer_weight_law_020.csv"
)

ORBIT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_flux_axis_integer_weight_orbits_020.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_flux_axis_integer_weight_law_020.npz"
)

TOLERANCE = 1e-9
RATIO_TOLERANCE = 1e-7


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

    threshold = max(
        tolerance,
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * singular_values[0],
    )

    return int(
        np.count_nonzero(
            singular_values > threshold
        )
    )


def stabilizer_face_orbits(
    stabilizer_indices: set[int],
    face_targets: np.ndarray,
) -> list[list[int]]:
    adjacency = {
        face: set()
        for face in range(20)
    }

    for group_index in stabilizer_indices:
        action = face_targets[group_index]

        for source, target in enumerate(action):
            target_int = int(target)

            adjacency[source].add(target_int)
            adjacency[target_int].add(source)

    unseen = set(range(20))
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

    # Canonical order is by descending orbit size: 12, 6, 2.
    orbits.sort(
        key=lambda orbit: (
            -len(orbit),
            orbit[0],
        )
    )

    return orbits


def normalized_orbit_indicator_basis(
    orbits: list[list[int]],
) -> np.ndarray:
    basis = np.zeros(
        (20, len(orbits)),
        dtype=np.float64,
    )

    for column, orbit in enumerate(orbits):
        value = 1.0 / math.sqrt(len(orbit))

        for face in orbit:
            basis[face, column] = value

    return basis


def nullspace(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(
        matrix,
        full_matrices=True,
    )

    threshold = max(
        tolerance,
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * (
            singular_values[0]
            if singular_values.size
            else 1.0
        ),
    )

    rank = int(
        np.count_nonzero(
            singular_values > threshold
        )
    )

    return vh[rank:].T


def primitive_integer_ratio(
    values: list[float],
) -> tuple[list[int], list[Fraction]]:
    nonzero = [
        abs(value)
        for value in values
        if abs(value) > TOLERANCE
    ]

    if not nonzero:
        raise RuntimeError("cannot form ratio from zero values")

    scale = min(nonzero)

    fractions = [
        Fraction(value / scale).limit_denominator(120)
        for value in values
    ]

    denominator_lcm = 1

    for value in fractions:
        denominator_lcm = math.lcm(
            denominator_lcm,
            value.denominator,
        )

    integers = [
        value.numerator
        * (
            denominator_lcm
            // value.denominator
        )
        for value in fractions
    ]

    gcd_value = 0

    for value in integers:
        gcd_value = math.gcd(
            gcd_value,
            abs(value),
        )

    if gcd_value > 0:
        integers = [
            value // gcd_value
            for value in integers
        ]

    # Canonical sign: the size-2 summit class is positive.
    if integers[-1] < 0:
        integers = [-value for value in integers]
        fractions = [-value for value in fractions]

    return integers, fractions


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    AXIS_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    ORBIT_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    axis_register = json.loads(
        AXIS_REGISTER_PATH.read_text(encoding="utf-8")
    )

    relation = json.loads(
        RELATION_PATH.read_text(encoding="utf-8")
    )

    axis_data = np.load(
        AXIS_REGISTER_NPZ_PATH
    )

    geometry_data = np.load(
        GEOMETRY_NPZ_PATH
    )

    action_data = np.load(
        ACTION_PATH
    )

    axis_matrix = np.array(
        axis_data["axis_matrix"],
        dtype=np.float64,
    )

    flux_basis = np.array(
        geometry_data["face_flux_basis"],
        dtype=np.float64,
    )

    flux_projector = np.array(
        geometry_data["face_flux_projector"],
        dtype=np.float64,
    )

    flux_representation = np.array(
        geometry_data["face_flux_representation"],
        dtype=np.float64,
    )

    face_targets = np.array(
        action_data["face_target"],
        dtype=np.int64,
    )

    face_signs = np.array(
        action_data["face_sign"],
        dtype=np.int8,
    )

    if axis_matrix.shape != (4, 20):
        raise RuntimeError(
            f"unexpected axis matrix shape: {axis_matrix.shape}"
        )

    identity_face = np.eye(
        20,
        dtype=np.float64,
    )

    axis_rows = []
    orbit_rows = []

    recovered_patterns = []
    intersection_lines = []

    ratio_profile = Counter()
    orbit_size_profile = Counter()
    intersection_dimension_profile = Counter()

    global_residuals = {
        "axis_pattern_stabilizer_invariance": 0.0,
        "orbit_constancy": 0.0,
        "intersection_reconstruction": 0.0,
        "axis_pattern_alignment": 0.0,
        "zero_sum": 0.0,
        "unit_norm": 0.0,
        "exact_weight_formula": 0.0,
    }

    for axis_id in range(20):
        axis_coordinate = axis_matrix[:, axis_id]

        axis_pattern = (
            flux_basis @ axis_coordinate
        )

        stabilizer = {
            index
            for index in range(480)
            if max_abs(
                flux_representation[index]
                @ axis_coordinate
                - axis_coordinate
            )
            < TOLERANCE
        }

        if len(stabilizer) != 24:
            raise RuntimeError(
                f"axis {axis_id} stabilizer order "
                f"{len(stabilizer)}, expected 24"
            )

        orbits = stabilizer_face_orbits(
            stabilizer,
            face_targets,
        )

        orbit_sizes = [
            len(orbit)
            for orbit in orbits
        ]

        orbit_size_profile[
            tuple(orbit_sizes)
        ] += 1

        if orbit_sizes != [12, 6, 2]:
            raise RuntimeError(
                f"axis {axis_id} orbit sizes "
                f"{orbit_sizes}, expected [12, 6, 2]"
            )

        indicator_basis = (
            normalized_orbit_indicator_basis(
                orbits
            )
        )

        # The invariant orbit-constant space is three-dimensional.
        # Its intersection with the four-flux space is the nullspace of
        # (I-P_flux) restricted to that orbit-constant space.
        escape_matrix = (
            identity_face - flux_projector
        ) @ indicator_basis

        intersection_coefficients = nullspace(
            escape_matrix
        )

        intersection_dimension = int(
            intersection_coefficients.shape[1]
        )

        intersection_dimension_profile[
            intersection_dimension
        ] += 1

        if intersection_dimension != 1:
            raise RuntimeError(
                f"axis {axis_id} intersection dimension "
                f"{intersection_dimension}, expected 1"
            )

        coefficient = (
            intersection_coefficients[:, 0]
        )

        recovered_pattern = (
            indicator_basis @ coefficient
        )

        recovered_pattern /= np.linalg.norm(
            recovered_pattern
        )

        correlation = float(
            np.dot(
                recovered_pattern,
                axis_pattern,
            )
        )

        if correlation < 0:
            recovered_pattern *= -1.0
            coefficient *= -1.0
            correlation *= -1.0

        # Canonical sign is positive on the size-2 class.
        size_two_mean = float(
            np.mean(
                recovered_pattern[
                    orbits[2]
                ]
            )
        )

        if size_two_mean < 0:
            recovered_pattern *= -1.0
            coefficient *= -1.0
            axis_pattern *= -1.0

        orbit_values = [
            float(
                np.mean(
                    recovered_pattern[orbit]
                )
            )
            for orbit in orbits
        ]

        orbit_spreads = [
            max_abs(
                recovered_pattern[orbit]
                - orbit_values[index]
            )
            for index, orbit in enumerate(
                orbits
            )
        ]

        integer_ratio, rational_ratio = (
            primitive_integer_ratio(
                orbit_values
            )
        )

        ratio_key = tuple(integer_ratio)
        ratio_profile[ratio_key] += 1

        expected_scale = (
            1.0 / (6.0 * math.sqrt(5.0))
        )

        expected_values = np.array(
            [
                expected_scale,
                -4.0 * expected_scale,
                6.0 * expected_scale,
            ],
            dtype=np.float64,
        )

        # Stabilizer action on the signed face pattern.
        stabilizer_residual = 0.0

        for group_index in stabilizer:
            transformed = np.empty_like(
                recovered_pattern
            )

            transformed[
                face_targets[group_index]
            ] = (
                face_signs[group_index]
                * recovered_pattern
            )

            stabilizer_residual = max(
                stabilizer_residual,
                max_abs(
                    transformed
                    - recovered_pattern
                ),
            )

        zero_sum_residual = abs(
            float(
                np.sum(
                    recovered_pattern
                )
            )
        )

        unit_norm_residual = abs(
            float(
                np.linalg.norm(
                    recovered_pattern
                )
            )
            - 1.0
        )

        intersection_reconstruction_residual = (
            max_abs(
                (
                    identity_face
                    - flux_projector
                )
                @ recovered_pattern
            )
        )

        alignment_residual = min(
            max_abs(
                recovered_pattern
                - axis_pattern
            ),
            max_abs(
                recovered_pattern
                + axis_pattern
            ),
        )

        exact_weight_residual = max_abs(
            np.array(
                orbit_values,
                dtype=np.float64,
            )
            - expected_values
        )

        global_residuals[
            "axis_pattern_stabilizer_invariance"
        ] = max(
            global_residuals[
                "axis_pattern_stabilizer_invariance"
            ],
            stabilizer_residual,
        )

        global_residuals[
            "orbit_constancy"
        ] = max(
            global_residuals[
                "orbit_constancy"
            ],
            max(orbit_spreads),
        )

        global_residuals[
            "intersection_reconstruction"
        ] = max(
            global_residuals[
                "intersection_reconstruction"
            ],
            intersection_reconstruction_residual,
        )

        global_residuals[
            "axis_pattern_alignment"
        ] = max(
            global_residuals[
                "axis_pattern_alignment"
            ],
            alignment_residual,
        )

        global_residuals[
            "zero_sum"
        ] = max(
            global_residuals["zero_sum"],
            zero_sum_residual,
        )

        global_residuals[
            "unit_norm"
        ] = max(
            global_residuals["unit_norm"],
            unit_norm_residual,
        )

        global_residuals[
            "exact_weight_formula"
        ] = max(
            global_residuals[
                "exact_weight_formula"
            ],
            exact_weight_residual,
        )

        axis_rows.append(
            {
                "axis_id": axis_id,
                "stabilizer_order": len(
                    stabilizer
                ),
                "orbit_sizes": json.dumps(
                    orbit_sizes
                ),
                "intersection_dimension": (
                    intersection_dimension
                ),
                "weight_12": orbit_values[0],
                "weight_6": orbit_values[1],
                "weight_2": orbit_values[2],
                "primitive_integer_ratio": (
                    json.dumps(
                        integer_ratio
                    )
                ),
                "rational_ratio": json.dumps(
                    [
                        str(value)
                        for value in rational_ratio
                    ]
                ),
                "zero_sum_residual": (
                    zero_sum_residual
                ),
                "unit_norm_residual": (
                    unit_norm_residual
                ),
                "intersection_reconstruction_max_abs": (
                    intersection_reconstruction_residual
                ),
                "axis_pattern_alignment_max_abs": (
                    alignment_residual
                ),
                "exact_weight_formula_max_abs": (
                    exact_weight_residual
                ),
                "all_checks_pass": (
                    integer_ratio == [1, -4, 6]
                    and intersection_dimension == 1
                    and stabilizer_residual
                    < TOLERANCE
                    and max(orbit_spreads)
                    < TOLERANCE
                    and zero_sum_residual
                    < TOLERANCE
                    and unit_norm_residual
                    < TOLERANCE
                    and intersection_reconstruction_residual
                    < TOLERANCE
                    and alignment_residual
                    < TOLERANCE
                    and exact_weight_residual
                    < RATIO_TOLERANCE
                ),
            }
        )

        for orbit_id, (
            orbit,
            value,
            ratio_value,
        ) in enumerate(
            zip(
                orbits,
                orbit_values,
                integer_ratio,
            )
        ):
            orbit_rows.append(
                {
                    "axis_id": axis_id,
                    "orbit_id": orbit_id,
                    "orbit_size": len(orbit),
                    "faces": json.dumps(
                        orbit
                    ),
                    "weight": value,
                    "integer_weight": (
                        ratio_value
                    ),
                    "orbit_sum": (
                        len(orbit)
                        * value
                    ),
                }
            )

        recovered_patterns.append(
            recovered_pattern
        )

        intersection_lines.append(
            coefficient
        )

    recovered_pattern_matrix = np.column_stack(
        recovered_patterns
    )

    intersection_line_matrix = np.column_stack(
        intersection_lines
    )

    all_axes_pass = all(
        row["all_checks_pass"]
        for row in axis_rows
    )

    primitive_balance = (
        12 * 1
        + 6 * (-4)
        + 2 * 6
    )

    primitive_squared_norm = (
        12 * (1**2)
        + 6 * ((-4) ** 2)
        + 2 * (6**2)
    )

    normalization_denominator = math.sqrt(
        primitive_squared_norm
    )

    expected_normalization_denominator = (
        6.0 * math.sqrt(5.0)
    )

    checks = {
        "input_018_audit_pass": (
            axis_register.get("audit_pass")
            is True
        ),
        "input_019_audit_pass": (
            relation.get("audit_pass")
            is True
        ),
        "twenty_signed_axes_audited": (
            len(axis_rows) == 20
        ),
        "every_axis_has_orbit_sizes_12_6_2": (
            orbit_size_profile
            == {
                (12, 6, 2): 20
            }
        ),
        "every_orbit_constant_space_meets_flux_space_in_one_line": (
            intersection_dimension_profile
            == {1: 20}
        ),
        "every_axis_has_integer_ratio_1_minus4_6": (
            ratio_profile
            == {
                (1, -4, 6): 20
            }
        ),
        "primitive_integer_weights_balance_to_zero": (
            primitive_balance == 0
        ),
        "primitive_squared_norm_is_180": (
            primitive_squared_norm == 180
        ),
        "normalization_is_1_over_6_sqrt5": (
            abs(
                normalization_denominator
                - expected_normalization_denominator
            )
            < TOLERANCE
        ),
        "all_axis_audits_pass": (
            all_axes_pass
        ),
        "all_recovered_patterns_span_four_dimensions": (
            numerical_rank(
                recovered_pattern_matrix
            )
            == 4
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_statement = (
        "For every signed axis q in the native twenty-axis register, "
        "its stabilizer has three face orbits of sizes 12, 6, and 2. "
        "The three-dimensional orbit-constant face space intersects "
        "the four-dimensional flux representation in exactly one "
        "line. The unit vector on that line, oriented positively on "
        "the two-face orbit, has constant orbit weights "
        "(1, -4, 6)/(6 sqrt(5))."
    )

    payload = {
        "artifact_id": (
            "native_g60_flux_axis_integer_weight_law_020"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": audit_pass,
        "verdict": (
            "native_g60_flux_axis_integer_weight_law_identified"
            if audit_pass
            else "native_g60_flux_axis_integer_weight_law_failed"
        ),
        "inputs": {
            "axis_register": str(
                AXIS_REGISTER_PATH.relative_to(
                    ROOT
                )
            ),
            "axis_register_npz": str(
                AXIS_REGISTER_NPZ_PATH.relative_to(
                    ROOT
                )
            ),
            "axis_face_relation": str(
                RELATION_PATH.relative_to(
                    ROOT
                )
            ),
            "four_flux_geometry_npz": str(
                GEOMETRY_NPZ_PATH.relative_to(
                    ROOT
                )
            ),
            "signed_cochain_actions": str(
                ACTION_PATH.relative_to(ROOT)
            ),
        },
        "theorem": {
            "statement": theorem_statement,
            "axis_count": 20,
            "face_orbit_sizes": [
                12,
                6,
                2,
            ],
            "orbit_constant_space_dimension": 3,
            "flux_space_dimension": 4,
            "intersection_dimension": 1,
            "primitive_integer_weights": [
                1,
                -4,
                6,
            ],
            "primitive_balance": (
                "12(1) + 6(-4) + 2(6) = 0"
            ),
            "primitive_squared_norm": (
                primitive_squared_norm
            ),
            "unit_normalization": (
                "1/(6 sqrt(5))"
            ),
            "unit_orbit_weights": [
                "1/(6 sqrt(5))",
                "-4/(6 sqrt(5))",
                "6/(6 sqrt(5))",
            ],
        },
        "checks": checks,
        "profiles": {
            "orbit_size_profile": {
                str(key): value
                for key, value in sorted(
                    orbit_size_profile.items()
                )
            },
            "intersection_dimension_profile": {
                str(key): value
                for key, value in sorted(
                    intersection_dimension_profile.items()
                )
            },
            "primitive_ratio_profile": {
                str(key): value
                for key, value in sorted(
                    ratio_profile.items()
                )
            },
        },
        "global_maximum_residuals": (
            global_residuals
        ),
        "axis_rows": axis_rows,
        "earned_interpretation": {
            "integer_weight_law_is_uniform_across_all_axes": (
                audit_pass
            ),
            "weight_line_is_symmetry_forced": (
                audit_pass
            ),
            "overall_unit_normalization_is_mathematical": (
                audit_pass
            ),
            "external_constitutive_amplitude_remains_free": (
                True
            ),
            "physical_tension_claim": False,
            "physical_flux_claim": False,
        },
        "outputs": {
            "axis_csv": str(
                AXIS_CSV_OUT.relative_to(ROOT)
            ),
            "orbit_csv": str(
                ORBIT_CSV_OUT.relative_to(ROOT)
            ),
            "law_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "integer_weight_law_derived": (
                audit_pass
            ),
            "unique_orbit_constant_flux_line_proved": (
                audit_pass
            ),
            "law_verified_for_all_twenty_signed_axes": (
                audit_pass
            ),
            "overall_constitutive_scale_selected": (
                False
            ),
            "metric_distance_derived": False,
            "physical_energy_claim": False,
            "physical_flux_claim": False,
            "physical_thread_claim": False,
            "force_claim": False,
            "physical_claim": False,
        },
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

    with AXIS_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                axis_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(axis_rows)

    with ORBIT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                orbit_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(orbit_rows)

    np.savez_compressed(
        NPZ_OUT,
        recovered_pattern_matrix=(
            recovered_pattern_matrix
        ),
        intersection_line_coefficients=(
            intersection_line_matrix
        ),
        primitive_integer_weights=np.array(
            [1, -4, 6],
            dtype=np.int64,
        ),
        face_orbit_sizes=np.array(
            [12, 6, 2],
            dtype=np.int64,
        ),
        unit_orbit_weights=np.array(
            [
                1.0 / (6.0 * math.sqrt(5.0)),
                -4.0 / (6.0 * math.sqrt(5.0)),
                6.0 / (6.0 * math.sqrt(5.0)),
            ],
            dtype=np.float64,
        ),
        tolerance=np.array(
            [TOLERANCE]
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "axis_count:",
        len(axis_rows),
    )
    print(
        "orbit_size_profile:",
        dict(
            sorted(
                orbit_size_profile.items()
            )
        ),
    )
    print(
        "intersection_dimension_profile:",
        dict(
            sorted(
                intersection_dimension_profile.items()
            )
        ),
    )
    print(
        "primitive_ratio_profile:",
        dict(
            sorted(
                ratio_profile.items()
            )
        ),
    )
    print(
        "primitive_balance:",
        primitive_balance,
    )
    print(
        "primitive_squared_norm:",
        primitive_squared_norm,
    )
    print(
        "normalization_denominator:",
        normalization_denominator,
    )
    print(
        "recovered_pattern_rank:",
        numerical_rank(
            recovered_pattern_matrix
        ),
    )
    print(
        "all_axis_audits_pass:",
        all_axes_pass,
    )
    print(
        "global_maximum_residuals:",
        global_residuals,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", ORBIT_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
