from __future__ import annotations

import csv
import json
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

STRUCTURE_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_residual_zero_40_line_structure_037.json"
)

STRUCTURE_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_residual_zero_40_line_structure_037.npz"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

FOUR_GEOMETRY_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_face_geometry_017.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_residual_zero_30_line_census_038.json"
)

LINE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_residual_zero_30_lines_038.csv"
)

ORBIT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_residual_zero_30_orbits_038.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_residual_zero_30_gram_038.csv"
)

ACTION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_residual_zero_30_action_038.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_residual_zero_30_line_census_038.npz"
)

PROJECTIVE_MATCH_TOLERANCE = 3e-6
SIGNATURE_ROUND_DIGITS = 10
RATIONAL_DENOMINATOR_LIMIT = 100000


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(array)
        )
    )


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.array(
        vector,
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0.0:
        raise RuntimeError(
            "cannot normalize zero vector"
        )

    return vector / norm


def canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = normalized(vector)

    pivot = int(
        np.argmax(
            np.abs(result)
        )
    )

    if result[pivot] < 0.0:
        result *= -1.0

    return result


def projective_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    overlap = abs(
        float(
            np.dot(
                normalized(first),
                normalized(second),
            )
        )
    )

    overlap = float(
        np.clip(
            overlap,
            -1.0,
            1.0,
        )
    )

    return float(
        np.sqrt(
            max(
                0.0,
                2.0 - 2.0 * overlap,
            )
        )
    )


def nearest_line(
    point: np.ndarray,
    lines: np.ndarray,
) -> tuple[float, int]:
    distances = np.array(
        [
            projective_distance(
                point,
                line,
            )
            for line in lines
        ],
        dtype=np.float64,
    )

    index = int(
        np.argmin(distances)
    )

    return (
        float(distances[index]),
        index,
    )


def rational_label(value: float) -> str:
    return str(
        Fraction(float(value))
        .limit_denominator(
            RATIONAL_DENOMINATOR_LIMIT
        )
    )


def register_covariant(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    overlaps = (
        axis_lines @ point
    )

    return np.einsum(
        "i,ia,ib->ab",
        overlaps**2,
        axis_lines,
        axis_lines,
    )


def register_moment(
    point: np.ndarray,
    axis_lines: np.ndarray,
    power: int,
) -> float:
    overlaps = (
        axis_lines @ point
    )

    return float(
        np.sum(
            overlaps**power
        )
    )


def pencil_singular_values(
    slices: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    matrix = np.einsum(
        "r,rab->ab",
        point,
        slices,
    )

    return np.linalg.svd(
        matrix,
        compute_uv=False,
    )


def residual_vector(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    covariant = register_covariant(
        point,
        axis_lines,
    )

    s4 = register_moment(
        point,
        axis_lines,
        4,
    )

    return (
        covariant @ point
        - s4 * point
    )


def line_signature(
    line_id: int,
    point: np.ndarray,
    axis_lines: np.ndarray,
    slices: np.ndarray,
) -> dict:
    point = canonical_sign(point)

    s4 = register_moment(
        point,
        axis_lines,
        4,
    )

    s6 = register_moment(
        point,
        axis_lines,
        6,
    )

    s8 = register_moment(
        point,
        axis_lines,
        8,
    )

    covariant = register_covariant(
        point,
        axis_lines,
    )

    covariant_eigenvalues = (
        np.linalg.eigvalsh(
            covariant
        )
    )

    singular_values = (
        pencil_singular_values(
            slices,
            point,
        )
    )

    residual_norm = float(
        np.linalg.norm(
            residual_vector(
                point,
                axis_lines,
            )
        )
    )

    signature_key = tuple(
        round(
            value,
            SIGNATURE_ROUND_DIGITS,
        )
        for value in (
            s4,
            s6,
            s8,
            *covariant_eigenvalues.tolist(),
            *singular_values.tolist(),
        )
    )

    return {
        "line_id": line_id,
        "point": point,
        "s4": s4,
        "s6": s6,
        "s8": s8,
        "covariant_eigenvalues": (
            covariant_eigenvalues
        ),
        "singular_values": (
            singular_values
        ),
        "operator_norm": float(
            singular_values[0]
        ),
        "residual_norm": (
            residual_norm
        ),
        "signature_key": signature_key,
    }


def gram_profile(
    lines: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[str, int],
    list[dict],
]:
    gram = lines @ lines.T
    counter = Counter()
    rows = []

    for first in range(
        len(lines)
    ):
        for second in range(
            first + 1,
            len(lines),
        ):
            signed_value = float(
                gram[first, second]
            )

            absolute_value = abs(
                signed_value
            )

            rational = rational_label(
                absolute_value
            )

            counter[rational] += 1

            rows.append(
                {
                    "first_line": first,
                    "second_line": second,
                    "signed_gram": (
                        signed_value
                    ),
                    "absolute_gram": (
                        absolute_value
                    ),
                    "absolute_rational": (
                        rational
                    ),
                }
            )

    return (
        gram,
        dict(
            sorted(
                counter.items()
            )
        ),
        rows,
    )


def build_projective_action(
    representations: np.ndarray,
    lines: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    group_count = len(
        representations
    )

    line_count = len(lines)

    action = np.empty(
        (
            group_count,
            line_count,
        ),
        dtype=np.int64,
    )

    distances = np.empty(
        (
            group_count,
            line_count,
        ),
        dtype=np.float64,
    )

    rows = []

    for group_index in range(
        group_count
    ):
        representation = (
            representations[
                group_index
            ]
        )

        for line_id, line in enumerate(
            lines
        ):
            image = (
                representation @ line
            )

            distance, target = (
                nearest_line(
                    image,
                    lines,
                )
            )

            action[
                group_index,
                line_id,
            ] = target

            distances[
                group_index,
                line_id,
            ] = distance

            rows.append(
                {
                    "group_index": (
                        group_index
                    ),
                    "source_line": line_id,
                    "target_line": target,
                    "projective_distance": (
                        distance
                    ),
                    "action_pass": (
                        distance
                        < PROJECTIVE_MATCH_TOLERANCE
                    ),
                }
            )

    return action, distances, rows


def action_orbits(
    action: np.ndarray,
) -> list[list[int]]:
    line_count = action.shape[1]

    unseen = set(
        range(line_count)
    )

    orbits = []

    while unseen:
        start = min(unseen)
        reached = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            targets = {
                int(target)
                for target in action[
                    :,
                    current,
                ]
            }

            for target in targets:
                if target not in reached:
                    reached.add(target)
                    queue.append(target)

        unseen -= reached

        orbits.append(
            sorted(reached)
        )

    orbits.sort(
        key=lambda orbit: (
            -len(orbit),
            orbit[0],
        )
    )

    return orbits


def stabilizer_order(
    action: np.ndarray,
    line_id: int,
) -> int:
    return int(
        np.count_nonzero(
            action[:, line_id]
            == line_id
        )
    )


def signature_clusters(
    signatures: list[dict],
) -> list[list[int]]:
    groups: dict[
        tuple,
        list[int],
    ] = {}

    for signature in signatures:
        key = signature[
            "signature_key"
        ]

        groups.setdefault(
            key,
            [],
        ).append(
            signature[
                "line_id"
            ]
        )

    clusters = list(
        groups.values()
    )

    clusters.sort(
        key=lambda cluster: (
            -len(cluster),
            cluster[0],
        )
    )

    return clusters


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LINE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ORBIT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ACTION_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    structure_receipt = json.loads(
        STRUCTURE_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    structure_data = np.load(
        STRUCTURE_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    geometry_data = np.load(
        FOUR_GEOMETRY_NPZ_PATH
    )

    axis_lines = np.array(
        structure_data["axis_lines"],
        dtype=np.float64,
    )

    nonaxis_lines = np.array(
        structure_data["nonaxis_roots"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    representations = np.array(
        geometry_data[
            "face_flux_representation"
        ],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis shape: {axis_lines.shape}"
        )

    if nonaxis_lines.shape != (30, 4):
        raise RuntimeError(
            "unexpected nonaxis shape: "
            f"{nonaxis_lines.shape}"
        )

    if representations.shape != (
        480,
        4,
        4,
    ):
        raise RuntimeError(
            "unexpected representation shape: "
            f"{representations.shape}"
        )

    signatures = [
        line_signature(
            line_id,
            line,
            axis_lines,
            slices,
        )
        for line_id, line in enumerate(
            nonaxis_lines
        )
    ]

    signature_groups = (
        signature_clusters(
            signatures
        )
    )

    (
        gram,
        absolute_gram_profile,
        gram_rows,
    ) = gram_profile(
        nonaxis_lines
    )

    (
        action,
        action_distances,
        action_rows,
    ) = build_projective_action(
        representations,
        nonaxis_lines,
    )

    group_orbits = action_orbits(
        action
    )

    maximum_action_distance = float(
        np.max(
            action_distances
        )
    )

    action_closed = (
        maximum_action_distance
        < PROJECTIVE_MATCH_TOLERANCE
    )

    line_to_orbit = {}

    orbit_rows = []

    for orbit_id, orbit in enumerate(
        group_orbits
    ):
        stabilizers = [
            stabilizer_order(
                action,
                line_id,
            )
            for line_id in orbit
        ]

        operator_norms = [
            signatures[
                line_id
            ]["operator_norm"]
            for line_id in orbit
        ]

        s4_values = [
            signatures[
                line_id
            ]["s4"]
            for line_id in orbit
        ]

        for line_id in orbit:
            line_to_orbit[
                line_id
            ] = orbit_id

        orbit_rows.append(
            {
                "orbit_id": orbit_id,
                "orbit_size": len(
                    orbit
                ),
                "line_ids": json.dumps(
                    orbit
                ),
                "stabilizer_orders": json.dumps(
                    sorted(
                        set(
                            stabilizers
                        )
                    )
                ),
                "minimum_operator_norm": min(
                    operator_norms
                ),
                "maximum_operator_norm": max(
                    operator_norms
                ),
                "minimum_s4": min(
                    s4_values
                ),
                "maximum_s4": max(
                    s4_values
                ),
                "signature_cluster_count": len(
                    {
                        signatures[
                            line_id
                        ]["signature_key"]
                        for line_id in orbit
                    }
                ),
            }
        )

    line_rows = []

    for signature in signatures:
        line_id = signature[
            "line_id"
        ]

        covariant_eigenvalues = (
            signature[
                "covariant_eigenvalues"
            ]
        )

        singular_values = (
            signature[
                "singular_values"
            ]
        )

        line_rows.append(
            {
                "line_id": line_id,
                "group_orbit_id": (
                    line_to_orbit[
                        line_id
                    ]
                ),
                "stabilizer_order": (
                    stabilizer_order(
                        action,
                        line_id,
                    )
                ),
                "s4": signature["s4"],
                "s4_rational": (
                    rational_label(
                        signature["s4"]
                    )
                ),
                "s6": signature["s6"],
                "s6_rational": (
                    rational_label(
                        signature["s6"]
                    )
                ),
                "s8": signature["s8"],
                "s8_rational": (
                    rational_label(
                        signature["s8"]
                    )
                ),
                "covariant_eigenvalue_0": float(
                    covariant_eigenvalues[0]
                ),
                "covariant_eigenvalue_1": float(
                    covariant_eigenvalues[1]
                ),
                "covariant_eigenvalue_2": float(
                    covariant_eigenvalues[2]
                ),
                "covariant_eigenvalue_3": float(
                    covariant_eigenvalues[3]
                ),
                "operator_norm": (
                    signature[
                        "operator_norm"
                    ]
                ),
                "one_third_gap": (
                    1.0 / 3.0
                    - signature[
                        "operator_norm"
                    ]
                ),
                "singular_value_0": float(
                    singular_values[0]
                ),
                "singular_value_1": float(
                    singular_values[1]
                ),
                "singular_value_2": float(
                    singular_values[2]
                ),
                "singular_value_3": float(
                    singular_values[3]
                ),
                "singular_value_4": float(
                    singular_values[4]
                ),
                "singular_value_5": float(
                    singular_values[5]
                ),
                "residual_norm": (
                    signature[
                        "residual_norm"
                    ]
                ),
                "f0": float(
                    signature["point"][0]
                ),
                "f1": float(
                    signature["point"][1]
                ),
                "f2": float(
                    signature["point"][2]
                ),
                "f3": float(
                    signature["point"][3]
                ),
            }
        )

    operator_norms = np.array(
        [
            signature[
                "operator_norm"
            ]
            for signature in signatures
        ],
        dtype=np.float64,
    )

    checks = {
        "input_037_artifact_present": (
            structure_receipt.get(
                "artifact_id"
            )
            == (
                "native_g60_cross_flux_residual_zero_40_line_structure_037"
            )
        ),
        "nonaxis_line_count_is_30": (
            len(nonaxis_lines) == 30
        ),
        "all_residuals_are_zero": (
            max(
                signature[
                    "residual_norm"
                ]
                for signature in signatures
            )
            < 1e-8
        ),
        "full_group_action_closes_on_thirty_lines": (
            action_closed
        ),
        "all_action_rows_pass": all(
            row["action_pass"]
            for row in action_rows
        ),
        "orbit_stabilizer_orders_are_consistent": all(
            all(
                480
                == len(orbit)
                * stabilizer_order(
                    action,
                    line_id,
                )
                for line_id in orbit
            )
            for orbit in group_orbits
        ),
        "no_nonaxis_line_reaches_one_third": (
            float(
                np.max(
                    operator_norms
                )
            )
            < 1.0 / 3.0
            - 1e-8
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = (
        audit_pass
        and action_closed
    )

    verdict = (
        "native_g60_cross_flux_nonaxis_self_aligned_thirty_line_group_census_resolved"
        if theorem_pass
        else "native_g60_cross_flux_nonaxis_thirty_line_census_incomplete"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_residual_zero_30_line_census_038"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "line_count": 30,
        "group": {
            "order": 480,
            "action_closed": (
                action_closed
            ),
            "maximum_projective_action_residual": (
                maximum_action_distance
            ),
            "orbit_count": len(
                group_orbits
            ),
            "orbit_sizes": [
                len(orbit)
                for orbit in group_orbits
            ],
            "orbits": orbit_rows,
        },
        "signature_census": {
            "signature_cluster_count": len(
                signature_groups
            ),
            "signature_cluster_sizes": [
                len(cluster)
                for cluster in signature_groups
            ],
            "signature_clusters": (
                signature_groups
            ),
        },
        "operator_norm": {
            "minimum": float(
                np.min(
                    operator_norms
                )
            ),
            "maximum": float(
                np.max(
                    operator_norms
                )
            ),
            "mean": float(
                np.mean(
                    operator_norms
                )
            ),
            "distinct_rounded_values": sorted(
                {
                    round(
                        value,
                        12,
                    )
                    for value in (
                        operator_norms
                    )
                }
            ),
        },
        "projective_gram": {
            "absolute_overlap_profile": (
                absolute_gram_profile
            ),
            "rank": int(
                np.linalg.matrix_rank(
                    gram,
                    tol=1e-9,
                )
            ),
            "eigenvalues": (
                np.linalg.eigvalsh(
                    gram
                )
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "thirty_nonaxis_lines_form_native_group_register": (
                theorem_pass
            ),
            "thirty_line_layer_is_single_group_orbit": (
                len(group_orbits)
                == 1
            ),
            "thirty_line_layer_has_multiple_symmetry_orbits": (
                len(group_orbits)
                > 1
            ),
            "nonaxis_self_alignment_implies_extremality": (
                False
            ),
        },
        "boundary": {
            "recovered_thirty_line_layer_classified": (
                theorem_pass
            ),
            "complete_real_zero_locus_proved": (
                False
            ),
            "coordinate_construction_law_derived": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "line_csv": str(
                LINE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "orbit_csv": str(
                ORBIT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "gram_csv": str(
                GRAM_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "action_csv": str(
                ACTION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "census_npz": str(
                NPZ_OUT.relative_to(
                    ROOT
                )
            ),
        },
    }

    JSON_OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    for path, rows in (
        (
            LINE_CSV_OUT,
            line_rows,
        ),
        (
            ORBIT_CSV_OUT,
            orbit_rows,
        ),
        (
            GRAM_CSV_OUT,
            gram_rows,
        ),
        (
            ACTION_CSV_OUT,
            action_rows,
        ),
    ):
        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[0]
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        nonaxis_lines=nonaxis_lines,
        group_action=action,
        group_action_distances=(
            action_distances
        ),
        projective_gram=gram,
        operator_norms=operator_norms,
        s4=np.array(
            [
                signature["s4"]
                for signature in signatures
            ],
            dtype=np.float64,
        ),
        s6=np.array(
            [
                signature["s6"]
                for signature in signatures
            ],
            dtype=np.float64,
        ),
        s8=np.array(
            [
                signature["s8"]
                for signature in signatures
            ],
            dtype=np.float64,
        ),
        covariant_eigenvalues=np.array(
            [
                signature[
                    "covariant_eigenvalues"
                ]
                for signature in signatures
            ],
            dtype=np.float64,
        ),
        pencil_singular_values=np.array(
            [
                signature[
                    "singular_values"
                ]
                for signature in signatures
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "group_action_maximum_residual:",
        maximum_action_distance,
    )
    print(
        "group_orbit_sizes:",
        [
            len(orbit)
            for orbit in group_orbits
        ],
    )
    print(
        "orbit_stabilizer_orders:",
        [
            sorted(
                {
                    stabilizer_order(
                        action,
                        line_id,
                    )
                    for line_id in orbit
                }
            )
            for orbit in group_orbits
        ],
    )
    print(
        "signature_cluster_sizes:",
        [
            len(cluster)
            for cluster in signature_groups
        ],
    )
    print(
        "operator_norm_distinct_values:",
        sorted(
            {
                round(
                    value,
                    12,
                )
                for value in operator_norms
            }
        ),
    )
    print(
        "operator_norm_range:",
        float(
            np.min(
                operator_norms
            )
        ),
        float(
            np.max(
                operator_norms
            )
        ),
    )
    print(
        "absolute_gram_profile:",
        absolute_gram_profile,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", LINE_CSV_OUT)
    print("wrote:", ORBIT_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", ACTION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
