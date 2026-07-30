from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]

ZERO_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_covariant_residual_zero_locus_036.json"
)

ZERO_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_covariant_residual_zero_locus_036.npz"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_residual_zero_40_line_structure_037.json"
)

MATCH_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_residual_zero_weak_pair_matching_037.csv"
)

LINE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_residual_zero_lines_037.csv"
)

PAIR_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_axis_pair_candidates_037.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_residual_zero_40_line_structure_037.npz"
)

PROJECTIVE_TOLERANCE = 2e-6
AXIS_MATCH_TOLERANCE = 2e-6
PAIR_MATCH_TOLERANCE = 3e-6
SPAN_TOLERANCE = 3e-6
EQUAL_WEIGHT_TOLERANCE = 3e-6

WEAK_OVERLAP = 1.0 / 6.0
STRONG_OVERLAP = 2.0 / 3.0


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

    return float(np.max(np.abs(array)))


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.array(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        raise RuntimeError("cannot normalize zero vector")

    return vector / norm


def canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = normalized(vector)
    pivot = int(np.argmax(np.abs(result)))

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

    overlap = float(np.clip(overlap, -1.0, 1.0))

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
            projective_distance(point, line)
            for line in lines
        ],
        dtype=np.float64,
    )

    index = int(np.argmin(distances))

    return float(distances[index]), index


def projective_deduplicate(
    points: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    representatives = []

    for point in points:
        point = canonical_sign(point)

        if not representatives:
            representatives.append(point)
            continue

        distance, _ = nearest_line(
            point,
            np.array(
                representatives,
                dtype=np.float64,
            ),
        )

        if distance > tolerance:
            representatives.append(point)

    return np.array(
        representatives,
        dtype=np.float64,
    )


def residual_vector(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    point = normalized(point)
    overlaps = axis_lines @ point

    covariant = np.einsum(
        "i,ia,ib->ab",
        overlaps**2,
        axis_lines,
        axis_lines,
    )

    s4 = float(
        np.sum(
            overlaps**4
        )
    )

    return covariant @ point - s4 * point


def operator_norm(
    slices: np.ndarray,
    direction: np.ndarray,
) -> float:
    matrix = np.einsum(
        "r,rab->ab",
        direction,
        slices,
    )

    return float(
        np.linalg.svd(
            matrix,
            compute_uv=False,
        )[0]
    )


def build_pair_candidates(
    axis_lines: np.ndarray,
) -> list[dict]:
    records = []

    for first in range(len(axis_lines)):
        for second in range(first + 1, len(axis_lines)):
            q_first = axis_lines[first]
            q_second = axis_lines[second]

            signed_overlap = float(
                np.dot(
                    q_first,
                    q_second,
                )
            )

            absolute_overlap = abs(
                signed_overlap
            )

            sign = (
                1.0
                if signed_overlap >= 0.0
                else -1.0
            )

            acute = canonical_sign(
                q_first + sign * q_second
            )

            obtuse = canonical_sign(
                q_first - sign * q_second
            )

            if abs(
                absolute_overlap
                - WEAK_OVERLAP
            ) < 1e-8:
                overlap_class = "weak_1_over_6"
            elif abs(
                absolute_overlap
                - STRONG_OVERLAP
            ) < 1e-8:
                overlap_class = "strong_2_over_3"
            else:
                overlap_class = "other"

            records.append(
                {
                    "pair_id": len(records),
                    "first_axis": first,
                    "second_axis": second,
                    "signed_overlap": signed_overlap,
                    "absolute_overlap": absolute_overlap,
                    "overlap_class": overlap_class,
                    "acute": acute,
                    "obtuse": obtuse,
                }
            )

    return records


def match_roots_to_pairs(
    roots: np.ndarray,
    pair_records: list[dict],
    axis_lines: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    pair_count = len(pair_records)

    cost = np.zeros(
        (len(roots), pair_count),
        dtype=np.float64,
    )

    branch = np.empty(
        (len(roots), pair_count),
        dtype=object,
    )

    for root_id, root in enumerate(roots):
        for pair_index, pair in enumerate(pair_records):
            acute_distance = projective_distance(
                root,
                pair["acute"],
            )

            obtuse_distance = projective_distance(
                root,
                pair["obtuse"],
            )

            if acute_distance <= obtuse_distance:
                cost[root_id, pair_index] = (
                    acute_distance
                )

                branch[root_id, pair_index] = (
                    "acute"
                )
            else:
                cost[root_id, pair_index] = (
                    obtuse_distance
                )

                branch[root_id, pair_index] = (
                    "obtuse"
                )

    row_indices, column_indices = (
        linear_sum_assignment(cost)
    )

    rows = []

    for root_id, pair_index in zip(
        row_indices,
        column_indices,
    ):
        root = roots[root_id]
        pair = pair_records[pair_index]
        selected_branch = str(
            branch[root_id, pair_index]
        )

        candidate = (
            pair["acute"]
            if selected_branch == "acute"
            else pair["obtuse"]
        )

        axis_pair_matrix = np.column_stack(
            [
                axis_lines[
                    pair["first_axis"]
                ],
                axis_lines[
                    pair["second_axis"]
                ],
            ]
        )

        coefficients, _, _, _ = (
            np.linalg.lstsq(
                axis_pair_matrix,
                root,
                rcond=None,
            )
        )

        reconstructed = (
            axis_pair_matrix
            @ coefficients
        )

        span_residual = min(
            np.linalg.norm(
                reconstructed - root
            ),
            np.linalg.norm(
                reconstructed + root
            ),
        )

        coefficient_magnitudes = np.abs(
            coefficients
        )

        equal_weight_residual = abs(
            coefficient_magnitudes[0]
            - coefficient_magnitudes[1]
        )

        rows.append(
            {
                "root_id": int(root_id),
                "pair_index": int(
                    pair_index
                ),
                "pair_id": int(
                    pair["pair_id"]
                ),
                "first_axis": int(
                    pair["first_axis"]
                ),
                "second_axis": int(
                    pair["second_axis"]
                ),
                "signed_overlap": float(
                    pair["signed_overlap"]
                ),
                "absolute_overlap": float(
                    pair["absolute_overlap"]
                ),
                "overlap_class": (
                    pair["overlap_class"]
                ),
                "selected_branch": (
                    selected_branch
                ),
                "projective_distance": float(
                    cost[
                        root_id,
                        pair_index,
                    ]
                ),
                "candidate_coordinate_residual": (
                    max_abs(
                        canonical_sign(root)
                        - canonical_sign(candidate)
                    )
                ),
                "span_residual": float(
                    span_residual
                ),
                "coefficient_first": float(
                    coefficients[0]
                ),
                "coefficient_second": float(
                    coefficients[1]
                ),
                "absolute_coefficient_ratio": float(
                    coefficient_magnitudes[0]
                    / max(
                        coefficient_magnitudes[1],
                        1e-15,
                    )
                ),
                "equal_weight_residual": float(
                    equal_weight_residual
                ),
            }
        )

    rows.sort(
        key=lambda row: row["root_id"]
    )

    return cost, branch, rows


def gram_profile(lines: np.ndarray) -> dict:
    counter = Counter()

    gram = lines @ lines.T

    for first in range(len(lines)):
        for second in range(
            first + 1,
            len(lines),
        ):
            value = abs(
                float(
                    gram[first, second]
                )
            )

            counter[
                round(value, 12)
            ] += 1

    return {
        str(key): value
        for key, value in sorted(
            counter.items()
        )
    }


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MATCH_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LINE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PAIR_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    zero_receipt = json.loads(
        ZERO_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    zero_data = np.load(
        ZERO_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    axis_lines = np.array(
        zero_data["axis_lines"],
        dtype=np.float64,
    )

    raw_roots = np.array(
        zero_data["root_representatives"],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    roots = projective_deduplicate(
        raw_roots,
        PROJECTIVE_TOLERANCE,
    )

    axis_root_rows = []
    axis_root_indices = []
    nonaxis_root_indices = []

    for root_id, root in enumerate(roots):
        axis_distance, axis_id = nearest_line(
            root,
            axis_lines,
        )

        residual_norm = float(
            np.linalg.norm(
                residual_vector(
                    root,
                    axis_lines,
                )
            )
        )

        sigma = operator_norm(
            slices,
            root,
        )

        is_axis = (
            axis_distance
            < AXIS_MATCH_TOLERANCE
        )

        if is_axis:
            axis_root_indices.append(
                root_id
            )
        else:
            nonaxis_root_indices.append(
                root_id
            )

        axis_root_rows.append(
            {
                "root_id": root_id,
                "is_native_axis": is_axis,
                "nearest_axis_id": axis_id,
                "axis_distance": (
                    axis_distance
                ),
                "residual_norm": (
                    residual_norm
                ),
                "operator_norm": sigma,
                "one_third_gap": (
                    1.0 / 3.0 - sigma
                ),
                "f0": float(root[0]),
                "f1": float(root[1]),
                "f2": float(root[2]),
                "f3": float(root[3]),
            }
        )

    axis_roots = roots[
        axis_root_indices
    ]

    nonaxis_roots = roots[
        nonaxis_root_indices
    ]

    pair_records = build_pair_candidates(
        axis_lines
    )

    weak_pairs = [
        pair
        for pair in pair_records
        if pair["overlap_class"]
        == "weak_1_over_6"
    ]

    strong_pairs = [
        pair
        for pair in pair_records
        if pair["overlap_class"]
        == "strong_2_over_3"
    ]

    (
        weak_cost,
        weak_branch,
        weak_match_rows,
    ) = match_roots_to_pairs(
        nonaxis_roots,
        weak_pairs,
        axis_lines,
    )

    (
        strong_cost,
        strong_branch,
        strong_match_rows,
    ) = match_roots_to_pairs(
        nonaxis_roots[:len(strong_pairs)],
        strong_pairs,
        axis_lines,
    )

    maximum_weak_match_distance = max(
        row["projective_distance"]
        for row in weak_match_rows
    )

    maximum_weak_span_residual = max(
        row["span_residual"]
        for row in weak_match_rows
    )

    maximum_equal_weight_residual = max(
        row["equal_weight_residual"]
        for row in weak_match_rows
    )

    weak_branch_counts = dict(
        Counter(
            row["selected_branch"]
            for row in weak_match_rows
        )
    )

    maximum_strong_match_distance = max(
        row["projective_distance"]
        for row in strong_match_rows
    )

    nonaxis_operator_norms = np.array(
        [
            operator_norm(
                slices,
                root,
            )
            for root in nonaxis_roots
        ],
        dtype=np.float64,
    )

    axis_operator_norms = np.array(
        [
            operator_norm(
                slices,
                root,
            )
            for root in axis_roots
        ],
        dtype=np.float64,
    )

    pair_rows = []

    for pair in pair_records:
        pair_rows.append(
            {
                "pair_id": (
                    pair["pair_id"]
                ),
                "first_axis": (
                    pair["first_axis"]
                ),
                "second_axis": (
                    pair["second_axis"]
                ),
                "signed_overlap": (
                    pair["signed_overlap"]
                ),
                "absolute_overlap": (
                    pair["absolute_overlap"]
                ),
                "overlap_class": (
                    pair["overlap_class"]
                ),
                "acute_residual_norm": float(
                    np.linalg.norm(
                        residual_vector(
                            pair["acute"],
                            axis_lines,
                        )
                    )
                ),
                "obtuse_residual_norm": float(
                    np.linalg.norm(
                        residual_vector(
                            pair["obtuse"],
                            axis_lines,
                        )
                    )
                ),
                "acute_operator_norm": (
                    operator_norm(
                        slices,
                        pair["acute"],
                    )
                ),
                "obtuse_operator_norm": (
                    operator_norm(
                        slices,
                        pair["obtuse"],
                    )
                ),
            }
        )

    checks = {
        "input_036_census_completed": (
            zero_receipt.get(
                "artifact_id"
            )
            == (
                "native_g60_cross_flux_covariant_residual_zero_locus_036"
            )
        ),
        "recovered_projective_root_count_is_40": (
            len(roots) == 40
        ),
        "native_axis_root_count_is_10": (
            len(axis_roots) == 10
        ),
        "nonaxis_root_count_is_30": (
            len(nonaxis_roots) == 30
        ),
        "weak_overlap_pair_count_is_30": (
            len(weak_pairs) == 30
        ),
        "strong_overlap_pair_count_is_15": (
            len(strong_pairs) == 15
        ),
        "weak_pair_matching_is_bijective": (
            len(weak_match_rows)
            == 30
            and len(
                {
                    row["pair_index"]
                    for row in weak_match_rows
                }
            )
            == 30
        ),
        "all_nonaxis_roots_match_weak_pair_bisectors": (
            maximum_weak_match_distance
            < PAIR_MATCH_TOLERANCE
        ),
        "all_nonaxis_roots_lie_in_matched_axis_pair_spans": (
            maximum_weak_span_residual
            < SPAN_TOLERANCE
        ),
        "all_nonaxis_roots_are_equal_weight_pair_combinations": (
            maximum_equal_weight_residual
            < EQUAL_WEIGHT_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_residual_zero_register_is_ten_axes_plus_thirty_weak_pair_bisectors"
        if theorem_pass
        else "native_g60_cross_flux_residual_zero_40_line_structure_not_resolved"
    )

    theorem_statement = (
        "Within the recovered forty-line projective zero census, "
        "ten lines are the native flux axes and the remaining thirty "
        "match bijectively with the thirty native axis pairs having "
        "absolute overlap 1/6. Each non-axis zero is the normalized "
        "equal-weight acute or obtuse bisector of its matched weak "
        "axis pair."
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_residual_zero_40_line_structure_037"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "theorem": {
            "statement": theorem_statement,
            "decomposition": (
                "40 = 10 native axes + "
                "30 weak-overlap pair bisectors"
            ),
            "native_axis_count": (
                len(axis_roots)
            ),
            "nonaxis_zero_count": (
                len(nonaxis_roots)
            ),
            "weak_pair_count": (
                len(weak_pairs)
            ),
        },
        "matching": {
            "maximum_projective_distance": (
                maximum_weak_match_distance
            ),
            "maximum_pair_span_residual": (
                maximum_weak_span_residual
            ),
            "maximum_equal_weight_residual": (
                maximum_equal_weight_residual
            ),
            "branch_counts": (
                weak_branch_counts
            ),
            "strong_pair_control_maximum_matching_distance": (
                maximum_strong_match_distance
            ),
        },
        "operator_norms": {
            "native_axes": {
                "minimum": float(
                    np.min(
                        axis_operator_norms
                    )
                ),
                "maximum": float(
                    np.max(
                        axis_operator_norms
                    )
                ),
            },
            "nonaxis_self_aligned_lines": {
                "minimum": float(
                    np.min(
                        nonaxis_operator_norms
                    )
                ),
                "maximum": float(
                    np.max(
                        nonaxis_operator_norms
                    )
                ),
                "mean": float(
                    np.mean(
                        nonaxis_operator_norms
                    )
                ),
                "maximum_one_third_gap": float(
                    np.max(
                        1.0 / 3.0
                        - nonaxis_operator_norms
                    )
                ),
                "minimum_one_third_gap": float(
                    np.min(
                        1.0 / 3.0
                        - nonaxis_operator_norms
                    )
                ),
            },
        },
        "gram_profiles": {
            "native_ten_axes": (
                gram_profile(
                    axis_lines
                )
            ),
            "nonaxis_thirty_lines": (
                gram_profile(
                    nonaxis_roots
                )
            ),
            "full_forty_line_register": (
                gram_profile(
                    roots
                )
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "forty_line_census_has_ten_plus_thirty_structure": (
                theorem_pass
            ),
            "thirty_nonaxis_lines_are_indexed_by_weak_axis_pairs": (
                theorem_pass
            ),
            "self_alignment_implies_cross_flux_extremality": (
                False
            ),
            "native_axis_lines_remain_the_observed_one_third_extremizers": (
                True
            ),
        },
        "boundary": {
            "structure_of_recovered_forty_line_census_resolved": (
                theorem_pass
            ),
            "complete_real_zero_locus_proved": (
                False
            ),
            "forty_lines_proved_to_be_all_real_projective_zeros": (
                False
            ),
            "nonaxis_lines_are_global_norm_maximizers": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "matching_csv": str(
                MATCH_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "line_csv": str(
                LINE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "pair_csv": str(
                PAIR_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "structure_npz": str(
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

    with MATCH_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                weak_match_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            weak_match_rows
        )

    with LINE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                axis_root_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            axis_root_rows
        )

    with PAIR_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                pair_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            pair_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        recovered_roots=roots,
        axis_roots=axis_roots,
        nonaxis_roots=nonaxis_roots,
        weak_matching_cost=weak_cost,
        weak_matching_branch=weak_branch,
        nonaxis_operator_norms=(
            nonaxis_operator_norms
        ),
        axis_operator_norms=(
            axis_operator_norms
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "recovered_root_count:",
        len(roots),
    )
    print(
        "axis/nonaxis_counts:",
        len(axis_roots),
        len(nonaxis_roots),
    )
    print(
        "weak/strong_pair_counts:",
        len(weak_pairs),
        len(strong_pairs),
    )
    print(
        "maximum_weak_pair_match_distance:",
        maximum_weak_match_distance,
    )
    print(
        "maximum_pair_span_residual:",
        maximum_weak_span_residual,
    )
    print(
        "maximum_equal_weight_residual:",
        maximum_equal_weight_residual,
    )
    print(
        "weak_pair_branch_counts:",
        weak_branch_counts,
    )
    print(
        "strong_pair_control_distance:",
        maximum_strong_match_distance,
    )
    print(
        "axis_operator_norm_range:",
        float(
            np.min(
                axis_operator_norms
            )
        ),
        float(
            np.max(
                axis_operator_norms
            )
        ),
    )
    print(
        "nonaxis_operator_norm_range:",
        float(
            np.min(
                nonaxis_operator_norms
            )
        ),
        float(
            np.max(
                nonaxis_operator_norms
            )
        ),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", MATCH_CSV_OUT)
    print("wrote:", LINE_CSV_OUT)
    print("wrote:", PAIR_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
