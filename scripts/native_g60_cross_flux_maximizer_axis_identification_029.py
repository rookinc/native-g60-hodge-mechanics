from __future__ import annotations

import csv
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]

AXIS_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_four_flux_axis_register_018.json"
)

AXIS_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_four_flux_axis_register_018.npz"
)

LOCUS_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_pencil_maximizer_locus_028.json"
)

LOCUS_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_pencil_maximizer_locus_028.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_maximizer_axis_identification_029.json"
)

MATCH_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_maximizer_axis_matching_029.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_line_gram_029.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_maximizer_axis_identification_029.npz"
)

PROJECTIVE_CLUSTER_TOLERANCE = 1e-6
MATCH_TOLERANCE = 2e-6
GRAM_TOLERANCE = 2e-6
LAW_TOLERANCE = 2e-6

EXPECTED_FRAME_CONSTANT = 5.0 / 2.0


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
    dot = abs(
        float(
            np.dot(
                first,
                second,
            )
        )
    )

    dot = min(
        max(dot, -1.0),
        1.0,
    )

    return float(
        np.sqrt(
            max(
                0.0,
                2.0 - 2.0 * dot,
            )
        )
    )


def cluster_projective_points(
    points: np.ndarray,
    tolerance: float,
) -> list[list[int]]:
    unseen = set(
        range(len(points))
    )

    clusters = []

    while unseen:
        seed = min(unseen)
        cluster = {seed}
        frontier = [seed]

        while frontier:
            current = frontier.pop()

            additions = {
                candidate
                for candidate in unseen
                if projective_distance(
                    points[current],
                    points[candidate],
                )
                <= tolerance
            }

            for candidate in additions:
                if candidate not in cluster:
                    cluster.add(candidate)
                    frontier.append(candidate)

        unseen -= cluster

        clusters.append(
            sorted(cluster)
        )

    clusters.sort(
        key=lambda cluster: (
            -len(cluster),
            cluster[0],
        )
    )

    return clusters


def projective_cluster_representatives(
    points: np.ndarray,
    tolerance: float,
) -> tuple[
    np.ndarray,
    list[list[int]],
]:
    clusters = cluster_projective_points(
        points,
        tolerance,
    )

    representatives = []

    for cluster in clusters:
        reference = canonical_sign(
            points[cluster[0]]
        )

        aligned = []

        for index in cluster:
            point = normalized(
                points[index]
            )

            if np.dot(
                point,
                reference,
            ) < 0.0:
                point *= -1.0

            aligned.append(point)

        representative = canonical_sign(
            np.mean(
                np.array(
                    aligned,
                    dtype=np.float64,
                ),
                axis=0,
            )
        )

        representatives.append(
            representative
        )

    return (
        np.array(
            representatives,
            dtype=np.float64,
        ),
        clusters,
    )


def rational_label(
    value: float,
    denominator_limit: int = 144,
) -> str:
    return str(
        Fraction(float(value))
        .limit_denominator(
            denominator_limit
        )
    )


def match_projective_registers(
    maximizers: np.ndarray,
    axes: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    cost = np.zeros(
        (
            len(maximizers),
            len(axes),
        ),
        dtype=np.float64,
    )

    for first in range(
        len(maximizers)
    ):
        for second in range(
            len(axes)
        ):
            cost[first, second] = (
                projective_distance(
                    maximizers[first],
                    axes[second],
                )
            )

    row_indices, column_indices = (
        linear_sum_assignment(cost)
    )

    permutation = np.empty(
        len(maximizers),
        dtype=np.int64,
    )

    aligned_maximizers = np.empty_like(
        maximizers
    )

    match_rows = []

    for row, column in zip(
        row_indices,
        column_indices,
    ):
        permutation[row] = column

        maximizer = maximizers[row].copy()
        axis = axes[column].copy()

        orientation_sign = 1

        if np.dot(
            maximizer,
            axis,
        ) < 0.0:
            maximizer *= -1.0
            orientation_sign = -1

        aligned_maximizers[column] = (
            maximizer
        )

        coordinate_residual = max_abs(
            maximizer - axis
        )

        match_rows.append(
            {
                "maximizer_line_id": int(row),
                "axis_line_id": int(column),
                "orientation_sign": (
                    orientation_sign
                ),
                "projective_distance": float(
                    cost[row, column]
                ),
                "coordinate_max_abs_residual": (
                    coordinate_residual
                ),
                "match_pass": (
                    cost[row, column]
                    < MATCH_TOLERANCE
                ),
            }
        )

    match_rows.sort(
        key=lambda row: (
            row["axis_line_id"]
        )
    )

    return (
        cost,
        permutation,
        aligned_maximizers,
        match_rows,
    )


def gram_profile(
    gram: np.ndarray,
) -> dict:
    signed_counter = Counter()
    absolute_counter = Counter()

    signed_values = []
    absolute_values = []

    for first in range(
        gram.shape[0]
    ):
        for second in range(
            first + 1,
            gram.shape[1],
        ):
            value = float(
                gram[first, second]
            )

            signed_values.append(value)
            absolute_values.append(
                abs(value)
            )

            signed_key = rational_label(
                value
            )

            absolute_key = rational_label(
                abs(value)
            )

            signed_counter[
                signed_key
            ] += 1

            absolute_counter[
                absolute_key
            ] += 1

    return {
        "signed_profile": dict(
            sorted(
                signed_counter.items()
            )
        ),
        "absolute_profile": dict(
            sorted(
                absolute_counter.items()
            )
        ),
        "signed_values": np.array(
            signed_values,
            dtype=np.float64,
        ),
        "absolute_values": np.array(
            absolute_values,
            dtype=np.float64,
        ),
    }


def write_gram_csv(
    path: Path,
    native_gram: np.ndarray,
    maximizer_gram: np.ndarray,
) -> list[dict]:
    rows = []

    for first in range(10):
        for second in range(10):
            native_value = float(
                native_gram[first, second]
            )

            maximizer_value = float(
                maximizer_gram[first, second]
            )

            rows.append(
                {
                    "first_line": first,
                    "second_line": second,
                    "native_signed_gram": (
                        native_value
                    ),
                    "maximizer_signed_gram": (
                        maximizer_value
                    ),
                    "native_absolute_gram": abs(
                        native_value
                    ),
                    "maximizer_absolute_gram": abs(
                        maximizer_value
                    ),
                    "signed_residual": (
                        maximizer_value
                        - native_value
                    ),
                    "absolute_residual": (
                        abs(maximizer_value)
                        - abs(native_value)
                    ),
                    "native_rational": (
                        rational_label(
                            native_value
                        )
                    ),
                    "native_absolute_rational": (
                        rational_label(
                            abs(native_value)
                        )
                    ),
                }
            )

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

    return rows


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MATCH_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAM_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    axis_receipt = json.loads(
        AXIS_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    locus_receipt = json.loads(
        LOCUS_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    axis_data = np.load(
        AXIS_NPZ_PATH
    )

    locus_data = np.load(
        LOCUS_NPZ_PATH
    )

    signed_axis_matrix = np.array(
        axis_data["axis_matrix"],
        dtype=np.float64,
    )

    terminal_points = np.array(
        locus_data["terminal_points"],
        dtype=np.float64,
    )

    if signed_axis_matrix.shape == (4, 20):
        signed_axes = (
            signed_axis_matrix.T
        )
    elif signed_axis_matrix.shape == (20, 4):
        signed_axes = (
            signed_axis_matrix
        )
    else:
        raise RuntimeError(
            "unexpected signed axis matrix shape: "
            f"{signed_axis_matrix.shape}"
        )

    signed_axes = np.array(
        [
            normalized(axis)
            for axis in signed_axes
        ],
        dtype=np.float64,
    )

    (
        axis_lines,
        axis_clusters,
    ) = projective_cluster_representatives(
        signed_axes,
        PROJECTIVE_CLUSTER_TOLERANCE,
    )

    (
        maximizer_lines,
        maximizer_clusters,
    ) = projective_cluster_representatives(
        terminal_points,
        PROJECTIVE_CLUSTER_TOLERANCE,
    )

    (
        matching_cost,
        permutation,
        aligned_maximizers,
        match_rows,
    ) = match_projective_registers(
        maximizer_lines,
        axis_lines,
    )

    matching_distances = np.array(
        [
            row["projective_distance"]
            for row in match_rows
        ],
        dtype=np.float64,
    )

    matching_coordinate_residuals = np.array(
        [
            row[
                "coordinate_max_abs_residual"
            ]
            for row in match_rows
        ],
        dtype=np.float64,
    )

    native_line_matrix = axis_lines

    maximizer_line_matrix = (
        aligned_maximizers
    )

    native_gram = (
        native_line_matrix
        @ native_line_matrix.T
    )

    maximizer_gram = (
        maximizer_line_matrix
        @ maximizer_line_matrix.T
    )

    native_absolute_gram = np.abs(
        native_gram
    )

    maximizer_absolute_gram = np.abs(
        maximizer_gram
    )

    gram_signed_residual = max_abs(
        maximizer_gram
        - native_gram
    )

    gram_absolute_residual = max_abs(
        maximizer_absolute_gram
        - native_absolute_gram
    )

    native_profile = gram_profile(
        native_gram
    )

    maximizer_profile = gram_profile(
        maximizer_gram
    )

    native_frame_operator = (
        native_line_matrix.T
        @ native_line_matrix
    )

    maximizer_frame_operator = (
        maximizer_line_matrix.T
        @ maximizer_line_matrix
    )

    target_frame_operator = (
        EXPECTED_FRAME_CONSTANT
        * np.eye(4)
    )

    native_frame_residual = max_abs(
        native_frame_operator
        - target_frame_operator
    )

    maximizer_frame_residual = max_abs(
        maximizer_frame_operator
        - target_frame_operator
    )

    native_gram_quadratic_residual = (
        max_abs(
            native_gram
            @ native_gram
            - EXPECTED_FRAME_CONSTANT
            * native_gram
        )
    )

    maximizer_gram_quadratic_residual = (
        max_abs(
            maximizer_gram
            @ maximizer_gram
            - EXPECTED_FRAME_CONSTANT
            * maximizer_gram
        )
    )

    native_gram_eigenvalues = (
        np.linalg.eigvalsh(
            native_gram
        )
    )

    maximizer_gram_eigenvalues = (
        np.linalg.eigvalsh(
            maximizer_gram
        )
    )

    gram_rows = write_gram_csv(
        GRAM_CSV_OUT,
        native_gram,
        maximizer_gram,
    )

    with MATCH_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                match_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            match_rows
        )

    expected_absolute_profile = {
        "1/6": 30,
        "2/3": 15,
    }

    checks = {
        "input_018_audit_pass": (
            axis_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "input_028_audit_pass": (
            locus_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "signed_axis_count_is_20": (
            len(signed_axes) == 20
        ),
        "native_unoriented_axis_count_is_10": (
            len(axis_lines) == 10
        ),
        "maximizer_projective_cluster_count_is_10": (
            len(maximizer_lines) == 10
        ),
        "matching_is_bijective": (
            sorted(
                permutation.tolist()
            )
            == list(range(10))
        ),
        "all_projective_matches_resolved": (
            max_abs(
                matching_distances
            )
            < MATCH_TOLERANCE
        ),
        "all_coordinate_matches_resolved": (
            max_abs(
                matching_coordinate_residuals
            )
            < MATCH_TOLERANCE
        ),
        "signed_gram_matrices_match": (
            gram_signed_residual
            < GRAM_TOLERANCE
        ),
        "absolute_gram_matrices_match": (
            gram_absolute_residual
            < GRAM_TOLERANCE
        ),
        "native_axis_lines_form_tight_frame": (
            native_frame_residual
            < LAW_TOLERANCE
        ),
        "maximizer_lines_form_same_tight_frame": (
            maximizer_frame_residual
            < LAW_TOLERANCE
        ),
        "native_line_gram_satisfies_quadratic_law": (
            native_gram_quadratic_residual
            < LAW_TOLERANCE
        ),
        "maximizer_line_gram_satisfies_quadratic_law": (
            maximizer_gram_quadratic_residual
            < LAW_TOLERANCE
        ),
        "native_absolute_overlap_profile_is_30_and_15": (
            native_profile[
                "absolute_profile"
            ]
            == expected_absolute_profile
        ),
        "maximizer_absolute_overlap_profile_matches": (
            maximizer_profile[
                "absolute_profile"
            ]
            == expected_absolute_profile
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_maximizers_equal_ten_native_axis_lines"
        if theorem_pass
        else "native_g60_cross_flux_maximizer_axis_identification_failed"
    )

    theorem_statement = (
        "The ten projective terminal maximizer clusters of the "
        "cross-flux operator pencil match bijectively with the ten "
        "unoriented lines of the native twenty-axis flux register. "
        "Writing their unit representatives as the rows of L10, "
        "the line Gram matrix R10=L10 L10^T has diagonal one, "
        "absolute off-diagonal overlaps 1/6 on thirty pairs and "
        "2/3 on fifteen pairs, and satisfies "
        "R10^2=(5/2)R10. Equivalently, "
        "L10^T L10=(5/2)I4."
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_maximizer_axis_identification_029"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "theorem": {
            "statement": theorem_statement,
            "maximizer_projective_line_count": 10,
            "native_unoriented_axis_line_count": 10,
            "identification": (
                "maximizer locus equals native unoriented flux-axis register"
            ),
        },
        "notation": {
            "native_incidence_gram": (
                "Q = M M^T, with M of shape 15 by 30"
            ),
            "four_flux_line_register": (
                "L10 has shape 10 by 4"
            ),
            "four_flux_line_gram": (
                "R10 = L10 L10^T"
            ),
            "notation_collision_avoided": (
                "R10 is not the native Q"
            ),
        },
        "matching": {
            "maximum_projective_distance": float(
                np.max(
                    matching_distances
                )
            ),
            "maximum_coordinate_residual": float(
                np.max(
                    matching_coordinate_residuals
                )
            ),
            "permutation_maximizer_to_axis": (
                permutation
            ),
            "all_matches_pass": all(
                row["match_pass"]
                for row in match_rows
            ),
        },
        "line_gram": {
            "native_signed_gram": (
                native_gram
            ),
            "native_absolute_gram": (
                native_absolute_gram
            ),
            "maximizer_signed_gram": (
                maximizer_gram
            ),
            "maximizer_absolute_gram": (
                maximizer_absolute_gram
            ),
            "signed_gram_maximum_residual": (
                gram_signed_residual
            ),
            "absolute_gram_maximum_residual": (
                gram_absolute_residual
            ),
            "absolute_off_diagonal_profile": (
                native_profile[
                    "absolute_profile"
                ]
            ),
            "signed_off_diagonal_profile": (
                native_profile[
                    "signed_profile"
                ]
            ),
            "native_eigenvalues": (
                native_gram_eigenvalues
            ),
            "maximizer_eigenvalues": (
                maximizer_gram_eigenvalues
            ),
        },
        "tight_frame_law": {
            "frame_constant": (
                "5/2"
            ),
            "register_law": (
                "L10^T L10 = (5/2) I4"
            ),
            "gram_law": (
                "R10^2 = (5/2) R10"
            ),
            "native_frame_residual": (
                native_frame_residual
            ),
            "maximizer_frame_residual": (
                maximizer_frame_residual
            ),
            "native_gram_quadratic_residual": (
                native_gram_quadratic_residual
            ),
            "maximizer_gram_quadratic_residual": (
                maximizer_gram_quadratic_residual
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "ten_maximizer_lines_are_native_flux_axis_lines": (
                theorem_pass
            ),
            "line_register_has_native_gram_law": (
                theorem_pass
            ),
            "one_third_extremizers_live_on_native_axis_register": (
                theorem_pass
            ),
            "sharp_one_third_upper_bound_proved": (
                False
            ),
        },
        "boundary": {
            "maximizer_terminal_set_identified_with_native_axes": (
                theorem_pass
            ),
            "all_global_maximizers_proved_to_be_in_register": (
                False
            ),
            "sharp_operator_norm_upper_bound_proved": (
                False
            ),
            "R10_identified_with_original_Q": False,
            "physical_claim": False,
        },
        "outputs": {
            "match_csv": str(
                MATCH_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "gram_csv": str(
                GRAM_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "identification_npz": str(
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

    np.savez_compressed(
        NPZ_OUT,
        native_line_matrix=(
            native_line_matrix
        ),
        maximizer_line_matrix=(
            maximizer_line_matrix
        ),
        native_signed_gram=(
            native_gram
        ),
        native_absolute_gram=(
            native_absolute_gram
        ),
        maximizer_signed_gram=(
            maximizer_gram
        ),
        maximizer_absolute_gram=(
            maximizer_absolute_gram
        ),
        native_frame_operator=(
            native_frame_operator
        ),
        maximizer_frame_operator=(
            maximizer_frame_operator
        ),
        matching_cost=matching_cost,
        matching_permutation=(
            permutation
        ),
        axis_clusters=np.array(
            axis_clusters,
            dtype=object,
        ),
        maximizer_clusters=np.array(
            maximizer_clusters,
            dtype=object,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "native_unoriented_axis_count:",
        len(axis_lines),
    )
    print(
        "maximizer_projective_cluster_count:",
        len(maximizer_lines),
    )
    print(
        "maximum_projective_matching_distance:",
        float(
            np.max(
                matching_distances
            )
        ),
    )
    print(
        "maximum_coordinate_matching_residual:",
        float(
            np.max(
                matching_coordinate_residuals
            )
        ),
    )
    print(
        "signed_gram_residual:",
        gram_signed_residual,
    )
    print(
        "absolute_gram_residual:",
        gram_absolute_residual,
    )
    print(
        "absolute_overlap_profile:",
        native_profile[
            "absolute_profile"
        ],
    )
    print(
        "signed_overlap_profile:",
        native_profile[
            "signed_profile"
        ],
    )
    print(
        "native_frame_operator:",
        native_frame_operator.tolist(),
    )
    print(
        "native_frame_residual:",
        native_frame_residual,
    )
    print(
        "native_gram_eigenvalues:",
        native_gram_eigenvalues.tolist(),
    )
    print(
        "native_gram_quadratic_residual:",
        native_gram_quadratic_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", MATCH_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
