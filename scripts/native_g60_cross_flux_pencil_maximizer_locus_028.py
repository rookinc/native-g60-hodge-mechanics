from __future__ import annotations

import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PENCIL_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_operator_pencil_026.json"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

START_CSV_PATH = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_operator_pencil_starts_026.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_pencil_maximizer_locus_028.json"
)

POINT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_pencil_maximizer_points_028.csv"
)

QUADRATIC_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_pencil_quadratic_relations_028.csv"
)

GENERIC_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_pencil_generic_comparison_028.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_pencil_maximizer_locus_028.npz"
)

TARGET_NORM = 1.0 / 3.0
TARGET_TOLERANCE = 1e-8

QUADRATIC_NULL_RELATIVE_TOLERANCE = 1e-8
QUADRATIC_RELATION_TOLERANCE = 2e-7

GENERIC_PROBE_COUNT = 20000
RANDOM_SEED = 46028

SIGN_CLUSTER_TOLERANCES = (
    1e-10,
    1e-8,
    1e-6,
    1e-4,
)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def normalized(
    vector: np.ndarray,
) -> np.ndarray:
    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0.0:
        raise RuntimeError(
            "cannot normalize zero vector"
        )

    return vector / norm


def canonical_sign(
    vector: np.ndarray,
) -> np.ndarray:
    result = normalized(vector)

    pivot = int(
        np.argmax(
            np.abs(result)
        )
    )

    if result[pivot] < 0.0:
        result *= -1.0

    return result


def max_abs(
    array: np.ndarray,
) -> float:
    if array.size == 0:
        return 0.0

    return float(
        np.max(
            np.abs(array)
        )
    )


def pencil_matrix(
    slices: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "r,rab->ab",
        f,
        slices,
    )


def operator_norm(
    slices: np.ndarray,
    f: np.ndarray,
) -> float:
    return float(
        np.linalg.svd(
            pencil_matrix(
                slices,
                f,
            ),
            compute_uv=False,
        )[0]
    )


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

    return math.sqrt(
        max(
            0.0,
            2.0 - 2.0 * dot,
        )
    )


def load_terminal_points() -> tuple[
    np.ndarray,
    list[dict],
]:
    rows = []

    with START_CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for raw in reader:
            point = canonical_sign(
                np.array(
                    [
                        float(raw["f0"]),
                        float(raw["f1"]),
                        float(raw["f2"]),
                        float(raw["f3"]),
                    ],
                    dtype=np.float64,
                )
            )

            rows.append(
                {
                    "start_id": int(
                        raw["start_id"]
                    ),
                    "sigma": float(
                        raw["sigma"]
                    ),
                    "target_gap": float(
                        raw["target_gap"]
                    ),
                    "stationarity_residual": float(
                        raw[
                            "tangent_stationarity_residual"
                        ]
                    ),
                    "point": point,
                }
            )

    points = np.array(
        [
            row["point"]
            for row in rows
        ],
        dtype=np.float64,
    )

    return points, rows


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


def quadratic_feature_labels() -> list[str]:
    return [
        "f0^2",
        "f1^2",
        "f2^2",
        "f3^2",
        "2*f0*f1",
        "2*f0*f2",
        "2*f0*f3",
        "2*f1*f2",
        "2*f1*f3",
        "2*f2*f3",
    ]


def quadratic_features(
    points: np.ndarray,
) -> np.ndarray:
    f0 = points[:, 0]
    f1 = points[:, 1]
    f2 = points[:, 2]
    f3 = points[:, 3]

    return np.column_stack(
        [
            f0 * f0,
            f1 * f1,
            f2 * f2,
            f3 * f3,
            2.0 * f0 * f1,
            2.0 * f0 * f2,
            2.0 * f0 * f3,
            2.0 * f1 * f2,
            2.0 * f1 * f3,
            2.0 * f2 * f3,
        ]
    )


def vector_to_symmetric_matrix(
    coefficients: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros(
        (4, 4),
        dtype=np.float64,
    )

    matrix[0, 0] = coefficients[0]
    matrix[1, 1] = coefficients[1]
    matrix[2, 2] = coefficients[2]
    matrix[3, 3] = coefficients[3]

    matrix[0, 1] = matrix[1, 0] = (
        coefficients[4]
    )
    matrix[0, 2] = matrix[2, 0] = (
        coefficients[5]
    )
    matrix[0, 3] = matrix[3, 0] = (
        coefficients[6]
    )
    matrix[1, 2] = matrix[2, 1] = (
        coefficients[7]
    )
    matrix[1, 3] = matrix[3, 1] = (
        coefficients[8]
    )
    matrix[2, 3] = matrix[3, 2] = (
        coefficients[9]
    )

    return matrix


def rational_profile(
    coefficients: np.ndarray,
    denominator_limit: int = 720,
) -> list[str]:
    scale = float(
        np.max(
            np.abs(coefficients)
        )
    )

    if scale == 0.0:
        return [
            "0"
            for _ in coefficients
        ]

    normalized_coefficients = (
        coefficients / scale
    )

    return [
        str(
            Fraction(float(value))
            .limit_denominator(
                denominator_limit
            )
        )
        for value in normalized_coefficients
    ]


def fit_quadratic_relations(
    points: np.ndarray,
) -> tuple[
    list[dict],
    np.ndarray,
    np.ndarray,
    dict,
]:
    design = quadratic_features(
        points
    )

    _, singular_values, right_t = np.linalg.svd(
        design,
        full_matrices=False,
    )

    threshold = max(
        QUADRATIC_NULL_RELATIVE_TOLERANCE
        * singular_values[0],
        1e-12,
    )

    null_indices = [
        index
        for index, value in enumerate(
            singular_values
        )
        if value < threshold
    ]

    relation_vectors = (
        right_t[null_indices]
        if null_indices
        else np.empty(
            (0, 10),
            dtype=np.float64,
        )
    )

    relation_matrices = np.array(
        [
            vector_to_symmetric_matrix(
                vector
            )
            for vector in relation_vectors
        ],
        dtype=np.float64,
    )

    relation_rows = []

    for relation_id, (
        vector,
        matrix,
    ) in enumerate(
        zip(
            relation_vectors,
            relation_matrices,
        )
    ):
        values = design @ vector

        eigenvalues = np.linalg.eigvalsh(
            matrix
        )

        relation_rows.append(
            {
                "relation_id": relation_id,
                "singular_value": float(
                    singular_values[
                        null_indices[
                            relation_id
                        ]
                    ]
                ),
                "maximum_terminal_residual": (
                    max_abs(values)
                ),
                "matrix_rank": int(
                    np.linalg.matrix_rank(
                        matrix,
                        tol=1e-9,
                    )
                ),
                "matrix_trace": float(
                    np.trace(matrix)
                ),
                "minimum_matrix_eigenvalue": float(
                    eigenvalues[0]
                ),
                "maximum_matrix_eigenvalue": float(
                    eigenvalues[-1]
                ),
                "coefficient_vector": (
                    vector.tolist()
                ),
                "rational_profile_scaled": (
                    rational_profile(
                        vector
                    )
                ),
            }
        )

    summary = {
        "design_shape": list(
            design.shape
        ),
        "design_rank": int(
            np.linalg.matrix_rank(
                design,
                tol=threshold,
            )
        ),
        "singular_values": (
            singular_values.tolist()
        ),
        "null_threshold": threshold,
        "quadratic_nullity": len(
            null_indices
        ),
        "maximum_relation_terminal_residual": (
            max(
                (
                    row[
                        "maximum_terminal_residual"
                    ]
                    for row in relation_rows
                ),
                default=None,
            )
        ),
    }

    return (
        relation_rows,
        relation_vectors,
        relation_matrices,
        summary,
    )


def quadratic_relation_score(
    points: np.ndarray,
    relation_vectors: np.ndarray,
) -> np.ndarray:
    if relation_vectors.shape[0] == 0:
        return np.zeros(
            len(points),
            dtype=np.float64,
        )

    values = (
        quadratic_features(points)
        @ relation_vectors.T
    )

    return np.linalg.norm(
        values,
        axis=1,
    )


def generic_sphere_comparison(
    slices: np.ndarray,
    relation_vectors: np.ndarray,
) -> tuple[
    list[dict],
    dict,
]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    points = np.array(
        [
            normalized(
                rng.normal(size=4)
            )
            for _ in range(
                GENERIC_PROBE_COUNT
            )
        ],
        dtype=np.float64,
    )

    relation_scores = (
        quadratic_relation_score(
            points,
            relation_vectors,
        )
    )

    sigma_values = np.array(
        [
            operator_norm(
                slices,
                point,
            )
            for point in points
        ],
        dtype=np.float64,
    )

    target_gaps = (
        TARGET_NORM
        - sigma_values
    )

    comparison_rows = []

    for probe_id in range(
        min(
            GENERIC_PROBE_COUNT,
            4096,
        )
    ):
        comparison_rows.append(
            {
                "probe_id": probe_id,
                "f0": float(
                    points[probe_id, 0]
                ),
                "f1": float(
                    points[probe_id, 1]
                ),
                "f2": float(
                    points[probe_id, 2]
                ),
                "f3": float(
                    points[probe_id, 3]
                ),
                "sigma": float(
                    sigma_values[probe_id]
                ),
                "target_gap": float(
                    target_gaps[probe_id]
                ),
                "quadratic_relation_score": float(
                    relation_scores[probe_id]
                ),
            }
        )

    near_maximum_mask = (
        target_gaps
        < TARGET_TOLERANCE
    )

    summary = {
        "probe_count": (
            GENERIC_PROBE_COUNT
        ),
        "maximum_sigma": float(
            np.max(sigma_values)
        ),
        "minimum_target_gap": float(
            np.min(target_gaps)
        ),
        "near_maximum_count": int(
            np.count_nonzero(
                near_maximum_mask
            )
        ),
        "minimum_quadratic_relation_score": float(
            np.min(
                relation_scores
            )
        ),
        "median_quadratic_relation_score": float(
            np.median(
                relation_scores
            )
        ),
        "maximum_quadratic_relation_score": float(
            np.max(
                relation_scores
            )
        ),
        "correlation_relation_score_target_gap": (
            float(
                np.corrcoef(
                    relation_scores,
                    target_gaps,
                )[0, 1]
            )
            if (
                np.std(relation_scores)
                > 0.0
                and np.std(target_gaps)
                > 0.0
            )
            else None
        ),
    }

    return comparison_rows, summary


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    POINT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QUADRATIC_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GENERIC_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pencil_receipt = json.loads(
        PENCIL_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    points, source_rows = (
        load_terminal_points()
    )

    sigma_values = np.array(
        [
            row["sigma"]
            for row in source_rows
        ],
        dtype=np.float64,
    )

    point_norm_residual = max_abs(
        np.linalg.norm(
            points,
            axis=1,
        )
        - 1.0
    )

    centered_points = (
        points
        - np.mean(
            points,
            axis=0,
        )
    )

    point_singular_values = np.linalg.svd(
        centered_points,
        compute_uv=False,
    )

    covariance = (
        points.T @ points
        / len(points)
    )

    covariance_eigenvalues = (
        np.linalg.eigvalsh(
            covariance
        )
    )

    cluster_profiles = {}

    for tolerance in (
        SIGN_CLUSTER_TOLERANCES
    ):
        clusters = cluster_projective_points(
            points,
            tolerance,
        )

        cluster_profiles[
            f"{tolerance:.0e}"
        ] = {
            "cluster_count": len(
                clusters
            ),
            "largest_cluster_size": max(
                len(cluster)
                for cluster in clusters
            ),
            "cluster_size_profile": sorted(
                [
                    len(cluster)
                    for cluster in clusters
                ],
                reverse=True,
            ),
        }

    pairwise_distances = []

    for first in range(len(points)):
        for second in range(
            first + 1,
            len(points),
        ):
            pairwise_distances.append(
                projective_distance(
                    points[first],
                    points[second],
                )
            )

    pairwise_distances = np.array(
        pairwise_distances,
        dtype=np.float64,
    )

    (
        relation_rows,
        relation_vectors,
        relation_matrices,
        relation_summary,
    ) = fit_quadratic_relations(
        points
    )

    terminal_relation_scores = (
        quadratic_relation_score(
            points,
            relation_vectors,
        )
    )

    (
        generic_rows,
        generic_summary,
    ) = generic_sphere_comparison(
        slices,
        relation_vectors,
    )

    point_rows = []

    for source_row, score in zip(
        source_rows,
        terminal_relation_scores,
    ):
        point = source_row["point"]

        point_rows.append(
            {
                "start_id": (
                    source_row["start_id"]
                ),
                "sigma": (
                    source_row["sigma"]
                ),
                "target_gap": (
                    source_row["target_gap"]
                ),
                "stationarity_residual": (
                    source_row[
                        "stationarity_residual"
                    ]
                ),
                "f0": float(point[0]),
                "f1": float(point[1]),
                "f2": float(point[2]),
                "f3": float(point[3]),
                "quadratic_relation_score": float(
                    score
                ),
            }
        )

    terminal_all_near_target = bool(
        np.all(
            np.abs(
                sigma_values
                - TARGET_NORM
            )
            < TARGET_TOLERANCE
        )
    )

    quadratic_relation_found = (
        relation_vectors.shape[0] > 0
    )

    quadratic_relations_resolved = (
        quadratic_relation_found
        and max_abs(
            terminal_relation_scores
        )
        < QUADRATIC_RELATION_TOLERANCE
    )

    checks = {
        "input_026_audit_pass": (
            pencil_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "terminal_point_count_is_256": (
            len(points) == 256
        ),
        "all_terminal_points_are_unit": (
            point_norm_residual
            < 1e-10
        ),
        "all_terminal_points_reach_one_third": (
            terminal_all_near_target
        ),
        "terminal_points_span_four_coordinates": (
            np.linalg.matrix_rank(
                points,
                tol=1e-10,
            )
            == 4
        ),
        "quadratic_fit_completed": (
            relation_summary[
                "design_shape"
            ]
            == [256, 10]
        ),
        "generic_comparison_completed": (
            generic_summary[
                "probe_count"
            ]
            == GENERIC_PROBE_COUNT
        ),
    }

    audit_pass = all(
        checks.values()
    )

    if quadratic_relations_resolved:
        verdict = (
            "native_g60_cross_flux_maximizer_locus_"
            "has_resolved_quadratic_relations"
        )
    elif quadratic_relation_found:
        verdict = (
            "native_g60_cross_flux_maximizer_locus_"
            "has_approximate_quadratic_relations"
        )
    else:
        verdict = (
            "native_g60_cross_flux_maximizer_locus_"
            "not_quadratically_resolved"
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_pencil_maximizer_locus_028"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": False,
        "verdict": verdict,
        "terminal_set": {
            "point_count": len(
                points
            ),
            "all_reach_one_third": (
                terminal_all_near_target
            ),
            "maximum_target_gap_abs": float(
                np.max(
                    np.abs(
                        sigma_values
                        - TARGET_NORM
                    )
                )
            ),
            "unit_norm_maximum_residual": (
                point_norm_residual
            ),
            "coordinate_rank": int(
                np.linalg.matrix_rank(
                    points,
                    tol=1e-10,
                )
            ),
            "centered_rank": int(
                np.linalg.matrix_rank(
                    centered_points,
                    tol=1e-10,
                )
            ),
            "centered_singular_values": (
                point_singular_values
            ),
            "covariance": covariance,
            "covariance_eigenvalues": (
                covariance_eigenvalues
            ),
            "projective_pairwise_distance": {
                "minimum": float(
                    np.min(
                        pairwise_distances
                    )
                ),
                "median": float(
                    np.median(
                        pairwise_distances
                    )
                ),
                "maximum": float(
                    np.max(
                        pairwise_distances
                    )
                ),
            },
            "cluster_profiles": (
                cluster_profiles
            ),
        },
        "quadratic_fit": {
            **relation_summary,
            "relation_found": (
                quadratic_relation_found
            ),
            "relations_resolved_on_terminal_set": (
                quadratic_relations_resolved
            ),
            "terminal_score_maximum": (
                max_abs(
                    terminal_relation_scores
                )
            ),
            "relation_rows": (
                relation_rows
            ),
        },
        "generic_comparison": (
            generic_summary
        ),
        "checks": checks,
        "earned_interpretation": {
            "maximizer_terminal_set_collected": (
                audit_pass
            ),
            "finite_projective_orbit_identified": (
                cluster_profiles[
                    "1e-08"
                ][
                    "cluster_count"
                ]
                < len(points)
            ),
            "quadratic_locus_candidate_found": (
                quadratic_relation_found
            ),
            "quadratic_locus_proved": False,
            "complete_maximizer_locus_classified": (
                False
            ),
        },
        "boundary": {
            "terminal_optimizer_locus_audited": (
                audit_pass
            ),
            "all_global_maximizers_found": False,
            "terminal_set_equals_complete_locus": (
                False
            ),
            "quadratic_equations_exactly_derived": (
                False
            ),
            "sharp_one_third_upper_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "point_csv": str(
                POINT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "quadratic_csv": str(
                QUADRATIC_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "generic_csv": str(
                GENERIC_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "locus_npz": str(
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

    with POINT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                point_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(point_rows)

    if relation_rows:
        flattened_relation_rows = []

        labels = quadratic_feature_labels()

        for row in relation_rows:
            flattened = {
                key: value
                for key, value in row.items()
                if key not in (
                    "coefficient_vector",
                    "rational_profile_scaled",
                )
            }

            for label, value, rational in zip(
                labels,
                row["coefficient_vector"],
                row[
                    "rational_profile_scaled"
                ],
            ):
                safe_label = (
                    label
                    .replace("*", "_")
                    .replace("^", "_pow_")
                )

                flattened[
                    f"coefficient_{safe_label}"
                ] = value

                flattened[
                    f"rational_{safe_label}"
                ] = rational

            flattened_relation_rows.append(
                flattened
            )

        with QUADRATIC_CSV_OUT.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    flattened_relation_rows[0]
                ),
            )
            writer.writeheader()
            writer.writerows(
                flattened_relation_rows
            )
    else:
        QUADRATIC_CSV_OUT.write_text(
            "relation_id,status\n"
            "0,no_quadratic_nullspace_found\n",
            encoding="utf-8",
        )

    with GENERIC_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                generic_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            generic_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        terminal_points=points,
        terminal_sigmas=(
            sigma_values
        ),
        covariance=covariance,
        covariance_eigenvalues=(
            covariance_eigenvalues
        ),
        quadratic_relation_vectors=(
            relation_vectors
        ),
        quadratic_relation_matrices=(
            relation_matrices
        ),
        quadratic_design_singular_values=np.array(
            relation_summary[
                "singular_values"
            ],
            dtype=np.float64,
        ),
        terminal_relation_scores=(
            terminal_relation_scores
        ),
        pairwise_projective_distances=(
            pairwise_distances
        ),
        target_norm=np.array(
            [TARGET_NORM],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", False)
    print("verdict:", verdict)
    print(
        "terminal_point_count:",
        len(points),
    )
    print(
        "all_terminal_points_reach_one_third:",
        terminal_all_near_target,
    )
    print(
        "coordinate_rank:",
        int(
            np.linalg.matrix_rank(
                points,
                tol=1e-10,
            )
        ),
    )
    print(
        "centered_rank:",
        int(
            np.linalg.matrix_rank(
                centered_points,
                tol=1e-10,
            )
        ),
    )
    print(
        "covariance_eigenvalues:",
        covariance_eigenvalues.tolist(),
    )
    print(
        "projective_distance_min/median/max:",
        float(
            np.min(
                pairwise_distances
            )
        ),
        float(
            np.median(
                pairwise_distances
            )
        ),
        float(
            np.max(
                pairwise_distances
            )
        ),
    )
    print(
        "cluster_profiles:",
        cluster_profiles,
    )
    print(
        "quadratic_design_singular_values:",
        relation_summary[
            "singular_values"
        ],
    )
    print(
        "quadratic_nullity:",
        relation_summary[
            "quadratic_nullity"
        ],
    )
    print(
        "terminal_quadratic_score_max:",
        max_abs(
            terminal_relation_scores
        ),
    )
    print(
        "generic_comparison:",
        generic_summary,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", POINT_CSV_OUT)
    print("wrote:", QUADRATIC_CSV_OUT)
    print("wrote:", GENERIC_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
