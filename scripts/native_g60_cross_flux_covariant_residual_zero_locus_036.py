import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize




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

    overlap = min(
        max(overlap, -1.0),
        1.0,
    )

    return math.sqrt(
        max(
            0.0,
            2.0 - 2.0 * overlap,
        )
    )


def nearest_axis_distance(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> tuple[float, int]:
    distances = np.array(
        [
            projective_distance(
                point,
                axis,
            )
            for axis in axis_lines
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


def s4_value(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> float:
    overlaps = (
        axis_lines @ point
    )

    return float(
        np.sum(
            overlaps**4
        )
    )


def residual_vector(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    point = normalized(point)

    covariant = register_covariant(
        point,
        axis_lines,
    )

    s4 = s4_value(
        point,
        axis_lines,
    )

    return (
        covariant @ point
        - s4 * point
    )


def root_system(
    raw_point: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    point = np.array(
        raw_point,
        dtype=np.float64,
    )

    norm_constraint = (
        float(
            np.dot(point, point)
        )
        - 1.0
    )

    norm = float(
        np.linalg.norm(point)
    )

    if norm < 1e-12:
        residual = np.full(
            4,
            1e3,
            dtype=np.float64,
        )
    else:
        unit = point / norm

        covariant = register_covariant(
            unit,
            axis_lines,
        )

        s4 = s4_value(
            unit,
            axis_lines,
        )

        residual = (
            covariant @ unit
            - s4 * unit
        )

    return np.concatenate(
        [
            residual,
            np.array(
                [norm_constraint],
                dtype=np.float64,
            ),
        ]
    )


def solve_root_from_start(
    initial: np.ndarray,
    axis_lines: np.ndarray,
) -> dict:
    result = least_squares(
        root_system,
        initial,
        args=(axis_lines,),
        method="trf",
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=MAX_FUNCTION_EVALUATIONS,
    )

    point = canonical_sign(
        result.x
    )

    residual = residual_vector(
        point,
        axis_lines,
    )

    residual_norm = float(
        np.linalg.norm(residual)
    )

    norm_residual = abs(
        float(
            np.dot(
                result.x,
                result.x,
            )
            - 1.0
        )
    )

    axis_distance, axis_id = (
        nearest_axis_distance(
            point,
            axis_lines,
        )
    )

    return {
        "point": point,
        "residual_norm": residual_norm,
        "residual_max_abs": max_abs(
            residual
        ),
        "norm_residual": norm_residual,
        "axis_distance": axis_distance,
        "nearest_axis_id": axis_id,
        "success": bool(
            result.success
        ),
        "status": int(
            result.status
        ),
        "message": str(
            result.message
        ),
        "function_evaluations": int(
            result.nfev
        ),
        "root_pass": (
            residual_norm
            < ROOT_RESIDUAL_TOLERANCE
            and norm_residual
            < ROOT_RESIDUAL_TOLERANCE
        ),
    }


def cluster_projective_roots(
    roots: list[dict],
    tolerance: float,
) -> list[list[int]]:
    points = np.array(
        [
            root["point"]
            for root in roots
        ],
        dtype=np.float64,
    )

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


def cluster_representative(
    roots: list[dict],
    cluster: list[int],
) -> np.ndarray:
    reference = roots[
        cluster[0]
    ]["point"]

    aligned = []

    for index in cluster:
        point = roots[
            index
        ]["point"].copy()

        if np.dot(
            point,
            reference,
        ) < 0.0:
            point *= -1.0

        aligned.append(point)

    return canonical_sign(
        np.mean(
            np.array(
                aligned,
                dtype=np.float64,
            ),
            axis=0,
        )
    )


def tangent_basis(
    axis: np.ndarray,
) -> np.ndarray:
    axis = normalized(axis)

    projector = (
        np.eye(4)
        - np.outer(axis, axis)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        projector
    )

    basis = eigenvectors[
        :,
        eigenvalues > 0.5
    ]

    if basis.shape != (4, 3):
        raise RuntimeError(
            f"unexpected tangent basis shape: {basis.shape}"
        )

    return basis


def tangent_residual_jacobian(
    axis: np.ndarray,
    axis_lines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = tangent_basis(axis)

    jacobian = np.zeros(
        (4, 3),
        dtype=np.float64,
    )

    for column in range(3):
        tangent = basis[
            :,
            column,
        ]

        plus = normalized(
            axis
            + JACOBIAN_STEP
            * tangent
        )

        minus = normalized(
            axis
            - JACOBIAN_STEP
            * tangent
        )

        jacobian[
            :,
            column,
        ] = (
            residual_vector(
                plus,
                axis_lines,
            )
            - residual_vector(
                minus,
                axis_lines,
            )
        ) / (
            2.0 * JACOBIAN_STEP
        )

    singular_values = np.linalg.svd(
        jacobian,
        compute_uv=False,
    )

    return jacobian, singular_values


def exclusion_objective(
    raw_point: np.ndarray,
    axis_lines: np.ndarray,
) -> float:
    point = normalized(
        raw_point
    )

    residual = residual_vector(
        point,
        axis_lines,
    )

    residual_energy = float(
        np.dot(
            residual,
            residual,
        )
    )

    axis_distance, _ = (
        nearest_axis_distance(
            point,
            axis_lines,
        )
    )

    exclusion_violation = max(
        0.0,
        EXCLUSION_RADIUS
        - axis_distance,
    )

    return (
        residual_energy
        + EXCLUSION_PENALTY
        * exclusion_violation**2
    )


def run_exclusion_search(
    axis_lines: np.ndarray,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    rows = []

    best_valid_residual = float("inf")
    best_valid_distance = None
    valid_count = 0

    for start_id in range(
        EXCLUSION_START_COUNT
    ):
        initial = normalized(
            rng.normal(size=4)
        )

        result = minimize(
            exclusion_objective,
            initial,
            args=(axis_lines,),
            method="Nelder-Mead",
            options={
                "maxiter": 5000,
                "xatol": 1e-12,
                "fatol": 1e-16,
            },
        )

        point = canonical_sign(
            result.x
        )

        residual = residual_vector(
            point,
            axis_lines,
        )

        residual_norm = float(
            np.linalg.norm(
                residual
            )
        )

        axis_distance, axis_id = (
            nearest_axis_distance(
                point,
                axis_lines,
            )
        )

        valid_exclusion = (
            axis_distance
            >= EXCLUSION_RADIUS
            - 1e-6
        )

        if valid_exclusion:
            valid_count += 1

            if residual_norm < best_valid_residual:
                best_valid_residual = (
                    residual_norm
                )

                best_valid_distance = (
                    axis_distance
                )

        rows.append(
            {
                "start_id": start_id,
                "residual_norm": (
                    residual_norm
                ),
                "residual_energy": (
                    residual_norm**2
                ),
                "axis_distance": (
                    axis_distance
                ),
                "nearest_axis_id": (
                    axis_id
                ),
                "valid_exclusion": (
                    valid_exclusion
                ),
                "objective": float(
                    result.fun
                ),
                "success": bool(
                    result.success
                ),
                "iteration_count": int(
                    result.nit
                ),
                "f0": float(point[0]),
                "f1": float(point[1]),
                "f2": float(point[2]),
                "f3": float(point[3]),
            }
        )

        if (
            start_id == 0
            or (start_id + 1) % 64 == 0
            or start_id + 1
            == EXCLUSION_START_COUNT
        ):
            print(
                "\rexclusion_search:",
                f"{start_id + 1}/{EXCLUSION_START_COUNT}",
                "best_valid_residual:",
                best_valid_residual,
                end="",
                flush=True,
            )

    print()

    summary = {
        "start_count": (
            EXCLUSION_START_COUNT
        ),
        "exclusion_radius": (
            EXCLUSION_RADIUS
        ),
        "valid_exclusion_count": (
            valid_count
        ),
        "best_valid_residual_norm": (
            best_valid_residual
            if np.isfinite(
                best_valid_residual
            )
            else None
        ),
        "best_valid_axis_distance": (
            best_valid_distance
        ),
        "non_axis_zero_found": (
            np.isfinite(
                best_valid_residual
            )
            and best_valid_residual
            < ROOT_RESIDUAL_TOLERANCE
        ),
    }

    return rows, summary



ROOT = Path(__file__).resolve().parents[1]

ORIENTATION_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_covariant_orientation_035.json"
)

ORIENTATION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_covariant_orientation_035.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_covariant_residual_zero_locus_036.json"
)

ROOT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_residual_roots_036.csv"
)

START_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_residual_starts_036.csv"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_residual_axis_jacobians_036.csv"
)

EXCLUSION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_covariant_residual_exclusion_search_036.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_covariant_residual_zero_locus_036.npz"
)

RANDOM_SEED = 46036

ROOT_START_COUNT = 2048
EXCLUSION_START_COUNT = 512

ROOT_RESIDUAL_TOLERANCE = 1e-9
PROJECTIVE_CLUSTER_TOLERANCE = 2e-6
AXIS_MATCH_TOLERANCE = 2e-6

JACOBIAN_STEP = 1e-6
JACOBIAN_RANK_TOLERANCE = 1e-7

EXCLUSION_RADIUS = 0.05
EXCLUSION_PENALTY = 100.0

MAX_FUNCTION_EVALUATIONS = 4000

def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ROOT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    START_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AXIS_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXCLUSION_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    orientation_receipt = json.loads(
        ORIENTATION_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        ORIENTATION_NPZ_PATH
    )

    axis_lines = np.array(
        data["axis_lines"],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis line shape: {axis_lines.shape}"
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    solved_rows = []
    roots = []

    started_at = time.monotonic()

    initial_points = [
        axis.copy()
        for axis in axis_lines
    ]

    while len(
        initial_points
    ) < ROOT_START_COUNT:
        initial_points.append(
            normalized(
                rng.normal(size=4)
            )
        )

    for start_id, initial in enumerate(
        initial_points
    ):
        result = solve_root_from_start(
            initial,
            axis_lines,
        )

        solved_rows.append(
            {
                "start_id": start_id,
                "root_pass": (
                    result["root_pass"]
                ),
                "residual_norm": (
                    result["residual_norm"]
                ),
                "residual_max_abs": (
                    result[
                        "residual_max_abs"
                    ]
                ),
                "norm_residual": (
                    result["norm_residual"]
                ),
                "axis_distance": (
                    result["axis_distance"]
                ),
                "nearest_axis_id": (
                    result[
                        "nearest_axis_id"
                    ]
                ),
                "success": (
                    result["success"]
                ),
                "status": (
                    result["status"]
                ),
                "function_evaluations": (
                    result[
                        "function_evaluations"
                    ]
                ),
                "f0": float(
                    result["point"][0]
                ),
                "f1": float(
                    result["point"][1]
                ),
                "f2": float(
                    result["point"][2]
                ),
                "f3": float(
                    result["point"][3]
                ),
            }
        )

        if result["root_pass"]:
            roots.append(result)

        if (
            start_id == 0
            or (start_id + 1) % 128 == 0
            or start_id + 1
            == ROOT_START_COUNT
        ):
            elapsed = (
                time.monotonic()
                - started_at
            )

            print(
                "\rroot_search:",
                f"{start_id + 1}/{ROOT_START_COUNT}",
                "accepted_roots:",
                len(roots),
                "elapsed:",
                f"{elapsed:.1f}s",
                end="",
                flush=True,
            )

    print()

    clusters = cluster_projective_roots(
        roots,
        PROJECTIVE_CLUSTER_TOLERANCE,
    )

    cluster_rows = []
    representatives = []

    for cluster_id, cluster in enumerate(
        clusters
    ):
        representative = (
            cluster_representative(
                roots,
                cluster,
            )
        )

        representatives.append(
            representative
        )

        residual = residual_vector(
            representative,
            axis_lines,
        )

        axis_distance, axis_id = (
            nearest_axis_distance(
                representative,
                axis_lines,
            )
        )

        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(
                    cluster
                ),
                "residual_norm": float(
                    np.linalg.norm(
                        residual
                    )
                ),
                "axis_distance": (
                    axis_distance
                ),
                "nearest_axis_id": (
                    axis_id
                ),
                "matches_native_axis": (
                    axis_distance
                    < AXIS_MATCH_TOLERANCE
                ),
                "f0": float(
                    representative[0]
                ),
                "f1": float(
                    representative[1]
                ),
                "f2": float(
                    representative[2]
                ),
                "f3": float(
                    representative[3]
                ),
            }
        )

    axis_rows = []
    jacobians = []
    jacobian_singular_values = []

    for axis_id, axis in enumerate(
        axis_lines
    ):
        jacobian, singular_values = (
            tangent_residual_jacobian(
                axis,
                axis_lines,
            )
        )

        jacobians.append(jacobian)
        jacobian_singular_values.append(
            singular_values
        )

        axis_rows.append(
            {
                "axis_id": axis_id,
                "jacobian_rank": int(
                    np.count_nonzero(
                        singular_values
                        > JACOBIAN_RANK_TOLERANCE
                    )
                ),
                "minimum_singular_value": float(
                    singular_values[-1]
                ),
                "maximum_singular_value": float(
                    singular_values[0]
                ),
                "locally_isolated": (
                    np.count_nonzero(
                        singular_values
                        > JACOBIAN_RANK_TOLERANCE
                    )
                    == 3
                ),
            }
        )

    exclusion_rows, exclusion_summary = (
        run_exclusion_search(
            axis_lines
        )
    )

    all_clusters_match_axes = (
        len(cluster_rows) > 0
        and all(
            row[
                "matches_native_axis"
            ]
            for row in cluster_rows
        )
    )

    matched_axis_ids = sorted(
        {
            row["nearest_axis_id"]
            for row in cluster_rows
            if row[
                "matches_native_axis"
            ]
        }
    )

    all_ten_axes_recovered = (
        matched_axis_ids
        == list(range(10))
    )

    all_axes_locally_isolated = all(
        row["locally_isolated"]
        for row in axis_rows
    )

    checks = {
        "input_035_theorem_pass": (
            orientation_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "axis_line_count_is_10": (
            len(axis_lines) == 10
        ),
        "root_search_completed": (
            len(solved_rows)
            == ROOT_START_COUNT
        ),
        "at_least_one_root_recovered": (
            len(roots) > 0
        ),
        "all_recovered_root_clusters_match_native_axes": (
            all_clusters_match_axes
        ),
        "all_ten_native_axis_lines_recovered": (
            all_ten_axes_recovered
        ),
        "all_native_axes_locally_isolated": (
            all_axes_locally_isolated
        ),
        "exclusion_search_completed": (
            len(exclusion_rows)
            == EXCLUSION_START_COUNT
        ),
        "no_non_axis_zero_found": (
            not exclusion_summary[
                "non_axis_zero_found"
            ]
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    verdict = (
        "native_g60_cross_flux_residual_zero_census_recovers_only_ten_axes"
        if audit_pass
        else "native_g60_cross_flux_residual_zero_locus_census_incomplete"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_covariant_residual_zero_locus_036"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "system": {
            "equation": (
                "C(f)f-S4(f)f=0"
            ),
            "sphere_constraint": (
                "||f||^2=1"
            ),
            "projective_identification": (
                "f equivalent to -f"
            ),
        },
        "root_search": {
            "start_count": (
                ROOT_START_COUNT
            ),
            "accepted_root_count": (
                len(roots)
            ),
            "projective_cluster_count": (
                len(clusters)
            ),
            "matched_axis_ids": (
                matched_axis_ids
            ),
            "all_ten_axes_recovered": (
                all_ten_axes_recovered
            ),
            "all_clusters_match_axes": (
                all_clusters_match_axes
            ),
            "cluster_rows": (
                cluster_rows
            ),
        },
        "axis_jacobian": {
            "all_axes_locally_isolated": (
                all_axes_locally_isolated
            ),
            "minimum_tangent_jacobian_singular_value": min(
                row[
                    "minimum_singular_value"
                ]
                for row in axis_rows
            ),
            "axis_rows": axis_rows,
        },
        "exclusion_search": (
            exclusion_summary
        ),
        "checks": checks,
        "earned_interpretation": {
            "all_recovered_real_projective_zeros_are_native_axes": (
                audit_pass
            ),
            "all_ten_native_axes_recovered": (
                all_ten_axes_recovered
            ),
            "native_axis_zeros_are_locally_isolated": (
                all_axes_locally_isolated
            ),
            "non_axis_zero_found": (
                exclusion_summary[
                    "non_axis_zero_found"
                ]
            ),
            "complete_real_zero_locus_proved": (
                False
            ),
        },
        "boundary": {
            "multistart_zero_locus_census_completed": (
                audit_pass
            ),
            "local_isolation_proved_numerically": (
                all_axes_locally_isolated
            ),
            "global_algebraic_zero_locus_proved": (
                False
            ),
            "r_equals_zero_iff_native_axis_proved": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "root_csv": str(
                ROOT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "start_csv": str(
                START_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "axis_csv": str(
                AXIS_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "exclusion_csv": str(
                EXCLUSION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "zero_locus_npz": str(
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
            ROOT_CSV_OUT,
            cluster_rows,
        ),
        (
            START_CSV_OUT,
            solved_rows,
        ),
        (
            AXIS_CSV_OUT,
            axis_rows,
        ),
        (
            EXCLUSION_CSV_OUT,
            exclusion_rows,
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
        root_representatives=np.array(
            representatives,
            dtype=np.float64,
        ),
        root_cluster_sizes=np.array(
            [
                len(cluster)
                for cluster in clusters
            ],
            dtype=np.int64,
        ),
        axis_jacobians=np.array(
            jacobians,
            dtype=np.float64,
        ),
        axis_jacobian_singular_values=np.array(
            jacobian_singular_values,
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "accepted_root_count:",
        len(roots),
    )
    print(
        "projective_cluster_count:",
        len(clusters),
    )
    print(
        "matched_axis_ids:",
        matched_axis_ids,
    )
    print(
        "all_clusters_match_native_axes:",
        all_clusters_match_axes,
    )
    print(
        "all_axes_locally_isolated:",
        all_axes_locally_isolated,
    )
    print(
        "minimum_axis_jacobian_singular_value:",
        payload[
            "axis_jacobian"
        ][
            "minimum_tangent_jacobian_singular_value"
        ],
    )
    print(
        "exclusion_summary:",
        exclusion_summary,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", ROOT_CSV_OUT)
    print("wrote:", START_CSV_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", EXCLUSION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
