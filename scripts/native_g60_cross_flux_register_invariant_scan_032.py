from __future__ import annotations

import csv
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]

HESSIAN_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_axis_local_hessian_031.json"
)

IDENTIFICATION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_maximizer_axis_identification_029.npz"
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
    / "native_g60_cross_flux_register_invariant_scan_032.json"
)

SAMPLE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_register_invariant_samples_032.csv"
)

OPTIMIZATION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_register_invariant_optimizations_032.csv"
)

REGRESSION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_register_invariant_regressions_032.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_register_invariant_scan_032.npz"
)

RANDOM_SEED = 46032
RANDOM_SAMPLE_COUNT = 50000
SAVED_SAMPLE_COUNT = 5000

OPTIMIZATION_START_COUNT = 256
MAX_ITERATIONS = 3000

TARGET_OPERATOR_NORM = 1.0 / 3.0
TARGET_TOP_EIGENVALUE = 1.0 / 9.0
EXPECTED_AXIS_S4 = 115.0 / 72.0

UNIT_TOLERANCE = 2e-10
INVARIANT_TOLERANCE = 2e-9
OPTIMIZATION_TOLERANCE = 2e-7


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


def rational_label(
    value: float,
    denominator_limit: int = 100000,
) -> str:
    return str(
        Fraction(float(value))
        .limit_denominator(
            denominator_limit
        )
    )


def pencil_matrix(
    slices: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "r,rab->ab",
        direction,
        slices,
    )


def operator_norm(
    slices: np.ndarray,
    direction: np.ndarray,
) -> float:
    return float(
        np.linalg.svd(
            pencil_matrix(
                slices,
                direction,
            ),
            compute_uv=False,
        )[0]
    )


def projective_distance_to_register(
    directions: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    absolute_overlaps = np.abs(
        directions @ axis_lines.T
    )

    maximum_overlaps = np.max(
        absolute_overlaps,
        axis=1,
    )

    maximum_overlaps = np.clip(
        maximum_overlaps,
        -1.0,
        1.0,
    )

    return np.sqrt(
        np.maximum(
            0.0,
            2.0
            - 2.0
            * maximum_overlaps,
        )
    )


def register_moments(
    directions: np.ndarray,
    axis_lines: np.ndarray,
) -> dict[str, np.ndarray]:
    overlaps = (
        directions @ axis_lines.T
    )

    return {
        "s2": np.sum(
            overlaps**2,
            axis=1,
        ),
        "s4": np.sum(
            overlaps**4,
            axis=1,
        ),
        "s6": np.sum(
            overlaps**6,
            axis=1,
        ),
        "s8": np.sum(
            overlaps**8,
            axis=1,
        ),
    }


def batch_operator_data(
    slices: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.einsum(
        "nr,rab->nab",
        directions,
        slices,
    )

    singular_values = np.linalg.svd(
        matrices,
        compute_uv=False,
    )

    operator_norms = (
        singular_values[:, 0]
    )

    return (
        operator_norms,
        singular_values,
    )


def regression_summary(
    name: str,
    design: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, np.ndarray]:
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    prediction = design @ coefficients
    residual = target - prediction

    target_centered = (
        target - np.mean(target)
    )

    denominator = float(
        np.dot(
            target_centered,
            target_centered,
        )
    )

    r_squared = (
        1.0
        - float(
            np.dot(
                residual,
                residual,
            )
        )
        / denominator
        if denominator > 0.0
        else 1.0
    )

    row = {
        "model": name,
        "feature_names": json.dumps(
            feature_names
        ),
        "coefficient_count": len(
            coefficients
        ),
        "coefficients": json.dumps(
            coefficients.tolist()
        ),
        "r_squared": r_squared,
        "root_mean_square_residual": float(
            np.sqrt(
                np.mean(
                    residual**2
                )
            )
        ),
        "maximum_absolute_residual": max_abs(
            residual
        ),
        "minimum_residual": float(
            np.min(residual)
        ),
        "maximum_residual": float(
            np.max(residual)
        ),
    }

    return row, prediction


def binned_spread(
    invariant: np.ndarray,
    target: np.ndarray,
    bin_count: int = 200,
) -> dict:
    minimum = float(
        np.min(invariant)
    )

    maximum = float(
        np.max(invariant)
    )

    edges = np.linspace(
        minimum,
        maximum,
        bin_count + 1,
    )

    indices = np.clip(
        np.digitize(
            invariant,
            edges,
        )
        - 1,
        0,
        bin_count - 1,
    )

    maximum_target_spread = 0.0
    populated_bin_count = 0
    widest_bin = None

    for bin_index in range(bin_count):
        values = target[
            indices == bin_index
        ]

        if len(values) < 2:
            continue

        populated_bin_count += 1

        spread = float(
            np.max(values)
            - np.min(values)
        )

        if spread > maximum_target_spread:
            maximum_target_spread = spread
            widest_bin = {
                "bin_index": bin_index,
                "invariant_minimum": float(
                    edges[bin_index]
                ),
                "invariant_maximum": float(
                    edges[
                        bin_index + 1
                    ]
                ),
                "sample_count": len(
                    values
                ),
                "target_spread": spread,
            }

    return {
        "bin_count": bin_count,
        "populated_bin_count": (
            populated_bin_count
        ),
        "maximum_target_spread": (
            maximum_target_spread
        ),
        "widest_bin": widest_bin,
    }


def axis_anchored_affine_envelope(
    s4: np.ndarray,
    top_eigenvalue: np.ndarray,
) -> dict:
    delta_s4 = (
        s4 - EXPECTED_AXIS_S4
    )

    delta_lambda = (
        top_eigenvalue
        - TARGET_TOP_EIGENVALUE
    )

    valid = (
        delta_s4 < -1e-12
    )

    ratios = (
        delta_lambda[valid]
        / delta_s4[valid]
    )

    candidate_slope = float(
        np.min(ratios)
    )

    predicted_upper = (
        TARGET_TOP_EIGENVALUE
        + candidate_slope
        * delta_s4
    )

    slack = (
        predicted_upper
        - top_eigenvalue
    )

    return {
        "form": (
            "lambda_max <= 1/9 "
            "+ beta*(S4-115/72)"
        ),
        "candidate_beta": (
            candidate_slope
        ),
        "candidate_beta_rational": (
            rational_label(
                candidate_slope
            )
        ),
        "minimum_sample_slack": float(
            np.min(slack)
        ),
        "maximum_sample_slack": float(
            np.max(slack)
        ),
        "active_sample_count_at_1e_10": int(
            np.count_nonzero(
                slack < 1e-10
            )
        ),
        "ratio_minimum": float(
            np.min(ratios)
        ),
        "ratio_median": float(
            np.median(ratios)
        ),
        "ratio_maximum": float(
            np.max(ratios)
        ),
    }


def negative_s4_objective(
    raw_direction: np.ndarray,
    axis_lines: np.ndarray,
) -> float:
    direction = normalized(
        raw_direction
    )

    overlaps = (
        axis_lines @ direction
    )

    return -float(
        np.sum(
            overlaps**4
        )
    )


def optimize_s4_maxima(
    axis_lines: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    initial_directions = [
        axis.copy()
        for axis in axis_lines
    ]

    while (
        len(initial_directions)
        < OPTIMIZATION_START_COUNT
    ):
        initial_directions.append(
            normalized(
                rng.normal(size=4)
            )
        )

    rows = []
    terminal_points = []

    started_at = time.monotonic()

    for start_id, initial in enumerate(
        initial_directions
    ):
        result = minimize(
            negative_s4_objective,
            initial,
            args=(axis_lines,),
            method="BFGS",
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": 1e-12,
            },
        )

        direction = canonical_sign(
            result.x
        )

        s4_value = -negative_s4_objective(
            direction,
            axis_lines,
        )

        distances = (
            projective_distance_to_register(
                direction.reshape(1, 4),
                axis_lines,
            )
        )

        distance = float(
            distances[0]
        )

        terminal_points.append(
            direction
        )

        rows.append(
            {
                "start_id": start_id,
                "s4": s4_value,
                "axis_s4_gap": (
                    EXPECTED_AXIS_S4
                    - s4_value
                ),
                "projective_distance_to_axis_register": (
                    distance
                ),
                "success": bool(
                    result.success
                ),
                "status": int(
                    result.status
                ),
                "message": str(
                    result.message
                ),
                "iteration_count": int(
                    result.nit
                ),
                "evaluation_count": int(
                    result.nfev
                ),
                "f0": float(
                    direction[0]
                ),
                "f1": float(
                    direction[1]
                ),
                "f2": float(
                    direction[2]
                ),
                "f3": float(
                    direction[3]
                ),
            }
        )

        if (
            start_id == 0
            or (start_id + 1) % 32 == 0
            or start_id + 1
            == OPTIMIZATION_START_COUNT
        ):
            elapsed = (
                time.monotonic()
                - started_at
            )

            best_s4 = max(
                row["s4"]
                for row in rows
            )

            print(
                "\rs4_optimization:",
                f"{start_id + 1}/{OPTIMIZATION_START_COUNT}",
                "best_s4:",
                best_s4,
                "elapsed:",
                f"{elapsed:.1f}s",
                end="",
                flush=True,
            )

    print()

    return (
        rows,
        np.array(
            terminal_points,
            dtype=np.float64,
        ),
    )


def axis_moment_profile(
    axis_lines: np.ndarray,
) -> dict:
    moments = register_moments(
        axis_lines,
        axis_lines,
    )

    profile = {}

    for name, values in moments.items():
        profile[name] = {
            "minimum": float(
                np.min(values)
            ),
            "maximum": float(
                np.max(values)
            ),
            "maximum_residual": float(
                np.max(values)
                - np.min(values)
            ),
            "mean": float(
                np.mean(values)
            ),
            "rational": rational_label(
                float(
                    np.mean(values)
                )
            ),
            "values": values,
        }

    return profile


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OPTIMIZATION_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGRESSION_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hessian_receipt = json.loads(
        HESSIAN_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    identification_data = np.load(
        IDENTIFICATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    axis_lines = np.array(
        identification_data[
            "native_line_matrix"
        ],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis line shape: {axis_lines.shape}"
        )

    if slices.shape != (4, 6, 6):
        raise RuntimeError(
            f"unexpected slices shape: {slices.shape}"
        )

    frame_operator = (
        axis_lines.T
        @ axis_lines
    )

    frame_residual = max_abs(
        frame_operator
        - 2.5 * np.eye(4)
    )

    axis_profile = axis_moment_profile(
        axis_lines
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    random_directions = rng.normal(
        size=(
            RANDOM_SAMPLE_COUNT,
            4,
        )
    )

    random_directions /= np.linalg.norm(
        random_directions,
        axis=1,
        keepdims=True,
    )

    moments = register_moments(
        random_directions,
        axis_lines,
    )

    (
        operator_norms,
        singular_values,
    ) = batch_operator_data(
        slices,
        random_directions,
    )

    top_eigenvalues = (
        operator_norms**2
    )

    distances = (
        projective_distance_to_register(
            random_directions,
            axis_lines,
        )
    )

    s2_residual = max_abs(
        moments["s2"] - 2.5
    )

    correlation_matrix = np.corrcoef(
        np.column_stack(
            [
                top_eigenvalues,
                moments["s4"],
                moments["s6"],
                moments["s8"],
                distances,
            ]
        ),
        rowvar=False,
    )

    regression_rows = []
    regression_predictions = {}

    designs = {
        "s4_affine": (
            np.column_stack(
                [
                    np.ones(
                        RANDOM_SAMPLE_COUNT
                    ),
                    moments["s4"],
                ]
            ),
            [
                "1",
                "S4",
            ],
        ),
        "s4_quadratic": (
            np.column_stack(
                [
                    np.ones(
                        RANDOM_SAMPLE_COUNT
                    ),
                    moments["s4"],
                    moments["s4"] ** 2,
                ]
            ),
            [
                "1",
                "S4",
                "S4^2",
            ],
        ),
        "s4_s6_affine": (
            np.column_stack(
                [
                    np.ones(
                        RANDOM_SAMPLE_COUNT
                    ),
                    moments["s4"],
                    moments["s6"],
                ]
            ),
            [
                "1",
                "S4",
                "S6",
            ],
        ),
        "s4_s6_s8_affine": (
            np.column_stack(
                [
                    np.ones(
                        RANDOM_SAMPLE_COUNT
                    ),
                    moments["s4"],
                    moments["s6"],
                    moments["s8"],
                ]
            ),
            [
                "1",
                "S4",
                "S6",
                "S8",
            ],
        ),
    }

    for name, (
        design,
        feature_names,
    ) in designs.items():
        row, prediction = (
            regression_summary(
                name,
                design,
                top_eigenvalues,
                feature_names,
            )
        )

        regression_rows.append(row)
        regression_predictions[
            name
        ] = prediction

    s4_binned_spread = binned_spread(
        moments["s4"],
        top_eigenvalues,
    )

    affine_envelope = (
        axis_anchored_affine_envelope(
            moments["s4"],
            top_eigenvalues,
        )
    )

    (
        optimization_rows,
        optimization_terminal_points,
    ) = optimize_s4_maxima(
        axis_lines
    )

    maximum_optimized_s4 = max(
        row["s4"]
        for row in optimization_rows
    )

    maximum_optimization_axis_distance = max(
        row[
            "projective_distance_to_axis_register"
        ]
        for row in optimization_rows
    )

    near_axis_terminal_count = sum(
        row[
            "projective_distance_to_axis_register"
        ]
        < OPTIMIZATION_TOLERANCE
        for row in optimization_rows
    )

    all_optimizations_reach_axis_s4 = all(
        abs(
            row["s4"]
            - EXPECTED_AXIS_S4
        )
        < OPTIMIZATION_TOLERANCE
        for row in optimization_rows
    )

    sample_maximum_s4 = float(
        np.max(
            moments["s4"]
        )
    )

    sample_maximum_operator_norm = float(
        np.max(
            operator_norms
        )
    )

    sample_maximum_top_eigenvalue = float(
        np.max(
            top_eigenvalues
        )
    )

    checks = {
        "input_031_theorem_pass": (
            hessian_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "axis_line_count_is_10": (
            len(axis_lines) == 10
        ),
        "register_is_tight_frame": (
            frame_residual
            < INVARIANT_TOLERANCE
        ),
        "random_s2_is_constant_five_halves": (
            s2_residual
            < INVARIANT_TOLERANCE
        ),
        "axis_s4_is_115_over_72": (
            abs(
                axis_profile["s4"][
                    "mean"
                ]
                - EXPECTED_AXIS_S4
            )
            < INVARIANT_TOLERANCE
        ),
        "all_s4_optimizations_reach_axis_value": (
            all_optimizations_reach_axis_s4
        ),
        "all_s4_optimization_endpoints_reach_axis_register": (
            near_axis_terminal_count
            == OPTIMIZATION_START_COUNT
        ),
        "random_sample_does_not_exceed_axis_s4": (
            sample_maximum_s4
            <= EXPECTED_AXIS_S4
            + OPTIMIZATION_TOLERANCE
        ),
        "random_sample_does_not_exceed_one_third": (
            sample_maximum_operator_norm
            <= TARGET_OPERATOR_NORM
            + OPTIMIZATION_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    s4_alone_determines_top_eigenvalue = (
        s4_binned_spread[
            "maximum_target_spread"
        ]
        < 1e-8
    )

    theorem_pass = False

    verdict = (
        "native_g60_cross_flux_register_s4_axis_maximum_candidate_identified"
        if audit_pass
        else "native_g60_cross_flux_register_invariant_scan_failed"
    )

    sample_rows = []

    saved_indices = np.linspace(
        0,
        RANDOM_SAMPLE_COUNT - 1,
        SAVED_SAMPLE_COUNT,
        dtype=np.int64,
    )

    for sample_id, index in enumerate(
        saved_indices
    ):
        direction = (
            random_directions[index]
        )

        sample_rows.append(
            {
                "sample_id": sample_id,
                "source_index": int(index),
                "f0": float(
                    direction[0]
                ),
                "f1": float(
                    direction[1]
                ),
                "f2": float(
                    direction[2]
                ),
                "f3": float(
                    direction[3]
                ),
                "operator_norm": float(
                    operator_norms[index]
                ),
                "top_eigenvalue": float(
                    top_eigenvalues[index]
                ),
                "s2": float(
                    moments["s2"][index]
                ),
                "s4": float(
                    moments["s4"][index]
                ),
                "s6": float(
                    moments["s6"][index]
                ),
                "s8": float(
                    moments["s8"][index]
                ),
                "projective_distance_to_axis_register": float(
                    distances[index]
                ),
            }
        )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_register_invariant_scan_032"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "register": {
            "line_count": 10,
            "ambient_dimension": 4,
            "frame_law": (
                "L10^T L10 = (5/2) I4"
            ),
            "frame_residual": (
                frame_residual
            ),
            "quadratic_moment": (
                "S2(f)=5/2 for unit f"
            ),
            "random_s2_maximum_residual": (
                s2_residual
            ),
        },
        "axis_moments": {
            name: {
                key: value
                for key, value in (
                    data.items()
                )
                if key != "values"
            }
            for name, data in (
                axis_profile.items()
            )
        },
        "axis_s4_formula": {
            "formula": (
                "1 + 3(2/3)^4 + 6(1/6)^4"
            ),
            "value": (
                EXPECTED_AXIS_S4
            ),
            "rational": "115/72",
        },
        "random_scan": {
            "sample_count": (
                RANDOM_SAMPLE_COUNT
            ),
            "maximum_s4": (
                sample_maximum_s4
            ),
            "maximum_operator_norm": (
                sample_maximum_operator_norm
            ),
            "maximum_top_eigenvalue": (
                sample_maximum_top_eigenvalue
            ),
            "minimum_axis_distance": float(
                np.min(distances)
            ),
            "correlation_labels": [
                "top_eigenvalue",
                "S4",
                "S6",
                "S8",
                "axis_distance",
            ],
            "correlation_matrix": (
                correlation_matrix
            ),
        },
        "s4_optimization": {
            "start_count": (
                OPTIMIZATION_START_COUNT
            ),
            "maximum_terminal_s4": (
                maximum_optimized_s4
            ),
            "maximum_axis_value_residual": max(
                abs(
                    row["s4"]
                    - EXPECTED_AXIS_S4
                )
                for row in (
                    optimization_rows
                )
            ),
            "near_axis_terminal_count": (
                near_axis_terminal_count
            ),
            "maximum_terminal_axis_distance": (
                maximum_optimization_axis_distance
            ),
            "all_starts_reach_axis_s4": (
                all_optimizations_reach_axis_s4
            ),
        },
        "top_eigenvalue_regression": {
            "models": (
                regression_rows
            ),
            "s4_binned_spread": (
                s4_binned_spread
            ),
            "s4_alone_determines_top_eigenvalue": (
                s4_alone_determines_top_eigenvalue
            ),
        },
        "sampled_axis_anchored_envelope": (
            affine_envelope
        ),
        "checks": checks,
        "earned_interpretation": {
            "s4_is_first_nonconstant_native_line_register_moment": (
                True
            ),
            "native_axes_are_s4_maximizer_candidates": (
                audit_pass
            ),
            "s4_maximum_globally_proved": (
                False
            ),
            "s4_alone_controls_operator_norm": (
                s4_alone_determines_top_eigenvalue
            ),
            "sampled_affine_upper_envelope_found": (
                audit_pass
            ),
            "global_one_third_bound_proved": (
                False
            ),
        },
        "boundary": {
            "native_register_invariant_scan_completed": (
                audit_pass
            ),
            "axis_s4_value_exactly_computed": (
                audit_pass
            ),
            "global_s4_maximum_proved": (
                False
            ),
            "sampled_affine_envelope_is_theorem": (
                False
            ),
            "global_operator_norm_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "sample_csv": str(
                SAMPLE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "optimization_csv": str(
                OPTIMIZATION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "regression_csv": str(
                REGRESSION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "scan_npz": str(
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

    with SAMPLE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                sample_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            sample_rows
        )

    with OPTIMIZATION_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                optimization_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            optimization_rows
        )

    with REGRESSION_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                regression_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            regression_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        slices=slices,
        random_directions=(
            random_directions
        ),
        operator_norms=operator_norms,
        singular_values=singular_values,
        top_eigenvalues=(
            top_eigenvalues
        ),
        s2=moments["s2"],
        s4=moments["s4"],
        s6=moments["s6"],
        s8=moments["s8"],
        axis_distances=distances,
        optimization_terminal_points=(
            optimization_terminal_points
        ),
        optimization_terminal_s4=np.array(
            [
                row["s4"]
                for row in (
                    optimization_rows
                )
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "frame_residual:",
        frame_residual,
    )
    print(
        "random_s2_residual:",
        s2_residual,
    )
    print(
        "axis_moment_profile:",
        {
            name: {
                "value": data["mean"],
                "rational": data["rational"],
            }
            for name, data in (
                axis_profile.items()
            )
        },
    )
    print(
        "random_maximum_s4:",
        sample_maximum_s4,
    )
    print(
        "random_maximum_operator_norm:",
        sample_maximum_operator_norm,
    )
    print(
        "s4_optimization_all_reach_axes:",
        all_optimizations_reach_axis_s4,
    )
    print(
        "s4_optimization_maximum_axis_distance:",
        maximum_optimization_axis_distance,
    )
    print(
        "correlation_matrix:",
        correlation_matrix.tolist(),
    )
    print(
        "regression_models:",
        [
            {
                "model": row["model"],
                "r_squared": row[
                    "r_squared"
                ],
                "max_residual": row[
                    "maximum_absolute_residual"
                ],
            }
            for row in (
                regression_rows
            )
        ],
    )
    print(
        "s4_binned_spread:",
        s4_binned_spread,
    )
    print(
        "axis_anchored_affine_envelope:",
        affine_envelope,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SAMPLE_CSV_OUT)
    print("wrote:", OPTIMIZATION_CSV_OUT)
    print("wrote:", REGRESSION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
