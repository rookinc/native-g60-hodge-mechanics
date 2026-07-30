from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize


ROOT = Path(__file__).resolve().parents[1]

AXIS_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_axis_extremal_certificate_030.json"
)

AXIS_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_axis_extremal_certificate_030.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_axis_local_hessian_031.json"
)

AXIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_axis_local_hessian_031.csv"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_axis_hessian_coefficients_031.csv"
)

DIRECTION_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_axis_hessian_directions_031.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_axis_local_hessian_031.npz"
)

HIGH_EIGENVALUE = 1.0 / 9.0
LOW_EIGENVALUE = 1.0 / 144.0
SPECTRAL_GAP = (
    HIGH_EIGENVALUE
    - LOW_EIGENVALUE
)

HIGH_SINGULAR_VALUE = 1.0 / 3.0

FIRST_ORDER_TOLERANCE = 2e-9
COEFFICIENT_TOLERANCE = 2e-9
NEGATIVITY_TOLERANCE = 1e-10
FINITE_DIFFERENCE_TOLERANCE = 2e-5

RANDOM_SEED = 46031
RANDOM_DIRECTION_COUNT = 4096
OPTIMIZATION_START_COUNT = 64

FINITE_DIFFERENCE_ANGLES = (
    1e-3,
    5e-4,
    2e-4,
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


def canonical_columns(
    basis: np.ndarray,
) -> np.ndarray:
    result = basis.copy()

    for column in range(
        result.shape[1]
    ):
        pivot = int(
            np.argmax(
                np.abs(
                    result[:, column]
                )
            )
        )

        if result[
            pivot,
            column,
        ] < 0.0:
            result[:, column] *= -1.0

    return result


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

    return canonical_columns(basis)


def top_and_low_bases(
    gram: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        gram
    )

    low_basis = eigenvectors[:, :4]
    high_basis = eigenvectors[:, 4:]

    return (
        canonical_columns(high_basis),
        canonical_columns(low_basis),
    )


def effective_second_order_operator(
    slices: np.ndarray,
    axis: np.ndarray,
    tangent: np.ndarray,
    high_basis: np.ndarray,
    low_basis: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    axis_matrix = pencil_matrix(
        slices,
        axis,
    )

    tangent_matrix = pencil_matrix(
        slices,
        tangent,
    )

    base_gram = (
        axis_matrix.T
        @ axis_matrix
    )

    first_order = (
        axis_matrix.T
        @ tangent_matrix
        + tangent_matrix.T
        @ axis_matrix
    )

    second_order_direct = (
        tangent_matrix.T
        @ tangent_matrix
        - base_gram
    )

    first_order_high = (
        high_basis.T
        @ first_order
        @ high_basis
    )

    high_to_low = (
        high_basis.T
        @ first_order
        @ low_basis
    )

    effective = (
        high_basis.T
        @ second_order_direct
        @ high_basis
        + (
            high_to_low
            @ high_to_low.T
        )
        / SPECTRAL_GAP
    )

    effective = 0.5 * (
        effective
        + effective.T
    )

    return (
        effective,
        first_order_high,
        first_order,
    )


def build_quadratic_coefficient_tensor(
    slices: np.ndarray,
    axis: np.ndarray,
    tangent_basis_matrix: np.ndarray,
    high_basis: np.ndarray,
    low_basis: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    diagonal_values = []

    first_order_high_values = []

    for index in range(3):
        effective, first_high, _ = (
            effective_second_order_operator(
                slices,
                axis,
                tangent_basis_matrix[:, index],
                high_basis,
                low_basis,
            )
        )

        diagonal_values.append(
            effective
        )

        first_order_high_values.append(
            first_high
        )

    coefficients = np.zeros(
        (3, 3, 2, 2),
        dtype=np.float64,
    )

    for index in range(3):
        coefficients[
            index,
            index,
        ] = diagonal_values[index]

    for first in range(3):
        for second in range(
            first + 1,
            3,
        ):
            tangent_sum = normalized(
                tangent_basis_matrix[:, first]
                + tangent_basis_matrix[:, second]
            )

            effective_sum, _, _ = (
                effective_second_order_operator(
                    slices,
                    axis,
                    tangent_sum,
                    high_basis,
                    low_basis,
                )
            )

            cross = (
                effective_sum
                - 0.5
                * diagonal_values[first]
                - 0.5
                * diagonal_values[second]
            )

            coefficients[
                first,
                second,
            ] = cross

            coefficients[
                second,
                first,
            ] = cross

    return (
        coefficients,
        np.array(
            first_order_high_values,
            dtype=np.float64,
        ),
    )


def evaluate_quadratic_operator(
    coefficients: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    result = np.einsum(
        "i,j,ijab->ab",
        coordinates,
        coordinates,
        coefficients,
    )

    return 0.5 * (
        result + result.T
    )


def curvature_from_effective(
    effective: np.ndarray,
) -> tuple[float, float, float]:
    eigenvalues = np.linalg.eigvalsh(
        effective
    )

    largest_effective = float(
        eigenvalues[-1]
    )

    smallest_effective = float(
        eigenvalues[0]
    )

    singular_drop_coefficient = (
        -largest_effective
        / (
            2.0
            * HIGH_SINGULAR_VALUE
        )
    )

    return (
        smallest_effective,
        largest_effective,
        singular_drop_coefficient,
    )


def geodesic_point(
    axis: np.ndarray,
    tangent: np.ndarray,
    angle: float,
) -> np.ndarray:
    return (
        np.cos(angle) * axis
        + np.sin(angle) * tangent
    )


def finite_difference_curvature(
    slices: np.ndarray,
    axis: np.ndarray,
    tangent: np.ndarray,
    angle: float,
) -> float:
    plus = operator_norm(
        slices,
        geodesic_point(
            axis,
            tangent,
            angle,
        ),
    )

    minus = operator_norm(
        slices,
        geodesic_point(
            axis,
            tangent,
            -angle,
        ),
    )

    centered = 0.5 * (
        plus + minus
    )

    return (
        HIGH_SINGULAR_VALUE
        - centered
    ) / angle**2


def spherical_coordinates(
    raw: np.ndarray,
) -> np.ndarray:
    return normalized(
        np.array(
            raw,
            dtype=np.float64,
        )
    )


def worst_curvature_objective(
    raw: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    coordinates = spherical_coordinates(
        raw
    )

    effective = evaluate_quadratic_operator(
        coefficients,
        coordinates,
    )

    _, _, curvature = (
        curvature_from_effective(
            effective
        )
    )

    return curvature


def optimize_minimum_curvature(
    coefficients: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    results = []

    initial_points = [
        np.eye(3)[index]
        for index in range(3)
    ]

    while len(
        initial_points
    ) < OPTIMIZATION_START_COUNT:
        initial_points.append(
            normalized(
                rng.normal(size=3)
            )
        )

    for initial in initial_points:
        result = minimize(
            worst_curvature_objective,
            initial,
            args=(coefficients,),
            method="Nelder-Mead",
            options={
                "maxiter": 4000,
                "xatol": 1e-12,
                "fatol": 1e-14,
            },
        )

        coordinates = spherical_coordinates(
            result.x
        )

        effective = evaluate_quadratic_operator(
            coefficients,
            coordinates,
        )

        (
            minimum_effective,
            maximum_effective,
            curvature,
        ) = curvature_from_effective(
            effective
        )

        results.append(
            {
                "coordinates": coordinates,
                "minimum_effective_eigenvalue": (
                    minimum_effective
                ),
                "maximum_effective_eigenvalue": (
                    maximum_effective
                ),
                "curvature": curvature,
                "success": bool(
                    result.success
                ),
                "iteration_count": int(
                    result.nit
                ),
            }
        )

    best = min(
        results,
        key=lambda row: row["curvature"],
    )

    return {
        "minimum_curvature_candidate": (
            best["curvature"]
        ),
        "worst_direction_coordinates": (
            best["coordinates"]
        ),
        "worst_effective_minimum_eigenvalue": (
            best[
                "minimum_effective_eigenvalue"
            ]
        ),
        "worst_effective_maximum_eigenvalue": (
            best[
                "maximum_effective_eigenvalue"
            ]
        ),
        "rounded_terminal_curvatures": sorted(
            {
                round(
                    row["curvature"],
                    12,
                )
                for row in results
            }
        ),
        "all_runs_successful": all(
            row["success"]
            for row in results
        ),
    }


def random_direction_audit(
    slices: np.ndarray,
    axis_id: int,
    axis: np.ndarray,
    tangent_basis_matrix: np.ndarray,
    coefficients: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    rows = []

    minimum_curvature = float("inf")
    maximum_effective_eigenvalue = -float(
        "inf"
    )
    maximum_finite_difference_residual = 0.0
    nonpositive_curvature_count = 0

    for direction_id in range(
        RANDOM_DIRECTION_COUNT
    ):
        coordinates = normalized(
            rng.normal(size=3)
        )

        tangent = normalized(
            tangent_basis_matrix
            @ coordinates
        )

        effective = evaluate_quadratic_operator(
            coefficients,
            coordinates,
        )

        (
            minimum_effective,
            maximum_effective,
            curvature,
        ) = curvature_from_effective(
            effective
        )

        finite_values = [
            finite_difference_curvature(
                slices,
                axis,
                tangent,
                angle,
            )
            for angle in (
                FINITE_DIFFERENCE_ANGLES
            )
        ]

        finite_curvature = float(
            finite_values[-1]
        )

        finite_difference_residual = abs(
            finite_curvature
            - curvature
        )

        minimum_curvature = min(
            minimum_curvature,
            curvature,
        )

        maximum_effective_eigenvalue = max(
            maximum_effective_eigenvalue,
            maximum_effective,
        )

        maximum_finite_difference_residual = max(
            maximum_finite_difference_residual,
            finite_difference_residual,
        )

        if curvature <= 0.0:
            nonpositive_curvature_count += 1

        if direction_id < 512:
            rows.append(
                {
                    "axis_id": axis_id,
                    "direction_id": (
                        direction_id
                    ),
                    "tangent_coordinate_0": float(
                        coordinates[0]
                    ),
                    "tangent_coordinate_1": float(
                        coordinates[1]
                    ),
                    "tangent_coordinate_2": float(
                        coordinates[2]
                    ),
                    "effective_minimum_eigenvalue": (
                        minimum_effective
                    ),
                    "effective_maximum_eigenvalue": (
                        maximum_effective
                    ),
                    "analytic_drop_coefficient": (
                        curvature
                    ),
                    "finite_difference_drop_coefficient": (
                        finite_curvature
                    ),
                    "finite_difference_residual": (
                        finite_difference_residual
                    ),
                }
            )

    summary = {
        "minimum_sampled_curvature": (
            minimum_curvature
        ),
        "maximum_sampled_effective_eigenvalue": (
            maximum_effective_eigenvalue
        ),
        "maximum_finite_difference_residual": (
            maximum_finite_difference_residual
        ),
        "nonpositive_curvature_count": (
            nonpositive_curvature_count
        ),
    }

    return rows, summary


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AXIS_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COEFFICIENT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIRECTION_CSV_OUT.parent.mkdir(
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

    data = np.load(
        AXIS_NPZ_PATH
    )

    axis_lines = np.array(
        data["axis_lines"],
        dtype=np.float64,
    )

    slices = np.array(
        data["slices"],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis line shape: {axis_lines.shape}"
        )

    if slices.shape != (4, 6, 6):
        raise RuntimeError(
            f"unexpected slice shape: {slices.shape}"
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    axis_rows = []
    coefficient_rows = []
    direction_rows = []

    tangent_bases = []
    high_bases = []
    low_bases = []
    coefficient_tensors = []

    for axis_id, axis in enumerate(
        axis_lines
    ):
        axis = normalized(axis)

        axis_matrix = pencil_matrix(
            slices,
            axis,
        )

        base_gram = (
            axis_matrix.T
            @ axis_matrix
        )

        high_basis, low_basis = (
            top_and_low_bases(
                base_gram
            )
        )

        tangent_basis_matrix = (
            tangent_basis(axis)
        )

        (
            coefficient_tensor,
            first_order_high_values,
        ) = build_quadratic_coefficient_tensor(
            slices,
            axis,
            tangent_basis_matrix,
            high_basis,
            low_basis,
        )

        first_order_high_residual = (
            max_abs(
                first_order_high_values
            )
        )

        random_rows, random_summary = (
            random_direction_audit(
                slices,
                axis_id,
                axis,
                tangent_basis_matrix,
                coefficient_tensor,
                rng,
            )
        )

        direction_rows.extend(
            random_rows
        )

        optimization = (
            optimize_minimum_curvature(
                coefficient_tensor,
                rng,
            )
        )

        worst_coordinates = np.array(
            optimization[
                "worst_direction_coordinates"
            ],
            dtype=np.float64,
        )

        worst_tangent = normalized(
            tangent_basis_matrix
            @ worst_coordinates
        )

        direct_effective, _, _ = (
            effective_second_order_operator(
                slices,
                axis,
                worst_tangent,
                high_basis,
                low_basis,
            )
        )

        reconstructed_effective = (
            evaluate_quadratic_operator(
                coefficient_tensor,
                worst_coordinates,
            )
        )

        reconstruction_residual = max_abs(
            direct_effective
            - reconstructed_effective
        )

        minimum_curvature = float(
            optimization[
                "minimum_curvature_candidate"
            ]
        )

        maximum_effective_eigenvalue = float(
            optimization[
                "worst_effective_maximum_eigenvalue"
            ]
        )

        axis_pass = (
            first_order_high_residual
            < FIRST_ORDER_TOLERANCE
            and reconstruction_residual
            < COEFFICIENT_TOLERANCE
            and minimum_curvature
            > NEGATIVITY_TOLERANCE
            and maximum_effective_eigenvalue
            < -NEGATIVITY_TOLERANCE
            and random_summary[
                "nonpositive_curvature_count"
            ]
            == 0
            and random_summary[
                "maximum_finite_difference_residual"
            ]
            < FINITE_DIFFERENCE_TOLERANCE
        )

        axis_rows.append(
            {
                "axis_id": axis_id,
                "first_order_high_block_max_abs": (
                    first_order_high_residual
                ),
                "quadratic_reconstruction_max_abs": (
                    reconstruction_residual
                ),
                "minimum_curvature_candidate": (
                    minimum_curvature
                ),
                "worst_effective_maximum_eigenvalue": (
                    maximum_effective_eigenvalue
                ),
                "worst_effective_minimum_eigenvalue": (
                    optimization[
                        "worst_effective_minimum_eigenvalue"
                    ]
                ),
                "minimum_sampled_curvature": (
                    random_summary[
                        "minimum_sampled_curvature"
                    ]
                ),
                "maximum_finite_difference_residual": (
                    random_summary[
                        "maximum_finite_difference_residual"
                    ]
                ),
                "nonpositive_sampled_curvature_count": (
                    random_summary[
                        "nonpositive_curvature_count"
                    ]
                ),
                "rounded_terminal_curvatures": json.dumps(
                    optimization[
                        "rounded_terminal_curvatures"
                    ]
                ),
                "axis_pass": axis_pass,
            }
        )

        for first, second in product(
            range(3),
            repeat=2,
        ):
            matrix = (
                coefficient_tensor[
                    first,
                    second,
                ]
            )

            coefficient_rows.append(
                {
                    "axis_id": axis_id,
                    "tangent_first": first,
                    "tangent_second": second,
                    "matrix_00": float(
                        matrix[0, 0]
                    ),
                    "matrix_01": float(
                        matrix[0, 1]
                    ),
                    "matrix_10": float(
                        matrix[1, 0]
                    ),
                    "matrix_11": float(
                        matrix[1, 1]
                    ),
                    "frobenius_norm": float(
                        np.linalg.norm(
                            matrix,
                            ord="fro",
                        )
                    ),
                }
            )

        tangent_bases.append(
            tangent_basis_matrix
        )
        high_bases.append(
            high_basis
        )
        low_bases.append(
            low_basis
        )
        coefficient_tensors.append(
            coefficient_tensor
        )

        print(
            "axis:",
            axis_id + 1,
            "/10",
            "pass:",
            axis_pass,
            "min_curvature:",
            minimum_curvature,
            "max_effective:",
            maximum_effective_eigenvalue,
            "fd_residual:",
            random_summary[
                "maximum_finite_difference_residual"
            ],
            flush=True,
        )

    all_axes_pass = all(
        row["axis_pass"]
        for row in axis_rows
    )

    global_minimum_curvature = min(
        row[
            "minimum_curvature_candidate"
        ]
        for row in axis_rows
    )

    global_maximum_effective_eigenvalue = max(
        row[
            "worst_effective_maximum_eigenvalue"
        ]
        for row in axis_rows
    )

    global_first_order_residual = max(
        row[
            "first_order_high_block_max_abs"
        ]
        for row in axis_rows
    )

    global_reconstruction_residual = max(
        row[
            "quadratic_reconstruction_max_abs"
        ]
        for row in axis_rows
    )

    global_finite_difference_residual = max(
        row[
            "maximum_finite_difference_residual"
        ]
        for row in axis_rows
    )

    checks = {
        "input_030_theorem_pass": (
            axis_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "axis_count_is_10": (
            len(axis_rows) == 10
        ),
        "first_order_top_blocks_vanish": (
            global_first_order_residual
            < FIRST_ORDER_TOLERANCE
        ),
        "quadratic_coefficient_reconstruction_resolved": (
            global_reconstruction_residual
            < COEFFICIENT_TOLERANCE
        ),
        "all_effective_second_order_operators_negative": (
            global_maximum_effective_eigenvalue
            < -NEGATIVITY_TOLERANCE
        ),
        "all_singular_drop_curvatures_positive": (
            global_minimum_curvature
            > NEGATIVITY_TOLERANCE
        ),
        "finite_difference_confirmation_resolved": (
            global_finite_difference_residual
            < FINITE_DIFFERENCE_TOLERANCE
        ),
        "all_axis_audits_pass": (
            all_axes_pass
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_ten_axes_are_analytic_strict_local_maxima"
        if theorem_pass
        else "native_g60_cross_flux_axis_local_hessian_not_resolved"
    )

    theorem_statement = (
        "For each native cross-flux axis q, the first-order "
        "perturbation vanishes on the two-dimensional top singular "
        "subspace. The degenerate second-order effective operator "
        "H_q(t) is negative definite for every tested and globally "
        "optimized unit tangent direction t. Therefore the operator "
        "norm has positive quadratic drop away from each axis."
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_axis_local_hessian_031"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "perturbation_model": {
            "geodesic": (
                "f(theta)=cos(theta)q+sin(theta)t"
            ),
            "top_eigenvalue": "1/9",
            "lower_eigenvalue": "1/144",
            "spectral_gap": "5/48",
            "effective_second_order_operator": (
                "P(A(t)^T A(t)-B_q)P "
                "+ P B1(t)(I-P)B1(t)P/(5/48)"
            ),
            "singular_drop_coefficient": (
                "c_q(t)=-(3/2)lambda_max(H_q(t))"
            ),
        },
        "theorem": {
            "statement": theorem_statement,
            "axis_count": 10,
            "global_minimum_drop_curvature_candidate": (
                global_minimum_curvature
            ),
            "global_maximum_effective_eigenvalue": (
                global_maximum_effective_eigenvalue
            ),
        },
        "checks": checks,
        "axis_rows": axis_rows,
        "global_residuals": {
            "first_order_top_block": (
                global_first_order_residual
            ),
            "quadratic_reconstruction": (
                global_reconstruction_residual
            ),
            "finite_difference_confirmation": (
                global_finite_difference_residual
            ),
        },
        "boundary": {
            "degenerate_second_order_reduction_constructed": (
                audit_pass
            ),
            "all_ten_axes_strict_local_maxima": (
                theorem_pass
            ),
            "global_operator_norm_upper_bound_proved": (
                False
            ),
            "complete_global_equality_locus_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "axis_csv": str(
                AXIS_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "coefficient_csv": str(
                COEFFICIENT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "direction_csv": str(
                DIRECTION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "hessian_npz": str(
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

    with COEFFICIENT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                coefficient_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            coefficient_rows
        )

    with DIRECTION_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                direction_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            direction_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        axis_lines=axis_lines,
        slices=slices,
        tangent_bases=np.array(
            tangent_bases,
            dtype=np.float64,
        ),
        high_bases=np.array(
            high_bases,
            dtype=np.float64,
        ),
        low_bases=np.array(
            low_bases,
            dtype=np.float64,
        ),
        effective_quadratic_coefficients=np.array(
            coefficient_tensors,
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "global_minimum_drop_curvature:",
        global_minimum_curvature,
    )
    print(
        "global_maximum_effective_eigenvalue:",
        global_maximum_effective_eigenvalue,
    )
    print(
        "global_first_order_residual:",
        global_first_order_residual,
    )
    print(
        "global_quadratic_reconstruction_residual:",
        global_reconstruction_residual,
    )
    print(
        "global_finite_difference_residual:",
        global_finite_difference_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", AXIS_CSV_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", DIRECTION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
