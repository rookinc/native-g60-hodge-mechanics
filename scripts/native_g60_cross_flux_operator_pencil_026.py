from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]

CERTIFICATE_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_tensor_norm_certificate_025.json"
)

OBSTRUCTION_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_coercivity_obstruction_024.json"
)

OBSTRUCTION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_coercivity_obstruction_024.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_operator_pencil_026.json"
)

PAIR_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_operator_pencil_pairs_026.csv"
)

START_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_operator_pencil_starts_026.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_operator_pencil_probes_026.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

RANDOM_SEED = 46026
RANDOM_PROBE_COUNT = 20000
OPTIMIZATION_START_COUNT = 256
MAX_ITERATIONS = 2000

TARGET_NORM = 1.0 / 3.0
TOLERANCE = 1e-10
OPTIMIZATION_TOLERANCE = 2e-9


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
    result = vector.copy()

    pivot = int(
        np.argmax(
            np.abs(result)
        )
    )

    if result[pivot] < 0.0:
        result *= -1.0

    return result


def pencil_matrix(
    slices: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "r,rab->ab",
        f,
        slices,
    )


def pencil_singular_data(
    slices: np.ndarray,
    f: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    matrix = pencil_matrix(
        slices,
        f,
    )

    left, singular_values, right_t = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    return (
        float(singular_values[0]),
        left[:, 0],
        right_t[0],
        singular_values,
    )


def sphere_objective_and_gradient(
    raw_f: np.ndarray,
    slices: np.ndarray,
) -> tuple[float, np.ndarray]:
    raw_norm = float(
        np.linalg.norm(raw_f)
    )

    if raw_norm == 0.0:
        return 1e6, np.zeros_like(raw_f)

    f = raw_f / raw_norm

    sigma, left, right, _ = (
        pencil_singular_data(
            slices,
            f,
        )
    )

    gradient_on_f = np.array(
        [
            float(
                left
                @ slices[index]
                @ right
            )
            for index in range(4)
        ],
        dtype=np.float64,
    )

    tangent_projector = (
        np.eye(4)
        - np.outer(f, f)
    )

    gradient_on_raw = (
        tangent_projector
        @ gradient_on_f
        / raw_norm
    )

    return (
        -sigma,
        -gradient_on_raw,
    )


def optimize_pencil_direction(
    slices: np.ndarray,
    initial_f: np.ndarray,
) -> dict:
    result = minimize(
        sphere_objective_and_gradient,
        initial_f,
        args=(slices,),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": MAX_ITERATIONS,
            "maxls": 100,
            "ftol": 1e-15,
            "gtol": 1e-12,
            "maxcor": 30,
        },
    )

    f = canonical_sign(
        normalized(
            np.array(
                result.x,
                dtype=np.float64,
            )
        )
    )

    sigma, left, right, singular_values = (
        pencil_singular_data(
            slices,
            f,
        )
    )

    gradient = np.array(
        [
            float(
                left
                @ slices[index]
                @ right
            )
            for index in range(4)
        ],
        dtype=np.float64,
    )

    tangent_residual = float(
        np.linalg.norm(
            gradient
            - np.dot(
                gradient,
                f,
            )
            * f
        )
    )

    return {
        "f": f,
        "sigma": sigma,
        "left": canonical_sign(left),
        "right": canonical_sign(right),
        "singular_values": singular_values,
        "tangent_stationarity_residual": (
            tangent_residual
        ),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iteration_count": int(result.nit),
        "evaluation_count": int(result.nfev),
    }


def build_pair_algebra(
    slices: np.ndarray,
) -> tuple[
    list[dict],
    np.ndarray,
    np.ndarray,
]:
    pair_rows = []
    symmetric_products = []
    skew_products = []

    for first in range(4):
        for second in range(first, 4):
            left_product = (
                slices[first].T
                @ slices[second]
            )

            reverse_product = (
                slices[second].T
                @ slices[first]
            )

            symmetric_product = (
                0.5
                * (
                    left_product
                    + reverse_product
                )
            )

            skew_product = (
                0.5
                * (
                    left_product
                    - reverse_product
                )
            )

            symmetric_products.append(
                symmetric_product
            )

            skew_products.append(
                skew_product
            )

            scalar_part = float(
                np.trace(
                    symmetric_product
                )
                / 6.0
            )

            scalar_residual = (
                symmetric_product
                - scalar_part
                * np.eye(6)
            )

            pair_rows.append(
                {
                    "first_slice": first,
                    "second_slice": second,
                    "same_slice": (
                        first == second
                    ),
                    "symmetric_product_trace": float(
                        np.trace(
                            symmetric_product
                        )
                    ),
                    "symmetric_product_frobenius_norm": float(
                        np.linalg.norm(
                            symmetric_product,
                            ord="fro",
                        )
                    ),
                    "symmetric_product_rank": int(
                        np.linalg.matrix_rank(
                            symmetric_product,
                            tol=1e-10,
                        )
                    ),
                    "scalar_part": scalar_part,
                    "scalar_identity_residual_frobenius": float(
                        np.linalg.norm(
                            scalar_residual,
                            ord="fro",
                        )
                    ),
                    "skew_product_frobenius_norm": float(
                        np.linalg.norm(
                            skew_product,
                            ord="fro",
                        )
                    ),
                    "skew_product_rank": int(
                        np.linalg.matrix_rank(
                            skew_product,
                            tol=1e-10,
                        )
                    ),
                }
            )

    return (
        pair_rows,
        np.array(
            symmetric_products,
            dtype=np.float64,
        ),
        np.array(
            skew_products,
            dtype=np.float64,
        ),
    )


def random_sphere_probe(
    slices: np.ndarray,
) -> tuple[
    list[dict],
    dict,
    np.ndarray,
]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    probe_rows = []
    best_sigma = -float("inf")
    best_f = None

    sigma_values = np.empty(
        RANDOM_PROBE_COUNT,
        dtype=np.float64,
    )

    for probe_id in range(
        RANDOM_PROBE_COUNT
    ):
        f = normalized(
            rng.normal(size=4)
        )

        sigma, _, _, singular_values = (
            pencil_singular_data(
                slices,
                f,
            )
        )

        sigma_values[probe_id] = sigma

        if sigma > best_sigma:
            best_sigma = sigma
            best_f = f.copy()

        if probe_id < 4096:
            probe_rows.append(
                {
                    "probe_id": probe_id,
                    "sigma_max": sigma,
                    "sigma_second": float(
                        singular_values[1]
                    ),
                    "sigma_minimum": float(
                        singular_values[-1]
                    ),
                    "target_gap": (
                        TARGET_NORM
                        - sigma
                    ),
                    "f0": float(f[0]),
                    "f1": float(f[1]),
                    "f2": float(f[2]),
                    "f3": float(f[3]),
                }
            )

    summary = {
        "probe_count": (
            RANDOM_PROBE_COUNT
        ),
        "minimum_sigma": float(
            np.min(sigma_values)
        ),
        "mean_sigma": float(
            np.mean(sigma_values)
        ),
        "maximum_sigma": float(
            np.max(sigma_values)
        ),
        "target_minus_maximum": float(
            TARGET_NORM
            - np.max(sigma_values)
        ),
        "quantiles": {
            "q01": float(
                np.quantile(
                    sigma_values,
                    0.01,
                )
            ),
            "q25": float(
                np.quantile(
                    sigma_values,
                    0.25,
                )
            ),
            "q50": float(
                np.quantile(
                    sigma_values,
                    0.50,
                )
            ),
            "q75": float(
                np.quantile(
                    sigma_values,
                    0.75,
                )
            ),
            "q99": float(
                np.quantile(
                    sigma_values,
                    0.99,
                )
            ),
        },
    }

    return (
        probe_rows,
        summary,
        best_f,
    )


def multistart_optimize(
    slices: np.ndarray,
    seed_direction: np.ndarray,
) -> tuple[
    list[dict],
    list[dict],
]:
    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    initial_directions = [
        normalized(
            seed_direction
        )
    ]

    for basis_index in range(4):
        basis = np.zeros(
            4,
            dtype=np.float64,
        )
        basis[basis_index] = 1.0
        initial_directions.append(basis)

    while (
        len(initial_directions)
        < OPTIMIZATION_START_COUNT
    ):
        initial_directions.append(
            normalized(
                rng.normal(size=4)
            )
        )

    start_rows = []
    results = []

    started_at = time.monotonic()

    for start_id, initial_f in enumerate(
        initial_directions
    ):
        result = optimize_pencil_direction(
            slices,
            initial_f,
        )

        results.append(result)

        start_rows.append(
            {
                "start_id": start_id,
                "sigma": result["sigma"],
                "target_gap": (
                    TARGET_NORM
                    - result["sigma"]
                ),
                "success": result["success"],
                "iteration_count": (
                    result["iteration_count"]
                ),
                "evaluation_count": (
                    result["evaluation_count"]
                ),
                "tangent_stationarity_residual": (
                    result[
                        "tangent_stationarity_residual"
                    ]
                ),
                "f0": float(
                    result["f"][0]
                ),
                "f1": float(
                    result["f"][1]
                ),
                "f2": float(
                    result["f"][2]
                ),
                "f3": float(
                    result["f"][3]
                ),
            }
        )

        if (
            start_id == 0
            or (start_id + 1) % 16 == 0
            or start_id + 1
            == OPTIMIZATION_START_COUNT
        ):
            best_so_far = max(
                item["sigma"]
                for item in results
            )

            elapsed = (
                time.monotonic()
                - started_at
            )

            print(
                "\roperator_pencil_start:",
                f"{start_id + 1}/{OPTIMIZATION_START_COUNT}",
                "best_sigma:",
                best_so_far,
                "elapsed:",
                f"{elapsed:.1f}s",
                end="",
                flush=True,
            )

    print()

    return start_rows, results


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PAIR_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    START_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROBE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    obstruction_receipt = json.loads(
        OBSTRUCTION_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    certificate_receipt = json.loads(
        CERTIFICATE_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        OBSTRUCTION_NPZ_PATH
    )

    tensor = np.array(
        data["tensor"],
        dtype=np.float64,
    )

    witness_f = normalized(
        np.array(
            data["witness_f"],
            dtype=np.float64,
        )
    )

    witness_u = normalized(
        np.array(
            data["witness_u"],
            dtype=np.float64,
        )
    )

    witness_v = normalized(
        np.array(
            data["witness_v"],
            dtype=np.float64,
        )
    )

    witness_gain = float(
        np.array(
            data["witness_gain"],
            dtype=np.float64,
        )[0]
    )

    if tensor.shape != (4, 6, 6):
        raise RuntimeError(
            f"unexpected tensor shape: {tensor.shape}"
        )

    slices = tensor.copy()

    slice_frobenius_gram = np.einsum(
        "rab,sab->rs",
        slices,
        slices,
    )

    slice_frobenius_eigenvalues = (
        np.linalg.eigvalsh(
            slice_frobenius_gram
        )
    )

    pair_rows, symmetric_products, skew_products = (
        build_pair_algebra(
            slices
        )
    )

    symmetric_product_span_rank = int(
        np.linalg.matrix_rank(
            symmetric_products.reshape(
                len(symmetric_products),
                -1,
            ),
            tol=1e-10,
        )
    )

    skew_product_span_rank = int(
        np.linalg.matrix_rank(
            skew_products.reshape(
                len(skew_products),
                -1,
            ),
            tol=1e-10,
        )
    )

    witness_sigma, witness_left, witness_right, witness_singular_values = (
        pencil_singular_data(
            slices,
            witness_f,
        )
    )

    witness_left_alignment = max(
        abs(
            float(
                np.dot(
                    witness_left,
                    witness_u,
                )
            )
        ),
        abs(
            float(
                np.dot(
                    witness_left,
                    witness_v,
                )
            )
        ),
    )

    witness_right_alignment = max(
        abs(
            float(
                np.dot(
                    witness_right,
                    witness_u,
                )
            )
        ),
        abs(
            float(
                np.dot(
                    witness_right,
                    witness_v,
                )
            )
        ),
    )

    (
        probe_rows,
        probe_summary,
        best_random_f,
    ) = random_sphere_probe(
        slices
    )

    start_rows, optimization_results = (
        multistart_optimize(
            slices,
            best_random_f,
        )
    )

    best_result = max(
        optimization_results,
        key=lambda item: item["sigma"],
    )

    optimized_sigma = float(
        best_result["sigma"]
    )

    optimized_f = np.array(
        best_result["f"],
        dtype=np.float64,
    )

    optimized_left = np.array(
        best_result["left"],
        dtype=np.float64,
    )

    optimized_right = np.array(
        best_result["right"],
        dtype=np.float64,
    )

    optimized_singular_values = np.array(
        best_result["singular_values"],
        dtype=np.float64,
    )

    rounded_terminal_sigmas = sorted(
        {
            round(
                result["sigma"],
                12,
            )
            for result in optimization_results
        }
    )

    near_target_count = sum(
        abs(
            result["sigma"]
            - TARGET_NORM
        )
        < 1e-8
        for result in optimization_results
    )

    maximum_stationarity_residual_near_target = max(
        result[
            "tangent_stationarity_residual"
        ]
        for result in optimization_results
        if abs(
            result["sigma"]
            - TARGET_NORM
        )
        < 1e-8
    )

    same_slice_scalar_residuals = [
        row[
            "scalar_identity_residual_frobenius"
        ]
        for row in pair_rows
        if row["same_slice"]
    ]

    cross_slice_scalar_residuals = [
        row[
            "scalar_identity_residual_frobenius"
        ]
        for row in pair_rows
        if not row["same_slice"]
    ]

    simple_clifford_identity_holds = (
        max(
            same_slice_scalar_residuals
            + cross_slice_scalar_residuals
        )
        < TOLERANCE
    )

    checks = {
        "input_024_theorem_pass": (
            obstruction_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "input_025_search_completed": (
            certificate_receipt.get(
                "artifact_id"
            )
            == (
                "native_g60_cross_flux_tensor_norm_certificate_025"
            )
        ),
        "tensor_shape_is_4_by_6_by_6": (
            tensor.shape == (4, 6, 6)
        ),
        "witness_pencil_recovers_obstruction_gain": (
            abs(
                witness_sigma
                - witness_gain
            )
            < 1e-9
        ),
        "witness_pencil_reaches_one_third": (
            abs(
                witness_sigma
                - TARGET_NORM
            )
            < 1e-9
        ),
        "random_probe_does_not_exceed_one_third": (
            probe_summary[
                "maximum_sigma"
            ]
            <= TARGET_NORM
            + OPTIMIZATION_TOLERANCE
        ),
        "optimized_value_reaches_one_third": (
            abs(
                optimized_sigma
                - TARGET_NORM
            )
            < OPTIMIZATION_TOLERANCE
        ),
        "optimized_value_does_not_exceed_one_third": (
            optimized_sigma
            <= TARGET_NORM
            + OPTIMIZATION_TOLERANCE
        ),
        "near_target_stationarity_resolved": (
            maximum_stationarity_residual_near_target
            < 1e-7
        ),
        "all_optimization_starts_completed": (
            len(start_rows)
            == OPTIMIZATION_START_COUNT
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    verdict = (
        "native_g60_cross_flux_operator_pencil_supports_one_third_norm_candidate"
        if audit_pass
        else "native_g60_cross_flux_operator_pencil_audit_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_operator_pencil_026"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "operator_pencil": {
            "definition": (
                "A(f) = sum_r f_r A_r"
            ),
            "tensor_norm_identity": (
                "sigma_star = max_{||f||=1} ||A(f)||_op"
            ),
            "slice_count": 4,
            "slice_shape": [6, 6],
            "target_norm": (
                TARGET_NORM
            ),
        },
        "slice_algebra": {
            "frobenius_gram": (
                slice_frobenius_gram
            ),
            "frobenius_gram_eigenvalues": (
                slice_frobenius_eigenvalues
            ),
            "symmetric_product_span_rank": (
                symmetric_product_span_rank
            ),
            "skew_product_span_rank": (
                skew_product_span_rank
            ),
            "simple_clifford_identity_holds": (
                simple_clifford_identity_holds
            ),
            "maximum_same_slice_scalar_identity_residual": max(
                same_slice_scalar_residuals
            ),
            "maximum_cross_slice_scalar_identity_residual": max(
                cross_slice_scalar_residuals
            ),
        },
        "known_witness": {
            "gain_from_024": (
                witness_gain
            ),
            "pencil_operator_norm": (
                witness_sigma
            ),
            "singular_values": (
                witness_singular_values
            ),
            "left_alignment_with_known_inputs": (
                witness_left_alignment
            ),
            "right_alignment_with_known_inputs": (
                witness_right_alignment
            ),
            "f": witness_f,
        },
        "random_probe": (
            probe_summary
        ),
        "optimization": {
            "start_count": (
                OPTIMIZATION_START_COUNT
            ),
            "near_target_count": (
                near_target_count
            ),
            "rounded_terminal_sigmas": (
                rounded_terminal_sigmas
            ),
            "best_sigma": (
                optimized_sigma
            ),
            "best_target_gap": (
                TARGET_NORM
                - optimized_sigma
            ),
            "best_f": (
                optimized_f
            ),
            "best_left_singular_vector": (
                optimized_left
            ),
            "best_right_singular_vector": (
                optimized_right
            ),
            "best_singular_values": (
                optimized_singular_values
            ),
            "best_tangent_stationarity_residual": (
                best_result[
                    "tangent_stationarity_residual"
                ]
            ),
            "maximum_near_target_stationarity_residual": (
                maximum_stationarity_residual_near_target
            ),
        },
        "checks": checks,
        "earned_interpretation": {
            "tensor_norm_reduced_to_four_parameter_operator_pencil": (
                audit_pass
            ),
            "one_third_candidate_recovered_from_pencil": (
                audit_pass
            ),
            "one_third_upper_bound_numerically_supported": (
                audit_pass
            ),
            "simple_clifford_identity_found": (
                simple_clifford_identity_holds
            ),
            "sharp_tensor_norm_proved": False,
        },
        "boundary": {
            "operator_pencil_constructed": (
                audit_pass
            ),
            "multistart_sphere_search_completed": (
                audit_pass
            ),
            "counterexample_above_one_third_found": (
                optimized_sigma
                > TARGET_NORM
                + OPTIMIZATION_TOLERANCE
            ),
            "global_operator_norm_bound_proved": (
                False
            ),
            "sharp_tensor_norm_proved": False,
            "exact_polynomial_identity_derived": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "pair_csv": str(
                PAIR_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "start_csv": str(
                START_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "operator_pencil_npz": str(
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
        writer.writerows(pair_rows)

    with START_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                start_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(start_rows)

    with PROBE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                probe_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(probe_rows)

    np.savez_compressed(
        NPZ_OUT,
        slices=slices,
        slice_frobenius_gram=(
            slice_frobenius_gram
        ),
        symmetric_products=(
            symmetric_products
        ),
        skew_products=(
            skew_products
        ),
        witness_f=witness_f,
        witness_singular_values=(
            witness_singular_values
        ),
        optimized_f=optimized_f,
        optimized_left=optimized_left,
        optimized_right=optimized_right,
        optimized_singular_values=(
            optimized_singular_values
        ),
        target_norm=np.array(
            [TARGET_NORM],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "slice_frobenius_gram:",
        slice_frobenius_gram.tolist(),
    )
    print(
        "slice_frobenius_gram_eigenvalues:",
        slice_frobenius_eigenvalues.tolist(),
    )
    print(
        "simple_clifford_identity_holds:",
        simple_clifford_identity_holds,
    )
    print(
        "symmetric_product_span_rank:",
        symmetric_product_span_rank,
    )
    print(
        "skew_product_span_rank:",
        skew_product_span_rank,
    )
    print(
        "witness_pencil_sigma:",
        witness_sigma,
    )
    print(
        "random_probe_maximum_sigma:",
        probe_summary[
            "maximum_sigma"
        ],
    )
    print(
        "optimized_sigma:",
        optimized_sigma,
    )
    print(
        "optimized_target_gap:",
        TARGET_NORM
        - optimized_sigma,
    )
    print(
        "near_target_count:",
        near_target_count,
        "/",
        OPTIMIZATION_START_COUNT,
    )
    print(
        "rounded_terminal_sigmas:",
        rounded_terminal_sigmas,
    )
    print(
        "best_singular_values:",
        optimized_singular_values.tolist(),
    )
    print(
        "best_tangent_stationarity_residual:",
        best_result[
            "tangent_stationarity_residual"
        ],
    )
    print("wrote:", JSON_OUT)
    print("wrote:", PAIR_CSV_OUT)
    print("wrote:", START_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
