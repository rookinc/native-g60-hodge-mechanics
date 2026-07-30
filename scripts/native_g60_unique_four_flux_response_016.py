from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

ANATOMY_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_six_channel_anatomy_015.json"
)

ANATOMY_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_six_channel_anatomy_015.npz"
)

RESOLUTION_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_six_flux_channel_resolution_014.npz"
)

STATIC_SOLVER_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_static_field_solver_008.npz"
)

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
)

B1_PATH = (
    SOURCE_ROOT
    / "native_g60_B1_vertex_edge_004.csv"
)

B2_PATH = (
    SOURCE_ROOT
    / "native_g60_B2_edge_face_004.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_unique_four_flux_response_016.json"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_unique_four_flux_response_probes_016.csv"
)

FACE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_unique_four_flux_face_patterns_016.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_unique_four_flux_response_016.npz"
)

TOLERANCE = 1e-9


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

    threshold = max(
        tolerance,
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * singular_values[0],
    )

    return int(np.count_nonzero(singular_values > threshold))


def deterministic_normalize(
    matrix: np.ndarray,
) -> tuple[np.ndarray, dict]:
    norm = float(np.linalg.norm(matrix, ord="fro"))

    if norm <= TOLERANCE:
        raise RuntimeError("cannot normalize zero channel")

    normalized = matrix / norm

    pivot_flat = int(np.argmax(np.abs(normalized)))
    pivot_row, pivot_column = np.unravel_index(
        pivot_flat,
        normalized.shape,
    )

    if normalized[pivot_row, pivot_column] < 0:
        normalized *= -1.0

    return normalized, {
        "original_frobenius_norm": norm,
        "normalized_frobenius_norm": float(
            np.linalg.norm(normalized, ord="fro")
        ),
        "pivot_row": int(pivot_row),
        "pivot_column": int(pivot_column),
        "pivot_value": float(
            normalized[pivot_row, pivot_column]
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    FACE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    anatomy = json.loads(
        ANATOMY_PATH.read_text(encoding="utf-8")
    )

    anatomy_data = np.load(ANATOMY_NPZ_PATH)
    resolution_data = np.load(RESOLUTION_NPZ_PATH)
    static_data = np.load(STATIC_SOLVER_PATH)
    projector_data = np.load(PROJECTOR_PATH)

    target_dimensions = [
        int(value)
        for value in anatomy_data["target_dimensions"]
    ]

    if target_dimensions != [4, 4, 5, 6]:
        raise RuntimeError(
            f"unexpected target dimensions: {target_dimensions}"
        )

    target_four_channel_basis = np.array(
        anatomy_data["target_0_channel_basis"],
        dtype=np.float64,
    )

    if target_four_channel_basis.shape != (1, 19, 36):
        raise RuntimeError(
            "expected unique target-0 channel shape "
            f"(1, 19, 36), found {target_four_channel_basis.shape}"
        )

    channel_raw = target_four_channel_basis[0]

    channel, normalization = deterministic_normalize(
        channel_raw
    )

    coexact_basis = np.array(
        resolution_data["coexact_basis_edges"],
        dtype=np.float64,
    )

    target_four_basis = np.array(
        resolution_data["target_sector_0_basis"],
        dtype=np.float64,
    )

    target_four_projector = np.array(
        resolution_data["target_sector_0_projector"],
        dtype=np.float64,
    )

    coexact_solver = np.array(
        static_data["coexact_solver"],
        dtype=np.float64,
    )

    delta1 = np.array(
        static_data["Delta1"],
        dtype=np.float64,
    )

    p_harmonic = np.array(
        projector_data["P_harmonic"],
        dtype=np.float64,
    )

    p_coexact = np.array(
        projector_data["P_coexact"],
        dtype=np.float64,
    )

    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)
    d1 = b2.T

    if coexact_basis.shape != (120, 19):
        raise RuntimeError(
            f"unexpected coexact basis shape: {coexact_basis.shape}"
        )

    if target_four_basis.shape != (19, 4):
        raise RuntimeError(
            f"unexpected target-four basis shape: {target_four_basis.shape}"
        )

    channel_target_residual = max_abs(
        target_four_projector @ channel - channel
    )

    channel_rank = numerical_rank(channel)

    input_pair_rows = []
    source_coordinate_columns = []
    source_edge_columns = []
    response_edge_columns = []
    face_flux_columns = []

    global_residuals = {
        "source_in_target_four": 0.0,
        "source_is_coexact": 0.0,
        "source_has_no_harmonic_part": 0.0,
        "static_equation": 0.0,
        "response_is_coexact": 0.0,
        "response_divergence": 0.0,
        "face_flux_zero_sum": 0.0,
        "face_flux_reconstruction": 0.0,
    }

    for a_index in range(6):
        for b_index in range(6):
            pair_id = 6 * a_index + b_index

            tensor_coordinate = np.zeros(
                36,
                dtype=np.float64,
            )
            tensor_coordinate[pair_id] = 1.0

            source_coordinates = (
                channel @ tensor_coordinate
            )

            source_edges = (
                coexact_basis @ source_coordinates
            )

            response_edges = (
                coexact_solver @ source_edges
            )

            face_flux = d1 @ response_edges

            reconstructed_source = (
                delta1 @ response_edges
            )

            residuals = {
                "source_in_target_four": max_abs(
                    target_four_projector
                    @ source_coordinates
                    - source_coordinates
                ),
                "source_is_coexact": max_abs(
                    p_coexact @ source_edges
                    - source_edges
                ),
                "source_has_no_harmonic_part": max_abs(
                    p_harmonic @ source_edges
                ),
                "static_equation": max_abs(
                    reconstructed_source
                    - source_edges
                ),
                "response_is_coexact": max_abs(
                    p_coexact @ response_edges
                    - response_edges
                ),
                "response_divergence": max_abs(
                    b1 @ response_edges
                ),
                "face_flux_zero_sum": abs(
                    float(np.sum(face_flux))
                ),
                "face_flux_reconstruction": max_abs(
                    d1 @ response_edges
                    - face_flux
                ),
            }

            for name, value in residuals.items():
                global_residuals[name] = max(
                    global_residuals[name],
                    value,
                )

            positive_face_count = int(
                np.count_nonzero(face_flux > TOLERANCE)
            )
            negative_face_count = int(
                np.count_nonzero(face_flux < -TOLERANCE)
            )
            zero_face_count = (
                20
                - positive_face_count
                - negative_face_count
            )

            maximum_face = int(np.argmax(face_flux))
            minimum_face = int(np.argmin(face_flux))

            input_pair_rows.append(
                {
                    "pair_id": pair_id,
                    "six_a_coordinate": a_index,
                    "six_b_coordinate": b_index,
                    "source_coordinate_norm": float(
                        np.linalg.norm(source_coordinates)
                    ),
                    "source_edge_norm": float(
                        np.linalg.norm(source_edges)
                    ),
                    "response_edge_norm": float(
                        np.linalg.norm(response_edges)
                    ),
                    "face_flux_norm": float(
                        np.linalg.norm(face_flux)
                    ),
                    "positive_face_count": positive_face_count,
                    "negative_face_count": negative_face_count,
                    "zero_face_count": zero_face_count,
                    "maximum_face_id": maximum_face,
                    "maximum_face_value": float(
                        face_flux[maximum_face]
                    ),
                    "minimum_face_id": minimum_face,
                    "minimum_face_value": float(
                        face_flux[minimum_face]
                    ),
                    "face_flux_sum": float(
                        np.sum(face_flux)
                    ),
                    **{
                        name + "_max_abs": value
                        for name, value in residuals.items()
                    },
                    "all_checks_pass": all(
                        value < TOLERANCE
                        for value in residuals.values()
                    ),
                }
            )

            source_coordinate_columns.append(
                source_coordinates
            )
            source_edge_columns.append(source_edges)
            response_edge_columns.append(response_edges)
            face_flux_columns.append(face_flux)

    source_coordinate_matrix = np.column_stack(
        source_coordinate_columns
    )

    source_edge_matrix = np.column_stack(
        source_edge_columns
    )

    response_edge_matrix = np.column_stack(
        response_edge_columns
    )

    face_flux_matrix = np.column_stack(
        face_flux_columns
    )

    source_coordinate_rank = numerical_rank(
        source_coordinate_matrix
    )
    source_edge_rank = numerical_rank(
        source_edge_matrix
    )
    response_edge_rank = numerical_rank(
        response_edge_matrix
    )
    face_flux_rank = numerical_rank(
        face_flux_matrix
    )

    all_probe_checks_pass = all(
        row["all_checks_pass"]
        for row in input_pair_rows
    )

    face_flux_column_sums = np.sum(
        face_flux_matrix,
        axis=0,
    )

    face_flux_row_sums = np.sum(
        face_flux_matrix,
        axis=1,
    )

    checks = {
        "input_015_audit_pass": (
            anatomy.get("audit_pass") is True
        ),
        "unique_target_four_channel_loaded": (
            target_four_channel_basis.shape == (1, 19, 36)
        ),
        "channel_normalized_to_unit_frobenius_norm": (
            abs(
                normalization[
                    "normalized_frobenius_norm"
                ]
                - 1.0
            )
            < TOLERANCE
        ),
        "deterministic_channel_sign_is_positive": (
            normalization["pivot_value"] > 0.0
        ),
        "channel_image_lies_in_target_four": (
            channel_target_residual < TOLERANCE
        ),
        "channel_rank_is_4": (
            channel_rank == 4
        ),
        "all_36_coordinate_pairs_tested": (
            len(input_pair_rows) == 36
        ),
        "all_coordinate_pair_probes_pass": (
            all_probe_checks_pass
        ),
        "source_coordinate_span_is_4": (
            source_coordinate_rank == 4
        ),
        "source_edge_span_is_4": (
            source_edge_rank == 4
        ),
        "response_edge_span_is_4": (
            response_edge_rank == 4
        ),
        "face_flux_span_is_4": (
            face_flux_rank == 4
        ),
        "all_face_flux_patterns_have_zero_total": (
            max_abs(face_flux_column_sums)
            < TOLERANCE
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_unique_four_flux_response_016"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_unique_four_dimensional_cross_flux_response_exported"
            if audit_pass
            else "native_g60_unique_four_flux_response_failed"
        ),
        "inputs": {
            "cross_channel_anatomy": str(
                ANATOMY_PATH.relative_to(ROOT)
            ),
            "cross_channel_anatomy_npz": str(
                ANATOMY_NPZ_PATH.relative_to(ROOT)
            ),
            "channel_resolution_npz": str(
                RESOLUTION_NPZ_PATH.relative_to(ROOT)
            ),
            "static_solver": str(
                STATIC_SOLVER_PATH.relative_to(ROOT)
            ),
        },
        "channel": {
            "domain": "V6a tensor V6b",
            "domain_dimension": 36,
            "target": "unique cross target C4",
            "target_dimension": 4,
            "hom_dimension": 1,
            "map_rank": channel_rank,
            "normalization": normalization,
            "target_residual_max_abs": (
                channel_target_residual
            ),
            "canonical_status": (
                "unique equivariant line, normalized reproducibly; "
                "overall physical scale remains unselected"
            ),
        },
        "response_pipeline": {
            "bilinear_source": (
                "j4(u,v) = T4(u tensor v)"
            ),
            "coexact_edge_source": (
                "J4 = coexact_basis j4"
            ),
            "static_response": (
                "A4 = Delta1_coexact_inverse J4"
            ),
            "face_flux": "F4 = d1 A4",
        },
        "span_dimensions": {
            "source_coordinates": source_coordinate_rank,
            "coexact_edge_sources": source_edge_rank,
            "static_responses": response_edge_rank,
            "face_flux_patterns": face_flux_rank,
        },
        "checks": checks,
        "probe_audit": {
            "probe_count": len(input_pair_rows),
            "all_probes_pass": all_probe_checks_pass,
            "global_maximum_residuals": (
                global_residuals
            ),
        },
        "face_flux": {
            "face_count": 20,
            "pattern_count": 36,
            "pattern_rank": face_flux_rank,
            "all_pattern_sums_zero": (
                max_abs(face_flux_column_sums)
                < TOLERANCE
            ),
            "maximum_pattern_sum_abs": max_abs(
                face_flux_column_sums
            ),
            "aggregate_face_sum_profile": [
                float(value)
                for value in face_flux_row_sums
            ],
        },
        "outputs": {
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(ROOT)
            ),
            "face_pattern_csv": str(
                FACE_CSV_OUT.relative_to(ROOT)
            ),
            "response_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "unique_equivariant_four_channel_isolated": (
                audit_pass
            ),
            "reproducible_unit_normalization_selected": (
                audit_pass
            ),
            "static_combinatorial_flux_response_exported": (
                audit_pass
            ),
            "overall_constitutive_scale_selected": False,
            "input_dynamics_selected": False,
            "physical_flux_claim": False,
            "electromagnetism_claim": False,
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

    with PROBE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(input_pair_rows[0]),
        )
        writer.writeheader()
        writer.writerows(input_pair_rows)

    with FACE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["face_id"]
            + [
                f"pair_{index:02d}"
                for index in range(36)
            ]
        )

        for face_id in range(20):
            writer.writerow(
                [face_id]
                + [
                    f"{float(value):.17g}"
                    for value in face_flux_matrix[
                        face_id,
                        :,
                    ]
                ]
            )

    np.savez_compressed(
        NPZ_OUT,
        unique_four_channel=channel,
        coexact_basis_edges=coexact_basis,
        target_four_basis=target_four_basis,
        target_four_projector=target_four_projector,
        source_coordinate_matrix=source_coordinate_matrix,
        source_edge_matrix=source_edge_matrix,
        response_edge_matrix=response_edge_matrix,
        face_flux_matrix=face_flux_matrix,
        normalization_pivot=np.array(
            [
                normalization["pivot_row"],
                normalization["pivot_column"],
            ],
            dtype=np.int64,
        ),
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "channel_shape/rank:",
        list(channel.shape),
        channel_rank,
    )
    print(
        "channel_normalization:",
        normalization,
    )
    print(
        "channel_target_residual_max_abs:",
        channel_target_residual,
    )
    print(
        "probe_count/all_pass:",
        len(input_pair_rows),
        all_probe_checks_pass,
    )
    print(
        "source_coordinate/source_edge/response/flux_ranks:",
        source_coordinate_rank,
        source_edge_rank,
        response_edge_rank,
        face_flux_rank,
    )
    print(
        "face_flux_pattern_sum_max_abs:",
        max_abs(face_flux_column_sums),
    )
    print(
        "probe_global_maximum_residuals:",
        global_residuals,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", FACE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
