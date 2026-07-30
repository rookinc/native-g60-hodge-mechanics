from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

HARMONIC_REP_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_representation_006.npz"
)

IRREDUCIBLE_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_irreducible_projectors_006c.npz"
)

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
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
    / "native_g60_six_sector_flux_intertwiner_013.json"
)

CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_six_sector_flux_intertwiner_013.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_six_sector_flux_intertwiner_013.npz"
)

TOLERANCE = 1e-8


def max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0

    return float(np.max(np.abs(array)))


def orthonormal_basis_from_projector(
    projector: np.ndarray,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (projector + projector.T)
    )

    selected = eigenvalues > 0.5
    basis = eigenvectors[:, selected]

    for column in range(basis.shape[1]):
        vector = basis[:, column]
        pivot = int(np.argmax(np.abs(vector)))

        if vector[pivot] < 0:
            basis[:, column] *= -1.0

    return basis


def apply_signed_edge_action(
    basis: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(basis)

    transformed[target, :] = (
        sign[:, None] * basis
    )

    return transformed


def restricted_representation(
    basis: np.ndarray,
    edge_targets: np.ndarray,
    edge_signs: np.ndarray,
) -> np.ndarray:
    result = np.empty(
        (
            edge_targets.shape[0],
            basis.shape[1],
            basis.shape[1],
        ),
        dtype=np.float64,
    )

    for index in range(edge_targets.shape[0]):
        transformed = apply_signed_edge_action(
            basis,
            edge_targets[index],
            edge_signs[index],
        )

        result[index] = basis.T @ transformed

    return result


def character(
    representation: np.ndarray,
) -> np.ndarray:
    return np.trace(
        representation,
        axis1=1,
        axis2=2,
    )


def character_inner_product(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    return float(
        np.dot(left, right)
        / len(left)
    )


def nearest_integer(
    value: float,
    tolerance: float = 1e-7,
) -> int | None:
    candidate = int(round(value))

    if abs(value - candidate) <= tolerance:
        return candidate

    return None


def tensor_representation(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    group_order = left.shape[0]
    left_dimension = left.shape[1]
    right_dimension = right.shape[1]

    result = np.empty(
        (
            group_order,
            left_dimension * right_dimension,
            left_dimension * right_dimension,
        ),
        dtype=np.float64,
    )

    for index in range(group_order):
        result[index] = np.kron(
            left[index],
            right[index],
        )

    return result


def deterministic_seed(
    output_dimension: int,
    input_dimension: int,
    seed_id: int,
) -> np.ndarray:
    row = np.arange(
        1,
        output_dimension + 1,
        dtype=np.float64,
    )[:, None]

    column = np.arange(
        1,
        input_dimension + 1,
        dtype=np.float64,
    )[None, :]

    if seed_id == 0:
        return np.sin(row * column)

    if seed_id == 1:
        return np.cos(row + 2.0 * column)

    if seed_id == 2:
        return ((row + column) % 7.0) - 3.0

    if seed_id == 3:
        return np.sin(row) * np.cos(column)

    raise ValueError(seed_id)


def reynolds_intertwiner(
    output_representation: np.ndarray,
    input_representation: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(seed)

    for output_rho, input_rho in zip(
        output_representation,
        input_representation,
    ):
        result += (
            output_rho
            @ seed
            @ input_rho.T
        )

    result /= output_representation.shape[0]

    return result


def intertwiner_residual(
    intertwiner: np.ndarray,
    output_representation: np.ndarray,
    input_representation: np.ndarray,
) -> float:
    return max(
        max_abs(
            output_rho @ intertwiner
            - intertwiner @ input_rho
        )
        for output_rho, input_rho in zip(
            output_representation,
            input_representation,
        )
    )


def numerical_rank(
    matrix: np.ndarray,
    tolerance: float = TOLERANCE,
) -> int:
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    return int(
        np.count_nonzero(
            singular_values > tolerance
        )
    )


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    harmonic_payload = np.load(HARMONIC_REP_PATH)

    harmonic_representation = np.array(
        harmonic_payload["rho_harmonic"],
        dtype=np.float64,
    )

    irreducible_payload = np.load(
        IRREDUCIBLE_PATH,
        allow_pickle=True,
    )

    sector_projectors = [
        np.array(value, dtype=np.float64)
        for value in irreducible_payload[
            "sector_projector"
        ]
    ]

    sector_dimensions = [
        int(value)
        for value in irreducible_payload[
            "sector_dimensions"
        ]
    ]

    projector_payload = np.load(PROJECTOR_PATH)

    p_coexact = np.array(
        projector_payload["P_coexact"],
        dtype=np.float64,
    )

    action_payload = np.load(ACTION_PATH)

    edge_targets = np.array(
        action_payload["edge_target"],
        dtype=np.int64,
    )

    edge_signs = np.array(
        action_payload["edge_sign"],
        dtype=np.int8,
    )

    six_indices = [
        index
        for index, dimension in enumerate(
            sector_dimensions
        )
        if dimension == 6
    ]

    if len(six_indices) != 2:
        raise RuntimeError(
            "expected exactly two 6-dimensional sectors, "
            f"found {six_indices}"
        )

    six_a_index, six_b_index = six_indices

    six_a_basis_h = orthonormal_basis_from_projector(
        sector_projectors[six_a_index]
    )

    six_b_basis_h = orthonormal_basis_from_projector(
        sector_projectors[six_b_index]
    )

    # The sector projectors live in the 42-dimensional harmonic
    # coordinate space. Their restricted representations can be
    # obtained directly from the harmonic representation.
    six_a_representation = np.empty(
        (480, 6, 6),
        dtype=np.float64,
    )

    six_b_representation = np.empty(
        (480, 6, 6),
        dtype=np.float64,
    )

    for index in range(480):
        six_a_representation[index] = (
            six_a_basis_h.T
            @ harmonic_representation[index]
            @ six_a_basis_h
        )

        six_b_representation[index] = (
            six_b_basis_h.T
            @ harmonic_representation[index]
            @ six_b_basis_h
        )

    coexact_basis = orthonormal_basis_from_projector(
        p_coexact
    )

    if coexact_basis.shape != (120, 19):
        raise RuntimeError(
            f"unexpected coexact basis shape {coexact_basis.shape}"
        )

    coexact_representation = restricted_representation(
        coexact_basis,
        edge_targets,
        edge_signs,
    )

    chi_a = character(six_a_representation)
    chi_b = character(six_b_representation)
    chi_c = character(coexact_representation)

    coupling_specs = [
        (
            "six_a_tensor_six_a",
            six_a_representation,
            six_a_representation,
            chi_a * chi_a,
        ),
        (
            "six_a_tensor_six_b",
            six_a_representation,
            six_b_representation,
            chi_a * chi_b,
        ),
        (
            "six_b_tensor_six_b",
            six_b_representation,
            six_b_representation,
            chi_b * chi_b,
        ),
    ]

    result_rows = []
    intertwiner_outputs: dict[str, np.ndarray] = {}
    tensor_character_outputs: dict[str, np.ndarray] = {}

    for (
        coupling_name,
        left_representation,
        right_representation,
        tensor_character,
    ) in coupling_specs:
        hom_dimension_raw = character_inner_product(
            tensor_character,
            chi_c,
        )

        hom_dimension = nearest_integer(
            hom_dimension_raw
        )

        tensor_rep = tensor_representation(
            left_representation,
            right_representation,
        )

        accepted_intertwiner = None
        accepted_seed_id = None
        accepted_rank = 0
        accepted_residual = None
        singular_values = np.array([], dtype=np.float64)

        if hom_dimension is not None and hom_dimension > 0:
            for seed_id in range(4):
                seed = deterministic_seed(
                    19,
                    36,
                    seed_id,
                )

                candidate = reynolds_intertwiner(
                    coexact_representation,
                    tensor_rep,
                    seed,
                )

                candidate_singular_values = np.linalg.svd(
                    candidate,
                    compute_uv=False,
                )

                candidate_rank = numerical_rank(candidate)
                candidate_residual = intertwiner_residual(
                    candidate,
                    coexact_representation,
                    tensor_rep,
                )

                if (
                    candidate_rank > 0
                    and candidate_residual < TOLERANCE
                ):
                    accepted_intertwiner = candidate
                    accepted_seed_id = seed_id
                    accepted_rank = candidate_rank
                    accepted_residual = candidate_residual
                    singular_values = candidate_singular_values
                    break

        result_rows.append(
            {
                "coupling_name": coupling_name,
                "input_dimension": 36,
                "output_dimension": 19,
                "hom_dimension_raw": hom_dimension_raw,
                "hom_dimension_integer": hom_dimension,
                "symmetry_allows_coupling": (
                    hom_dimension is not None
                    and hom_dimension > 0
                ),
                "canonical_up_to_scale": (
                    hom_dimension == 1
                ),
                "selected_seed_id": accepted_seed_id,
                "intertwiner_rank": accepted_rank,
                "intertwiner_residual_max_abs": (
                    accepted_residual
                ),
                "largest_singular_value": (
                    float(singular_values[0])
                    if len(singular_values)
                    else None
                ),
                "smallest_nonzero_singular_value": (
                    float(
                        singular_values[
                            accepted_rank - 1
                        ]
                    )
                    if accepted_rank > 0
                    else None
                ),
            }
        )

        tensor_character_outputs[
            coupling_name
        ] = tensor_character

        if accepted_intertwiner is not None:
            intertwiner_outputs[
                coupling_name
            ] = accepted_intertwiner

    cross_row = next(
        row
        for row in result_rows
        if row["coupling_name"]
        == "six_a_tensor_six_b"
    )

    coexact_character_norm = character_inner_product(
        chi_c,
        chi_c,
    )

    checks = {
        "two_six_dimensional_sectors_found": (
            len(six_indices) == 2
        ),
        "six_a_basis_shape_is_42_by_6": (
            six_a_basis_h.shape == (42, 6)
        ),
        "six_b_basis_shape_is_42_by_6": (
            six_b_basis_h.shape == (42, 6)
        ),
        "coexact_basis_shape_is_120_by_19": (
            coexact_basis.shape == (120, 19)
        ),
        "six_a_representation_shape_is_480_by_6_by_6": (
            six_a_representation.shape == (480, 6, 6)
        ),
        "six_b_representation_shape_is_480_by_6_by_6": (
            six_b_representation.shape == (480, 6, 6)
        ),
        "coexact_representation_shape_is_480_by_19_by_19": (
            coexact_representation.shape == (480, 19, 19)
        ),
        "all_hom_dimensions_are_integer": all(
            row["hom_dimension_integer"] is not None
            for row in result_rows
        ),
        "all_constructed_intertwiners_are_equivariant": all(
            (
                not row["symmetry_allows_coupling"]
                or (
                    row["intertwiner_rank"] > 0
                    and row[
                        "intertwiner_residual_max_abs"
                    ]
                    is not None
                    and row[
                        "intertwiner_residual_max_abs"
                    ]
                    < TOLERANCE
                )
            )
            for row in result_rows
        ),
        "cross_six_permission_resolved": (
            cross_row["hom_dimension_integer"]
            is not None
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_six_sector_flux_intertwiner_013"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_six_sector_flux_permission_resolved"
            if audit_pass
            else "native_g60_six_sector_flux_audit_failed"
        ),
        "construction": {
            "target_space": (
                "19-dimensional coexact edge-field sector"
            ),
            "tested_domains": [
                "V6a tensor V6a",
                "V6a tensor V6b",
                "V6b tensor V6b",
            ],
            "permission_test": (
                "dim Hom_Gamma(domain, coexact) "
                "= <chi_domain, chi_coexact>"
            ),
            "explicit_map_method": (
                "Reynolds averaging of deterministic seed maps"
            ),
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "coexact_representation": {
            "dimension": 19,
            "character_norm_raw": (
                coexact_character_norm
            ),
            "character_norm_integer": nearest_integer(
                coexact_character_norm
            ),
        },
        "coupling_rows": result_rows,
        "cross_six_result": {
            "hom_dimension": (
                cross_row["hom_dimension_integer"]
            ),
            "symmetry_allows_flux_coupling": (
                cross_row[
                    "symmetry_allows_coupling"
                ]
            ),
            "canonical_up_to_scale": (
                cross_row["canonical_up_to_scale"]
            ),
            "explicit_intertwiner_rank": (
                cross_row["intertwiner_rank"]
            ),
            "interpretation": (
                "A positive Hom dimension means native symmetry "
                "permits a bilinear cross-six source in the coexact "
                "sector. Zero means the proposed coupling is forbidden "
                "without additional structure."
            ),
        },
        "outputs": {
            "coupling_csv": str(
                CSV_OUT.relative_to(ROOT)
            ),
            "intertwiner_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "symmetry_permission_test_completed": audit_pass,
            "explicit_equivariant_maps_exported_when_available": (
                audit_pass
            ),
            "bilinear_normalization_selected": False,
            "source_dynamics_selected": False,
            "physical_flux_claim": False,
            "electromagnetism_claim": False,
            "force_claim": False,
            "physical_claim": False,
        },
    }

    JSON_OUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(result_rows[0]),
        )

        writer.writeheader()
        writer.writerows(result_rows)

    npz_payload: dict[str, np.ndarray] = {
        "six_a_basis_harmonic_coordinates": six_a_basis_h,
        "six_b_basis_harmonic_coordinates": six_b_basis_h,
        "coexact_basis_edges": coexact_basis,
        "six_a_character": chi_a,
        "six_b_character": chi_b,
        "coexact_character": chi_c,
        "tolerance": np.array([TOLERANCE]),
    }

    for name, value in tensor_character_outputs.items():
        npz_payload[
            name + "_tensor_character"
        ] = value

    for name, value in intertwiner_outputs.items():
        npz_payload[
            name + "_intertwiner"
        ] = value

    np.savez_compressed(
        NPZ_OUT,
        **npz_payload,
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "coexact_character_norm:",
        coexact_character_norm,
        nearest_integer(coexact_character_norm),
    )

    print("\ncoupling permissions:")

    for row in result_rows:
        print(
            row["coupling_name"],
            "Hom_dimension=",
            row["hom_dimension_integer"],
            "canonical_up_to_scale=",
            row["canonical_up_to_scale"],
            "intertwiner_rank=",
            row["intertwiner_rank"],
            "residual=",
            row["intertwiner_residual_max_abs"],
        )

    print("\ncross_six_result:")
    print(
        "Hom_dimension:",
        cross_row["hom_dimension_integer"],
    )
    print(
        "symmetry_allows_flux_coupling:",
        cross_row["symmetry_allows_coupling"],
    )
    print(
        "canonical_up_to_scale:",
        cross_row["canonical_up_to_scale"],
    )
    print(
        "explicit_intertwiner_rank:",
        cross_row["intertwiner_rank"],
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
