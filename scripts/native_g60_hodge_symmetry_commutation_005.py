from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

DECOMPOSITION_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_hodge_decomposition_003.json"
)

SURFACE_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_orientation_004.json"
)

SYMMETRY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_symmetry_and_kernel_005.json"
)

ISO_PATH = (
    SOURCE_ROOT
    / "native_g60_fiber_product_isomorphism_044.json"
)

SUPPLEMENT_PATH = (
    SOURCE_ROOT
    / "symmetry_source_manifest_005.json"
)

B1_PATH = SOURCE_ROOT / "native_g60_B1_vertex_edge_004.csv"
B2_PATH = SOURCE_ROOT / "native_g60_B2_edge_face_004.csv"

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
)

DELTA1_PATH = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_Delta1_uniform_002.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_hodge_symmetry_commutation_005.json"
)

CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_hodge_symmetry_commutation_005.csv"
)

ACTION_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_signed_cochain_actions_005.npz"
)

TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_matrix_csv(
    path: Path,
    dtype: type = float,
) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    return np.array(
        [
            [dtype(value) for value in row[1:]]
            for row in rows[1:]
        ],
        dtype=np.float64 if dtype is float else np.int64,
    )


def max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0

    return float(np.max(np.abs(matrix)))


def recover_oriented_edges(
    b1: np.ndarray,
) -> list[tuple[int, int]]:
    edges = []

    for column in range(b1.shape[1]):
        negative = np.flatnonzero(b1[:, column] == -1)
        positive = np.flatnonzero(b1[:, column] == 1)

        if len(negative) != 1 or len(positive) != 1:
            raise RuntimeError(
                f"invalid oriented edge column {column}"
            )

        edges.append(
            (int(negative[0]), int(positive[0]))
        )

    return edges


def signed_edge_action(
    permutation: np.ndarray,
    edges: list[tuple[int, int]],
    edge_index: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    target = np.empty(len(edges), dtype=np.int64)
    sign = np.empty(len(edges), dtype=np.int8)

    for source, (u, v) in enumerate(edges):
        image_u = int(permutation[u])
        image_v = int(permutation[v])

        canonical = (
            (image_u, image_v)
            if image_u < image_v
            else (image_v, image_u)
        )

        target[source] = edge_index[canonical]
        sign[source] = (
            1 if (image_u, image_v) == canonical else -1
        )

    return target, sign


def conjugate_by_signed_permutation(
    matrix: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(matrix)

    transformed[np.ix_(target, target)] = (
        sign[:, None]
        * matrix
        * sign[None, :]
    )

    return transformed


def transform_rows(
    matrix: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(matrix)

    transformed[target, :] = (
        sign[:, None] * matrix
    )

    return transformed


def transform_columns(
    matrix: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    """
    Compute matrix @ P for the signed basis action

        basis_source -> sign[source] * basis_target[source].

    Column source of matrix @ P is therefore sign[source]
    times column target[source] of matrix.
    """
    return (
        matrix[:, target]
        * sign[None, :]
    )


def unsigned_transform_rows(
    matrix: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(matrix)
    transformed[target, :] = matrix
    return transformed


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    ACTION_OUT.parent.mkdir(parents=True, exist_ok=True)

    decomposition = json.loads(
        DECOMPOSITION_PATH.read_text(encoding="utf-8")
    )

    surface = json.loads(
        SURFACE_PATH.read_text(encoding="utf-8")
    )

    symmetry = json.loads(
        SYMMETRY_PATH.read_text(encoding="utf-8")
    )

    supplement = json.loads(
        SUPPLEMENT_PATH.read_text(encoding="utf-8")
    )

    isomorphism = json.loads(
        ISO_PATH.read_text(encoding="utf-8")
    )

    b1 = read_matrix_csv(B1_PATH, int)
    b2 = read_matrix_csv(B2_PATH, int)
    delta1 = read_matrix_csv(DELTA1_PATH, float)

    projector_payload = np.load(PROJECTOR_PATH)

    p_exact = np.array(
        projector_payload["P_exact"],
        dtype=np.float64,
    )
    p_harmonic = np.array(
        projector_payload["P_harmonic"],
        dtype=np.float64,
    )
    p_coexact = np.array(
        projector_payload["P_coexact"],
        dtype=np.float64,
    )

    mapping_rows = isomorphism["mapping_rows"]
    face_action_rows = surface["face_action"][
        "face_action_rows"
    ]

    if len(mapping_rows) != 480:
        raise RuntimeError("expected 480 native permutations")

    if len(face_action_rows) != 480:
        raise RuntimeError("expected 480 face-action rows")

    permutations = np.array(
        [
            row["actual_permutation"]
            for row in mapping_rows
        ],
        dtype=np.int64,
    )

    edges = recover_oriented_edges(b1)

    edge_index = {
        tuple(sorted(edge)): index
        for index, edge in enumerate(edges)
    }

    vertex_targets = permutations.copy()

    edge_targets = np.empty((480, 120), dtype=np.int64)
    edge_signs = np.empty((480, 120), dtype=np.int8)

    face_targets = np.empty((480, 20), dtype=np.int64)
    face_signs = np.empty((480, 20), dtype=np.int8)

    audit_rows = []

    global_maxima = Counter()

    for actual_index in range(480):
        vertex_target = vertex_targets[actual_index]

        edge_target, edge_sign = signed_edge_action(
            vertex_target,
            edges,
            edge_index,
        )

        action_row = face_action_rows[actual_index]

        face_target = np.array(
            action_row["face_permutation"],
            dtype=np.int64,
        )

        face_sign = np.array(
            action_row["orientation_signs"],
            dtype=np.int8,
        )

        edge_targets[actual_index] = edge_target
        edge_signs[actual_index] = edge_sign
        face_targets[actual_index] = face_target
        face_signs[actual_index] = face_sign

        # B1 P1 = P0 B1
        b1_p1 = transform_columns(
            b1,
            edge_target,
            edge_sign,
        )

        p0_b1 = unsigned_transform_rows(
            b1,
            vertex_target,
        )

        b1_chain_residual = b1_p1 - p0_b1

        # B2 P2 = P1 B2
        b2_p2 = transform_columns(
            b2,
            face_target,
            face_sign,
        )

        p1_b2 = transform_rows(
            b2,
            edge_target,
            edge_sign,
        )

        b2_chain_residual = b2_p2 - p1_b2

        delta1_conjugated = (
            conjugate_by_signed_permutation(
                delta1,
                edge_target,
                edge_sign,
            )
        )

        exact_conjugated = (
            conjugate_by_signed_permutation(
                p_exact,
                edge_target,
                edge_sign,
            )
        )

        harmonic_conjugated = (
            conjugate_by_signed_permutation(
                p_harmonic,
                edge_target,
                edge_sign,
            )
        )

        coexact_conjugated = (
            conjugate_by_signed_permutation(
                p_coexact,
                edge_target,
                edge_sign,
            )
        )

        residuals = {
            "b1_chain_max_abs": max_abs(
                b1_chain_residual
            ),
            "b2_chain_max_abs": max_abs(
                b2_chain_residual
            ),
            "delta1_commutation_max_abs": max_abs(
                delta1_conjugated - delta1
            ),
            "p_exact_commutation_max_abs": max_abs(
                exact_conjugated - p_exact
            ),
            "p_harmonic_commutation_max_abs": max_abs(
                harmonic_conjugated - p_harmonic
            ),
            "p_coexact_commutation_max_abs": max_abs(
                coexact_conjugated - p_coexact
            ),
        }

        for name, value in residuals.items():
            global_maxima[name] = max(
                float(global_maxima.get(name, 0.0)),
                value,
            )

        edge_sign_profile = Counter(
            int(value)
            for value in edge_sign
        )

        face_sign_profile = Counter(
            int(value)
            for value in face_sign
        )

        audit_rows.append(
            {
                "actual_index": actual_index,
                "orientation_type": action_row[
                    "orientation_type"
                ],
                "edge_positive_count": edge_sign_profile[1],
                "edge_negative_count": edge_sign_profile[-1],
                "face_positive_count": face_sign_profile[1],
                "face_negative_count": face_sign_profile[-1],
                **residuals,
                "all_checks_pass": all(
                    value < TOLERANCE
                    for value in residuals.values()
                ),
            }
        )

    all_rows_pass = all(
        row["all_checks_pass"]
        for row in audit_rows
    )

    orientation_profile = Counter(
        row["orientation_type"]
        for row in audit_rows
    )

    checks = {
        "input_decomposition_pass": (
            decomposition.get("audit_pass") is True
        ),
        "input_surface_pass": (
            surface.get("audit_pass") is True
        ),
        "input_symmetry_pass": (
            symmetry.get("audit_pass") is True
        ),
        "supplement_hash_match": (
            supplement.get("hash_match") is True
            and sha256(ISO_PATH)
            == supplement["expected_sha256"]
        ),
        "native_permutation_count_is_480": (
            permutations.shape == (480, 60)
        ),
        "edge_action_shape_is_480_by_120": (
            edge_targets.shape == (480, 120)
        ),
        "face_action_shape_is_480_by_20": (
            face_targets.shape == (480, 20)
        ),
        "all_vertex_actions_are_permutations": all(
            sorted(row.tolist()) == list(range(60))
            for row in vertex_targets
        ),
        "all_edge_actions_are_permutations": all(
            sorted(row.tolist()) == list(range(120))
            for row in edge_targets
        ),
        "all_face_actions_are_permutations": all(
            sorted(row.tolist()) == list(range(20))
            for row in face_targets
        ),
        "b1_equivariant_for_all_480": (
            global_maxima["b1_chain_max_abs"]
            < TOLERANCE
        ),
        "b2_equivariant_for_all_480": (
            global_maxima["b2_chain_max_abs"]
            < TOLERANCE
        ),
        "delta1_commutes_for_all_480": (
            global_maxima[
                "delta1_commutation_max_abs"
            ]
            < TOLERANCE
        ),
        "p_exact_commutes_for_all_480": (
            global_maxima[
                "p_exact_commutation_max_abs"
            ]
            < TOLERANCE
        ),
        "p_harmonic_commutes_for_all_480": (
            global_maxima[
                "p_harmonic_commutation_max_abs"
            ]
            < TOLERANCE
        ),
        "p_coexact_commutes_for_all_480": (
            global_maxima[
                "p_coexact_commutation_max_abs"
            ]
            < TOLERANCE
        ),
        "every_automorphism_passes": all_rows_pass,
        "orientation_profile_is_240_240": (
            orientation_profile
            == {
                "preserving": 240,
                "reversing": 240,
            }
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_hodge_symmetry_commutation_005"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_uniform_hodge_decomposition_is_full_group_equivariant"
            if audit_pass
            else "native_g60_hodge_symmetry_commutation_failed"
        ),
        "inputs": {
            "hodge_decomposition": str(
                DECOMPOSITION_PATH.relative_to(ROOT)
            ),
            "surface_action": str(
                SURFACE_PATH.relative_to(ROOT)
            ),
            "surface_symmetry": str(
                SYMMETRY_PATH.relative_to(ROOT)
            ),
            "native_vertex_permutations": str(
                ISO_PATH.relative_to(ROOT)
            ),
            "symmetry_source_supplement": str(
                SUPPLEMENT_PATH.relative_to(ROOT)
            ),
        },
        "construction": {
            "vertex_action": (
                "native 60-point permutation action"
            ),
            "edge_action": (
                "induced signed action on canonical oriented edges"
            ),
            "face_action": (
                "imported signed action on coherently oriented faces"
            ),
            "group_element_count": 480,
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "global_maximum_residuals": dict(
            sorted(global_maxima.items())
        ),
        "orientation_profile": dict(
            sorted(orientation_profile.items())
        ),
        "action_dimensions": {
            "C0": 60,
            "C1": 120,
            "C2": 20,
        },
        "equivariance": {
            "boundary_1": "B1 P1(g) = P0(g) B1",
            "boundary_2": "B2 P2(g) = P1(g) B2",
            "laplacian_1": (
                "P1(g) Delta1 P1(g)^T = Delta1"
            ),
            "exact_projector": (
                "P1(g) P_exact P1(g)^T = P_exact"
            ),
            "harmonic_projector": (
                "P1(g) P_harmonic P1(g)^T = P_harmonic"
            ),
            "coexact_projector": (
                "P1(g) P_coexact P1(g)^T = P_coexact"
            ),
        },
        "automorphism_rows": audit_rows,
        "outputs": {
            "audit_csv": str(
                CSV_OUT.relative_to(ROOT)
            ),
            "signed_actions_npz": str(
                ACTION_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "full_native_group_commutation_audited": (
                audit_pass
            ),
            "uniform_hodge_decomposition_is_native_equivariant": (
                audit_pass
            ),
            "harmonic_representation_decomposed": False,
            "canonical_metric_derived": False,
            "physical_constitutive_law_derived": False,
            "maxwell_claim": False,
            "physical_energy_claim": False,
            "physical_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False,
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
            fieldnames=list(audit_rows[0]),
        )

        writer.writeheader()
        writer.writerows(audit_rows)

    np.savez_compressed(
        ACTION_OUT,
        vertex_target=vertex_targets,
        edge_target=edge_targets,
        edge_sign=edge_signs,
        face_target=face_targets,
        face_sign=face_signs,
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print("group_element_count:", len(audit_rows))
    print(
        "orientation_profile:",
        dict(sorted(orientation_profile.items())),
    )
    print(
        "all_automorphisms_pass:",
        all_rows_pass,
    )

    for name, value in sorted(global_maxima.items()):
        print(name + ":", value)

    print("wrote:", JSON_OUT)
    print("wrote:", CSV_OUT)
    print("wrote:", ACTION_OUT)


if __name__ == "__main__":
    main()
