from __future__ import annotations

import csv
import json
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

PROJECTOR_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
)

B1_PATH = SOURCE_ROOT / "native_g60_B1_vertex_edge_004.csv"
B2_PATH = SOURCE_ROOT / "native_g60_B2_edge_face_004.csv"

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_harmonic_1_form_basis_004.json"
)

BASIS_CSV_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_harmonic_1_form_basis_004.csv"
)

BASIS_NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_harmonic_1_form_basis_004.npz"
)

MODE_AUDIT_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_harmonic_1_form_mode_audit_004.csv"
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


def max_abs(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0

    return float(np.max(np.abs(matrix)))


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


def write_basis_csv(
    path: Path,
    basis: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["edge_id"]
            + [
                f"h{index:02d}"
                for index in range(basis.shape[1])
            ]
        )

        for edge_index, row in enumerate(basis):
            writer.writerow(
                [f"e{edge_index:03d}"]
                + [
                    f"{float(value):.17g}"
                    for value in row
                ]
            )


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    BASIS_NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODE_AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)

    decomposition = json.loads(
        DECOMPOSITION_PATH.read_text(encoding="utf-8")
    )

    projector_payload = np.load(PROJECTOR_PATH)

    p_harmonic = np.array(
        projector_payload["P_harmonic"],
        dtype=np.float64,
    )

    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)

    d1 = b2.T

    delta1 = (
        b1.T @ b1
        + b2 @ b2.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(p_harmonic)

    selected = eigenvalues > 0.5

    harmonic_basis = eigenvectors[:, selected]

    if harmonic_basis.shape != (120, 42):
        raise RuntimeError(
            "expected harmonic basis shape (120, 42), found "
            f"{harmonic_basis.shape}"
        )

    # Fix arbitrary eigenvector signs deterministically.
    for column in range(harmonic_basis.shape[1]):
        vector = harmonic_basis[:, column]

        pivot = int(np.argmax(np.abs(vector)))

        if vector[pivot] < 0:
            harmonic_basis[:, column] *= -1.0

    gram = harmonic_basis.T @ harmonic_basis
    reconstructed_projector = (
        harmonic_basis @ harmonic_basis.T
    )

    divergence = b1 @ harmonic_basis
    curl = d1 @ harmonic_basis
    laplacian = delta1 @ harmonic_basis

    projector_fix = (
        p_harmonic @ harmonic_basis
        - harmonic_basis
    )

    mode_rows = []

    for mode_index in range(42):
        vector = harmonic_basis[:, mode_index]

        divergence_vector = divergence[:, mode_index]
        curl_vector = curl[:, mode_index]
        laplacian_vector = laplacian[:, mode_index]

        pivot_edge = int(np.argmax(np.abs(vector)))

        mode_rows.append(
            {
                "mode_id": mode_index,
                "norm": float(np.linalg.norm(vector)),
                "sum": float(np.sum(vector)),
                "mean": float(np.mean(vector)),
                "minimum": float(np.min(vector)),
                "maximum": float(np.max(vector)),
                "pivot_edge": pivot_edge,
                "pivot_value": float(vector[pivot_edge]),
                "support_above_1e_12": int(
                    np.count_nonzero(
                        np.abs(vector) > 1e-12
                    )
                ),
                "divergence_max_abs": max_abs(
                    divergence_vector
                ),
                "curl_max_abs": max_abs(curl_vector),
                "laplacian_max_abs": max_abs(
                    laplacian_vector
                ),
            }
        )

    checks = {
        "input_decomposition_audit_pass": (
            decomposition.get("audit_pass") is True
        ),
        "projector_shape_is_120_by_120": (
            p_harmonic.shape == (120, 120)
        ),
        "basis_shape_is_120_by_42": (
            harmonic_basis.shape == (120, 42)
        ),
        "basis_rank_is_42": (
            numerical_rank(harmonic_basis) == 42
        ),
        "basis_is_orthonormal": (
            max_abs(gram - np.eye(42)) < TOLERANCE
        ),
        "basis_reconstructs_harmonic_projector": (
            max_abs(
                reconstructed_projector - p_harmonic
            )
            < TOLERANCE
        ),
        "all_modes_are_divergence_free": (
            max_abs(divergence) < TOLERANCE
        ),
        "all_modes_are_curl_free": (
            max_abs(curl) < TOLERANCE
        ),
        "all_modes_are_laplacian_zero": (
            max_abs(laplacian) < TOLERANCE
        ),
        "harmonic_projector_fixes_basis": (
            max_abs(projector_fix) < TOLERANCE
        ),
        "all_mode_norms_are_one": all(
            abs(row["norm"] - 1.0) < TOLERANCE
            for row in mode_rows
        ),
        "deterministic_sign_rule_satisfied": all(
            row["pivot_value"] >= 0.0
            for row in mode_rows
        ),
    }

    audit_pass = all(checks.values())

    residuals = {
        "gram_identity_max_abs": max_abs(
            gram - np.eye(42)
        ),
        "projector_reconstruction_max_abs": max_abs(
            reconstructed_projector - p_harmonic
        ),
        "divergence_max_abs": max_abs(divergence),
        "curl_max_abs": max_abs(curl),
        "laplacian_max_abs": max_abs(laplacian),
        "projector_fix_max_abs": max_abs(projector_fix),
    }

    payload = {
        "artifact_id": (
            "native_g60_harmonic_1_form_basis_004"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_harmonic_1_form_basis_exported"
            if audit_pass
            else "native_g60_harmonic_1_form_basis_failed"
        ),
        "inputs": {
            "hodge_decomposition": str(
                DECOMPOSITION_PATH.relative_to(ROOT)
            ),
            "harmonic_projector": str(
                PROJECTOR_PATH.relative_to(ROOT)
            ),
            "b1": str(B1_PATH.relative_to(ROOT)),
            "b2": str(B2_PATH.relative_to(ROOT)),
        },
        "construction": {
            "basis_source": (
                "eigenvectors of P_harmonic with eigenvalue > 0.5"
            ),
            "basis_shape": [120, 42],
            "orthonormal": True,
            "deterministic_sign_rule": (
                "largest-absolute-value coordinate is nonnegative"
            ),
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "residuals": residuals,
        "harmonic_space": {
            "dimension": 42,
            "ambient_edge_dimension": 120,
            "conditions": [
                "B1 H = 0",
                "B2^T H = 0",
                "Delta1 H = 0",
                "H^T H = I42",
                "H H^T = P_harmonic"
            ],
        },
        "mode_audit": {
            "mode_count": len(mode_rows),
            "rows": mode_rows,
        },
        "outputs": {
            "basis_csv": str(
                BASIS_CSV_OUT.relative_to(ROOT)
            ),
            "basis_npz": str(
                BASIS_NPZ_OUT.relative_to(ROOT)
            ),
            "mode_audit_csv": str(
                MODE_AUDIT_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "harmonic_basis_exported": audit_pass,
            "basis_is_unique": False,
            "harmonic_subspace_is_unique": True,
            "native_group_action_on_basis_audited": False,
            "representation_decomposition_computed": False,
            "physical_mode_interpretation": False,
            "frequency_claim": False,
            "energy_claim": False,
            "maxwell_claim": False,
            "physical_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False,
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

    write_basis_csv(
        BASIS_CSV_OUT,
        harmonic_basis,
    )

    np.savez_compressed(
        BASIS_NPZ_OUT,
        H=harmonic_basis,
        P_harmonic=p_harmonic,
        gram=gram,
        tolerance=np.array([TOLERANCE]),
    )

    with MODE_AUDIT_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(mode_rows[0]),
        )

        writer.writeheader()
        writer.writerows(mode_rows)

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print("basis_shape:", list(harmonic_basis.shape))
    print(
        "basis_rank:",
        numerical_rank(harmonic_basis),
    )
    print(
        "gram_identity_max_abs:",
        residuals["gram_identity_max_abs"],
    )
    print(
        "projector_reconstruction_max_abs:",
        residuals[
            "projector_reconstruction_max_abs"
        ],
    )
    print(
        "divergence_max_abs:",
        residuals["divergence_max_abs"],
    )
    print(
        "curl_max_abs:",
        residuals["curl_max_abs"],
    )
    print(
        "laplacian_max_abs:",
        residuals["laplacian_max_abs"],
    )
    print(
        "projector_fix_max_abs:",
        residuals["projector_fix_max_abs"],
    )
    print(
        "mode_support_range:",
        min(
            row["support_above_1e_12"]
            for row in mode_rows
        ),
        max(
            row["support_above_1e_12"]
            for row in mode_rows
        ),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", BASIS_CSV_OUT)
    print("wrote:", BASIS_NPZ_OUT)
    print("wrote:", MODE_AUDIT_OUT)


if __name__ == "__main__":
    main()
