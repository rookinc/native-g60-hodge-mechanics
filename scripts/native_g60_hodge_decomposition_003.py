from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

BASELINE_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_uniform_hodge_baseline_002.json"
)

B1_PATH = SOURCE_ROOT / "native_g60_B1_vertex_edge_004.csv"
B2_PATH = SOURCE_ROOT / "native_g60_B2_edge_face_004.csv"

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_hodge_decomposition_003.json"
)

P_EXACT_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_P_exact_003.csv"
)

P_HARMONIC_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_P_harmonic_003.csv"
)

P_COEXACT_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_P_coexact_003.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_hodge_projectors_003.npz"
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


def write_float_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["row_id"]
            + [
                f"{prefix}{index:03d}"
                for index in range(matrix.shape[1])
            ]
        )

        for row_index, row in enumerate(matrix):
            writer.writerow(
                [f"{prefix}{row_index:03d}"]
                + [
                    f"{float(value):.17g}"
                    for value in row
                ]
            )


def spectral_projector(
    operator: np.ndarray,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray, dict]:
    eigenvalues, eigenvectors = np.linalg.eigh(operator)

    positive_mask = eigenvalues > tolerance
    zero_mask = np.abs(eigenvalues) <= tolerance

    basis = eigenvectors[:, positive_mask]
    projector = basis @ basis.T

    projector = 0.5 * (
        projector + projector.T
    )

    return projector, {
        "dimension": int(operator.shape[0]),
        "positive_eigenvalue_count": int(
            np.count_nonzero(positive_mask)
        ),
        "zero_eigenvalue_count": int(
            np.count_nonzero(zero_mask)
        ),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "smallest_positive_eigenvalue": (
            float(eigenvalues[positive_mask][0])
            if np.any(positive_mask)
            else None
        ),
        "maximum_eigenvalue": float(eigenvalues[-1]),
    }


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


def max_abs(
    matrix: np.ndarray,
) -> float:
    return float(np.max(np.abs(matrix)))


def frobenius_norm(
    matrix: np.ndarray,
) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def projector_audit(
    projector: np.ndarray,
) -> dict:
    idempotence_residual = (
        projector @ projector - projector
    )

    symmetry_residual = projector - projector.T

    trace_value = float(np.trace(projector))
    rank_value = numerical_rank(projector)

    eigenvalues = np.linalg.eigvalsh(projector)

    return {
        "rank": rank_value,
        "trace": trace_value,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "idempotence_max_abs": max_abs(
            idempotence_residual
        ),
        "idempotence_frobenius": frobenius_norm(
            idempotence_residual
        ),
        "symmetry_max_abs": max_abs(
            symmetry_residual
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(
        BASELINE_PATH.read_text(encoding="utf-8")
    )

    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)

    d0 = b1.T
    d1 = b2.T

    exact_operator = d0 @ d0.T
    coexact_operator = d1.T @ d1

    p_exact, exact_spectrum = spectral_projector(
        exact_operator
    )

    p_coexact, coexact_spectrum = spectral_projector(
        coexact_operator
    )

    identity = np.eye(120, dtype=np.float64)

    p_harmonic = (
        identity
        - p_exact
        - p_coexact
    )

    p_harmonic = 0.5 * (
        p_harmonic + p_harmonic.T
    )

    projectors = {
        "exact": p_exact,
        "harmonic": p_harmonic,
        "coexact": p_coexact,
    }

    audits = {
        name: projector_audit(projector)
        for name, projector in projectors.items()
    }

    exact_harmonic = p_exact @ p_harmonic
    exact_coexact = p_exact @ p_coexact
    harmonic_coexact = p_harmonic @ p_coexact

    completeness_residual = (
        p_exact
        + p_harmonic
        + p_coexact
        - identity
    )

    harmonic_closed_residual = d1 @ p_harmonic
    harmonic_coclosed_residual = b1 @ p_harmonic

    exact_closed_residual = d1 @ p_exact
    coexact_coclosed_residual = b1 @ p_coexact

    delta1 = exact_operator + coexact_operator
    harmonic_laplacian_residual = (
        delta1 @ p_harmonic
    )

    exact_projection_of_d0 = (
        p_exact @ d0 - d0
    )

    coexact_projection_of_d1t = (
        p_coexact @ d1.T - d1.T
    )

    checks = {
        "input_uniform_hodge_baseline_pass": (
            baseline.get("audit_pass") is True
        ),
        "p_exact_rank_is_59": (
            audits["exact"]["rank"] == 59
        ),
        "p_harmonic_rank_is_42": (
            audits["harmonic"]["rank"] == 42
        ),
        "p_coexact_rank_is_19": (
            audits["coexact"]["rank"] == 19
        ),
        "dimension_sum_is_120": (
            audits["exact"]["rank"]
            + audits["harmonic"]["rank"]
            + audits["coexact"]["rank"]
            == 120
        ),
        "p_exact_is_idempotent": (
            audits["exact"]["idempotence_max_abs"]
            < TOLERANCE
        ),
        "p_harmonic_is_idempotent": (
            audits["harmonic"]["idempotence_max_abs"]
            < TOLERANCE
        ),
        "p_coexact_is_idempotent": (
            audits["coexact"]["idempotence_max_abs"]
            < TOLERANCE
        ),
        "all_projectors_are_symmetric": all(
            audit["symmetry_max_abs"] < TOLERANCE
            for audit in audits.values()
        ),
        "exact_harmonic_product_is_zero": (
            max_abs(exact_harmonic) < TOLERANCE
        ),
        "exact_coexact_product_is_zero": (
            max_abs(exact_coexact) < TOLERANCE
        ),
        "harmonic_coexact_product_is_zero": (
            max_abs(harmonic_coexact) < TOLERANCE
        ),
        "projectors_sum_to_identity": (
            max_abs(completeness_residual)
            < TOLERANCE
        ),
        "harmonic_sector_is_closed": (
            max_abs(harmonic_closed_residual)
            < TOLERANCE
        ),
        "harmonic_sector_is_coclosed": (
            max_abs(harmonic_coclosed_residual)
            < TOLERANCE
        ),
        "harmonic_sector_is_laplacian_kernel": (
            max_abs(harmonic_laplacian_residual)
            < TOLERANCE
        ),
        "exact_sector_is_closed": (
            max_abs(exact_closed_residual)
            < TOLERANCE
        ),
        "coexact_sector_is_coclosed": (
            max_abs(coexact_coclosed_residual)
            < TOLERANCE
        ),
        "p_exact_fixes_image_d0": (
            max_abs(exact_projection_of_d0)
            < TOLERANCE
        ),
        "p_coexact_fixes_image_d1_transpose": (
            max_abs(coexact_projection_of_d1t)
            < TOLERANCE
        ),
        "projector_traces_match_dimensions": (
            abs(audits["exact"]["trace"] - 59)
            < 1e-8
            and abs(
                audits["harmonic"]["trace"] - 42
            )
            < 1e-8
            and abs(
                audits["coexact"]["trace"] - 19
            )
            < 1e-8
        ),
    }

    audit_pass = all(checks.values())

    residuals = {
        "p_exact_idempotence_max_abs": (
            audits["exact"]["idempotence_max_abs"]
        ),
        "p_harmonic_idempotence_max_abs": (
            audits["harmonic"][
                "idempotence_max_abs"
            ]
        ),
        "p_coexact_idempotence_max_abs": (
            audits["coexact"][
                "idempotence_max_abs"
            ]
        ),
        "exact_harmonic_max_abs": (
            max_abs(exact_harmonic)
        ),
        "exact_coexact_max_abs": (
            max_abs(exact_coexact)
        ),
        "harmonic_coexact_max_abs": (
            max_abs(harmonic_coexact)
        ),
        "completeness_max_abs": (
            max_abs(completeness_residual)
        ),
        "harmonic_closed_max_abs": (
            max_abs(harmonic_closed_residual)
        ),
        "harmonic_coclosed_max_abs": (
            max_abs(harmonic_coclosed_residual)
        ),
        "harmonic_laplacian_max_abs": (
            max_abs(harmonic_laplacian_residual)
        ),
        "exact_closed_max_abs": (
            max_abs(exact_closed_residual)
        ),
        "coexact_coclosed_max_abs": (
            max_abs(coexact_coclosed_residual)
        ),
        "p_exact_fixes_d0_max_abs": (
            max_abs(exact_projection_of_d0)
        ),
        "p_coexact_fixes_d1t_max_abs": (
            max_abs(coexact_projection_of_d1t)
        ),
    }

    payload = {
        "artifact_id": (
            "native_g60_hodge_decomposition_003"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_explicit_hodge_decomposition_constructed"
            if audit_pass
            else "native_g60_hodge_decomposition_failed"
        ),
        "inputs": {
            "uniform_hodge_baseline": str(
                BASELINE_PATH.relative_to(ROOT)
            ),
            "b1": str(B1_PATH.relative_to(ROOT)),
            "b2": str(B2_PATH.relative_to(ROOT)),
        },
        "construction": {
            "p_exact": (
                "orthogonal spectral projector onto "
                "image(d0)"
            ),
            "p_coexact": (
                "orthogonal spectral projector onto "
                "image(d1^T)"
            ),
            "p_harmonic": (
                "I120 - p_exact - p_coexact"
            ),
            "numerical_tolerance": TOLERANCE,
            "weight_model": (
                "uniform_identity_inner_products"
            ),
        },
        "checks": checks,
        "projector_audits": audits,
        "source_operator_spectra": {
            "exact_operator": exact_spectrum,
            "coexact_operator": coexact_spectrum,
        },
        "decomposition": {
            "edge_space_dimension": 120,
            "exact_dimension": (
                audits["exact"]["rank"]
            ),
            "harmonic_dimension": (
                audits["harmonic"]["rank"]
            ),
            "coexact_dimension": (
                audits["coexact"]["rank"]
            ),
            "statement": (
                "C1 = im(d0) orthogonal_direct_sum "
                "ker(Delta1) orthogonal_direct_sum "
                "im(d1^T)"
            ),
        },
        "residuals": residuals,
        "outputs": {
            "p_exact_csv": str(
                P_EXACT_OUT.relative_to(ROOT)
            ),
            "p_harmonic_csv": str(
                P_HARMONIC_OUT.relative_to(ROOT)
            ),
            "p_coexact_csv": str(
                P_COEXACT_OUT.relative_to(ROOT)
            ),
            "projector_npz": str(
                NPZ_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "explicit_hodge_projectors_constructed": (
                audit_pass
            ),
            "hodge_decomposition_proved_for_uniform_baseline": (
                audit_pass
            ),
            "harmonic_basis_exported": False,
            "native_group_commutation_audited": False,
            "metric_geometry_derived": False,
            "physical_constitutive_law_derived": False,
            "dynamics_defined": False,
            "maxwell_claim": False,
            "physical_energy_claim": False,
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

    write_float_matrix_csv(
        P_EXACT_OUT,
        p_exact,
        "e",
    )

    write_float_matrix_csv(
        P_HARMONIC_OUT,
        p_harmonic,
        "e",
    )

    write_float_matrix_csv(
        P_COEXACT_OUT,
        p_coexact,
        "e",
    )

    np.savez_compressed(
        NPZ_OUT,
        P_exact=p_exact,
        P_harmonic=p_harmonic,
        P_coexact=p_coexact,
        tolerance=np.array([TOLERANCE]),
    )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "projector_ranks:",
        audits["exact"]["rank"],
        audits["harmonic"]["rank"],
        audits["coexact"]["rank"],
    )
    print(
        "projector_traces:",
        audits["exact"]["trace"],
        audits["harmonic"]["trace"],
        audits["coexact"]["trace"],
    )
    print(
        "idempotence_max_abs:",
        residuals["p_exact_idempotence_max_abs"],
        residuals["p_harmonic_idempotence_max_abs"],
        residuals["p_coexact_idempotence_max_abs"],
    )
    print(
        "cross_products_max_abs:",
        residuals["exact_harmonic_max_abs"],
        residuals["exact_coexact_max_abs"],
        residuals["harmonic_coexact_max_abs"],
    )
    print(
        "completeness_max_abs:",
        residuals["completeness_max_abs"],
    )
    print(
        "harmonic_closed/coclosed/laplacian:",
        residuals["harmonic_closed_max_abs"],
        residuals["harmonic_coclosed_max_abs"],
        residuals["harmonic_laplacian_max_abs"],
    )
    print(
        "exact_closed/coexact_coclosed:",
        residuals["exact_closed_max_abs"],
        residuals["coexact_coclosed_max_abs"],
    )
    print("wrote:", JSON_OUT)
    print("wrote:", P_EXACT_OUT)
    print("wrote:", P_HARMONIC_OUT)
    print("wrote:", P_COEXACT_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
