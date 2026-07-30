from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = ROOT / "data" / "imported" / "project45"

IMPORT_RECEIPT_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cochain_complex_import_001.json"
)

B1_PATH = SOURCE_ROOT / "native_g60_B1_vertex_edge_004.csv"
B2_PATH = SOURCE_ROOT / "native_g60_B2_edge_face_004.csv"

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_uniform_hodge_baseline_002.json"
)

DELTA0_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_Delta0_uniform_002.csv"
)

DELTA1_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_Delta1_uniform_002.csv"
)

DELTA2_OUT = (
    ROOT
    / "artifacts"
    / "matrices"
    / "native_g60_Delta2_uniform_002.csv"
)

SPECTRUM_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_uniform_hodge_spectra_002.csv"
)


def read_matrix_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    return np.array(
        [
            [int(value) for value in row[1:]]
            for row in rows[1:]
        ],
        dtype=np.int64,
    )


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    row_prefix: str,
    column_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["row_id"]
            + [
                f"{column_prefix}{index:03d}"
                for index in range(matrix.shape[1])
            ]
        )

        for row_index, row in enumerate(matrix):
            writer.writerow(
                [f"{row_prefix}{row_index:03d}"]
                + [int(value) for value in row]
            )


def exact_rank(matrix: np.ndarray) -> int:
    singular_values = np.linalg.svd(
        matrix.astype(np.float64),
        compute_uv=False,
    )

    if singular_values.size == 0:
        return 0

    tolerance = (
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * singular_values[0]
    )

    return int(np.count_nonzero(singular_values > tolerance))


def spectrum_summary(
    matrix: np.ndarray,
    tolerance: float = 1e-9,
) -> dict:
    eigenvalues = np.linalg.eigvalsh(
        matrix.astype(np.float64)
    )

    eigenvalues[np.abs(eigenvalues) < tolerance] = 0.0

    zero_count = int(
        np.count_nonzero(np.abs(eigenvalues) <= tolerance)
    )

    negative_count = int(
        np.count_nonzero(eigenvalues < -tolerance)
    )

    positive = eigenvalues[eigenvalues > tolerance]

    rounded_profile = Counter(
        round(float(value), 10)
        for value in eigenvalues
    )

    return {
        "dimension": int(matrix.shape[0]),
        "zero_eigenvalue_multiplicity": zero_count,
        "positive_eigenvalue_count": int(len(positive)),
        "negative_eigenvalue_count": negative_count,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "smallest_positive_eigenvalue": (
            float(positive[0])
            if len(positive)
            else None
        ),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "trace": float(np.sum(eigenvalues)),
        "rounded_eigenvalue_profile": {
            str(value): count
            for value, count in sorted(
                rounded_profile.items()
            )
        },
        "eigenvalues": [
            float(value)
            for value in eigenvalues
        ],
    }


def frobenius_inner(
    left: np.ndarray,
    right: np.ndarray,
) -> int:
    return int(np.sum(left * right))


def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPECTRUM_OUT.parent.mkdir(parents=True, exist_ok=True)

    receipt = json.loads(
        IMPORT_RECEIPT_PATH.read_text(encoding="utf-8")
    )

    b1 = read_matrix_csv(B1_PATH)
    b2 = read_matrix_csv(B2_PATH)

    d0 = b1.T
    d1 = b2.T

    exact_term = d0 @ d0.T
    coexact_term = d1.T @ d1

    delta0 = b1 @ b1.T
    delta1 = exact_term + coexact_term
    delta2 = b2.T @ b2

    ranks = {
        "delta0": exact_rank(delta0),
        "delta1": exact_rank(delta1),
        "delta2": exact_rank(delta2),
        "delta1_exact_term": exact_rank(exact_term),
        "delta1_coexact_term": exact_rank(coexact_term),
    }

    kernels = {
        "delta0": 60 - ranks["delta0"],
        "delta1": 120 - ranks["delta1"],
        "delta2": 20 - ranks["delta2"],
    }

    spectra = {
        "delta0": spectrum_summary(delta0),
        "delta1": spectrum_summary(delta1),
        "delta2": spectrum_summary(delta2),
    }

    exact_coexact_product = exact_term @ coexact_term
    coexact_exact_product = coexact_term @ exact_term

    checks = {
        "input_import_receipt_pass": (
            receipt.get("audit_pass") is True
        ),
        "b1_shape_is_60_by_120": (
            b1.shape == (60, 120)
        ),
        "b2_shape_is_120_by_20": (
            b2.shape == (120, 20)
        ),
        "d0_shape_is_120_by_60": (
            d0.shape == (120, 60)
        ),
        "d1_shape_is_20_by_120": (
            d1.shape == (20, 120)
        ),
        "delta0_shape_is_60_by_60": (
            delta0.shape == (60, 60)
        ),
        "delta1_shape_is_120_by_120": (
            delta1.shape == (120, 120)
        ),
        "delta2_shape_is_20_by_20": (
            delta2.shape == (20, 20)
        ),
        "delta0_is_symmetric": np.array_equal(
            delta0,
            delta0.T,
        ),
        "delta1_is_symmetric": np.array_equal(
            delta1,
            delta1.T,
        ),
        "delta2_is_symmetric": np.array_equal(
            delta2,
            delta2.T,
        ),
        "delta0_rank_is_59": ranks["delta0"] == 59,
        "delta1_rank_is_78": ranks["delta1"] == 78,
        "delta2_rank_is_19": ranks["delta2"] == 19,
        "delta0_kernel_dimension_is_1": (
            kernels["delta0"] == 1
        ),
        "delta1_kernel_dimension_is_42": (
            kernels["delta1"] == 42
        ),
        "delta2_kernel_dimension_is_1": (
            kernels["delta2"] == 1
        ),
        "exact_sector_rank_is_59": (
            ranks["delta1_exact_term"] == 59
        ),
        "coexact_sector_rank_is_19": (
            ranks["delta1_coexact_term"] == 19
        ),
        "exact_and_coexact_terms_multiply_to_zero": (
            not np.any(exact_coexact_product)
            and not np.any(coexact_exact_product)
        ),
        "exact_and_coexact_terms_are_frobenius_orthogonal": (
            frobenius_inner(
                exact_term,
                coexact_term,
            )
            == 0
        ),
        "delta0_positive_semidefinite": (
            spectra["delta0"][
                "negative_eigenvalue_count"
            ]
            == 0
        ),
        "delta1_positive_semidefinite": (
            spectra["delta1"][
                "negative_eigenvalue_count"
            ]
            == 0
        ),
        "delta2_positive_semidefinite": (
            spectra["delta2"][
                "negative_eigenvalue_count"
            ]
            == 0
        ),
        "delta0_spectral_kernel_is_1": (
            spectra["delta0"][
                "zero_eigenvalue_multiplicity"
            ]
            == 1
        ),
        "delta1_spectral_kernel_is_42": (
            spectra["delta1"][
                "zero_eigenvalue_multiplicity"
            ]
            == 42
        ),
        "delta2_spectral_kernel_is_1": (
            spectra["delta2"][
                "zero_eigenvalue_multiplicity"
            ]
            == 1
        ),
        "dimension_partition_is_59_42_19": (
            ranks["delta1_exact_term"]
            + kernels["delta1"]
            + ranks["delta1_coexact_term"]
            == 120
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_uniform_hodge_baseline_002"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "native_g60_uniform_hodge_baseline_constructed"
            if audit_pass
            else "native_g60_uniform_hodge_baseline_failed"
        ),
        "inputs": {
            "cochain_import_receipt": str(
                IMPORT_RECEIPT_PATH.relative_to(ROOT)
            ),
            "b1": str(B1_PATH.relative_to(ROOT)),
            "b2": str(B2_PATH.relative_to(ROOT)),
        },
        "hodge_weights": {
            "star0": "I60",
            "star1": "I120",
            "star2": "I20",
            "weight_status": (
                "uniform_positive_combinatorial_baseline"
            ),
        },
        "operators": {
            "d0": "B1^T",
            "d1": "B2^T",
            "delta1": "d0^T",
            "delta2": "d1^T",
            "Delta0": "B1 B1^T",
            "Delta1": (
                "B1^T B1 + B2 B2^T"
            ),
            "Delta2": "B2^T B2",
        },
        "checks": checks,
        "ranks": ranks,
        "kernel_dimensions": kernels,
        "edge_space_partition": {
            "exact_dimension": (
                ranks["delta1_exact_term"]
            ),
            "harmonic_dimension": kernels["delta1"],
            "coexact_dimension": (
                ranks["delta1_coexact_term"]
            ),
            "total_dimension": 120,
            "dimension_sum": (
                ranks["delta1_exact_term"]
                + kernels["delta1"]
                + ranks["delta1_coexact_term"]
            ),
        },
        "orthogonality": {
            "exact_coexact_product_nonzero_count": int(
                np.count_nonzero(
                    exact_coexact_product
                )
            ),
            "coexact_exact_product_nonzero_count": int(
                np.count_nonzero(
                    coexact_exact_product
                )
            ),
            "frobenius_inner_product": (
                frobenius_inner(
                    exact_term,
                    coexact_term,
                )
            ),
        },
        "spectra": spectra,
        "outputs": {
            "delta0": str(
                DELTA0_OUT.relative_to(ROOT)
            ),
            "delta1": str(
                DELTA1_OUT.relative_to(ROOT)
            ),
            "delta2": str(
                DELTA2_OUT.relative_to(ROOT)
            ),
            "spectrum_csv": str(
                SPECTRUM_OUT.relative_to(ROOT)
            ),
        },
        "boundary": {
            "uniform_hodge_baseline_constructed": (
                audit_pass
            ),
            "hodge_decomposition_dimension_identity_verified": (
                audit_pass
            ),
            "explicit_hodge_projectors_constructed": False,
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

    write_matrix_csv(
        DELTA0_OUT,
        delta0,
        "v",
        "v",
    )

    write_matrix_csv(
        DELTA1_OUT,
        delta1,
        "e",
        "e",
    )

    write_matrix_csv(
        DELTA2_OUT,
        delta2,
        "f",
        "f",
    )

    with SPECTRUM_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "operator",
                "eigenvalue_index",
                "eigenvalue",
                "is_zero",
            ]
        )

        for operator_name, summary in spectra.items():
            for index, value in enumerate(
                summary["eigenvalues"]
            ):
                writer.writerow(
                    [
                        operator_name,
                        index,
                        f"{value:.16g}",
                        abs(value) <= 1e-9,
                    ]
                )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "Delta0_rank/kernel:",
        ranks["delta0"],
        kernels["delta0"],
    )
    print(
        "Delta1_rank/kernel:",
        ranks["delta1"],
        kernels["delta1"],
    )
    print(
        "Delta2_rank/kernel:",
        ranks["delta2"],
        kernels["delta2"],
    )
    print(
        "edge_partition_exact_harmonic_coexact:",
        payload["edge_space_partition"][
            "exact_dimension"
        ],
        payload["edge_space_partition"][
            "harmonic_dimension"
        ],
        payload["edge_space_partition"][
            "coexact_dimension"
        ],
    )
    print(
        "exact_coexact_products_nonzero:",
        payload["orthogonality"][
            "exact_coexact_product_nonzero_count"
        ],
        payload["orthogonality"][
            "coexact_exact_product_nonzero_count"
        ],
    )
    print(
        "frobenius_inner_product:",
        payload["orthogonality"][
            "frobenius_inner_product"
        ],
    )

    for name in ("delta0", "delta1", "delta2"):
        summary = spectra[name]

        print(
            name,
            "negative/zero/positive:",
            summary["negative_eigenvalue_count"],
            summary["zero_eigenvalue_multiplicity"],
            summary["positive_eigenvalue_count"],
        )
        print(
            name,
            "smallest_positive/max:",
            summary["smallest_positive_eigenvalue"],
            summary["maximum_eigenvalue"],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", DELTA0_OUT)
    print("wrote:", DELTA1_OUT)
    print("wrote:", DELTA2_OUT)
    print("wrote:", SPECTRUM_OUT)


if __name__ == "__main__":
    main()
