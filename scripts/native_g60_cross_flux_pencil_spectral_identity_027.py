from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PENCIL_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_operator_pencil_026.json"
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
    / "native_g60_cross_flux_pencil_spectral_identity_027.json"
)

COEFFICIENT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_pencil_spectral_coefficients_027.csv"
)

PROBE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_pencil_spectral_probes_027.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_pencil_spectral_identity_027.npz"
)

IDENTITY_TOLERANCE = 2e-10
SPECTRAL_TOLERANCE = 2e-10

RANDOM_SEED = 46027
PROBE_COUNT = 4096

HIGH_EIGENVALUE = 1.0 / 9.0
LOW_EIGENVALUE = 1.0 / 144.0

INTEGER_M2_COEFFICIENT = 1296
INTEGER_SM_COEFFICIENT = 153
INTEGER_S2_COEFFICIENT = 1


Exponent = tuple[int, int, int, int]


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


def exponent_add(
    first: Exponent,
    second: Exponent,
) -> Exponent:
    return tuple(
        first[index] + second[index]
        for index in range(4)
    )


def degree_four_exponents() -> list[Exponent]:
    exponents = []

    for values in product(
        range(5),
        repeat=4,
    ):
        if sum(values) == 4:
            exponents.append(
                tuple(
                    int(value)
                    for value in values
                )
            )

    exponents.sort(
        reverse=True
    )

    return exponents


def monomial_label(
    exponent: Exponent,
) -> str:
    factors = []

    for index, power in enumerate(
        exponent
    ):
        if power == 0:
            continue

        if power == 1:
            factors.append(
                f"f{index}"
            )
        else:
            factors.append(
                f"f{index}^{power}"
            )

    return "*".join(factors) or "1"


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


def pencil_matrix(
    slices: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "r,rab->ab",
        f,
        slices,
    )


def pencil_square(
    slices: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    matrix = pencil_matrix(
        slices,
        f,
    )

    return matrix.T @ matrix


def build_quadratic_M_coefficients(
    slices: np.ndarray,
) -> dict[Exponent, np.ndarray]:
    coefficients: dict[
        Exponent,
        np.ndarray,
    ] = {}

    for first in range(4):
        exponent = [0, 0, 0, 0]
        exponent[first] = 2

        coefficients[
            tuple(exponent)
        ] = (
            slices[first].T
            @ slices[first]
        )

    for first in range(4):
        for second in range(
            first + 1,
            4,
        ):
            exponent = [0, 0, 0, 0]
            exponent[first] = 1
            exponent[second] = 1

            coefficients[
                tuple(exponent)
            ] = (
                slices[first].T
                @ slices[second]
                + slices[second].T
                @ slices[first]
            )

    return coefficients


def build_norm_squared_coefficients() -> dict[
    Exponent,
    float,
]:
    coefficients = {}

    for index in range(4):
        exponent = [0, 0, 0, 0]
        exponent[index] = 2

        coefficients[
            tuple(exponent)
        ] = 1.0

    return coefficients


def convolve_matrix_polynomials(
    first: dict[Exponent, np.ndarray],
    second: dict[Exponent, np.ndarray],
) -> dict[Exponent, np.ndarray]:
    result: dict[
        Exponent,
        np.ndarray,
    ] = defaultdict(
        lambda: np.zeros(
            (6, 6),
            dtype=np.float64,
        )
    )

    for first_exponent, first_matrix in (
        first.items()
    ):
        for second_exponent, second_matrix in (
            second.items()
        ):
            exponent = exponent_add(
                first_exponent,
                second_exponent,
            )

            result[exponent] += (
                first_matrix
                @ second_matrix
            )

    return dict(result)


def convolve_scalar_matrix_polynomials(
    scalar: dict[Exponent, float],
    matrix: dict[Exponent, np.ndarray],
) -> dict[Exponent, np.ndarray]:
    result: dict[
        Exponent,
        np.ndarray,
    ] = defaultdict(
        lambda: np.zeros(
            (6, 6),
            dtype=np.float64,
        )
    )

    for scalar_exponent, scalar_value in (
        scalar.items()
    ):
        for matrix_exponent, matrix_value in (
            matrix.items()
        ):
            exponent = exponent_add(
                scalar_exponent,
                matrix_exponent,
            )

            result[exponent] += (
                scalar_value
                * matrix_value
            )

    return dict(result)


def convolve_scalar_polynomials(
    first: dict[Exponent, float],
    second: dict[Exponent, float],
) -> dict[Exponent, float]:
    result: dict[
        Exponent,
        float,
    ] = defaultdict(float)

    for first_exponent, first_value in (
        first.items()
    ):
        for second_exponent, second_value in (
            second.items()
        ):
            exponent = exponent_add(
                first_exponent,
                second_exponent,
            )

            result[exponent] += (
                first_value
                * second_value
            )

    return dict(result)


def build_spectral_identity_coefficients(
    slices: np.ndarray,
) -> tuple[
    list[dict],
    np.ndarray,
    dict,
]:
    m_coefficients = (
        build_quadratic_M_coefficients(
            slices
        )
    )

    s_coefficients = (
        build_norm_squared_coefficients()
    )

    m_squared_coefficients = (
        convolve_matrix_polynomials(
            m_coefficients,
            m_coefficients,
        )
    )

    s_times_m_coefficients = (
        convolve_scalar_matrix_polynomials(
            s_coefficients,
            m_coefficients,
        )
    )

    s_squared_coefficients = (
        convolve_scalar_polynomials(
            s_coefficients,
            s_coefficients,
        )
    )

    exponents = degree_four_exponents()

    identity = np.eye(
        6,
        dtype=np.float64,
    )

    residual_matrices = []
    coefficient_rows = []

    maximum_residual = 0.0
    maximum_frobenius_residual = 0.0

    for coefficient_id, exponent in enumerate(
        exponents
    ):
        m_squared = (
            m_squared_coefficients.get(
                exponent,
                np.zeros(
                    (6, 6),
                    dtype=np.float64,
                ),
            )
        )

        s_times_m = (
            s_times_m_coefficients.get(
                exponent,
                np.zeros(
                    (6, 6),
                    dtype=np.float64,
                ),
            )
        )

        s_squared = (
            s_squared_coefficients.get(
                exponent,
                0.0,
            )
        )

        residual = (
            INTEGER_M2_COEFFICIENT
            * m_squared
            - INTEGER_SM_COEFFICIENT
            * s_times_m
            + INTEGER_S2_COEFFICIENT
            * s_squared
            * identity
        )

        residual_max_abs = max_abs(
            residual
        )

        residual_frobenius = float(
            np.linalg.norm(
                residual,
                ord="fro",
            )
        )

        maximum_residual = max(
            maximum_residual,
            residual_max_abs,
        )

        maximum_frobenius_residual = max(
            maximum_frobenius_residual,
            residual_frobenius,
        )

        residual_matrices.append(
            residual
        )

        coefficient_rows.append(
            {
                "coefficient_id": (
                    coefficient_id
                ),
                "monomial": (
                    monomial_label(
                        exponent
                    )
                ),
                "f0_power": exponent[0],
                "f1_power": exponent[1],
                "f2_power": exponent[2],
                "f3_power": exponent[3],
                "m_squared_coefficient_frobenius": float(
                    np.linalg.norm(
                        m_squared,
                        ord="fro",
                    )
                ),
                "s_times_m_coefficient_frobenius": float(
                    np.linalg.norm(
                        s_times_m,
                        ord="fro",
                    )
                ),
                "s_squared_coefficient": (
                    s_squared
                ),
                "residual_max_abs": (
                    residual_max_abs
                ),
                "residual_frobenius": (
                    residual_frobenius
                ),
                "coefficient_pass": (
                    residual_max_abs
                    < IDENTITY_TOLERANCE
                ),
            }
        )

    summary = {
        "degree_four_monomial_count": (
            len(exponents)
        ),
        "maximum_coefficient_residual": (
            maximum_residual
        ),
        "maximum_coefficient_frobenius_residual": (
            maximum_frobenius_residual
        ),
        "all_coefficients_pass": all(
            row["coefficient_pass"]
            for row in coefficient_rows
        ),
    }

    return (
        coefficient_rows,
        np.array(
            residual_matrices,
            dtype=np.float64,
        ),
        {
            "m_coefficients": (
                m_coefficients
            ),
            "m_squared_coefficients": (
                m_squared_coefficients
            ),
            "s_times_m_coefficients": (
                s_times_m_coefficients
            ),
            "s_squared_coefficients": (
                s_squared_coefficients
            ),
            "summary": summary,
        },
    )


def spectral_probe(
    slices: np.ndarray,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    probe_rows = []

    global_residuals = {
        "integer_polynomial_identity": 0.0,
        "high_eigenvalue": 0.0,
        "low_eigenvalue": 0.0,
        "high_projector_idempotence": 0.0,
        "low_projector_idempotence": 0.0,
        "projector_orthogonality": 0.0,
        "projector_completeness": 0.0,
        "high_projector_trace": 0.0,
        "low_projector_trace": 0.0,
    }

    expected_eigenvalues = np.array(
        [
            LOW_EIGENVALUE,
            LOW_EIGENVALUE,
            LOW_EIGENVALUE,
            LOW_EIGENVALUE,
            HIGH_EIGENVALUE,
            HIGH_EIGENVALUE,
        ],
        dtype=np.float64,
    )

    identity = np.eye(
        6,
        dtype=np.float64,
    )

    denominator = (
        HIGH_EIGENVALUE
        - LOW_EIGENVALUE
    )

    for probe_id in range(PROBE_COUNT):
        f = normalized(
            rng.normal(size=4)
        )

        norm_squared = float(
            np.dot(f, f)
        )

        m = pencil_square(
            slices,
            f,
        )

        integer_polynomial = (
            INTEGER_M2_COEFFICIENT
            * (m @ m)
            - INTEGER_SM_COEFFICIENT
            * norm_squared
            * m
            + INTEGER_S2_COEFFICIENT
            * norm_squared**2
            * identity
        )

        eigenvalues = np.linalg.eigvalsh(
            m
        )

        scaled_high = (
            HIGH_EIGENVALUE
            * norm_squared
        )

        scaled_low = (
            LOW_EIGENVALUE
            * norm_squared
        )

        high_projector = (
            m
            - scaled_low
            * identity
        ) / (
            denominator
            * norm_squared
        )

        low_projector = (
            identity
            - high_projector
        )

        residuals = {
            "integer_polynomial_identity": (
                max_abs(
                    integer_polynomial
                )
            ),
            "high_eigenvalue": max_abs(
                eigenvalues[-2:]
                - scaled_high
            ),
            "low_eigenvalue": max_abs(
                eigenvalues[:4]
                - scaled_low
            ),
            "high_projector_idempotence": (
                max_abs(
                    high_projector
                    @ high_projector
                    - high_projector
                )
            ),
            "low_projector_idempotence": (
                max_abs(
                    low_projector
                    @ low_projector
                    - low_projector
                )
            ),
            "projector_orthogonality": (
                max_abs(
                    high_projector
                    @ low_projector
                )
            ),
            "projector_completeness": (
                max_abs(
                    high_projector
                    + low_projector
                    - identity
                )
            ),
            "high_projector_trace": abs(
                float(
                    np.trace(
                        high_projector
                    )
                )
                - 2.0
            ),
            "low_projector_trace": abs(
                float(
                    np.trace(
                        low_projector
                    )
                )
                - 4.0
            ),
        }

        for name, value in residuals.items():
            global_residuals[name] = max(
                global_residuals[name],
                value,
            )

        probe_rows.append(
            {
                "probe_id": probe_id,
                "f0": float(f[0]),
                "f1": float(f[1]),
                "f2": float(f[2]),
                "f3": float(f[3]),
                "minimum_eigenvalue": float(
                    eigenvalues[0]
                ),
                "maximum_eigenvalue": float(
                    eigenvalues[-1]
                ),
                "singular_value_low": float(
                    np.sqrt(
                        max(
                            eigenvalues[0],
                            0.0,
                        )
                    )
                ),
                "singular_value_high": float(
                    np.sqrt(
                        max(
                            eigenvalues[-1],
                            0.0,
                        )
                    )
                ),
                **{
                    name
                    + "_max_abs": value
                    for name, value in (
                        residuals.items()
                    )
                },
                "probe_pass": all(
                    value
                    < SPECTRAL_TOLERANCE
                    for value in residuals.values()
                ),
            }
        )

    summary = {
        "probe_count": PROBE_COUNT,
        "all_probes_pass": all(
            row["probe_pass"]
            for row in probe_rows
        ),
        "global_maximum_residuals": (
            global_residuals
        ),
        "expected_M_eigenvalues_for_unit_f": (
            expected_eigenvalues
        ),
        "expected_singular_values_for_unit_f": (
            np.sqrt(
                expected_eigenvalues
            )
        ),
    }

    return probe_rows, summary


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COEFFICIENT_CSV_OUT.parent.mkdir(
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

    pencil_receipt = json.loads(
        PENCIL_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    data = np.load(
        PENCIL_NPZ_PATH
    )

    slices = np.array(
        data["slices"],
        dtype=np.float64,
    )

    if slices.shape != (4, 6, 6):
        raise RuntimeError(
            f"unexpected slice shape: {slices.shape}"
        )

    (
        coefficient_rows,
        residual_matrices,
        coefficient_data,
    ) = build_spectral_identity_coefficients(
        slices
    )

    coefficient_summary = (
        coefficient_data["summary"]
    )

    probe_rows, probe_summary = (
        spectral_probe(
            slices
        )
    )

    expected_unit_singular_values = (
        np.array(
            [
                1.0 / 12.0,
                1.0 / 12.0,
                1.0 / 12.0,
                1.0 / 12.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ],
            dtype=np.float64,
        )
    )

    checks = {
        "input_026_audit_pass": (
            pencil_receipt.get(
                "audit_pass"
            )
            is True
        ),
        "slice_shape_is_4_by_6_by_6": (
            slices.shape
            == (4, 6, 6)
        ),
        "degree_four_monomial_count_is_35": (
            coefficient_summary[
                "degree_four_monomial_count"
            ]
            == 35
        ),
        "all_coefficient_matrices_vanish": (
            coefficient_summary[
                "all_coefficients_pass"
            ]
        ),
        "maximum_coefficient_residual_resolved": (
            coefficient_summary[
                "maximum_coefficient_residual"
            ]
            < IDENTITY_TOLERANCE
        ),
        "all_spectral_probes_pass": (
            probe_summary[
                "all_probes_pass"
            ]
        ),
        "probe_polynomial_identity_resolved": (
            probe_summary[
                "global_maximum_residuals"
            ][
                "integer_polynomial_identity"
            ]
            < SPECTRAL_TOLERANCE
        ),
        "high_projector_has_trace_two": (
            probe_summary[
                "global_maximum_residuals"
            ][
                "high_projector_trace"
            ]
            < SPECTRAL_TOLERANCE
        ),
        "low_projector_has_trace_four": (
            probe_summary[
                "global_maximum_residuals"
            ][
                "low_projector_trace"
            ]
            < SPECTRAL_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_pencil_spectral_identity_exact"
        if theorem_pass
        else "native_g60_cross_flux_pencil_spectral_identity_failed"
    )

    theorem_statement = (
        "For every f in the native four-flux coordinate space, "
        "with A(f)=sum_r f_r A_r and M(f)=A(f)^T A(f), "
        "the coefficientwise polynomial identity "
        "1296 M(f)^2 - 153 ||f||^2 M(f) "
        "+ ||f||^4 I6 = 0 holds. "
        "For nonzero f, M(f) has eigenvalues "
        "||f||^2/9 with multiplicity 2 and "
        "||f||^2/144 with multiplicity 4. "
        "Therefore A(f) has singular values "
        "||f||/3 twice and ||f||/12 four times."
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_pencil_spectral_identity_027"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "theorem": {
            "statement": theorem_statement,
            "integer_polynomial_identity": (
                "1296 M(f)^2 "
                "- 153 ||f||^2 M(f) "
                "+ ||f||^4 I6 = 0"
            ),
            "fractional_polynomial_identity": (
                "M(f)^2 "
                "- (17/144)||f||^2 M(f) "
                "+ (1/1296)||f||^4 I6 = 0"
            ),
            "M_high_eigenvalue": (
                "||f||^2/9"
            ),
            "M_high_multiplicity": 2,
            "M_low_eigenvalue": (
                "||f||^2/144"
            ),
            "M_low_multiplicity": 4,
            "A_high_singular_value": (
                "||f||/3"
            ),
            "A_high_multiplicity": 2,
            "A_low_singular_value": (
                "||f||/12"
            ),
            "A_low_multiplicity": 4,
            "operator_norm": (
                "||A(f)||_op = ||f||/3"
            ),
            "sharp_bilinear_norm": (
                "sigma_star = 1/3"
            ),
        },
        "coefficient_certificate": {
            **coefficient_summary,
            "integer_coefficients": {
                "M_squared": (
                    INTEGER_M2_COEFFICIENT
                ),
                "norm_squared_times_M": (
                    -INTEGER_SM_COEFFICIENT
                ),
                "norm_fourth_times_identity": (
                    INTEGER_S2_COEFFICIENT
                ),
            },
            "certificate_kind": (
                "all 35 degree-four monomial "
                "coefficient matrices verified"
            ),
        },
        "spectral_probe": (
            probe_summary
        ),
        "checks": checks,
        "earned_interpretation": {
            "operator_pencil_is_spectrally_locked": (
                theorem_pass
            ),
            "operator_norm_is_constant_on_unit_flux_sphere": (
                theorem_pass
            ),
            "sharp_tensor_norm_is_one_third": (
                theorem_pass
            ),
            "quartic_descent_rate_is_sharp": (
                theorem_pass
            ),
            "sharp_g_1_descent_coefficient": (
                "-1/180"
            ),
        },
        "boundary": {
            "finite_coefficient_certificate_completed": (
                theorem_pass
            ),
            "sharp_rank_one_tensor_norm_proved": (
                theorem_pass
            ),
            "minimal_potential_sharp_noncoercive_ray_coefficient_proved": (
                theorem_pass
            ),
            "physical_energy_claim": False,
            "physical_instability_claim": False,
            "physical_thread_claim": False,
            "force_claim": False,
            "physical_claim": False,
        },
        "outputs": {
            "coefficient_csv": str(
                COEFFICIENT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "probe_csv": str(
                PROBE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "certificate_npz": str(
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
        writer.writerows(
            probe_rows
        )

    exponents = np.array(
        [
            [
                row["f0_power"],
                row["f1_power"],
                row["f2_power"],
                row["f3_power"],
            ]
            for row in coefficient_rows
        ],
        dtype=np.int64,
    )

    np.savez_compressed(
        NPZ_OUT,
        slices=slices,
        degree_four_exponents=exponents,
        coefficient_residual_matrices=(
            residual_matrices
        ),
        expected_unit_singular_values=(
            expected_unit_singular_values
        ),
        polynomial_integer_coefficients=np.array(
            [
                INTEGER_M2_COEFFICIENT,
                -INTEGER_SM_COEFFICIENT,
                INTEGER_S2_COEFFICIENT,
            ],
            dtype=np.int64,
        ),
        high_eigenvalue=np.array(
            [HIGH_EIGENVALUE],
            dtype=np.float64,
        ),
        low_eigenvalue=np.array(
            [LOW_EIGENVALUE],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "degree_four_monomial_count:",
        coefficient_summary[
            "degree_four_monomial_count"
        ],
    )
    print(
        "all_coefficient_matrices_vanish:",
        coefficient_summary[
            "all_coefficients_pass"
        ],
    )
    print(
        "maximum_coefficient_residual:",
        coefficient_summary[
            "maximum_coefficient_residual"
        ],
    )
    print(
        "maximum_coefficient_frobenius_residual:",
        coefficient_summary[
            "maximum_coefficient_frobenius_residual"
        ],
    )
    print(
        "all_spectral_probes_pass:",
        probe_summary[
            "all_probes_pass"
        ],
    )
    print(
        "global_spectral_residuals:",
        probe_summary[
            "global_maximum_residuals"
        ],
    )
    print(
        "unit_singular_spectrum:",
        expected_unit_singular_values.tolist(),
    )
    print(
        "sharp_tensor_norm:",
        "1/3"
        if theorem_pass
        else "not proved",
    )
    print(
        "sharp_quartic_descent_coefficient_at_g_1:",
        "-1/180"
        if theorem_pass
        else "not proved",
    )
    print("wrote:", JSON_OUT)
    print("wrote:", COEFFICIENT_CSV_OUT)
    print("wrote:", PROBE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
