from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CENSUS_JSON_PATH = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_residual_zero_30_line_census_038.json"
)

CENSUS_NPZ_PATH = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_residual_zero_30_line_census_038.npz"
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
    / "native_g60_cross_flux_zero_orbit_squared_spectra_040.json"
)

ORBIT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_zero_orbit_squared_spectra_040.csv"
)

LINE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_zero_orbit_squared_spectra_lines_040.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_zero_orbit_squared_spectra_040.npz"
)

POLYNOMIAL_TOLERANCE = 2e-9
SPECTRAL_TOLERANCE = 2e-9
TRACE_TOLERANCE = 2e-9
RANK_TOLERANCE = 1e-9


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


def orbit_partition(
    action: np.ndarray,
) -> list[list[int]]:
    unseen = set(
        range(
            action.shape[1]
        )
    )

    orbits = []

    while unseen:
        seed = min(unseen)
        reached = {seed}
        frontier = deque([seed])

        while frontier:
            current = frontier.popleft()

            for target in action[:, current]:
                target_int = int(target)

                if target_int not in reached:
                    reached.add(target_int)
                    frontier.append(target_int)

        unseen -= reached
        orbits.append(
            sorted(reached)
        )

    orbits.sort(
        key=lambda orbit: (
            -len(orbit),
            orbit[0],
        )
    )

    return orbits


def pencil_gram(
    slices: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    matrix = np.einsum(
        "r,rab->ab",
        direction,
        slices,
    )

    gram = matrix.T @ matrix

    return 0.5 * (
        gram + gram.T
    )


def numerical_rank(
    matrix: np.ndarray,
) -> int:
    return int(
        np.count_nonzero(
            np.linalg.eigvalsh(
                matrix
            )
            > RANK_TOLERANCE
        )
    )


def expected_spectrum(
    orbit_size: int,
) -> tuple[np.ndarray, list[str]]:
    if orbit_size == 15:
        low = (
            3.0 - np.sqrt(5.0)
        ) / 48.0

        high = (
            3.0 + np.sqrt(5.0)
        ) / 48.0

        values = np.array(
            [
                0.0,
                0.0,
                low,
                low,
                high,
                high,
            ],
            dtype=np.float64,
        )

        expressions = [
            "0",
            "0",
            "(3-sqrt(5))/48",
            "(3-sqrt(5))/48",
            "(3+sqrt(5))/48",
            "(3+sqrt(5))/48",
        ]

        return values, expressions

    if orbit_size == 10:
        values = np.array(
            [
                0.0,
                0.0,
                1.0 / 16.0,
                1.0 / 16.0,
                1.0 / 16.0,
                1.0 / 16.0,
            ],
            dtype=np.float64,
        )

        expressions = [
            "0",
            "0",
            "1/16",
            "1/16",
            "1/16",
            "1/16",
        ]

        return values, expressions

    if orbit_size == 5:
        values = np.full(
            6,
            1.0 / 24.0,
            dtype=np.float64,
        )

        expressions = [
            "1/24"
            for _ in range(6)
        ]

        return values, expressions

    raise RuntimeError(
        f"unexpected orbit size: {orbit_size}"
    )


def polynomial_residual(
    gram: np.ndarray,
    orbit_size: int,
) -> tuple[np.ndarray, str]:
    identity = np.eye(
        6,
        dtype=np.float64,
    )

    if orbit_size == 15:
        residual = (
            gram
            @ (
                gram @ gram
                - (1.0 / 8.0) * gram
                + (1.0 / 576.0)
                * identity
            )
        )

        law = (
            "B(B^2-(1/8)B+(1/576)I6)=0"
        )

        return residual, law

    if orbit_size == 10:
        residual = (
            gram
            @ (
                gram
                - (1.0 / 16.0)
                * identity
            )
        )

        law = (
            "B(B-(1/16)I6)=0"
        )

        return residual, law

    if orbit_size == 5:
        residual = (
            gram
            - (1.0 / 24.0)
            * identity
        )

        law = (
            "B=(1/24)I6"
        )

        return residual, law

    raise RuntimeError(
        f"unexpected orbit size: {orbit_size}"
    )


def expected_rank(
    orbit_size: int,
) -> int:
    return {
        15: 4,
        10: 4,
        5: 6,
    }[orbit_size]


def expected_trace(
    orbit_size: int,
) -> float:
    # All pencil slices have fixed Frobenius norm squared 1/4
    # on the unit four-flux sphere.
    return 1.0 / 4.0


def exact_operator_norm_expression(
    orbit_size: int,
) -> str:
    return {
        15: "sqrt((3+sqrt(5))/48)",
        10: "1/4",
        5: "sqrt(6)/12",
    }[orbit_size]


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ORBIT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LINE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    census_receipt = json.loads(
        CENSUS_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    census_data = np.load(
        CENSUS_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    nonaxis_lines = np.array(
        census_data["nonaxis_lines"],
        dtype=np.float64,
    )

    group_action = np.array(
        census_data["group_action"],
        dtype=np.int64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    if nonaxis_lines.shape != (30, 4):
        raise RuntimeError(
            "unexpected nonaxis line shape: "
            f"{nonaxis_lines.shape}"
        )

    if slices.shape != (4, 6, 6):
        raise RuntimeError(
            f"unexpected slice shape: {slices.shape}"
        )

    orbits = orbit_partition(
        group_action
    )

    orbit_rows = []
    line_rows = []

    orbit_gram_representatives = []
    orbit_expected_spectra = []

    global_polynomial_residual = 0.0
    global_spectral_residual = 0.0
    global_trace_residual = 0.0

    for orbit_id, orbit in enumerate(
        orbits
    ):
        orbit_size = len(orbit)

        expected_values, expected_expressions = (
            expected_spectrum(
                orbit_size
            )
        )

        orbit_polynomial_residual = 0.0
        orbit_spectral_residual = 0.0
        orbit_trace_residual = 0.0
        orbit_rank_profile = []
        operator_norms = []

        representative_gram = None

        for line_id in orbit:
            gram = pencil_gram(
                slices,
                nonaxis_lines[line_id],
            )

            eigenvalues = np.linalg.eigvalsh(
                gram
            )

            residual_matrix, law = (
                polynomial_residual(
                    gram,
                    orbit_size,
                )
            )

            current_polynomial_residual = (
                max_abs(
                    residual_matrix
                )
            )

            current_spectral_residual = (
                max_abs(
                    eigenvalues
                    - expected_values
                )
            )

            current_trace_residual = abs(
                float(
                    np.trace(gram)
                )
                - expected_trace(
                    orbit_size
                )
            )

            rank = numerical_rank(
                gram
            )

            operator_norm = float(
                np.sqrt(
                    max(
                        eigenvalues[-1],
                        0.0,
                    )
                )
            )

            orbit_polynomial_residual = max(
                orbit_polynomial_residual,
                current_polynomial_residual,
            )

            orbit_spectral_residual = max(
                orbit_spectral_residual,
                current_spectral_residual,
            )

            orbit_trace_residual = max(
                orbit_trace_residual,
                current_trace_residual,
            )

            orbit_rank_profile.append(
                rank
            )

            operator_norms.append(
                operator_norm
            )

            if representative_gram is None:
                representative_gram = (
                    gram.copy()
                )

            line_rows.append(
                {
                    "line_id": line_id,
                    "orbit_id": orbit_id,
                    "orbit_size": orbit_size,
                    "rank": rank,
                    "trace": float(
                        np.trace(gram)
                    ),
                    "eigenvalue_0": float(
                        eigenvalues[0]
                    ),
                    "eigenvalue_1": float(
                        eigenvalues[1]
                    ),
                    "eigenvalue_2": float(
                        eigenvalues[2]
                    ),
                    "eigenvalue_3": float(
                        eigenvalues[3]
                    ),
                    "eigenvalue_4": float(
                        eigenvalues[4]
                    ),
                    "eigenvalue_5": float(
                        eigenvalues[5]
                    ),
                    "operator_norm": (
                        operator_norm
                    ),
                    "polynomial_residual": (
                        current_polynomial_residual
                    ),
                    "spectral_residual": (
                        current_spectral_residual
                    ),
                    "trace_residual": (
                        current_trace_residual
                    ),
                }
            )

        global_polynomial_residual = max(
            global_polynomial_residual,
            orbit_polynomial_residual,
        )

        global_spectral_residual = max(
            global_spectral_residual,
            orbit_spectral_residual,
        )

        global_trace_residual = max(
            global_trace_residual,
            orbit_trace_residual,
        )

        expected_rank_value = (
            expected_rank(
                orbit_size
            )
        )

        orbit_pass = (
            orbit_polynomial_residual
            < POLYNOMIAL_TOLERANCE
            and orbit_spectral_residual
            < SPECTRAL_TOLERANCE
            and orbit_trace_residual
            < TRACE_TOLERANCE
            and set(
                orbit_rank_profile
            )
            == {
                expected_rank_value
            }
        )

        orbit_rows.append(
            {
                "orbit_id": orbit_id,
                "orbit_size": orbit_size,
                "stabilizer_order": (
                    480 // orbit_size
                ),
                "polynomial_law": law,
                "expected_rank": (
                    expected_rank_value
                ),
                "observed_rank_profile": (
                    json.dumps(
                        sorted(
                            set(
                                orbit_rank_profile
                            )
                        )
                    )
                ),
                "expected_squared_spectrum": (
                    json.dumps(
                        expected_expressions
                    )
                ),
                "expected_operator_norm": (
                    exact_operator_norm_expression(
                        orbit_size
                    )
                ),
                "observed_operator_norm_minimum": min(
                    operator_norms
                ),
                "observed_operator_norm_maximum": max(
                    operator_norms
                ),
                "maximum_polynomial_residual": (
                    orbit_polynomial_residual
                ),
                "maximum_spectral_residual": (
                    orbit_spectral_residual
                ),
                "maximum_trace_residual": (
                    orbit_trace_residual
                ),
                "orbit_pass": orbit_pass,
            }
        )

        orbit_gram_representatives.append(
            representative_gram
        )

        orbit_expected_spectra.append(
            expected_values
        )

        print(
            "orbit:",
            orbit_id,
            "size:",
            orbit_size,
            "law:",
            law,
            "rank:",
            sorted(
                set(
                    orbit_rank_profile
                )
            ),
            "polynomial_residual:",
            orbit_polynomial_residual,
            "spectral_residual:",
            orbit_spectral_residual,
            flush=True,
        )

    checks = {
        "input_038_theorem_pass": (
            census_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "orbit_sizes_are_15_10_5": (
            [
                len(orbit)
                for orbit in orbits
            ]
            == [15, 10, 5]
        ),
        "all_orbit_polynomial_laws_pass": all(
            row["orbit_pass"]
            for row in orbit_rows
        ),
        "global_polynomial_residual_resolved": (
            global_polynomial_residual
            < POLYNOMIAL_TOLERANCE
        ),
        "global_spectral_residual_resolved": (
            global_spectral_residual
            < SPECTRAL_TOLERANCE
        ),
        "all_pencil_gram_traces_equal_one_quarter": (
            global_trace_residual
            < TRACE_TOLERANCE
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_zero_orbit_squared_spectra_exact"
        if theorem_pass
        else "native_g60_cross_flux_zero_orbit_squared_spectra_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_zero_orbit_squared_spectra_040"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "theorem": {
            "orbit_15": {
                "polynomial_law": (
                    "B(B^2-(1/8)B+(1/576)I6)=0"
                ),
                "squared_spectrum": [
                    "0",
                    "0",
                    "(3-sqrt(5))/48",
                    "(3-sqrt(5))/48",
                    "(3+sqrt(5))/48",
                    "(3+sqrt(5))/48",
                ],
                "operator_norm": (
                    "sqrt((3+sqrt(5))/48)"
                ),
                "rank": 4,
            },
            "orbit_10": {
                "polynomial_law": (
                    "B(B-(1/16)I6)=0"
                ),
                "squared_spectrum": [
                    "0",
                    "0",
                    "1/16",
                    "1/16",
                    "1/16",
                    "1/16",
                ],
                "operator_norm": "1/4",
                "rank": 4,
            },
            "orbit_5": {
                "polynomial_law": (
                    "B=(1/24)I6"
                ),
                "squared_spectrum": [
                    "1/24",
                    "1/24",
                    "1/24",
                    "1/24",
                    "1/24",
                    "1/24",
                ],
                "operator_norm": (
                    "sqrt(6)/12"
                ),
                "rank": 6,
            },
        },
        "orbit_rows": orbit_rows,
        "checks": checks,
        "global_residuals": {
            "polynomial_law": (
                global_polynomial_residual
            ),
            "squared_spectrum": (
                global_spectral_residual
            ),
            "trace_one_quarter": (
                global_trace_residual
            ),
        },
        "earned_interpretation": {
            "all_three_nonaxis_orbits_have_exact_squared_spectral_laws": (
                theorem_pass
            ),
            "fifteen_line_orbit_is_near_extremal_nonaxis_layer": (
                theorem_pass
            ),
            "ten_line_orbit_has_two_dimensional_kernel": (
                theorem_pass
            ),
            "five_line_orbit_is_isotropic_full_rank_layer": (
                theorem_pass
            ),
            "nonaxis_lines_reach_one_third": (
                False
            ),
        },
        "boundary": {
            "squared_spectral_polynomial_certificates_completed": (
                theorem_pass
            ),
            "coordinate_construction_law_derived": (
                False
            ),
            "complete_real_zero_locus_proved": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "orbit_csv": str(
                ORBIT_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "line_csv": str(
                LINE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "squared_spectra_npz": str(
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

    with ORBIT_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                orbit_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            orbit_rows
        )

    with LINE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                line_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            line_rows
        )

    np.savez_compressed(
        NPZ_OUT,
        nonaxis_lines=nonaxis_lines,
        group_action=group_action,
        orbit_representative_grams=np.array(
            orbit_gram_representatives,
            dtype=np.float64,
        ),
        orbit_expected_squared_spectra=np.array(
            orbit_expected_spectra,
            dtype=np.float64,
        ),
        orbit_sizes=np.array(
            [
                len(orbit)
                for orbit in orbits
            ],
            dtype=np.int64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "global_polynomial_residual:",
        global_polynomial_residual,
    )
    print(
        "global_spectral_residual:",
        global_spectral_residual,
    )
    print(
        "global_trace_residual:",
        global_trace_residual,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", ORBIT_CSV_OUT)
    print("wrote:", LINE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
