from __future__ import annotations

import csv
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp


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

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cross_flux_zero_orbit_spectra_039.json"
)

ORBIT_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_zero_orbit_spectra_039.csv"
)

GRAM_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_zero_orbit_gram_profiles_039.csv"
)

LINE_CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cross_flux_zero_orbit_lines_039.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts"
    / "npy"
    / "native_g60_cross_flux_zero_orbit_spectra_039.npz"
)

RATIONAL_DENOMINATOR_LIMIT = 1000000
RECONSTRUCTION_TOLERANCE = 2e-9
ORBIT_UNIFORMITY_TOLERANCE = 2e-8

ALGEBRAIC_CONSTANTS = [
    sp.sqrt(2),
    sp.sqrt(3),
    sp.sqrt(5),
    sp.sqrt(6),
]


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, sp.Basic):
        return str(value)

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


def rational_candidate(value: float) -> Fraction:
    return Fraction(
        float(value)
    ).limit_denominator(
        RATIONAL_DENOMINATOR_LIMIT
    )


def reconstruct_value(value: float) -> dict:
    rational = rational_candidate(value)
    rational_float = float(rational)
    rational_residual = abs(
        float(value) - rational_float
    )

    try:
        algebraic = sp.nsimplify(
            float(value),
            ALGEBRAIC_CONSTANTS,
            tolerance=1e-10,
            full=False,
        )

        algebraic_float = float(
            sp.N(
                algebraic,
                30,
            )
        )

        algebraic_residual = abs(
            float(value)
            - algebraic_float
        )
    except Exception:
        algebraic = None
        algebraic_float = None
        algebraic_residual = float("inf")

    if (
        rational_residual
        <= algebraic_residual
        and rational_residual
        < RECONSTRUCTION_TOLERANCE
    ):
        selected_kind = "rational"
        selected_expression = str(
            rational
        )
        selected_residual = (
            rational_residual
        )
    elif (
        algebraic is not None
        and algebraic_residual
        < RECONSTRUCTION_TOLERANCE
    ):
        selected_kind = "algebraic"
        selected_expression = str(
            algebraic
        )
        selected_residual = (
            algebraic_residual
        )
    else:
        selected_kind = "unresolved"
        selected_expression = repr(
            float(value)
        )
        selected_residual = min(
            rational_residual,
            algebraic_residual,
        )

    return {
        "value": float(value),
        "rational_candidate": str(
            rational
        ),
        "rational_residual": (
            rational_residual
        ),
        "algebraic_candidate": (
            None
            if algebraic is None
            else str(algebraic)
        ),
        "algebraic_residual": (
            None
            if algebraic is None
            else algebraic_residual
        ),
        "selected_kind": selected_kind,
        "selected_expression": (
            selected_expression
        ),
        "selected_residual": (
            selected_residual
        ),
    }


def register_covariant(
    point: np.ndarray,
    axis_lines: np.ndarray,
) -> np.ndarray:
    overlaps = (
        axis_lines @ point
    )

    return np.einsum(
        "i,ia,ib->ab",
        overlaps**2,
        axis_lines,
        axis_lines,
    )


def register_moment(
    point: np.ndarray,
    axis_lines: np.ndarray,
    power: int,
) -> float:
    overlaps = (
        axis_lines @ point
    )

    return float(
        np.sum(
            overlaps**power
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
        frontier = [seed]

        while frontier:
            current = frontier.pop()

            for target in action[:, current]:
                target_int = int(target)

                if target_int not in reached:
                    reached.add(
                        target_int
                    )
                    frontier.append(
                        target_int
                    )

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


def projective_gram_profile(
    lines: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[str, int],
    list[dict],
]:
    gram = (
        lines @ lines.T
    )

    counter = Counter()
    rows = []

    for first in range(
        len(lines)
    ):
        for second in range(
            first + 1,
            len(lines),
        ):
            signed_value = float(
                gram[
                    first,
                    second,
                ]
            )

            absolute_value = abs(
                signed_value
            )

            reconstruction = (
                reconstruct_value(
                    absolute_value
                )
            )

            key = reconstruction[
                "selected_expression"
            ]

            counter[key] += 1

            rows.append(
                {
                    "first_local_line": first,
                    "second_local_line": (
                        second
                    ),
                    "signed_gram": (
                        signed_value
                    ),
                    "absolute_gram": (
                        absolute_value
                    ),
                    "selected_kind": (
                        reconstruction[
                            "selected_kind"
                        ]
                    ),
                    "selected_expression": (
                        reconstruction[
                            "selected_expression"
                        ]
                    ),
                    "selected_residual": (
                        reconstruction[
                            "selected_residual"
                        ]
                    ),
                }
            )

    return (
        gram,
        dict(
            sorted(
                counter.items()
            )
        ),
        rows,
    )


def line_signature(
    point: np.ndarray,
    axis_lines: np.ndarray,
    slices: np.ndarray,
) -> dict:
    s4 = register_moment(
        point,
        axis_lines,
        4,
    )

    s6 = register_moment(
        point,
        axis_lines,
        6,
    )

    s8 = register_moment(
        point,
        axis_lines,
        8,
    )

    covariant = register_covariant(
        point,
        axis_lines,
    )

    covariant_eigenvalues = (
        np.linalg.eigvalsh(
            covariant
        )
    )

    pencil_matrix = np.einsum(
        "r,rab->ab",
        point,
        slices,
    )

    singular_values = np.linalg.svd(
        pencil_matrix,
        compute_uv=False,
    )

    residual = (
        covariant @ point
        - s4 * point
    )

    return {
        "s4": s4,
        "s6": s6,
        "s8": s8,
        "covariant_eigenvalues": (
            covariant_eigenvalues
        ),
        "pencil_singular_values": (
            singular_values
        ),
        "operator_norm": float(
            singular_values[0]
        ),
        "residual_norm": float(
            np.linalg.norm(
                residual
            )
        ),
    }


def uniformity_residual(
    values: np.ndarray,
) -> float:
    mean = np.mean(
        values,
        axis=0,
    )

    return max_abs(
        values - mean
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ORBIT_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAM_CSV_OUT.parent.mkdir(
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

    data = np.load(
        CENSUS_NPZ_PATH
    )

    axis_lines = np.array(
        data["axis_lines"],
        dtype=np.float64,
    )

    nonaxis_lines = np.array(
        data["nonaxis_lines"],
        dtype=np.float64,
    )

    action = np.array(
        data["group_action"],
        dtype=np.int64,
    )

    singular_value_table = np.array(
        data["pencil_singular_values"],
        dtype=np.float64,
    )

    covariant_eigenvalue_table = np.array(
        data["covariant_eigenvalues"],
        dtype=np.float64,
    )

    s4_table = np.array(
        data["s4"],
        dtype=np.float64,
    )

    s6_table = np.array(
        data["s6"],
        dtype=np.float64,
    )

    s8_table = np.array(
        data["s8"],
        dtype=np.float64,
    )

    if axis_lines.shape != (10, 4):
        raise RuntimeError(
            f"unexpected axis shape: {axis_lines.shape}"
        )

    if nonaxis_lines.shape != (30, 4):
        raise RuntimeError(
            "unexpected nonaxis shape: "
            f"{nonaxis_lines.shape}"
        )

    slices_path = (
        ROOT
        / "artifacts"
        / "npy"
        / "native_g60_cross_flux_operator_pencil_026.npz"
    )

    slices_data = np.load(
        slices_path
    )

    slices = np.array(
        slices_data["slices"],
        dtype=np.float64,
    )

    orbits = orbit_partition(
        action
    )

    orbit_rows = []
    line_rows = []
    gram_rows = []
    orbit_payloads = []

    for orbit_id, orbit in enumerate(
        orbits
    ):
        print(
            "orbit_progress:",
            f"{orbit_id + 1}/{len(orbits)}",
            "size:",
            len(orbit),
            "starting",
            flush=True,
        )

        orbit_lines = (
            nonaxis_lines[orbit]
        )

        representative_id = orbit[0]

        representative = (
            nonaxis_lines[
                representative_id
            ]
        )

        signature = line_signature(
            representative,
            axis_lines,
            slices,
        )

        orbit_s4 = s4_table[
            orbit
        ]

        orbit_s6 = s6_table[
            orbit
        ]

        orbit_s8 = s8_table[
            orbit
        ]

        orbit_covariant = (
            covariant_eigenvalue_table[
                orbit
            ]
        )

        orbit_singular = (
            singular_value_table[
                orbit
            ]
        )

        (
            orbit_gram,
            gram_profile,
            current_gram_rows,
        ) = projective_gram_profile(
            orbit_lines
        )

        for row in current_gram_rows:
            row["orbit_id"] = (
                orbit_id
            )

            row["orbit_size"] = (
                len(orbit)
            )

            row["first_global_line"] = (
                orbit[
                    row[
                        "first_local_line"
                    ]
                ]
            )

            row["second_global_line"] = (
                orbit[
                    row[
                        "second_local_line"
                    ]
                ]
            )

            gram_rows.append(row)

        s4_reconstruction = (
            reconstruct_value(
                signature["s4"]
            )
        )

        s6_reconstruction = (
            reconstruct_value(
                signature["s6"]
            )
        )

        s8_reconstruction = (
            reconstruct_value(
                signature["s8"]
            )
        )

        operator_reconstruction = (
            reconstruct_value(
                signature[
                    "operator_norm"
                ]
            )
        )

        covariant_reconstructions = [
            reconstruct_value(value)
            for value in signature[
                "covariant_eigenvalues"
            ]
        ]

        singular_reconstructions = [
            reconstruct_value(value)
            for value in signature[
                "pencil_singular_values"
            ]
        ]

        stabilizer_order = (
            480 // len(orbit)
        )

        orbit_uniformity = {
            "s4": uniformity_residual(
                orbit_s4
            ),
            "s6": uniformity_residual(
                orbit_s6
            ),
            "s8": uniformity_residual(
                orbit_s8
            ),
            "covariant_spectrum": (
                uniformity_residual(
                    orbit_covariant
                )
            ),
            "pencil_singular_spectrum": (
                uniformity_residual(
                    orbit_singular
                )
            ),
        }

        orbit_pass = (
            max(
                orbit_uniformity.values()
            )
            < ORBIT_UNIFORMITY_TOLERANCE
            and signature[
                "residual_norm"
            ]
            < RECONSTRUCTION_TOLERANCE
        )

        orbit_rows.append(
            {
                "orbit_id": orbit_id,
                "orbit_size": len(
                    orbit
                ),
                "stabilizer_order": (
                    stabilizer_order
                ),
                "representative_line_id": (
                    representative_id
                ),
                "s4": signature["s4"],
                "s4_expression": (
                    s4_reconstruction[
                        "selected_expression"
                    ]
                ),
                "s6": signature["s6"],
                "s6_expression": (
                    s6_reconstruction[
                        "selected_expression"
                    ]
                ),
                "s8": signature["s8"],
                "s8_expression": (
                    s8_reconstruction[
                        "selected_expression"
                    ]
                ),
                "operator_norm": (
                    signature[
                        "operator_norm"
                    ]
                ),
                "operator_norm_expression": (
                    operator_reconstruction[
                        "selected_expression"
                    ]
                ),
                "operator_norm_kind": (
                    operator_reconstruction[
                        "selected_kind"
                    ]
                ),
                "operator_norm_residual": (
                    operator_reconstruction[
                        "selected_residual"
                    ]
                ),
                "covariant_spectrum": json.dumps(
                    signature[
                        "covariant_eigenvalues"
                    ].tolist()
                ),
                "covariant_spectrum_expressions": json.dumps(
                    [
                        item[
                            "selected_expression"
                        ]
                        for item in (
                            covariant_reconstructions
                        )
                    ]
                ),
                "pencil_singular_spectrum": json.dumps(
                    signature[
                        "pencil_singular_values"
                    ].tolist()
                ),
                "pencil_singular_expressions": json.dumps(
                    [
                        item[
                            "selected_expression"
                        ]
                        for item in (
                            singular_reconstructions
                        )
                    ]
                ),
                "gram_absolute_profile": json.dumps(
                    gram_profile
                ),
                "gram_rank": int(
                    np.linalg.matrix_rank(
                        orbit_gram,
                        tol=1e-9,
                    )
                ),
                "gram_eigenvalues": json.dumps(
                    np.linalg.eigvalsh(
                        orbit_gram
                    ).tolist()
                ),
                "maximum_uniformity_residual": max(
                    orbit_uniformity.values()
                ),
                "orbit_pass": (
                    orbit_pass
                ),
            }
        )

        orbit_payloads.append(
            {
                "orbit_id": orbit_id,
                "orbit_size": len(
                    orbit
                ),
                "line_ids": orbit,
                "stabilizer_order": (
                    stabilizer_order
                ),
                "representative": (
                    representative
                ),
                "s4": s4_reconstruction,
                "s6": s6_reconstruction,
                "s8": s8_reconstruction,
                "operator_norm": (
                    operator_reconstruction
                ),
                "covariant_spectrum": (
                    covariant_reconstructions
                ),
                "pencil_singular_spectrum": (
                    singular_reconstructions
                ),
                "gram_absolute_profile": (
                    gram_profile
                ),
                "gram_eigenvalues": (
                    np.linalg.eigvalsh(
                        orbit_gram
                    )
                ),
                "uniformity_residuals": (
                    orbit_uniformity
                ),
            }
        )

        print(
            "orbit_progress:",
            f"{orbit_id + 1}/{len(orbits)}",
            "size:",
            len(orbit),
            "reconstruction_done",
            flush=True,
        )

        for line_id in orbit:
            signature_line = line_signature(
                nonaxis_lines[
                    line_id
                ],
                axis_lines,
                slices,
            )

            line_rows.append(
                {
                    "line_id": line_id,
                    "orbit_id": orbit_id,
                    "orbit_size": len(
                        orbit
                    ),
                    "stabilizer_order": (
                        stabilizer_order
                    ),
                    "s4": (
                        signature_line["s4"]
                    ),
                    "s6": (
                        signature_line["s6"]
                    ),
                    "s8": (
                        signature_line["s8"]
                    ),
                    "operator_norm": (
                        signature_line[
                            "operator_norm"
                        ]
                    ),
                    "residual_norm": (
                        signature_line[
                            "residual_norm"
                        ]
                    ),
                }
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
        "stabilizer_orders_are_32_48_96": (
            [
                480 // len(orbit)
                for orbit in orbits
            ]
            == [32, 48, 96]
        ),
        "all_orbit_signatures_are_uniform": all(
            row["orbit_pass"]
            for row in orbit_rows
        ),
        "all_three_operator_norms_are_distinct": (
            len(
                {
                    round(
                        row[
                            "operator_norm"
                        ],
                        10,
                    )
                    for row in orbit_rows
                }
            )
            == 3
        ),
        "all_nonaxis_operator_norms_below_one_third": all(
            row["operator_norm"]
            < 1.0 / 3.0
            - 1e-8
            for row in orbit_rows
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_zero_orbit_spectra_resolved"
        if theorem_pass
        else "native_g60_cross_flux_zero_orbit_spectra_incomplete"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_zero_orbit_spectra_039"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "orbit_count": len(
            orbits
        ),
        "orbit_sizes": [
            len(orbit)
            for orbit in orbits
        ],
        "stabilizer_orders": [
            480 // len(orbit)
            for orbit in orbits
        ],
        "orbits": orbit_payloads,
        "checks": checks,
        "earned_interpretation": {
            "three_nonaxis_symmetry_classes_have_distinct_exact_signatures": (
                theorem_pass
            ),
            "five_line_class_is_closest_nonaxis_layer_to_one_third": (
                orbit_rows[-1][
                    "operator_norm"
                ]
                == max(
                    row[
                        "operator_norm"
                    ]
                    for row in orbit_rows
                )
            ),
            "nonaxis_self_alignment_implies_extremality": (
                False
            ),
        },
        "boundary": {
            "orbitwise_spectral_census_completed": (
                theorem_pass
            ),
            "all_algebraic_candidates_exactly_proved": (
                False
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
            "gram_csv": str(
                GRAM_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "line_csv": str(
                LINE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "spectra_npz": str(
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

    with GRAM_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                gram_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            gram_rows
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
        axis_lines=axis_lines,
        nonaxis_lines=nonaxis_lines,
        orbit_ids=np.array(
            [
                orbit_id
                for orbit_id, orbit in enumerate(
                    orbits
                )
                for _ in orbit
            ],
            dtype=np.int64,
        ),
        orbit_representatives=np.array(
            [
                nonaxis_lines[
                    orbit[0]
                ]
                for orbit in orbits
            ],
            dtype=np.float64,
        ),
        orbit_s4=np.array(
            [
                row["s4"]
                for row in orbit_rows
            ],
            dtype=np.float64,
        ),
        orbit_s6=np.array(
            [
                row["s6"]
                for row in orbit_rows
            ],
            dtype=np.float64,
        ),
        orbit_s8=np.array(
            [
                row["s8"]
                for row in orbit_rows
            ],
            dtype=np.float64,
        ),
        orbit_operator_norms=np.array(
            [
                row["operator_norm"]
                for row in orbit_rows
            ],
            dtype=np.float64,
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "orbit_sizes:",
        [
            len(orbit)
            for orbit in orbits
        ],
    )
    print(
        "stabilizer_orders:",
        [
            480 // len(orbit)
            for orbit in orbits
        ],
    )

    for row in orbit_rows:
        print(
            "orbit=",
            row["orbit_id"],
            "size=",
            row["orbit_size"],
            "stabilizer=",
            row["stabilizer_order"],
            "S4=",
            row["s4_expression"],
            "S6=",
            row["s6_expression"],
            "S8=",
            row["s8_expression"],
            "operator_norm=",
            row[
                "operator_norm_expression"
            ],
            "singular_spectrum=",
            row[
                "pencil_singular_expressions"
            ],
            "covariant_spectrum=",
            row[
                "covariant_spectrum_expressions"
            ],
        )

    print("wrote:", JSON_OUT)
    print("wrote:", ORBIT_CSV_OUT)
    print("wrote:", GRAM_CSV_OUT)
    print("wrote:", LINE_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
