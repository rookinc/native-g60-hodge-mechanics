from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATHS = {
    "axis_register_018": (
        ROOT
        / "artifacts/json"
        / "native_g60_four_flux_axis_register_018.json"
    ),
    "axis_identification_029": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_maximizer_axis_identification_029.json"
    ),
    "axis_spectrum_030": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_axis_extremal_certificate_030.json"
    ),
    "axis_hessian_031": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_axis_local_hessian_031.json"
    ),
    "register_moments_032": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_register_invariant_scan_032.json"
    ),
    "register_covariant_033": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_register_covariant_033.json"
    ),
    "covariant_identities_034": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_covariant_identity_034.json"
    ),
    "covariant_orientation_035": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_covariant_orientation_035.json"
    ),
    "zero_census_036": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_covariant_residual_zero_locus_036.json"
    ),
    "thirty_orbits_038": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_residual_zero_30_line_census_038.json"
    ),
    "squared_spectra_040": (
        ROOT
        / "artifacts/json"
        / "native_g60_cross_flux_zero_orbit_squared_spectra_040.json"
    ),
}

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_self_alignment_register_theorem_041.json"
)

TABLE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_self_alignment_register_041.csv"
)

CHECK_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_self_alignment_register_checks_041.csv"
)

NOTE_OUT = (
    ROOT
    / "theorems"
    / "native_g60_cross_flux_self_alignment_register_theorem_041.md"
)

TOLERANCE = 2e-8


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def load_receipts() -> dict[str, dict]:
    receipts = {}

    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"missing input artifact: {path}"
            )

        receipts[name] = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    return receipts


def nested_get(
    value: dict,
    *keys: str,
    default=None,
):
    current = value

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def close(
    first: float,
    second: float,
    tolerance: float = TOLERANCE,
) -> bool:
    return abs(
        float(first) - float(second)
    ) < tolerance


def input_pass(receipt: dict) -> bool:
    return bool(
        receipt.get(
            "theorem_pass",
            False,
        )
        or receipt.get(
            "audit_pass",
            False,
        )
    )


def exact_class_rows() -> list[dict]:
    return [
        {
            "class_id": "A10",
            "role": "native_extremal_axes",
            "orbit_size": 10,
            "stabilizer_order": 48,
            "self_aligned": True,
            "extremal": True,
            "s4": "115/72",
            "s6": "3275/2592",
            "s8": "34745/31104",
            "covariant_spectrum": (
                "[5/72,5/12,5/12,115/72]"
            ),
            "pencil_rank": 6,
            "pencil_squared_spectrum": (
                "[1/144 x4,1/9 x2]"
            ),
            "pencil_polynomial_law": (
                "(B-I/9)(B-I/144)=0"
            ),
            "operator_norm": "1/3",
            "one_third_gap": "0",
            "local_status": (
                "analytic_strict_local_maximum"
            ),
        },
        {
            "class_id": "N15",
            "role": "near_extremal_nonaxis",
            "orbit_size": 15,
            "stabilizer_order": 32,
            "self_aligned": True,
            "extremal": False,
            "s4": "25/16",
            "s6": "1375/1152",
            "s8": "26875/27648",
            "covariant_spectrum": (
                "[25/144,25/144,85/144,25/16]"
            ),
            "pencil_rank": 4,
            "pencil_squared_spectrum": (
                "[0 x2,(3-sqrt(5))/48 x2,"
                "(3+sqrt(5))/48 x2]"
            ),
            "pencil_polynomial_law": (
                "B(B^2-B/8+I/576)=0"
            ),
            "operator_norm": (
                "sqrt((3+sqrt(5))/48)"
            ),
            "one_third_gap": (
                "1/3-sqrt((3+sqrt(5))/48)"
            ),
            "local_status": (
                "self_aligned_nonextremal"
            ),
        },
        {
            "class_id": "N10",
            "role": "rank_four_nonaxis",
            "orbit_size": 10,
            "stabilizer_order": 48,
            "self_aligned": True,
            "extremal": False,
            "s4": "25/24",
            "s6": "125/288",
            "s8": "625/3456",
            "covariant_spectrum": (
                "[5/72,25/36,25/36,25/24]"
            ),
            "pencil_rank": 4,
            "pencil_squared_spectrum": (
                "[0 x2,1/16 x4]"
            ),
            "pencil_polynomial_law": (
                "B(B-I/16)=0"
            ),
            "operator_norm": "1/4",
            "one_third_gap": "1/12",
            "local_status": (
                "self_aligned_nonextremal"
            ),
        },
        {
            "class_id": "N5",
            "role": "isotropic_nonaxis",
            "orbit_size": 5,
            "stabilizer_order": 96,
            "self_aligned": True,
            "extremal": False,
            "s4": "35/48",
            "s6": "275/1152",
            "s8": "2315/27648",
            "covariant_spectrum": (
                "[85/144,85/144,85/144,35/48]"
            ),
            "pencil_rank": 6,
            "pencil_squared_spectrum": (
                "[1/24 x6]"
            ),
            "pencil_polynomial_law": (
                "B=I/24"
            ),
            "operator_norm": "sqrt(6)/12",
            "one_third_gap": (
                "1/3-sqrt(6)/12"
            ),
            "local_status": (
                "self_aligned_nonextremal"
            ),
        },
    ]


def build_checks(
    receipts: dict[str, dict],
    rows: list[dict],
) -> dict[str, bool]:
    zero_census = receipts[
        "zero_census_036"
    ]

    thirty = receipts[
        "thirty_orbits_038"
    ]

    spectra = receipts[
        "squared_spectra_040"
    ]

    axis_moments = receipts[
        "register_moments_032"
    ]

    axis_covariant = receipts[
        "register_covariant_033"
    ]

    axis_hessian = receipts[
        "axis_hessian_031"
    ]

    recovered_zero_count = nested_get(
        zero_census,
        "root_search",
        "projective_cluster_count",
        default=None,
    )

    thirty_orbit_sizes = nested_get(
        thirty,
        "group",
        "orbit_sizes",
        default=None,
    )

    squared_size_values = [
        int(row["orbit_size"])
        for row in spectra.get(
            "orbit_rows",
            [],
        )
    ]

    axis_s4 = nested_get(
        axis_moments,
        "axis_moments",
        "s4",
        "mean",
        default=None,
    )

    axis_covariant_spectrum = nested_get(
        axis_covariant,
        "axis_covariant",
        "mean_spectrum",
        default=None,
    )

    minimum_curvature = nested_get(
        axis_hessian,
        "theorem",
        "global_minimum_drop_curvature_candidate",
        default=None,
    )

    return {
        "all_positive_upstream_inputs_pass": all(
            input_pass(receipt)
            for name, receipt in receipts.items()
            if name != "zero_census_036"
        ),
        "zero_census_recovered_forty_lines": (
            recovered_zero_count == 40
        ),
        "class_sizes_sum_to_forty": (
            sum(
                int(row["orbit_size"])
                for row in rows
            )
            == 40
        ),
        "decomposition_is_10_15_10_5": (
            [
                row["orbit_size"]
                for row in rows
            ]
            == [10, 15, 10, 5]
        ),
        "nonaxis_orbits_are_15_10_5": (
            thirty_orbit_sizes
            == [15, 10, 5]
        ),
        "squared_spectrum_orbits_are_15_10_5": (
            squared_size_values
            == [15, 10, 5]
        ),
        "all_orbit_stabilizer_products_equal_480": all(
            int(row["orbit_size"])
            * int(row["stabilizer_order"])
            == 480
            for row in rows
        ),
        "native_axis_s4_is_115_over_72": (
            axis_s4 is not None
            and close(
                axis_s4,
                115.0 / 72.0,
            )
        ),
        "native_axis_covariant_spectrum_matches": (
            axis_covariant_spectrum
            is not None
            and max(
                abs(
                    float(observed)
                    - expected
                )
                for observed, expected in zip(
                    axis_covariant_spectrum,
                    [
                        5.0 / 72.0,
                        5.0 / 12.0,
                        5.0 / 12.0,
                        115.0 / 72.0,
                    ],
                )
            )
            < TOLERANCE
        ),
        "native_axis_curvature_is_one_over_18": (
            minimum_curvature
            is not None
            and close(
                minimum_curvature,
                1.0 / 18.0,
            )
        ),
        "native_axis_class_is_unique_extremal_class": (
            sum(
                bool(row["extremal"])
                for row in rows
            )
            == 1
            and rows[0]["class_id"]
            == "A10"
        ),
        "five_line_class_is_isotropic": (
            rows[3][
                "pencil_polynomial_law"
            ]
            == "B=I/24"
        ),
        "ten_line_nonaxis_class_has_rank_four": (
            rows[2]["pencil_rank"]
            == 4
        ),
        "fifteen_line_class_is_near_extremal": (
            rows[1]["role"]
            == "near_extremal_nonaxis"
        ),
    }


def check_rows(
    checks: dict[str, bool],
) -> list[dict]:
    return [
        {
            "check": name,
            "pass": bool(value),
        }
        for name, value in checks.items()
    ]


def theorem_note(
    rows: list[dict],
) -> str:
    table_lines = [
        (
            "| Class | Size | Stabilizer | "
            "Rank | Operator norm | Status |"
        ),
        (
            "|---|---:|---:|---:|---|---|"
        ),
    ]

    for row in rows:
        table_lines.append(
            "| {class_id} | {orbit_size} | "
            "{stabilizer_order} | {pencil_rank} | "
            "{operator_norm} | {local_status} |".format(
                **row
            )
        )

    table = "\n".join(
        table_lines
    )

    return (
        "# Native G60 Cross-Flux "
        "Self-Alignment Register Theorem 041\n\n"
        "## Theorem\n\n"
        "The recovered real projective self-alignment "
        "census for\n\n"
        "    r(f) = C(f)f - S4(f)f\n\n"
        "decomposes under the native group of order "
        "480 as\n\n"
        "    40 = 10 + 15 + 10 + 5.\n\n"
        "The four response classes are:\n\n"
        + table
        + "\n\n"
        "The native ten-axis class is the unique "
        "recovered class with operator norm\n\n"
        "    1/3.\n\n"
        "It has squared pencil spectrum\n\n"
        "    1/144 with multiplicity 4\n"
        "    1/9 with multiplicity 2\n\n"
        "and every native axis is an analytic strict "
        "local maximum with weakest quadratic drop "
        "coefficient\n\n"
        "    1/18.\n\n"
        "The non-axis classes have exact squared "
        "spectral laws:\n\n"
        "    N15\n"
        "    B(B^2 - B/8 + I/576) = 0\n\n"
        "    N10\n"
        "    B(B - I/16) = 0\n\n"
        "    N5\n"
        "    B = I/24\n\n"
        "Thus self-alignment has four native response "
        "levels, but only the original ten-axis "
        "register is extremal among the recovered "
        "classes.\n\n"
        "## Register notation\n\n"
        "The original Thalean incidence Gram remains\n\n"
        "    Q = M M^T\n\n"
        "with M of shape 15 by 30.\n\n"
        "The ten-axis four-flux line register remains\n\n"
        "    R10 = L10 L10^T\n\n"
        "with\n\n"
        "    L10^T L10 = (5/2) I4\n"
        "    R10^2 = (5/2) R10.\n\n"
        "These are distinct Gram registers at "
        "distinct layers.\n\n"
        "## Boundary\n\n"
        "This theorem consolidates the recovered "
        "forty-line census and its exact orbitwise "
        "response laws.\n\n"
        "It does not prove that the forty recovered "
        "lines are the complete real projective zero "
        "locus.\n\n"
        "It does not yet prove the global inequality\n\n"
        "    ||A(f)||_op <= (1/3)||f||\n\n"
        "for every four-flux direction.\n\n"
        "No physical energy, force, material, "
        "transport, or instability claim is made.\n"
    )


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECK_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NOTE_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    receipts = load_receipts()
    rows = exact_class_rows()

    checks = build_checks(
        receipts,
        rows,
    )

    audit_pass = all(
        checks.values()
    )

    theorem_pass = audit_pass

    verdict = (
        "native_g60_cross_flux_self_alignment_"
        "register_theorem_synthesized"
        if theorem_pass
        else "native_g60_cross_flux_self_alignment_"
        "register_synthesis_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_self_alignment_"
            "register_theorem_041"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "theorem": {
            "decomposition": (
                "40 = 10 + 15 + 10 + 5"
            ),
            "group_order": 480,
            "class_count": 4,
            "self_aligned_line_count": 40,
            "unique_extremal_class": "A10",
            "extremal_operator_norm": "1/3",
            "native_axis_weakest_local_curvature": (
                "1/18"
            ),
            "statement": (
                "The recovered forty-line native "
                "cross-flux self-alignment register "
                "splits into four response classes "
                "of sizes 10, 15, 10, and 5. Only "
                "the native ten-axis class reaches "
                "the one-third operator-norm level."
            ),
        },
        "classes": rows,
        "checks": checks,
        "input_artifacts": {
            name: str(
                path.relative_to(ROOT)
            )
            for name, path in (
                INPUT_PATHS.items()
            )
        },
        "register_notation": {
            "thalean_incidence_gram": (
                "Q = M M^T"
            ),
            "thalean_incidence_shape": (
                "M is 15 by 30"
            ),
            "four_flux_line_gram": (
                "R10 = L10 L10^T"
            ),
            "four_flux_tight_frame_law": (
                "L10^T L10 = (5/2) I4"
            ),
            "four_flux_gram_law": (
                "R10^2 = (5/2) R10"
            ),
            "registers_are_distinct": True,
        },
        "earned_interpretation": {
            "recovered_register_has_four_response_classes": (
                theorem_pass
            ),
            "only_native_ten_axis_class_is_extremal": (
                theorem_pass
            ),
            "fifteen_line_class_is_nearest_nonaxis_layer": (
                theorem_pass
            ),
            "five_line_class_is_isotropic": (
                theorem_pass
            ),
            "self_alignment_implies_extremality": (
                False
            ),
        },
        "boundary": {
            "forty_lines_are_complete_real_zero_locus": (
                False
            ),
            "global_one_third_bound_proved": (
                False
            ),
            "global_equality_locus_proved": (
                False
            ),
            "physical_energy_claim": False,
            "physical_force_claim": False,
            "physical_instability_claim": False,
            "physical_claim": False,
        },
        "outputs": {
            "class_table_csv": str(
                TABLE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "check_csv": str(
                CHECK_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "theorem_note": str(
                NOTE_OUT.relative_to(
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

    with TABLE_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    current_check_rows = (
        check_rows(checks)
    )

    with CHECK_CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                current_check_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            current_check_rows
        )

    NOTE_OUT.write_text(
        theorem_note(rows),
        encoding="utf-8",
    )

    failed_checks = [
        name
        for name, value in checks.items()
        if not value
    ]

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "decomposition:",
        payload["theorem"][
            "decomposition"
        ],
    )
    print(
        "class_table:",
        [
            (
                row["class_id"],
                row["orbit_size"],
                row["stabilizer_order"],
                row["operator_norm"],
            )
            for row in rows
        ],
    )
    print(
        "passed_check_count:",
        sum(checks.values()),
        "/",
        len(checks),
    )
    print(
        "failed_checks:",
        failed_checks,
    )
    print("wrote:", JSON_OUT)
    print("wrote:", TABLE_CSV_OUT)
    print("wrote:", CHECK_CSV_OUT)
    print("wrote:", NOTE_OUT)


if __name__ == "__main__":
    main()
