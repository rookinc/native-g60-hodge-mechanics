from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

THEOREM_JSON_PATH = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_self_alignment_register_theorem_041.json"
)

ORIENTATION_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_covariant_orientation_035.npz"
)

PENCIL_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_operator_pencil_026.npz"
)

ZERO_NPZ_PATH = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_covariant_residual_zero_locus_036.npz"
)

JSON_OUT = (
    ROOT
    / "artifacts/json"
    / "native_g60_cross_flux_one_third_gap_probe_042.json"
)

SAMPLE_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_one_third_gap_samples_042.csv"
)

REGISTER_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_one_third_gap_register_042.csv"
)

REGRESSION_CSV_OUT = (
    ROOT
    / "artifacts/csv"
    / "native_g60_cross_flux_one_third_gap_regressions_042.csv"
)

NPZ_OUT = (
    ROOT
    / "artifacts/npy"
    / "native_g60_cross_flux_one_third_gap_probe_042.npz"
)

SAVED_SAMPLE_COUNT = 5000
PSD_TOLERANCE = 2e-10


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

    return float(np.max(np.abs(array)))


def pencil_grams(
    slices: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    matrices = np.einsum(
        "nr,rab->nab",
        directions,
        slices,
    )

    grams = np.einsum(
        "nai,naj->nij",
        matrices,
        matrices,
    )

    return 0.5 * (
        grams
        + np.swapaxes(
            grams,
            1,
            2,
        )
    )


def elementary_symmetric_values(
    eigenvalues: np.ndarray,
) -> np.ndarray:
    count = len(eigenvalues)

    values = np.zeros(
        count,
        dtype=np.float64,
    )

    values[0] = 1.0

    for eigenvalue in eigenvalues:
        for degree in range(
            count - 1,
            0,
            -1,
        ):
            values[degree] += (
                eigenvalue
                * values[
                    degree - 1
                ]
            )

    return values


def regression_summary(
    name: str,
    design: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, np.ndarray]:
    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    prediction = design @ coefficients
    residual = target - prediction

    centered = (
        target - np.mean(target)
    )

    denominator = float(
        np.dot(
            centered,
            centered,
        )
    )

    r_squared = (
        1.0
        - float(
            np.dot(
                residual,
                residual,
            )
        )
        / denominator
        if denominator > 0.0
        else 1.0
    )

    return (
        {
            "model": name,
            "feature_names": json.dumps(
                feature_names
            ),
            "design_rank": int(rank),
            "coefficients": json.dumps(
                coefficients.tolist()
            ),
            "r_squared": r_squared,
            "root_mean_square_residual": float(
                np.sqrt(
                    np.mean(
                        residual**2
                    )
                )
            ),
            "maximum_absolute_residual": max_abs(
                residual
            ),
            "minimum_residual": float(
                np.min(residual)
            ),
            "maximum_residual": float(
                np.max(residual)
            ),
        },
        prediction,
    )


def analyze_directions(
    slices: np.ndarray,
    directions: np.ndarray,
) -> dict[str, np.ndarray]:
    grams = pencil_grams(
        slices,
        directions,
    )

    norms_squared = np.einsum(
        "ni,ni->n",
        directions,
        directions,
    )

    gap_matrices = (
        norms_squared[:, None, None]
        / 9.0
        * np.eye(6)[None, :, :]
        - grams
    )

    gap_eigenvalues = np.linalg.eigvalsh(
        gap_matrices
    )

    gram_eigenvalues = np.linalg.eigvalsh(
        grams
    )

    determinants = np.linalg.det(
        gap_matrices
    )

    elementary = np.array(
        [
            elementary_symmetric_values(
                values
            )
            for values in gap_eigenvalues
        ],
        dtype=np.float64,
    )

    return {
        "grams": grams,
        "gram_eigenvalues": (
            gram_eigenvalues
        ),
        "gap_matrices": (
            gap_matrices
        ),
        "gap_eigenvalues": (
            gap_eigenvalues
        ),
        "minimum_gap": (
            gap_eigenvalues[:, 0]
        ),
        "second_gap": (
            gap_eigenvalues[:, 1]
        ),
        "determinant": determinants,
        "elementary": elementary,
        "operator_norm": np.sqrt(
            np.maximum(
                gram_eigenvalues[:, -1],
                0.0,
            )
        ),
    }


def build_regression_models(
    s4: np.ndarray,
    s6: np.ndarray,
    s8: np.ndarray,
    residual_squared: np.ndarray,
) -> dict[
    str,
    tuple[np.ndarray, list[str]],
]:
    ones = np.ones(
        len(s4),
        dtype=np.float64,
    )

    return {
        "moments_linear": (
            np.column_stack(
                [
                    ones,
                    s4,
                    s6,
                    s8,
                    residual_squared,
                ]
            ),
            [
                "1",
                "S4",
                "S6",
                "S8",
                "||r||^2",
            ],
        ),
        "moments_quadratic": (
            np.column_stack(
                [
                    ones,
                    s4,
                    s6,
                    s8,
                    residual_squared,
                    s4**2,
                    s4 * s6,
                    s4
                    * residual_squared,
                    s6**2,
                    residual_squared**2,
                ]
            ),
            [
                "1",
                "S4",
                "S6",
                "S8",
                "||r||^2",
                "S4^2",
                "S4*S6",
                "S4*||r||^2",
                "S6^2",
                "||r||^4",
            ],
        ),
        "moments_cubic": (
            np.column_stack(
                [
                    ones,
                    s4,
                    s6,
                    s8,
                    residual_squared,
                    s4**2,
                    s4 * s6,
                    s4
                    * residual_squared,
                    s6**2,
                    residual_squared**2,
                    s4**3,
                    s4**2 * s6,
                    s4 * s6**2,
                    s4**2
                    * residual_squared,
                    s4
                    * residual_squared**2,
                ]
            ),
            [
                "1",
                "S4",
                "S6",
                "S8",
                "||r||^2",
                "S4^2",
                "S4*S6",
                "S4*||r||^2",
                "S6^2",
                "||r||^4",
                "S4^3",
                "S4^2*S6",
                "S4*S6^2",
                "S4^2*||r||^2",
                "S4*||r||^4",
            ],
        ),
    }


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGISTER_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGRESSION_CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NPZ_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    theorem_receipt = json.loads(
        THEOREM_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    orientation_data = np.load(
        ORIENTATION_NPZ_PATH
    )

    pencil_data = np.load(
        PENCIL_NPZ_PATH
    )

    zero_data = np.load(
        ZERO_NPZ_PATH
    )

    directions = np.array(
        orientation_data[
            "random_directions"
        ],
        dtype=np.float64,
    )

    s4 = np.array(
        orientation_data["s4"],
        dtype=np.float64,
    )

    s6 = np.array(
        orientation_data["s6"],
        dtype=np.float64,
    )

    s8 = np.array(
        orientation_data["s8"],
        dtype=np.float64,
    )

    residual_squared = np.array(
        orientation_data[
            "residual_norm_squared"
        ],
        dtype=np.float64,
    )

    slices = np.array(
        pencil_data["slices"],
        dtype=np.float64,
    )

    register_lines = np.array(
        zero_data[
            "root_representatives"
        ],
        dtype=np.float64,
    )

    sample_analysis = analyze_directions(
        slices,
        directions,
    )

    register_analysis = analyze_directions(
        slices,
        register_lines,
    )

    minimum_sample_gap = float(
        np.min(
            sample_analysis[
                "minimum_gap"
            ]
        )
    )

    negative_sample_gap_count = int(
        np.count_nonzero(
            sample_analysis[
                "minimum_gap"
            ]
            < -PSD_TOLERANCE
        )
    )

    register_zero_gap_count = int(
        np.count_nonzero(
            np.abs(
                register_analysis[
                    "minimum_gap"
                ]
            )
            < PSD_TOLERANCE
        )
    )

    models = build_regression_models(
        s4,
        s6,
        s8,
        residual_squared,
    )

    regression_rows = []

    for target_name, target in (
        (
            "minimum_gap",
            sample_analysis[
                "minimum_gap"
            ],
        ),
        (
            "gap_determinant",
            sample_analysis[
                "determinant"
            ],
        ),
        (
            "second_elementary",
            sample_analysis[
                "elementary"
            ][:, 2],
        ),
    ):
        for model_name, (
            design,
            feature_names,
        ) in models.items():
            row, _ = regression_summary(
                (
                    target_name
                    + "__"
                    + model_name
                ),
                design,
                target,
                feature_names,
            )

            row["target"] = (
                target_name
            )

            regression_rows.append(row)

    saved_indices = np.linspace(
        0,
        len(directions) - 1,
        SAVED_SAMPLE_COUNT,
        dtype=np.int64,
    )

    sample_rows = []

    for sample_id, source_index in enumerate(
        saved_indices
    ):
        gaps = sample_analysis[
            "gap_eigenvalues"
        ][source_index]

        elementary = sample_analysis[
            "elementary"
        ][source_index]

        sample_rows.append(
            {
                "sample_id": sample_id,
                "source_index": int(
                    source_index
                ),
                "s4": float(
                    s4[source_index]
                ),
                "s6": float(
                    s6[source_index]
                ),
                "s8": float(
                    s8[source_index]
                ),
                "residual_norm_squared": float(
                    residual_squared[
                        source_index
                    ]
                ),
                "operator_norm": float(
                    sample_analysis[
                        "operator_norm"
                    ][source_index]
                ),
                "gap_0": float(gaps[0]),
                "gap_1": float(gaps[1]),
                "gap_2": float(gaps[2]),
                "gap_3": float(gaps[3]),
                "gap_4": float(gaps[4]),
                "gap_5": float(gaps[5]),
                "gap_determinant": float(
                    sample_analysis[
                        "determinant"
                    ][source_index]
                ),
                "elementary_1": float(
                    elementary[1]
                ),
                "elementary_2": float(
                    elementary[2]
                ),
                "elementary_3": float(
                    elementary[3]
                ),
                "elementary_4": float(
                    elementary[4]
                ),
                "elementary_5": float(
                    elementary[5]
                ),
            }
        )

    register_rows = []

    for line_id, line in enumerate(
        register_lines
    ):
        gaps = register_analysis[
            "gap_eigenvalues"
        ][line_id]

        register_rows.append(
            {
                "line_id": line_id,
                "operator_norm": float(
                    register_analysis[
                        "operator_norm"
                    ][line_id]
                ),
                "minimum_gap": float(
                    gaps[0]
                ),
                "second_gap": float(
                    gaps[1]
                ),
                "gap_determinant": float(
                    register_analysis[
                        "determinant"
                    ][line_id]
                ),
                "zero_gap_multiplicity": int(
                    np.count_nonzero(
                        np.abs(gaps)
                        < PSD_TOLERANCE
                    )
                ),
                "f0": float(line[0]),
                "f1": float(line[1]),
                "f2": float(line[2]),
                "f3": float(line[3]),
            }
        )

    best_regression = max(
        regression_rows,
        key=lambda row: row[
            "r_squared"
        ],
    )

    checks = {
        "input_041_theorem_pass": (
            theorem_receipt.get(
                "theorem_pass"
            )
            is True
        ),
        "sample_direction_count_matches": (
            len(directions)
            == len(s4)
            == len(s6)
            == len(s8)
        ),
        "register_line_count_is_40": (
            len(register_lines) == 40
        ),
        "no_sampled_negative_gap": (
            negative_sample_gap_count == 0
        ),
        "all_gap_traces_are_five_twelfths": (
            max_abs(
                np.sum(
                    sample_analysis[
                        "gap_eigenvalues"
                    ],
                    axis=1,
                )
                - 5.0 / 12.0
            )
            < PSD_TOLERANCE
        ),
        "regression_scan_completed": (
            len(regression_rows)
            == 9
        ),
    }

    audit_pass = all(
        checks.values()
    )

    theorem_pass = False

    verdict = (
        "native_g60_cross_flux_one_third_gap_probe_completed"
        if audit_pass
        else "native_g60_cross_flux_one_third_gap_probe_failed"
    )

    payload = {
        "artifact_id": (
            "native_g60_cross_flux_one_third_gap_probe_042"
        ),
        "audit_pass": audit_pass,
        "theorem_pass": theorem_pass,
        "verdict": verdict,
        "definition": {
            "pencil_gram": (
                "B(f)=A(f)^T A(f)"
            ),
            "one_third_gap_matrix": (
                "G(f)=(||f||^2/9)I6-B(f)"
            ),
            "target": (
                "prove G(f) is positive semidefinite"
            ),
        },
        "sample_scan": {
            "sample_count": len(
                directions
            ),
            "minimum_sample_gap": (
                minimum_sample_gap
            ),
            "negative_gap_count": (
                negative_sample_gap_count
            ),
            "maximum_sample_operator_norm": float(
                np.max(
                    sample_analysis[
                        "operator_norm"
                    ]
                )
            ),
            "minimum_gap_determinant": float(
                np.min(
                    sample_analysis[
                        "determinant"
                    ]
                )
            ),
        },
        "recovered_register": {
            "line_count": len(
                register_lines
            ),
            "zero_minimum_gap_count": (
                register_zero_gap_count
            ),
            "minimum_gap_range": [
                float(
                    np.min(
                        register_analysis[
                            "minimum_gap"
                        ]
                    )
                ),
                float(
                    np.max(
                        register_analysis[
                            "minimum_gap"
                        ]
                    )
                ),
            ],
        },
        "regression_scan": {
            "model_count": len(
                regression_rows
            ),
            "best_model": (
                best_regression[
                    "model"
                ]
            ),
            "best_target": (
                best_regression[
                    "target"
                ]
            ),
            "best_r_squared": (
                best_regression[
                    "r_squared"
                ]
            ),
            "best_maximum_residual": (
                best_regression[
                    "maximum_absolute_residual"
                ]
            ),
            "models": regression_rows,
        },
        "checks": checks,
        "earned_interpretation": {
            "sampled_global_one_third_bound_survives": (
                negative_sample_gap_count
                == 0
            ),
            "gap_matrix_is_the_correct_global_object": (
                True
            ),
            "compact_invariant_gap_formula_found": (
                best_regression[
                    "maximum_absolute_residual"
                ]
                < 1e-9
            ),
            "global_one_third_bound_proved": (
                False
            ),
        },
        "boundary": {
            "global_gap_probe_completed": (
                audit_pass
            ),
            "positive_semidefinite_gap_proved": (
                False
            ),
            "complete_equality_locus_proved": (
                False
            ),
            "physical_claim": False,
        },
        "outputs": {
            "sample_csv": str(
                SAMPLE_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "register_csv": str(
                REGISTER_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "regression_csv": str(
                REGRESSION_CSV_OUT.relative_to(
                    ROOT
                )
            ),
            "gap_npz": str(
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

    for path, rows in (
        (
            SAMPLE_CSV_OUT,
            sample_rows,
        ),
        (
            REGISTER_CSV_OUT,
            register_rows,
        ),
        (
            REGRESSION_CSV_OUT,
            regression_rows,
        ),
    ):
        with path.open(
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

    np.savez_compressed(
        NPZ_OUT,
        random_directions=directions,
        sample_gap_eigenvalues=(
            sample_analysis[
                "gap_eigenvalues"
            ]
        ),
        sample_gap_determinants=(
            sample_analysis[
                "determinant"
            ]
        ),
        sample_gap_elementary=(
            sample_analysis[
                "elementary"
            ]
        ),
        register_lines=register_lines,
        register_gap_eigenvalues=(
            register_analysis[
                "gap_eigenvalues"
            ]
        ),
        register_gap_determinants=(
            register_analysis[
                "determinant"
            ]
        ),
    )

    print("audit_pass:", audit_pass)
    print("theorem_pass:", theorem_pass)
    print("verdict:", verdict)
    print(
        "sample_count:",
        len(directions),
    )
    print(
        "minimum_sample_gap:",
        minimum_sample_gap,
    )
    print(
        "negative_sample_gap_count:",
        negative_sample_gap_count,
    )
    print(
        "maximum_sample_operator_norm:",
        payload[
            "sample_scan"
        ][
            "maximum_sample_operator_norm"
        ],
    )
    print(
        "register_zero_gap_count:",
        register_zero_gap_count,
    )
    print(
        "best_regression:",
        {
            "target": (
                best_regression[
                    "target"
                ]
            ),
            "model": (
                best_regression[
                    "model"
                ]
            ),
            "r_squared": (
                best_regression[
                    "r_squared"
                ]
            ),
            "max_residual": (
                best_regression[
                    "maximum_absolute_residual"
                ]
            ),
        },
    )
    print("wrote:", JSON_OUT)
    print("wrote:", SAMPLE_CSV_OUT)
    print("wrote:", REGISTER_CSV_OUT)
    print("wrote:", REGRESSION_CSV_OUT)
    print("wrote:", NPZ_OUT)


if __name__ == "__main__":
    main()
