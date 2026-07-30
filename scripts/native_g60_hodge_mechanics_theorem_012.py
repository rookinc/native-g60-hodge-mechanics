from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_ROOT = ROOT / "artifacts" / "json"

ARTIFACT_PATHS = {
    "001": (
        ARTIFACT_ROOT
        / "native_g60_cochain_complex_import_001.json"
    ),
    "002": (
        ARTIFACT_ROOT
        / "native_g60_uniform_hodge_baseline_002.json"
    ),
    "003": (
        ARTIFACT_ROOT
        / "native_g60_hodge_decomposition_003.json"
    ),
    "004": (
        ARTIFACT_ROOT
        / "native_g60_harmonic_1_form_basis_004.json"
    ),
    "005": (
        ARTIFACT_ROOT
        / "native_g60_hodge_symmetry_commutation_005.json"
    ),
    "006": (
        ARTIFACT_ROOT
        / "native_g60_harmonic_representation_decomposition_006.json"
    ),
    "006b": (
        ARTIFACT_ROOT
        / "native_g60_harmonic_irreducible_decomposition_006b.json"
    ),
    "006c": (
        ARTIFACT_ROOT
        / "native_g60_harmonic_multiplicity_split_006c.json"
    ),
    "007": (
        ARTIFACT_ROOT
        / "native_g60_gauge_equivalence_audit_007.json"
    ),
    "008": (
        ARTIFACT_ROOT
        / "native_g60_static_field_response_008.json"
    ),
    "009": (
        ARTIFACT_ROOT
        / "native_g60_wave_operator_baseline_009.json"
    ),
    "010": (
        ARTIFACT_ROOT
        / "native_g60_discrete_energy_conservation_010.json"
    ),
    "011": (
        ARTIFACT_ROOT
        / "native_g60_weighted_hodge_candidate_search_011.json"
    ),
}

JSON_OUT = (
    ARTIFACT_ROOT
    / "native_g60_hodge_mechanics_theorem_012.json"
)

THEOREM_OUT = (
    ROOT
    / "theorems"
    / "native_g60_hodge_mechanics_theorem_012.md"
)

RECEIPT_OUT = (
    ROOT
    / "receipts"
    / "native_g60_hodge_mechanics_theorem_012.txt"
)


def nested_get(
    payload: dict[str, Any],
    keys: list[str],
) -> Any:
    value: Any = payload

    for key in keys:
        if not isinstance(value, dict):
            raise KeyError(
                f"cannot descend through non-dict at {key}"
            )

        value = value[key]

    return value


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    THEOREM_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    RECEIPT_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts = {}

    for artifact_id, path in ARTIFACT_PATHS.items():
        if not path.exists():
            raise SystemExit(
                f"missing required artifact {artifact_id}: {path}"
            )

        artifacts[artifact_id] = json.loads(
            path.read_text(encoding="utf-8")
        )

    artifact_passes = {
        artifact_id: (
            payload.get("audit_pass") is True
        )
        for artifact_id, payload in artifacts.items()
    }

    dimensions = {
        "C0": nested_get(
            artifacts["001"],
            ["chain_complex", "c0_dimension"],
        ),
        "C1": nested_get(
            artifacts["001"],
            ["chain_complex", "c1_dimension"],
        ),
        "C2": nested_get(
            artifacts["001"],
            ["chain_complex", "c2_dimension"],
        ),
        "exact": nested_get(
            artifacts["003"],
            ["decomposition", "exact_dimension"],
        ),
        "harmonic": nested_get(
            artifacts["003"],
            ["decomposition", "harmonic_dimension"],
        ),
        "coexact": nested_get(
            artifacts["003"],
            ["decomposition", "coexact_dimension"],
        ),
        "gauge_orbit": nested_get(
            artifacts["007"],
            ["gauge_structure", "gauge_orbit_dimension"],
        ),
        "gauge_quotient": nested_get(
            artifacts["007"],
            ["gauge_structure", "gauge_quotient_dimension"],
        ),
        "static_range": nested_get(
            artifacts["008"],
            ["solver_spectrum", "rank"],
        ),
        "static_kernel": nested_get(
            artifacts["008"],
            ["solver_spectrum", "kernel_dimension"],
        ),
        "wave_zero_modes": nested_get(
            artifacts["009"],
            ["spectrum", "zero_mode_count"],
        ),
        "wave_positive_modes": nested_get(
            artifacts["009"],
            ["spectrum", "positive_mode_count"],
        ),
    }

    irreducible_dimensions = sorted(
        nested_get(
            artifacts["006c"],
            [
                "final_decomposition",
                "dimensions",
            ],
        )
    )

    weighted = artifacts["011"]

    checks = {
        "all_required_artifacts_exist": (
            len(artifacts) == len(ARTIFACT_PATHS)
        ),
        "all_required_artifacts_pass": all(
            artifact_passes.values()
        ),
        "cochain_dimensions_are_60_120_20": (
            dimensions["C0"] == 60
            and dimensions["C1"] == 120
            and dimensions["C2"] == 20
        ),
        "boundary_of_boundary_is_zero": (
            nested_get(
                artifacts["001"],
                [
                    "chain_complex",
                    "b1_b2_nonzero_entry_count",
                ],
            )
            == 0
            and nested_get(
                artifacts["001"],
                [
                    "chain_complex",
                    "d1_d0_nonzero_entry_count",
                ],
            )
            == 0
        ),
        "hodge_dimensions_are_59_42_19": (
            dimensions["exact"] == 59
            and dimensions["harmonic"] == 42
            and dimensions["coexact"] == 19
        ),
        "hodge_dimensions_sum_to_120": (
            dimensions["exact"]
            + dimensions["harmonic"]
            + dimensions["coexact"]
            == 120
        ),
        "explicit_projectors_constructed": (
            nested_get(
                artifacts["003"],
                [
                    "boundary",
                    "explicit_hodge_projectors_constructed",
                ],
            )
            is True
        ),
        "harmonic_basis_dimension_is_42": (
            nested_get(
                artifacts["004"],
                ["harmonic_space", "dimension"],
            )
            == 42
        ),
        "full_native_group_equivariance_passes": (
            nested_get(
                artifacts["005"],
                [
                    "boundary",
                    "full_native_group_commutation_audited",
                ],
            )
            is True
        ),
        "harmonic_representation_is_faithful": (
            nested_get(
                artifacts["006"],
                [
                    "representation",
                    "kernel_order",
                ],
            )
            == 1
        ),
        "harmonic_character_norm_is_8": (
            nested_get(
                artifacts["006"],
                [
                    "character",
                    "norm_integer",
                ],
            )
            == 8
        ),
        "harmonic_irreducible_dimensions_are_2_6_6_8_10_10": (
            irreducible_dimensions
            == [2, 6, 6, 8, 10, 10]
        ),
        "harmonic_irreducible_dimensions_sum_to_42": (
            sum(irreducible_dimensions) == 42
        ),
        "gauge_dimensions_are_59_and_61": (
            dimensions["gauge_orbit"] == 59
            and dimensions["gauge_quotient"] == 61
        ),
        "gauge_quotient_splits_42_plus_19": (
            dimensions["gauge_quotient"]
            == dimensions["harmonic"]
            + dimensions["coexact"]
        ),
        "static_range_and_kernel_are_78_and_42": (
            dimensions["static_range"] == 78
            and dimensions["static_kernel"] == 42
        ),
        "wave_zero_and_positive_modes_are_42_and_78": (
            dimensions["wave_zero_modes"] == 42
            and dimensions["wave_positive_modes"] == 78
        ),
        "homogeneous_energy_conservation_proved": (
            nested_get(
                artifacts["010"],
                [
                    "boundary",
                    "homogeneous_energy_conservation_proved",
                ],
            )
            is True
        ),
        "forced_work_balance_derived": (
            nested_get(
                artifacts["010"],
                [
                    "boundary",
                    "forced_work_balance_derived",
                ],
            )
            is True
        ),
        "cell_actions_are_transitive": (
            nested_get(
                weighted,
                [
                    "cell_action_orbits",
                    "vertex_orbit_sizes",
                ],
            )
            == [60]
            and nested_get(
                weighted,
                [
                    "cell_action_orbits",
                    "edge_orbit_sizes",
                ],
            )
            == [120]
            and nested_get(
                weighted,
                [
                    "cell_action_orbits",
                    "face_orbit_sizes",
                ],
            )
            == [20]
        ),
        "diagonal_hodge_family_has_three_positive_scalars": (
            nested_get(
                weighted,
                [
                    "exact_diagonal_classification",
                    "raw_parameter_count",
                ],
            )
            == 3
        ),
        "weighted_operators_have_two_effective_ratios": (
            nested_get(
                weighted,
                [
                    "exact_diagonal_classification",
                    "effective_operator_parameter_count",
                ],
            )
            == 2
        ),
        "native_symmetry_does_not_select_unique_numeric_weights": (
            nested_get(
                weighted,
                [
                    "earned_interpretation",
                    "native_symmetry_selects_unique_numeric_weights",
                ],
            )
            is False
        ),
    }

    theorem_pass = all(checks.values())

    theorem_statement = (
        "On the imported native genus-21 surface of AT4val[60,6], "
        "the full native automorphism group preserves a positive "
        "uniform discrete Hodge complex whose edge-cochain space "
        "decomposes orthogonally into exact, harmonic, and coexact "
        "subspaces of dimensions 59, 42, and 19. The harmonic "
        "subspace carries a faithful 42-dimensional group "
        "representation with irreducible dimensions "
        "2, 6, 6, 8, 10, and 10. Gauge equivalence removes exactly "
        "the 59-dimensional exact sector, leaving a 61-dimensional "
        "quotient formed by 42 harmonic and 19 coexact directions. "
        "The edge Laplacian has rank 78 and harmonic kernel dimension "
        "42, admits the stated static solvability condition, and "
        "defines a finite homogeneous second-order dynamics with a "
        "conserved quadratic invariant. Full native symmetry forces "
        "every positive diagonal Hodge structure to have the form "
        "star0=aI60, star1=bI120, star2=cI20, while leaving the two "
        "effective positive ratios b/a and c/b undetermined."
    )

    boundary_statement = (
        "The theorem concerns a finite combinatorial Hodge mechanics. "
        "It does not derive a metric geometry, physical units, "
        "constitutive ratios, electromagnetism, photons, spacetime, "
        "quantum mechanics, gravity, physical forces, physical energy, "
        "a simulation of the physical universe, or a unification."
    )

    payload = {
        "artifact_id": (
            "native_g60_hodge_mechanics_theorem_012"
        ),
        "theorem_pass": theorem_pass,
        "audit_pass": theorem_pass,
        "verdict": (
            "native_g60_finite_hodge_mechanics_theorem_packaged"
            if theorem_pass
            else "native_g60_hodge_mechanics_theorem_package_failed"
        ),
        "required_artifacts": {
            artifact_id: {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "audit_pass": artifact_passes[
                    artifact_id
                ],
                "verdict": artifacts[
                    artifact_id
                ].get("verdict"),
            }
            for artifact_id, path in (
                ARTIFACT_PATHS.items()
            )
        },
        "checks": checks,
        "theorem": {
            "title": (
                "Native finite Hodge mechanics on the "
                "genus-21 surface of AT4val[60,6]"
            ),
            "statement": theorem_statement,
            "chain_complex": {
                "dimensions": {
                    "C0": 60,
                    "C1": 120,
                    "C2": 20,
                },
                "operators": {
                    "d0": "B1^T",
                    "d1": "B2^T",
                    "identity": "d1 d0 = 0",
                },
            },
            "uniform_hodge_decomposition": {
                "exact_dimension": 59,
                "harmonic_dimension": 42,
                "coexact_dimension": 19,
                "dimension_identity": (
                    "120 = 59 + 42 + 19"
                ),
            },
            "harmonic_representation": {
                "dimension": 42,
                "faithful": True,
                "character_norm": 8,
                "irreducible_dimensions": (
                    irreducible_dimensions
                ),
                "decomposition": (
                    "2 + 6 + 6 + 8 + 10 + 10"
                ),
                "central_halfturn_split": {
                    "plus_dimension": 12,
                    "minus_dimension": 30,
                },
            },
            "gauge_structure": {
                "gauge_orbit_dimension": 59,
                "gauge_quotient_dimension": 61,
                "quotient_decomposition": (
                    "61 = 42 harmonic + 19 coexact"
                ),
                "curvature_complete_classifier": False,
            },
            "static_mechanics": {
                "equation": "Delta1 A = J",
                "operator_rank": 78,
                "harmonic_kernel_dimension": 42,
                "solvability_condition": (
                    "P_harmonic J = 0"
                ),
            },
            "finite_dynamics": {
                "equation": (
                    "A_double_dot + Delta1 A = J(t)"
                ),
                "zero_mode_dimension": 42,
                "oscillatory_dimension": 78,
                "homogeneous_energy_balance": (
                    "dE/dt = 0"
                ),
                "forced_energy_balance": (
                    "dE/dt = <A_dot,J>"
                ),
            },
            "symmetry_compatible_diagonal_family": {
                "star0": "a I60",
                "star1": "b I120",
                "star2": "c I20",
                "conditions": "a>0, b>0, c>0",
                "effective_ratios": [
                    "x=b/a",
                    "y=c/b",
                ],
                "unique_numeric_point_selected": False,
            },
        },
        "claim_boundary": {
            "statement": boundary_statement,
            "surface_source_remains_project45": True,
            "surface_rederived_in_project46": False,
            "finite_hodge_mechanics_established": theorem_pass,
            "uniform_hodge_theorem_established": theorem_pass,
            "full_symmetry_equivariance_established": theorem_pass,
            "gauge_structure_established": theorem_pass,
            "static_solvability_boundary_established": theorem_pass,
            "finite_wave_baseline_established": theorem_pass,
            "mathematical_energy_balance_established": theorem_pass,
            "symmetry_compatible_diagonal_family_classified": theorem_pass,
            "unique_constitutive_ratios_derived": False,
            "non_diagonal_positive_hodge_commutant_classified": False,
            "native_metric_derived": False,
            "physical_units_derived": False,
            "physical_constants_derived": False,
            "physical_time_scale_derived": False,
            "physical_energy_claim": False,
            "electromagnetism_claim": False,
            "maxwell_claim": False,
            "photon_claim": False,
            "spacetime_claim": False,
            "quantum_claim": False,
            "gravity_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False,
        },
        "next_frontier": {
            "primary": (
                "Classify the full positive self-adjoint commutant "
                "of the native cell actions, beyond diagonal weights."
            ),
            "secondary": (
                "Determine whether construction data selects the "
                "remaining positive ratios x=b/a and y=c/b."
            ),
            "forbidden_shortcut": (
                "Do not assign physical units or familiar field names "
                "to the free ratios without an independent derivation."
            ),
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

    markdown = rf"""# Native G60 Hodge Mechanics Theorem 012

## Status

- theorem_pass: `{str(theorem_pass).lower()}`
- verdict: `{payload["verdict"]}`

## Theorem

{theorem_statement}

## Exact finite structure

\[
C^0 \\xrightarrow{{d_0}} C^1 \\xrightarrow{{d_1}} C^2,
\\qquad
d_1d_0=0,
\]

with

\[
\\dim C^0=60,
\\qquad
\\dim C^1=120,
\\qquad
\\dim C^2=20.
\]

For the uniform positive Hodge baseline,

\[
C^1
=
\\operatorname{{im}}d_0
\\oplus
\\ker\\Delta_1
\\oplus
\\operatorname{{im}}d_1^{{\\mathsf T}},
\]

and

\[
120=59+42+19.
\]

## Harmonic symmetry

The native group acts faithfully on the harmonic sector, with concrete
irreducible dimensions

\[
42=2+6+6+8+10+10.
\]

The central half-turn splits the harmonic space as

\[
42=12_+ + 30_-.
\]

## Gauge quotient

\[
C^1/\\operatorname{{im}}d_0
\\cong
\\mathcal H^1
\\oplus
\\operatorname{{im}}d_1^{{\\mathsf T}},
\]

with

\[
61=42+19.
\]

Face curvature is gauge invariant but does not classify the harmonic
part of the gauge quotient.

## Static and finite dynamics

The static equation

\[
\\Delta_1A=J
\]

is solvable precisely when

\[
P_{{\\mathrm{{harmonic}}}}J=0.
\]

The bounded second-order system is

\[
\\ddot A+\\Delta_1A=J(t).
\]

For the homogeneous system,

\[
\\frac{{dE}}{{dt}}=0.
\]

For the forced system,

\[
\\frac{{dE}}{{dt}}
=
\\langle\\dot A,J\\rangle.
\]

## Symmetry-compatible diagonal family

Full native symmetry forces

\[
\\star_0=aI_{{60}},
\\qquad
\\star_1=bI_{{120}},
\\qquad
\\star_2=cI_{{20}},
\]

with

\[
a>0,
\\qquad
b>0,
\\qquad
c>0.
\]

The operators depend only on

\[
x=\\frac ba,
\\qquad
y=\\frac cb.
\]

Native symmetry determines the form of the diagonal mechanics, but not
the numerical point inside this positive two-ratio family.

## Boundary

{boundary_statement}

## Next frontier

Classify the full positive self-adjoint commutant beyond diagonal
weights, then test whether native construction data selects the free
ratios \(x\) and \(y\).
"""

    THEOREM_OUT.write_text(
        markdown,
        encoding="utf-8",
    )

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    receipt_lines = [
        "artifact_id: native_g60_hodge_mechanics_theorem_012",
        f"theorem_pass: {str(theorem_pass).lower()}",
        f"verdict: {payload['verdict']}",
        f"required_artifact_count: {len(artifacts)}",
        f"required_artifact_pass_count: {sum(artifact_passes.values())}",
        f"failed_check_count: {len(failed_checks)}",
        "failed_checks: "
        + (
            ",".join(failed_checks)
            if failed_checks
            else "none"
        ),
        "hodge_dimensions: 59,42,19",
        "harmonic_irreducible_dimensions: 2,6,6,8,10,10",
        "gauge_dimensions: 59,61",
        "static_rank_kernel: 78,42",
        "wave_zero_positive_modes: 42,78",
        "diagonal_hodge_parameters: a,b,c",
        "effective_operator_ratios: b/a,c/b",
        "physical_claim: false",
        "maxwell_claim: false",
        "universe_simulation_claim: false",
    ]

    RECEIPT_OUT.write_text(
        "\n".join(receipt_lines) + "\n",
        encoding="utf-8",
    )

    print("theorem_pass:", theorem_pass)
    print("audit_pass:", theorem_pass)
    print("verdict:", payload["verdict"])
    print(
        "required_artifact_pass_count:",
        sum(artifact_passes.values()),
        "/",
        len(artifact_passes),
    )
    print("failed_check_count:", len(failed_checks))
    print("failed_checks:", failed_checks)
    print(
        "hodge_dimensions:",
        dimensions["exact"],
        dimensions["harmonic"],
        dimensions["coexact"],
    )
    print(
        "harmonic_irreducible_dimensions:",
        irreducible_dimensions,
    )
    print(
        "gauge_orbit/quotient:",
        dimensions["gauge_orbit"],
        dimensions["gauge_quotient"],
    )
    print(
        "static_range/kernel:",
        dimensions["static_range"],
        dimensions["static_kernel"],
    )
    print(
        "wave_zero/positive_modes:",
        dimensions["wave_zero_modes"],
        dimensions["wave_positive_modes"],
    )
    print(
        "weighted_family:",
        "star0=aI60 star1=bI120 star2=cI20",
    )
    print(
        "effective_ratios:",
        "b/a",
        "c/b",
    )
    print("wrote:", JSON_OUT)
    print("wrote:", THEOREM_OUT)
    print("wrote:", RECEIPT_OUT)


if __name__ == "__main__":
    main()
