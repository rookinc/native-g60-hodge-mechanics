"""Certify the normalized four-flux response on all native ordered edge pairs."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from math import isqrt
from pathlib import Path

import numpy as np

from native_inputs import BASE_COMMIT, DECK_IDS, ROOT, load, write

VERSION = "0.1.0"


def integer_seed(observed, denominator):
    candidate = np.rint(observed * denominator).astype(np.int64)
    assert np.max(np.abs(observed - candidate / denominator)) < 1e-12
    return candidate


def character(projector, denominator, targets, signs):
    indices = np.arange(projector.shape[0])
    traces = np.sum(signs * projector[indices, targets], axis=1)
    assert np.all(traces % denominator == 0)
    return traces // denominator


def archive_bound(observed, primitive, square_norm):
    """Absolute error enclosure for stored binary64 values against W/sqrt(S)."""
    scale = 2**64
    inverse_lower = isqrt(scale**2 // square_norm)
    inverse_upper = inverse_lower + 1
    assert inverse_lower**2 * square_norm <= scale**2 < inverse_upper**2 * square_norm
    bound = 0
    for value, coefficient in zip(observed.flat, primitive.flat):
        numerator, denominator = float(value).as_integer_ratio()
        stored_lower = numerator * scale // denominator
        stored_upper = -(-numerator * scale // denominator)
        ends = (int(coefficient) * inverse_lower, int(coefficient) * inverse_upper)
        bound = max(bound, abs(stored_lower - max(ends)), abs(stored_upper - min(ends)))
    return Fraction(bound, scale)


def flux_audit(data):
    b1, b2 = data["b1"], data["b2"]
    actions = data["actions"]
    vertices = actions["vertex_target"]
    edges, edge_sign = actions["edge_target"], actions["edge_sign"]
    faces, face_sign = actions["face_target"], actions["face_sign"]
    group = {row["actual_index"]: row["actual_permutation"]
             for row in data["group"]["mapping_rows"]}
    assert len(group) == 480 and all(np.array_equal(vertices[i], g) for i, g in group.items())

    harmonic = data["harmonic"]["H"]
    coupling = data["coupling"]
    u, v = (harmonic @ coupling[name] for name in ("six_a_basis", "six_b_basis"))
    response_basis = coupling["four_flux_basis"]
    operator = (response_basis @ coupling["unique_four_channel"]).reshape(20, 6, 6)
    observed = np.einsum("oij,ei,fj->oef", operator, u, v)
    assert abs(float(np.sum(observed**2)) - 1) < 1e-12

    # These are proposals from the numerical archive. The equations below certify
    # exact subspaces and a unique normalized intertwiner, independent of rounding.
    da, db, dc = 800, 60, 30
    pa = integer_seed(u @ u.T, da)
    pb = integer_seed(v @ v.T, db)
    pc = integer_seed(response_basis @ response_basis.T, dc)
    geometry = data["geometry"]
    assert np.array_equal(pc, integer_seed(geometry["face_flux_projector"], dc))
    w = integer_seed(observed / np.max(np.abs(observed)), 16)
    group_size = len(group)
    assert np.max(np.abs(w)) == 16
    assert np.gcd.reduce(w.ravel()) == 1
    for projector, denominator, rank in ((pa, da, 6), (pb, db, 6), (pc, dc, 4)):
        assert np.array_equal(projector.T, projector)
        assert np.array_equal(projector @ projector, denominator * projector)
        assert np.trace(projector) == rank * denominator
    assert not np.any(pa @ pb)
    for sector in (pa, pb):
        assert not np.any(b1 @ sector) and not np.any(b2.T @ sector)
    assert not np.any(pc.sum(axis=0))
    assert np.array_equal((b2.T @ b2) @ pc, 10 * pc)

    for g in range(group_size):
        vp, ep, fp = vertices[g], edges[g], faces[g]
        es, fs = edge_sign[g], face_sign[g]
        assert all(np.array_equal(np.sort(p), np.arange(len(p))) for p in (vp, ep, fp))
        assert set(es) <= {-1, 1} and set(fs) <= {-1, 1}
        assert np.array_equal(b1[np.ix_(vp, ep)], b1 * es[None, :])
        assert np.array_equal(b2[np.ix_(ep, fp)], b2 * es[:, None] * fs[None, :])
        for projector in (pa, pb):
            assert np.array_equal(projector[np.ix_(ep, ep)], projector * es[:, None] * es[None, :])
        assert np.array_equal(pc[np.ix_(fp, fp)], pc * fs[:, None] * fs[None, :])
        assert np.array_equal(w[np.ix_(fp, ep, ep)], w * fs[:, None, None] * es[None, :, None] * es)

    assert all(np.array_equal(pa @ sector, da * sector) for sector in w)
    assert all(np.array_equal(sector @ pb, db * sector) for sector in w)
    flat = w.reshape(20, -1)
    assert np.array_equal(pc @ flat, dc * flat)
    chi_a = character(pa, da, edges, edge_sign)
    chi_b = character(pb, db, edges, edge_sign)
    chi_c = character(pc, dc, faces, face_sign)
    assert np.max(np.abs(geometry["face_flux_character"] - chi_c)) < 1e-12
    assert all(int(chi @ chi) == group_size for chi in (chi_a, chi_b, chi_c))
    assert int(chi_a @ chi_b) == 0
    multiplicity_numerator = int(np.sum(chi_c * chi_a * chi_b))
    assert multiplicity_numerator == group_size
    # The integer matrices are small. Every dot product here has absolute
    # intermediate sum below 2**32, well inside exact signed int64 arithmetic.
    square_norm = int(np.sum(w * w))
    assert square_norm == 6912000
    assert np.array_equal(4 * dc * (flat @ flat.T), square_norm * pc)

    deck_actions = {}
    epsilon = []
    for g in range(group_size):
        p = vertices[g]
        moved_b = p[vertices[DECK_IDS["b"]]]
        if np.array_equal(moved_b, vertices[DECK_IDS["b"]][p]):
            epsilon.append(1)
        else:
            assert np.array_equal(moved_b, vertices[DECK_IDS["ab"]][p])
            epsilon.append(-1)
    assert Counter(epsilon) == {-1: 240, 1: 240}
    sign_character_numerator = int(np.sum(np.asarray(epsilon) * chi_c**2))
    assert sign_character_numerator == 0
    for name, g in DECK_IDS.items():
        multiplier = 1 if name in ("1", "a") else -1
        image = np.empty_like(pc)
        image[faces[g], :] = face_sign[g, :, None] * pc
        assert np.array_equal(image, multiplier * pc)
        deck_actions[name] = multiplier

    squared = np.sum(w**2, axis=0)
    assert all(np.array_equal(squared[np.ix_(p, p)], squared) for p in edges)
    pending = set(range(120**2))
    orbit_rows = []
    while pending:
        index = min(pending)
        e, f = divmod(index, 120)
        orbit = set((120 * edges[:, e] + edges[:, f]).tolist())
        assert orbit <= pending
        pending -= orbit
        orbit_rows.append({
            "representative": [e, f], "orbit_size": len(orbit),
            "intensity": str(Fraction(int(squared[e, f]), square_norm)),
        })
    counts = Counter(row["intensity"] for row in orbit_rows)
    assert len(orbit_rows) == 38
    assert counts == {"0": 6, "1/38400": 12, "1/23040": 7, "1/9600": 6, "23/115200": 7}

    samples = []
    for e, f in ((0, 5), (0, 17)):
        assert pa[e, f] == pb[e, f] == 0
        assert pa[e, e] == pa[f, f] == da // 20
        assert pb[e, e] == pb[f, f] == db // 20
        samples.append({
            "edge_pair": [e, f],
            "sector_A_inner": "0", "sector_B_inner": "0",
            "tensor_norm_squared": "1/400",
            "response_squared_norm": str(Fraction(int(squared[e, f]), square_norm)),
            "unscaled_face_vector": w[:, e, f].tolist(),
        })
    delta = Fraction(int(squared[0, 5]) - int(squared[0, 17]), square_norm)
    assert delta == Fraction(11, 115200)

    bound = archive_bound(observed, w, square_norm)
    assert bound < Fraction(1, 10**14)
    raw = data["response"]["face_flux_matrix"]
    raw_archive_residual = float(np.max(np.abs(
        raw / np.linalg.norm(raw) - operator.reshape(20, 36)
    )))
    # The two source audits may fix opposite signs for the one-dimensional channel.
    raw_archive_residual = min(
        raw_archive_residual, float(np.max(np.abs(raw / np.linalg.norm(raw) + operator.reshape(20, 36))))
    )
    assert raw_archive_residual < 1e-12

    return {
        "audit": "TOM-QM-027.F", "version": VERSION, "status": "pass",
        "project46_commit": BASE_COMMIT, "input_lock": "../inputs.lock.json",
        "method": "Exact signed equivariance, exact projector support, one-dimensional Hom space, unit Frobenius norm.",
        "tensor_order": "first edge in six_a, second edge in six_b; 6*a+b",
        "projector_denominators": [da, db, dc],
        "primitive_integer_tensor_shape": list(w.shape),
        "primitive_integer_tensor_norm_squared": square_norm,
        "output_laplacian_eigenvalue": 10,
        "character_sum_numerator": multiplicity_numerator,
        "hom_dimension": multiplicity_numerator // group_size,
        "normalized_response_covariance_rows_checked": group_size,
        "exact_comparison_samples": samples,
        "exact_intensity_difference": str(delta),
        "ordered_pair_orbit_count": len(orbit_rows),
        "orbits_per_intensity": dict(counts),
        "orbit_representatives": orbit_rows,
        "deck_action_on_four_flux_from_identity": deck_actions,
        "receipt_exchange_sign_End_four_flux_multiplicity": sign_character_numerator // group_size,
        "controls": {
            "equal_input_norms_unequal_output": True,
            "b_and_ab_identical_four_flux_actions": True,
            "relative_sign_absent_from_End_four_flux": True,
            "pure_density_conjugation_under_either_is_identity": True,
        },
        "numeric": {
            "componentwise_absolute_error_bound_for_binary64_replay_vs_exact_channel": str(bound),
            "componentwise_bound_decimal": float(bound),
            "raw_016_vs_unit_Frobenius_021_up_to_overall_sign_observed_residual": raw_archive_residual,
        },
        "boundary": "Squared norms use the unit-Frobenius four-flux convention of Audit 021. These are not probabilities.",
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Pinned Project 46 checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output / "flux-exact.json", flux_audit(load(args.root)))
