"""Audit the reference-corrected two-register lift on the Project 46 cover."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v0.1.0"))

from audit_contacts import Cover, census, witness
from native_inputs import DECK_IDS, ROOT, load, write


LOCK = json.loads((HERE / "inputs.lock.json").read_text())
VERSION = "0.2.0"


def pinned_json(path, entry):
    data = path.read_bytes()
    if sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Native input hash mismatch for {path}")
    return json.loads(data)


def check_native_fixture(cover, root):
    entry = LOCK["native_turn_fixture"]
    fixture = pinned_json(root / entry["path"], entry)
    assert {tuple(sorted(edge)) for edge in fixture["edges"]} == cover.edges
    assert {tuple(row["p"]) for row in fixture["automorphisms"]} == set(cover.group.values())
    assert all(fixture["deck"]["identity" if name == "1" else name] == list(p)
               for name, p in cover.deck.items())
    a = cover.deck["a"]
    assert all(g[a[x]] == a[g[x]] for g in cover.group.values() for x in range(60))
    assert all(a[x] != x for x in range(60))


def equalized_paths(cover, short, long, contact_b):
    origin = short[0]
    assert long[0] == origin == short[-1] == long[-1]
    assert contact_b in cover.neighbors[origin]
    (triangle_site,) = cover.neighbors[origin] & cover.neighbors[contact_b]
    triangle = (origin, contact_b, triangle_site, origin)
    retrace = (origin, contact_b, origin)
    assert cover.holonomy(triangle) == cover.holonomy(retrace) == "1"
    assert origin not in cover.neighbors[origin]
    # Length one cannot be inserted. Lengths eight and nine would each
    # require a length-one insertion in one of the two existing branches.
    assert (len(short) - 1, len(long) - 1) == (7, 8)
    assert len(triangle[:-1] + short) == len(retrace[:-1] + long) == 11
    return triangle[:-1] + short, retrace[:-1] + long


def synchronous_lift(cover, walk, system, pointer):
    """One base itinerary drives both lifts through their native edge maps."""
    for edge in zip(walk, walk[1:]):
        transport = cover.edge_maps[edge]
        system, pointer = transport[system], transport[pointer]
    return system, pointer


def pointer_write(cover, route, reference, system, pointer):
    system, pointer = synchronous_lift(cover, route, system, pointer)
    pointer = cover.lift(tuple(reversed(reference)), pointer)[-1]
    system = cover.lift(tuple(reversed(route)), system)[-1]
    return system, pointer


def undo_pointer_write(cover, route, reference, system, pointer):
    system = cover.lift(route, system)[-1]
    pointer = cover.lift(reference, pointer)[-1]
    return synchronous_lift(cover, tuple(reversed(route)), system, pointer)


def verify_readout(cover, sample):
    origin, contact_b, _, _ = sample["contact_sites"]
    short, long = sorted(
        (tuple(row["quotient_walk"]) for row in sample["three_perimeters"]
         if row["deck_holonomy"] in ("b", "ab")), key=len
    )
    reference, alternative = equalized_paths(cover, short, long, contact_b)
    later = cover.compare(origin, contact_b)
    assert (cover.holonomy(reference), cover.holonomy(alternative)) == ("ab", "b")
    a = cover.deck["a"]
    tested = 0
    for i, g in cover.group.items():
        q = cover.quotient_actions[i]
        mapped_short, mapped_long, mapped_ref, mapped_alt, mapped_later = (
            tuple(q[x] for x in path) for path in (short, long, reference, alternative, later)
        )
        assert equalized_paths(cover, mapped_short, mapped_long, q[contact_b]) == (mapped_ref, mapped_alt)
        assert cover.compare(q[origin], q[contact_b]) == mapped_later
        for system, pointer, branch in product(cover.fibers[origin], cover.fibers[origin], (0, 1)):
            route = (mapped_ref, mapped_alt)[branch]
            system_in, pointer_in = g[system], g[pointer]
            result = pointer_write(cover, route, mapped_ref, system_in, pointer_in)
            pointer_out = (pointer_in, a[pointer_in])[branch]
            assert result == (system_in, pointer_out)
            assert undo_pointer_write(cover, route, mapped_ref, *result) == (system_in, pointer_in)
            later_blank = cover.lift(mapped_later, pointer_in)[-1]
            later_pointer = cover.lift(mapped_later, pointer_out)[-1]
            assert later_pointer == (later_blank, a[later_blank])[branch]
            tested += 1
    assert tested == 15360
    start = sample["source_g60_vertex_for_example"]
    later_map = cover.transport(later)
    display = []
    for route in (reference, alternative):
        system_out, pointer_out = pointer_write(cover, route, reference, start, start)
        display.append({
            "holonomy": cover.holonomy(route),
            "system_initial": start,
            "pointer_initial": start,
            "common_route": route,
            "system_after_return": cover.lift(route, start)[-1],
            "system_after_reverse_replay": system_out,
            "pointer_after_reference_correction": pointer_out,
            "pointer_at_later_contact": later_map[pointer_out],
        })
    fixed_route_controls = [
        (cover.lift(route, start)[-1], cover.lift(reference, start)[-1])
        for route in (reference, alternative)
    ]
    assert fixed_route_controls[0][0] != fixed_route_controls[1][0]
    assert fixed_route_controls[0][1] == fixed_route_controls[1][1]
    return {
        "checks_across_group_and_register_inputs": tested,
        "all_transports_reversed_exactly": True,
        "pointer_stays_distinguishable_through_later_lift": True,
        "pointer_action": ["1", "a"],
        "central_deck_involution_is_fixed_point_free": True,
        "pointer_synchronization": "The occupied route drives two lifts on independent carrier copies.",
        "required_control": "Common branch itinerary, fixed pointer reference, and independently addressable system reversal.",
        "balanced_itinerary_lengths": [len(reference) - 1, len(alternative) - 1],
        "minimum_for_based_flat_padding": 10,
        "examples": display,
        "control": "When the pointer follows one fixed itinerary it has the same endpoint for both histories.",
        "independent_itineraries_control_endpoints": fixed_route_controls,
    }


def triangle_census(vertices, edges):
    graph = {v: set() for v in vertices}
    for x, y in edges:
        assert x != y
        graph[x].add(y)
        graph[y].add(x)
    by_degree = Counter(map(len, graph.values()))
    by_triangles = Counter(
        sum(len(graph[v] & graph[w]) for w in graph[v]) // 2 for v in vertices
    )
    return {"vertices": len(vertices), "edges": len(edges),
            "degree_counts": dict(by_degree), "triangles_per_vertex": dict(by_triangles)}


def verify_quotient(cover, sample, host_path):
    pairs = list(product(range(60), repeat=2))
    orbit = {(x, m): min((p[x], p[m]) for p in cover.deck.values()) for x, m in pairs}
    vertices = set(orbit.values())
    assert set(Counter(orbit.values()).values()) == {4}
    first_edges = {tuple(sorted((orbit[x, m], orbit[y, m])))
                   for x, y in cover.edges for m in range(60)}
    second_edges = {tuple(sorted((orbit[m, x], orbit[m, y])))
                    for x, y in cover.edges for m in range(60)}
    assert not (first_edges & second_edges)
    pair_census = triangle_census(vertices, first_edges | second_edges)
    assert pair_census == {"vertices": 900, "edges": 3600,
                           "degree_counts": {8: 900}, "triangles_per_vertex": {4: 900}}
    # Closed shared transport is in the diagonal deck action and disappears.
    for x, m in pairs:
        for p in cover.deck.values():
            assert orbit[p[x], p[m]] == orbit[x, m]
    origin = sample["contact_sites"][0]
    b, ab, a = (cover.deck[name] for name in ("b", "ab", "a"))
    for x, m in product(cover.fibers[origin], repeat=2):
        # Relative return survives when the two registers take different routes.
        assert orbit[b[x], ab[m]] == orbit[x, a[m]]
        assert orbit[ab[x], ab[m]] == orbit[x, m]
        assert orbit[x, a[m]] != orbit[x, m]
        assert orbit[a[x], a[m]] == orbit[x, m]
    result = {
        "cartesian_pair_quotient": pair_census,
        "shared_deck_motion_disappears": True,
        "single_lane_relative_deck_motion_survives": True,
        "host_graph_isomorphism": "unchecked",
    }

    if host_path is not None:
        host = pinned_json(host_path, LOCK["g900_static_body"])
        assert host["schema"] == "g900.viewer.static_body"
        host_vertices = {row["id"] for row in host["vertices"]}
        host_edges = {tuple(sorted(edge)) for edge in host["edges"]}
        assert len(host["vertices"]) == len(host_vertices) == 900
        assert len(host["edges"]) == len(host_edges) == 3600
        host_census = triangle_census(host_vertices, host_edges)
        assert host_census["triangles_per_vertex"] == {2: 180, 3: 540, 4: 180}
        assert host_census["triangles_per_vertex"] != pair_census["triangles_per_vertex"]
        result["g900_static_body"] = host_census
        result["host_graph_isomorphism"] = "refuted_by_triangle_census"
    return result


def verify_compressions(cover, g30_path, history_path):
    # A partition into two pairs on a regular Klein fiber must pair every
    # vertex with its image under one of the three nonidentity deck elements.
    letters = ("a", "b", "ab")
    for site in range(15):
        stabilizers = [i for i, action in cover.quotient_actions.items() if action[site] == site]
        invariant = [name for name in letters if all(
            cover.deck_conjugations[i][name] == name for i in stabilizers
        )]
        assert invariant == ["a"]

    a = cover.deck["a"]
    for m in range(60):
        for name in letters:
            same_block = a[m] in (m, cover.deck[name][m])
            assert same_block == (name == "a")
    result = {
        "fully_covariant_two_state_deck_quotient": "a",
        "stable_two_state_deck_quotient_erases_pointer_flip": True,
        "b_ab_quotients_each_keep_flip_but_full_group_exchanges_them": True,
        "ready_vertex_and_its_a_mate_form_a_two_state_subspace": True,
        "project42_g30_crosswalk": "unchecked",
        "project41_phase_surface_crosswalk": "unchecked",
    }
    if g30_path is not None:
        source = pinned_json(g30_path, LOCK["g30_certificate"])
        assert source["audit_pass"] is True
        assert all(source["involution_a"][str(x)] == a[x] for x in range(60))
        projections = {int(x): int(v) for x, v in source["g60_vertex_to_g30_vertex"].items()}
        assert set(projections) == set(range(60))
        assert len(set(projections.values())) == 30
        assert all(projections[x] == projections[a[x]] for x in projections)
        quotient_edges = {tuple(sorted((projections[x], projections[y]))) for x, y in cover.edges}
        assert quotient_edges == {tuple(sorted(edge)) for edge in source["quotient_edges"]}
        result["project42_g30_crosswalk"] = "all_60_vertices_erase_pointer_flip"

    if history_path is not None:
        source = pinned_json(history_path, LOCK["history_groupoid"])
        assert source["theorem_pass"] is True
        measurements = source["measurements"]
        assert len(measurements["phase_cosets"]) == len(measurements["phase_isotropy"]) == 4
        group = cover.group
        index_for = {p: i for i, p in group.items()}
        for coset in measurements["phase_cosets"]:
            assert {index_for[tuple(a[x] for x in group[i])] for i in coset} == set(coset)
        for row in measurements["phase_isotropy"]:
            assert set(row["Kk"]) == {DECK_IDS["1"], DECK_IDS["a"]}
        result["project41_phase_surface_crosswalk"] = "a_fixes_all_four_phase_cosets_and_is_in_every_isotropy_kernel"
    return result


def run(root, output, host, g30, history):
    cover = Cover(load(root))
    check_native_fixture(cover, root)
    accepted, _ = census(cover)
    sample = witness(cover, accepted)
    readout = verify_readout(cover, sample)
    quotient = verify_quotient(cover, sample, host)
    compressions = verify_compressions(cover, g30, history)
    write(output / "readout.json", {
        "audit": "TOM-QM-027.READOUT", "version": VERSION,
        "parent_commit": LOCK["parent_audit_commit"],
        "input_lock": "../inputs.lock.json",
        "expanded_graph_model_status": "pass",
        "bare_g60_pointer_coupling_status": "blocked",
        "readout": readout, "pair_quotient_controls": quotient,
        "single_fiber_compression_controls": compressions,
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g900-static-body", type=Path)
    parser.add_argument("--g30-certificate", type=Path)
    parser.add_argument("--history-groupoid", type=Path)
    args = parser.parse_args()
    run(args.root, args.output, args.g900_static_body, args.g30_certificate, args.history_groupoid)
