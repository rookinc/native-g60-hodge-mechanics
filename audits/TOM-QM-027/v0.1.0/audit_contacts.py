"""Census of six-comparison contact sets on the native G60 -> G15 cover."""
from argparse import ArgumentParser
from collections import Counter, deque
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from native_inputs import BASE_COMMIT, DECK_IDS, ROOT, load, write

VERSION = "0.1.0"


def cycles(neighbors, length):
    def extend(path):
        if len(path) == length:
            if path[0] in neighbors[path[-1]] and path[1] < path[-1]:
                yield path
            return
        for next_site in sorted(neighbors[path[-1]]):
            if next_site > path[0] and next_site not in path:
                yield from extend((*path, next_site))

    for first in range(len(neighbors)):
        yield from extend((first,))


def shortest_paths(neighbors, origin):
    paths = {origin: [(origin,)]}
    distance = {origin: 0}
    queue = deque([origin])
    while queue:
        first = queue.popleft()
        for second in sorted(neighbors[first]):
            steps = distance[first] + 1
            if second not in distance:
                distance[second] = steps
                paths[second] = []
                queue.append(second)
            if distance[second] == steps:
                paths[second].extend((*path, second) for path in paths[first])
    return paths


class Cover:
    def __init__(self, data):
        b1, b2 = data["b1"], data["b2"]
        certificate = data["group"]
        self.group = {
            row["actual_index"]: tuple(row["actual_permutation"])
            for row in certificate["mapping_rows"]
        }
        vertex_set = set(range(60))
        assert b1.shape == (60, 120) and b2.shape == (120, 20)
        assert not np.any(b1 @ b2)
        assert len(self.group) == len(set(self.group.values())) == 480
        assert all(set(g) == vertex_set for g in self.group.values())
        assert all(np.array_equal(g, data["actions"]["vertex_target"][i])
                   for i, g in self.group.items())

        generators = [tuple(row["actual_permutation"])
                      for row in certificate["generator_correspondence"]]
        reached = {tuple(range(60))}
        queue = deque(reached)
        while queue:
            current = queue.popleft()
            for generator in generators:
                product = tuple(generator[x] for x in current)
                if product not in reached:
                    reached.add(product)
                    queue.append(product)
        assert reached == set(self.group.values())

        self.deck = {name: self.group[index] for name, index in DECK_IDS.items()}
        identity = self.deck["1"]
        a, b, ab = (self.deck[name] for name in ("a", "b", "ab"))
        assert identity == tuple(range(60))
        assert all(tuple(p[x] for x in p) == identity for p in self.deck.values())
        assert tuple(a[x] for x in b) == tuple(b[x] for x in a) == ab
        self.fibers = sorted({tuple(sorted(p[x] for p in self.deck.values()))
                              for x in vertex_set})
        assert len(self.fibers) == 15 and all(len(set(fiber)) == 4 for fiber in self.fibers)
        self.parent = {x: i for i, fiber in enumerate(self.fibers) for x in fiber}
        assert len(self.parent) == 60

        self.edges = set()
        for column in b1.T:
            low = np.flatnonzero(column == -1)
            high = np.flatnonzero(column == 1)
            assert len(low) == len(high) == 1 and np.count_nonzero(column) == 2
            self.edges.add(tuple(sorted((int(low[0]), int(high[0])))))
        assert len(self.edges) == 120
        for g in self.group.values():
            assert {tuple(sorted((g[u], g[v]))) for u, v in self.edges} == self.edges

        self.neighbors = [set() for _ in self.fibers]
        self.edge_maps = {}
        for u, v in sorted(self.edges):
            for x, y in ((u, v), (v, u)):
                first, second = self.parent[x], self.parent[y]
                assert first != second
                self.neighbors[first].add(second)
                edge_map = self.edge_maps.setdefault((first, second), {})
                assert x not in edge_map
                edge_map[x] = y
        assert len(self.edge_maps) == 60 and all(len(s) == 4 for s in self.neighbors)
        for (i, j), edge_map in self.edge_maps.items():
            assert set(edge_map) == set(self.fibers[i])
            assert set(edge_map.values()) == set(self.fibers[j])
            assert self.edge_maps[j, i] == {y: x for x, y in edge_map.items()}

        self.quotient_actions = {}
        self.deck_conjugations = {}
        for index, g in self.group.items():
            q = tuple(self.parent[g[fiber[0]]] for fiber in self.fibers)
            assert len(set(q)) == 15
            assert all({g[x] for x in fiber} == set(self.fibers[q[i]])
                       for i, fiber in enumerate(self.fibers))
            self.quotient_actions[index] = q
            self.deck_conjugations[index] = {}
            for name, p in self.deck.items():
                image = [label for label, candidate in self.deck.items()
                         if all(g[p[x]] == candidate[g[x]] for x in vertex_set)]
                assert len(image) == 1
                self.deck_conjugations[index][name] = image[0]
            conjugation = self.deck_conjugations[index]
            assert conjugation["a"] == "a" and {conjugation["b"], conjugation["ab"]} == {"b", "ab"}

        self.paths = [shortest_paths(self.neighbors, i) for i in range(15)]
        assert all(len(paths) == 15 for paths in self.paths)

    def lift(self, path, start):
        assert self.parent[start] == path[0]
        vertices = [start]
        for edge in zip(path, path[1:]):
            vertices.append(self.edge_maps[edge][vertices[-1]])
        return tuple(vertices)

    def transport(self, path):
        return {x: self.lift(path, x)[-1] for x in self.fibers[path[0]]}

    def holonomy(self, walk):
        assert walk[0] == walk[-1]
        transport = self.transport(walk)
        match = [name for name, p in self.deck.items()
                 if transport == {x: p[x] for x in transport}]
        assert len(match) == 1
        return match[0]

    def compare(self, first, second):
        (path,) = self.paths[first][second]
        return path

    def perimeter(self, contacts):
        return tuple(x for first, second in zip(contacts, (*contacts[1:], contacts[0]))
                     for x in self.compare(first, second)[:-1]) + (contacts[0],)


def census(cover):
    pair_counts = Counter()
    for first, second in combinations(range(15), 2):
        paths = cover.paths[first][second]
        # The common source order makes distinct maps directly comparable.
        maps = {tuple(cover.transport(path).values()) for path in paths}
        pair_counts[f"distance_{len(paths[0]) - 1}_paths_{len(paths)}_maps_{len(maps)}"] += 1

    accepted = {}
    spectra = Counter()
    two_branch_lengths = Counter()
    for contacts in combinations(range(15), 4):
        if any(len(cover.paths[i][j]) != 1 for i, j in combinations(contacts, 2)):
            continue
        routes = []
        for rest in permutations(contacts[1:]):
            if rest[0] < rest[-1]:
                order = (contacts[0], *rest)
                walk = cover.perimeter(order)
                routes.append((cover.holonomy(walk), order, walk))
        spectrum = sorted(label for label, _, _ in routes)
        spectra[",".join(spectrum)] += 1
        accepted[contacts] = routes
        if {"b", "ab"}.issubset(spectrum):
            lengths = sorted(len(walk) - 1 for label, _, walk in routes if label in ("b", "ab"))
            two_branch_lengths[tuple(lengths)] += 1

    simple_cycles = {}
    for length in (3, 4, 5, 6):
        simple_cycles[str(length)] = dict(Counter(
            cover.holonomy((*path, path[0])) for path in cycles(cover.neighbors, length)
        ))
    assert simple_cycles["3"] == {"1": 10} and simple_cycles["4"] == {}
    assert pair_counts == {
        "distance_1_paths_1_maps_1": 30,
        "distance_2_paths_1_maps_1": 60,
        "distance_3_paths_4_maps_4": 15,
    }
    assert len(accepted) == 405
    assert spectra == {"1,1,1": 75, "1,a,a": 30, "1,b,b": 90, "1,ab,ab": 90, "a,ab,b": 120}
    assert two_branch_lengths == {(7, 8): 120}

    return accepted, {
        "all_four_site_sets": 1365,
        "six_unambiguous_comparisons_sets": len(accepted),
        "pair_census": dict(pair_counts),
        "unordered_perimeter_holonomy_spectra": dict(spectra),
        "two_branch_length_census": {f"{a},{b}": n for (a, b), n in two_branch_lengths.items()},
        "simple_cycle_holonomy_census_by_length": simple_cycles,
        "fibers": cover.fibers,
        "quotient_edges": sorted((a, b) for a, b in cover.edge_maps if a < b),
    }


def verify_covariance(cover, accepted):
    edge_checks = pair_checks = quartet_checks = 0
    for index, g in cover.group.items():
        q = cover.quotient_actions[index]
        for (i, j), edge_map in cover.edge_maps.items():
            transported = cover.edge_maps[q[i], q[j]]
            for x, y in edge_map.items():
                assert g[y] == transported[g[x]]
                edge_checks += 1
        for i, j in permutations(range(15), 2):
            images = {tuple(q[x] for x in path) for path in cover.paths[i][j]}
            assert images == set(cover.paths[q[i]][q[j]])
            pair_checks += len(images)
        for contacts, routes in accepted.items():
            moved = tuple(sorted(q[x] for x in contacts))
            transformed = sorted(cover.deck_conjugations[index][label] for label, _, _ in routes)
            assert transformed == sorted(label for label, _, _ in accepted[moved])
            quartet_checks += 1
    local_exchange_counts = []
    for x in range(60):
        exchanging = [
            g for i, g in cover.group.items()
            if g[x] == x and cover.deck_conjugations[i]["b"] == "ab"
        ]
        assert all(g[cover.deck["b"][x]] == cover.deck["ab"][x] for g in exchanging)
        local_exchange_counts.append(len(exchanging))
    assert local_exchange_counts == [4] * 60
    return {
        "automorphisms": len(cover.group),
        "edge_lifts_checked": edge_checks,
        "shortest_paths_checked": pair_checks,
        "quartet_images_checked": quartet_checks,
        "b_ab_swapping_automorphisms": sum(
            conjugation["b"] == "ab" for conjugation in cover.deck_conjugations.values()
        ),
        "b_ab_swapping_automorphisms_fixing_each_source_vertex": 4,
        "unframed_invariant_scalar_of_start_and_return_cannot_discriminate_b_from_ab": True,
    }


def pair_selection_obstruction(cover):
    """Test if a covariant fiber comparison can exist even beyond shortest paths."""
    failures = []
    for i, j in permutations(range(15), 2):
        paths = cover.paths[i][j]
        if len(paths) == 1:
            continue
        deck_maps = [cover.transport(path) for path in paths]
        assert len({tuple(transport.values()) for transport in deck_maps}) == 4
        for transport in deck_maps:
            assert all(transport[p[x]] == p[y]
                       for p in cover.deck.values() for x, y in transport.items())
        stabilizer = [(index, g) for index, g in cover.group.items()
                      if cover.quotient_actions[index][i] == i
                      and cover.quotient_actions[index][j] == j]
        assert len(stabilizer) == 16
        for transport in deck_maps:
            conjugate_maps = {
                tuple(sorted((g[x], g[y]) for x, y in transport.items()))
                for _, g in stabilizer
            }
            assert conjugate_maps == {tuple(sorted(p.items())) for p in deck_maps}
        failures.append({
            "ordered_pair": [i, j],
            "pair_stabilizer_size": len(stabilizer),
            "orbit_size_of_every_deck_equivariant_comparison": 4,
        })
    assert len(failures) == 30
    return {
        "ordered_pairs_with_no_covariant_bijection": failures,
        "proof": "The pair stabilizer contains the regular deck group. Its four equivariant bijections are exactly the four geodesic lift maps, and the full pair stabilizer acts transitively on these. There is no fixed map.",
    }


def witness(cover, accepted):
    contacts, routes = min((sites, routes) for sites, routes in accepted.items()
                           if {label for label, _, _ in routes} == {"a", "b", "ab"})
    first, second, _, _ = contacts
    pair_paths = {f"{i}-{j}": cover.compare(i, j) for i, j in combinations(contacts, 2)}
    first_vertex = cover.fibers[first][0]  # display convention, checked for every start below
    branch_walks = sorted((walk for label, _, walk in routes if label in ("b", "ab")), key=len)
    short, long = branch_walks
    forward = cover.compare(first, second)

    rows = []
    for index, g in cover.group.items():
        q = cover.quotient_actions[index]
        moved_short, moved_long, moved_forward = (
            tuple(q[x] for x in path) for path in (short, long, forward)
        )
        for initial in cover.fibers[first]:
            x = g[initial]
            reference = cover.lift(moved_short + moved_forward[1:], x)[-1]
            ends = []
            for original, moved in ((short, moved_short), (long, moved_long)):
                original_end = cover.lift(original, initial)[-1]
                end = cover.lift(moved, x)[-1]
                assert end == g[original_end]
                assert cover.lift(tuple(reversed(moved)), end)[-1] == x
                read = cover.lift(moved_forward, end)[-1]
                assert cover.lift(tuple(reversed(moved_forward)), read)[-1] == end
                bit = int(read == reference)
                assert bit == int(original == short)
                for pointer in (0, 1):
                    assert (pointer ^ bit) ^ bit == pointer
                ends.append(read)
            assert len(set(ends)) == 2

    for label, order, walk in routes:
        for rotation in range(4):
            rotated = order[rotation:] + order[:rotation]
            for direction in (rotated, tuple(reversed(rotated))):
                assert cover.holonomy(cover.perimeter(direction)) == label
        lifted = cover.lift(walk, first_vertex)
        rows.append({
            "contact_order": (*order, first),
            "quotient_walk": walk,
            "lifted_g60_vertices": lifted,
            "edge_count": len(walk) - 1,
            "deck_holonomy": label,
            "later_vertex_at_second_contact": cover.lift(forward, lifted[-1])[-1],
        })
    assert len(list(permutations(contacts))) == 24
    for role_assignment in permutations(contacts):
        assert {tuple(sorted((a, b))) for a, b in combinations(role_assignment, 2)} == set(
            combinations(contacts, 2)
        )
        assert cover.holonomy(cover.perimeter(role_assignment)) in {"a", "b", "ab"}

    # Controls: out-and-back and any point-selected intra-fiber comparisons telescope.
    backtrack = forward + tuple(reversed(forward))[1:]
    assert cover.holonomy(backtrack) == "1"
    one_fiber = cover.fibers[first]
    for x, y in permutations(one_fiber, 2):
        assert sum(p[x] == y for p in cover.deck.values()) == 1
    for order in permutations(one_fiber):
        current = tuple(range(60))
        for x, y in zip(order, (*order[1:], order[0])):
            translation = next(p for p in cover.deck.values() if p[x] == y)
            current = tuple(translation[z] for z in current)
        assert current == tuple(range(60))
    assert cover.holonomy(cover.perimeter((0, 1, 2, 8))) == "1"

    return {
        "contact_sites": contacts,
        "source_g60_vertex_for_example": first_vertex,
        "six_pair_paths": pair_paths,
        "six_pair_maps": {
            name: cover.transport(path) for name, path in pair_paths.items()
        },
        "three_perimeters": rows,
        "readout": {
            "later_contact": second,
            "reference": "shorter registered b/ab branch followed by first-to-second comparison",
            "boolean": "later vertex equals reference vertex",
            "exact_group_start_cases": len(cover.group) * len(cover.fibers[first]),
        },
        "role_assignments_checked": 24,
        "reverse_route_replays": True,
        "controls": {
            "out_and_back": "1",
            "one_fiber_point_selected_deck_translations_all_role_orders": "1",
            "all_flat_quartet_0_1_2_8": "1",
            "distance_three_comparison": "four different maps from four tied geodesics",
        },
    }


def minimal_nonidentity_witness(cover, accepted):
    """Minimize a nonidentity perimeter and then a flat perimeter on the same sites."""
    lengths = [
        (len(active) - 1, len(flat) - 1, contacts, label, active, flat)
        for contacts, routes in accepted.items()
        for label, _, active in routes if label != "1"
        for identity, _, flat in routes if identity == "1"
    ]
    active_length, flat_length, sites, label, active, flat = min(lengths)
    assert (active_length, flat_length) == (5, 6)
    assert len(set(active[:-1])) == 5
    x = cover.fibers[sites[0]][0]
    later = cover.compare(sites[0], sites[1])
    branches = {}
    for name, path in ((label, active), ("1", flat)):
        assert cover.holonomy(path) == name
        lifted = cover.lift(path, x)
        branches[name] = {
            "quotient_walk": path, "lifted_g60_vertices": lifted,
            "later_g60_vertex": cover.lift(later, lifted[-1])[-1],
        }
    return {
        "contact_sites": sites,
        "active_and_flat_lengths": [active_length, flat_length],
        "branches": branches,
    }


def run(root, output):
    cover = Cover(load(root))
    accepted, inventory = census(cover)
    covariance = verify_covariance(cover, accepted)
    obstruction = pair_selection_obstruction(cover)
    sample = witness(cover, accepted)
    minimal = minimal_nonidentity_witness(cover, accepted)
    records = {
        "01-contact-admission": ("pass", "Four distinct quotient-site fibers", {
            "contact_sites": sample["contact_sites"], "fibers": inventory["fibers"],
        }),
        "02-pairwise-comparison": ("pass", "Unique-geodesic lift on the 405 admitted quartets", {
            "pair_census": inventory["pair_census"], "six_pair_paths": sample["six_pair_paths"],
            "six_pair_maps": sample["six_pair_maps"],
            "all_quartets_rule": "fail for 960 quartets containing a distance-three pair",
            "obstruction": obstruction,
        }),
        "03-covariance": ("pass", "Exact covariance for source graph automorphisms", covariance),
        "04-closure": ("pass", "All six comparisons are composable; loop closes in G15 with a deck residual", {
            "perimeters": sample["three_perimeters"],
            "shortest_nonidentity_versus_flat_on_same_sites": minimal,
            "loop_length_and_spectrum_census": inventory["unordered_perimeter_holonomy_spectra"],
            "direct_four_edge_loop": "fail: no simple G15 four-cycle",
        }),
        "05-retained-residue": ("pass", "Distinct positions persist through the specified subsequent lift", {
            "perimeters": sample["three_perimeters"],
            "history_scope": "user-selected ordered geodesic segments and one later comparison",
        }),
        "06-readout": ("blocked", "Native observer readout", {
            **sample["readout"],
            "reference_relative_graph_readout": "pass",
            "missing": "Source-defined operation that physically couples endpoint identity to a persistent pointer.",
        }),
        "07-reconstruction": ("pass", "Reverse transport recovers every start given the route", {
            "reverse_route_replays": sample["reverse_route_replays"],
            "discarded_by_endpoint_summary": "Full path, starting role when omitted, and direction; reversing a V4 loop leaves its deck element unchanged.",
        }),
        "08-observer-independence": ("pass", "All roles are selectable; no fixed origin or b/ab scalar label is assumed", {
            "role_assignments_checked": sample["role_assignments_checked"],
            "distinguishes_label_from_order": "Role reassignment either preserves the physical route or specifies a different perimeter.",
            "b_ab_swapping_automorphisms": covariance["b_ab_swapping_automorphisms"],
        }),
    }
    for key, (status, scope, evidence) in records.items():
        write(output / f"{key}.json", {
            "audit": f"TOM-QM-027.{key[:2]}", "version": VERSION, "status": status,
            "scope": scope, "evidence": evidence, "project46_commit": BASE_COMMIT,
            "input_lock": "../inputs.lock.json",
        })
    write(output / "contact-census.json", {
        "audit": "TOM-QM-027", "version": VERSION, "project46_commit": BASE_COMMIT,
        "inventory": inventory, "covariance": covariance, "witness": sample,
        "shortest_nonidentity_vs_flat_witness": minimal,
        "pair_selection_obstruction": obstruction,
        "native_observer_admitted": False, "quantum_bridge_status": "blocked",
    })
    write(output / "quantum-gate.json", {
        "audit": "TOM-QM-027.Q", "version": VERSION, "status": "blocked",
        "project46_commit": BASE_COMMIT,
        "first_unmet_native_observer_obligation": "06-readout",
        "available": "Covariant endpoint distinguishability with an explicit reference and reversible discrete transport.",
        "missing": "A source-defined coupling to a later readable, persistent record and an admissible state-space and observable correspondence.",
        "interference": "blocked", "relative_phase": "blocked",
        "physical_detector_record": "blocked", "born_probability": "blocked",
        "objective_collapse": "open",
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Pinned Project 46 checkout")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.output)
