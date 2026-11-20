"""Cross-check native operators against Project 42 voltage and Project 43 return rows."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json

from audit_contacts import Cover, census, cycles, minimal_nonidentity_witness
from native_inputs import BASE_COMMIT, ROOT, load, write


def load_pinned(path, name):
    context = json.loads(Path(__file__).with_name("inputs.lock.json").read_text())["context_consulted"]
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == context[name]["sha256"]
    data = json.loads(payload)
    assert data["audit_pass"] is True
    return data


def orientations(cycle):
    for order in (cycle, tuple(reversed(cycle))):
        for shift in range(len(order)):
            yield order[shift:] + order[:shift]


def crosswalk(root, voltage_file, automaton_file, output):
    cover = Cover(load(root))
    voltage = load_pinned(voltage_file, "voltage.json")
    automaton = load_pinned(automaton_file, "automaton-089.json")
    assert [tuple(fiber) for fiber in voltage["quotient"]["orbits"]] == cover.fibers
    for row in voltage["voltage_rows"]:
        i, j = row["source_quotient"], row["target_quotient"]
        x, y = row["source_representative"], row["target_vertex"]
        assert x == cover.fibers[i][0]
        assert cover.edge_maps[i, j][x] == y == cover.deck[row["voltage"]][cover.fibers[j][0]]
    assert len(voltage["voltage_rows"]) == len(cover.edge_maps) == 60

    expected_cycles = {min(orientations(cycle)) for cycle in cycles(cover.neighbors, 5)}
    recorded_cycles = set()
    prefixes = {}
    profile = Counter()
    for row in automaton["pentagon_rows"]:
        cycle = tuple(row["base_cycle"])
        canonical = min(orientations(cycle))
        assert canonical not in recorded_cycles
        recorded_cycles.add(canonical)
        label = row["computed_holonomy"]
        assert label == row["expected_holonomy"]
        assert cover.holonomy((*cycle, cycle[0])) == label
        for start_row in row["start_rows"]:
            start = cover.deck[start_row["start_group"]][cover.fibers[cycle[0]][0]]
            lift = cover.lift(cycle * 2 + (cycle[0],), start)
            assert lift[-1] == start
            assert min(orientations(lift[:-1])) == min(orientations(tuple(start_row["registered_cycle"])))
            assert cover.deck[label][start] == lift[5]
        for orientation in orientations(cycle):
            walk = (*orientation, orientation[0])
            prefix = walk[:3]
            assert prefix not in prefixes
            prefixes[prefix] = walk
            assert cover.compare(prefix[0], prefix[2]) == prefix
            assert cover.holonomy(walk) == label
            profile[label] += 1

    assert recorded_cycles == expected_cycles
    assert len(recorded_cycles) == 12
    assert profile == {"b": 60, "ab": 60}
    assert set(prefixes) == {path for i, paths in enumerate(cover.paths)
                             for j, geodesics in paths.items() if i != j
                             for path in geodesics if len(path) == 3}
    assert not any(len(set(i) & set(j)) >= 4 for i, j in combinations(recorded_cycles, 2))

    accepted, _ = census(cover)
    minimum = minimal_nonidentity_witness(cover, accepted)
    (active,) = [tuple(branch["quotient_walk"]) for label, branch in minimum["branches"].items()
                 if label != "1"]
    assert prefixes[active[:3]] == active
    write(output / "registered-return-crosswalk.json", {
        "audit": "TOM-QM-027.R", "version": "0.1.0", "status": "pass",
        "project46_commit": BASE_COMMIT, "input_lock": "../inputs.lock.json",
        "project42_voltage_rows_checked": 60,
        "project43_089_pentagons_checked": 12,
        "project43_089_lifted_start_rows_checked": 48,
        "oriented_rooted_return_words_checked": len(prefixes),
        "holonomy_profile": dict(profile),
        "all_native_distance_two_prefixes_admit_exactly_one_registered_word": True,
        "two_distinct_registered_pentagons_share_four_sites": False,
        "shortest_four_contact_nonidentity_vs_flat_witness": minimum,
        "provenance_boundary": "The 120 words were expanded from all Project 43 Audit 089 rows. Project 41 Audit 106 supplies a matching aggregate; Audit 094's original rows were unavailable. The 7/8-edge b/ab quartet is a distinct construction.",
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--voltage", type=Path, required=True)
    parser.add_argument("--automaton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    crosswalk(args.root, args.voltage, args.automaton, args.output)
