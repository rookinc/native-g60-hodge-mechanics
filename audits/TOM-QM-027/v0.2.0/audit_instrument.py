"""Test the conditional quantum instrument on the explicit two-register model."""

from argparse import ArgumentParser
from pathlib import Path

from sympy import I, Matrix, Rational, diag, eye, kronecker_product, sqrt, trace, zeros

from audit_readout import (
    Cover, ROOT, VERSION, census, check_native_fixture, load, pointer_write,
    verify_readout, witness, write,
)


X = Matrix([[0, 1], [1, 0]])
Y = Matrix([[0, -I], [I, 0]])


def basis(size, index):
    return eye(size)[:, index]


def density(vector):
    return vector * vector.H


def expectation(state, observable):
    return trace(state * observable).simplify()


def discard_environment(state, dimension):
    return Matrix(2, 2, lambda i, j: sum(
        state[dimension * i + k, dimension * j + k] for k in range(dimension)
    ))


def lifted_gate(cover, readout):
    sample = readout["examples"]
    routes = [tuple(row["common_route"]) for row in sample]
    reference = routes[0]
    initial = sample[0]["pointer_initial"]
    fiber = cover.fibers[reference[0]]
    n = len(fiber)
    initial_index = fiber.index(initial)
    flip_index = fiber.index(cover.deck["a"][initial])

    shuffle = zeros(n)
    for i, vertex in enumerate(fiber):
        shuffle[fiber.index(cover.deck["a"][vertex]), i] = 1
    gate = zeros(2 * n)
    for j, route in enumerate(routes):
        for i, m in enumerate(fiber):
            x, out = pointer_write(cover, route, reference, initial, m)
            assert x == initial
            gate[j * n + fiber.index(out), j * n + i] = 1
    assert gate.H * gate == eye(2 * n) and gate == diag(eye(n), shuffle)
    return gate, shuffle, initial_index, flip_index


def measure_pointer(gate, initial, flipped):
    n = gate.rows // 2
    preparation = kronecker_product(eye(2), basis(n, initial))
    readouts = [
        kronecker_product(eye(2), basis(n, k).H) * gate * preparation
        for k in range(n)
    ]
    assert readouts[initial] == diag(1, 0)
    assert readouts[flipped] == diag(0, 1)
    assert sum((k.H * k for k in readouts), zeros(2)) == eye(2)
    assert all(k.H * k == k and k * k == k and k.H == k for k in readouts)
    # Each outcome Choi matrix is a positive outer product (or zero).
    mapped = []
    for k in readouts:
        vector = Matrix(4, 1, list(k.T))
        choi_matrix = vector * vector.H
        assert choi_matrix.is_positive_semidefinite
        mapped.append(k)
    return mapped


def copy_pointer(gate, initial):
    """A separate observer bit takes the blank/nonblank result in the pointer basis."""
    n = gate.rows // 2
    copy = zeros(4 * n)
    for j in range(2):
        for p in range(n):
            for bit in range(2):
                source = 2 * (n * j + p) + bit
                target = 2 * (n * j + p) + (bit ^ int(p != initial))
                copy[target, source] = 1
    assert copy.H * copy == eye(4 * n)
    return copy


def phase_tests(gate, pointer_flip, initial, flipped, operators):
    n = pointer_flip.rows
    blank, marked = basis(n, initial), basis(n, flipped)
    assert (blank.H * marked)[0] == 0
    mixture = diag(Rational(9, 25), Rational(16, 25))
    joint_mix = gate * kronecker_product(mixture, density(blank)) * gate.H
    external_record = copy_pointer(gate, initial)
    undo_with_record = kronecker_product(gate.H, eye(2))
    clear_record = density(basis(2, 0))
    rows = []
    for name, phase, components in (
        ("0", 1, (Rational(24, 25), 0)),
        ("pi", -1, (Rational(-24, 25), 0)),
        ("pi/2", I, (0, Rational(24, 25))),
        ("-pi/2", -I, (0, Rational(-24, 25))),
    ):
        state = density(Matrix([Rational(3, 5), Rational(4, 5) * phase]))
        joint_before = kronecker_product(state, density(blank))
        recorded = gate * joint_before * gate.H
        marginals = discard_environment(recorded, n)
        assert marginals == mixture
        no_record = (expectation(state, X), expectation(state, Y))
        pointer_unread = (expectation(marginals, X), expectation(marginals, Y))
        joint_phase = tuple(expectation(recorded, kronecker_product(axis, pointer_flip))
                            for axis in (X, Y))
        mixed_phase = tuple(expectation(joint_mix, kronecker_product(axis, pointer_flip))
                            for axis in (X, Y))
        assert no_record == joint_phase == components
        assert pointer_unread == mixed_phase == (0, 0)
        assert gate.H * recorded * gate == joint_before
        assert discard_environment(gate.H * joint_mix * gate, n) == mixture

        # Pointer record copied to a separate system. Reversing only the route
        # and pointer returns the pointer to blank but leaves that record intact.
        copied = external_record * kronecker_product(recorded, clear_record) * external_record.H
        undone = undo_with_record * copied * undo_with_record.H
        assert discard_environment(undone, 2 * n) == mixture
        assert undone != kronecker_product(joint_before, clear_record)

        conditional_probabilities = []
        for j, k in ((0, operators[initial]), (1, operators[flipped])):
            outcome = k * state * k.H
            probability = trace(outcome)
            assert outcome / probability == density(basis(2, j))
            conditional_probabilities.append(str(probability))
        assert conditional_probabilities == ["9/25", "16/25"]
        rows.append({
            "relative_phase": name,
            "route_X_Y_before_registration": [str(x) for x in no_record],
            "route_X_Y_after_registration_pointer_unread": [str(x) for x in pointer_unread],
            "joint_XY_times_pointer_flip": [str(x) for x in joint_phase],
            "same_joint_observables_for_classical_mixture": [str(x) for x in mixed_phase],
            "pointer_born_probabilities": conditional_probabilities,
            "route_X_plus_probability_before": str((1 + no_record[0]) / 2),
            "route_X_plus_probability_pointer_unread": "1/2",
        })

    input_state = density(Matrix([Rational(3, 5), Rational(4, 5)]))
    for sign in (1, -1):
        pointer = density((blank + sign * marked) / sqrt(2))
        out = gate * kronecker_product(input_state, pointer) * gate.H
        kickback = diag(1, sign)
        assert out == kronecker_product(kickback * input_state * kickback.H, pointer)
    return rows


def run(root, output):
    cover = Cover(load(root))
    check_native_fixture(cover, root)
    accepted, _ = census(cover)
    readout = verify_readout(cover, witness(cover, accepted))
    gate, shuffle, initial, flipped = lifted_gate(cover, readout)
    projectors = measure_pointer(gate, initial, flipped)
    rows = phase_tests(gate, shuffle, initial, flipped, projectors)
    write(output / "conditional-instrument.json", {
        "audit": "TOM-QM-027.INSTRUMENT", "version": VERSION,
        "status": "pass_for_explicit_graph_extension",
        "native_unassisted_quantum_instrument_status": "blocked",
        "state_space": "Logical route qubit tensor the complex linear span of one native four-point pointer fiber.",
        "joint_unitary_dimension": gate.rows,
        "kraus_effect_completeness_checked": True,
        "complete_positivity_checked_by_choi_positive_semidefiniteness": True,
        "individual_maps_trace_nonincreasing": True,
        "each_selective_route_state_matches_pointer_record": True,
        "orthogonal_detector_records_when_prepared_at_one_vertex": True,
        "global_unitary_uncomputation_recovers_phase": True,
        "copy_to_external_register_prevents_recovery_by_partial_uncomputation": True,
        "pointer_in_flip_eigenstate_yields_no_which_path_record": True,
        "phase_tests": rows,
        "born_rule": "Adopted to compute outcome probabilities from the completely positive maps.",
        "physical_selection": "No physical device, preferred measurement basis, or independent collapse law is established.",
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
