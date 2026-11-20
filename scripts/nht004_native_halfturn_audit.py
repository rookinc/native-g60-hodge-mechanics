#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/imported/project45/native_g60_fiber_product_isomorphism_044.json"
OUTPUT = ROOT / "artifacts/json/nht004_native_halfturn_receipt.json"

data = json.loads(SOURCE.read_text())
assert data["audit_pass"] is True

rows = data["mapping_rows"]
assert len(rows) == 480

perms = {
    r["actual_index"]: tuple(r["actual_permutation"])
    for r in rows
}
orders = {
    r["actual_index"]: r["actual_order"]
    for r in rows
}

def compose(a, b):
    return tuple(a[b[i]] for i in range(len(a)))

identity = tuple(range(60))
by_perm = {p: i for i, p in perms.items()}

assert len(by_perm) == 480
assert perms[0] == identity

z_index = 326
z = perms[z_index]

assert z != identity
assert compose(z, z) == identity

central = all(
    compose(z, p) == compose(p, z)
    for p in perms.values()
)
assert central

roots = sorted(
    i for i, p in perms.items()
    if compose(p, p) == z
)

assert len(roots) == 20
assert all(orders[i] == 4 for i in roots)

pairs = sorted({
    tuple(sorted((
        i,
        by_perm[compose(perms[i], z)]
    )))
    for i in roots
})

assert len(pairs) == 10
assert len({i for pair in pairs for i in pair}) == 20

receipt = {
    "audit": "NHT-004A",
    "status": "pass",
    "source": str(SOURCE.relative_to(ROOT)),
    "group_order": 480,
    "central_element_index": z_index,
    "central_element_order": 2,
    "centrality_verified": central,
    "order_four_root_count": len(roots),
    "root_indices": roots,
    "quotient_halfturn_count": len(pairs),
    "lift_pairs": [list(p) for p in pairs],
    "order_two_lift_count_among_pairs": 0,
    "native_group_obstruction": True,
    "nested_chamber_obstruction_proved": False,
    "physical_claim": False
}

OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n")

print(json.dumps(receipt, indent=2))
print("NHT004_NATIVE_HALF_TURN_PASS")
