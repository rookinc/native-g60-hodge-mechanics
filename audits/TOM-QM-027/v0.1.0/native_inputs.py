"""Read the pinned Project 45 imports and Project 46 numerical seeds."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if not __debug__:
    raise RuntimeError("Run the audits with Python assertions enabled")

ROOT = Path(__file__).resolve().parents[3]
BASE_COMMIT = "4769486d7785c21530716d89e253f966f1dc5ab9"
IMPORT = "data/imported/project45/"
NPZ = "artifacts/npy/"
INPUTS = {
    "b1": IMPORT + "native_g60_B1_vertex_edge_004.csv",
    "b2": IMPORT + "native_g60_B2_edge_face_004.csv",
    "group": IMPORT + "native_g60_fiber_product_isomorphism_044.json",
    "actions": NPZ + "native_g60_signed_cochain_actions_005.npz",
    "harmonic": NPZ + "native_g60_harmonic_1_form_basis_004.npz",
    "coupling": NPZ + "native_g60_cross_flux_energy_coupling_021.npz",
    "response": NPZ + "native_g60_unique_four_flux_response_016.npz",
    "geometry": NPZ + "native_g60_four_flux_face_geometry_017.npz",
}
# Indices in the Project 42 044 permutation ledger, with names from
# Project 42's Klein voltage certificate and the observer plateau manuscript.
DECK_IDS = {"1": 0, "a": 326, "b": 124, "ab": 65}


def load(root):
    files = {name: root / path for name, path in INPUTS.items()}
    manifests = {
        name: {"path": INPUTS[name], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in files.items()
    }
    lock = json.loads(Path(__file__).with_name("inputs.lock.json").read_text())
    if manifests != lock["inputs"] or lock["project46_commit"] != BASE_COMMIT:
        raise ValueError("Project 46 source hashes differ from the pinned input lock")
    data = {}
    for name, path in files.items():
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            data[name] = np.asarray([row[1:] for row in rows[1:]], dtype=np.int64)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                data[name] = dict(archive)
        else:
            data[name] = json.loads(path.read_text())
    return data


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
