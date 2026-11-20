"""Replay the Project 46 Hodge and cross-flux producers in disposable copies."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

import numpy as np

from native_inputs import BASE_COMMIT, ROOT, write

PROGRAM_ID = re.compile(r"_(0(?:0[1-9]|1[0-9]|2[0-3])[a-c]?)\.py$")
VERDICT = re.compile(r"^(?:audit_pass|theorem_pass):\s*(True|False)\s*$", re.MULTILINE)
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
AUDIT_021 = "native_g60_cross_flux_energy_coupling_021.py"
LOCKED_BASIS_021 = """
import runpy
import numpy as np

with np.load('artifacts/npy/native_g60_cross_flux_energy_coupling_021.npz') as archive:
    bases = [archive[name] for name in ('six_a_basis', 'six_b_basis')]

def archived_basis(projector):
    (basis,) = [basis for basis in bases
                if np.max(np.abs(basis @ basis.T - projector)) < 1e-10]
    return basis

main = runpy.run_path('scripts/native_g60_cross_flux_energy_coupling_021.py')['main']
main.__globals__['orthonormal_basis_from_projector'] = archived_basis
main()
"""


def tree(archive, directory):
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(directory, filter="data")


def replay(directory, name, logs, *, locked_basis=False):
    if locked_basis:
        arguments = [sys.executable, "-c", LOCKED_BASIS_021]
    elif name.endswith(".py"):
        arguments = [sys.executable, "scripts/" + name]
    else:
        arguments = [sys.executable, "-m", "unittest", "discover", "-v"]
    try:
        run = subprocess.run(arguments, cwd=directory, env=ENV,
                             capture_output=True, text=True, timeout=180, check=False)
        output = run.stdout + run.stderr
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"").decode(errors="replace")
        output += (error.stderr or b"").decode(errors="replace")
        (logs / (name + ".log")).write_text(output)
        return {"script": name, "status": "timeout"}
    (logs / (name + ".log")).write_text(output)
    results = VERDICT.findall(output)
    if name == "unittest-discovery":
        match = re.search(r"Ran (\d+) tests?", output)
        count = int(match[1]) if match else None
        status = "pass" if run.returncode == 0 else "fail"
        if count == 0 and run.returncode in (0, 5):
            status = "no_tests"
        return {"script": name, "exit_code": run.returncode, "tests_found": count,
                "status": status}
    status = "fail" if run.returncode or "False" in results else "pass"
    if not results and not run.returncode:
        status = "no_verdict"
    report = {"script": name, "exit_code": run.returncode, "verdicts": results,
              "status": status}
    if status == "fail":
        report["diagnostic"] = re.findall(r"^(?:RuntimeError|verdict|channel_equivariance_max_abs):.*$",
                                          output, re.MULTILINE)
    return report


def run(root, output):
    archive = subprocess.check_output(["git", "-C", str(root), "archive", BASE_COMMIT])
    output.mkdir(parents=True, exist_ok=True)
    ordered_logs = output / "ordered"
    locked_logs = output / "locked"
    ordered_logs.mkdir(exist_ok=True)
    locked_logs.mkdir(exist_ok=True)
    with TemporaryDirectory() as directory:
        tree(archive, directory)
        programs = [(PROGRAM_ID.search(path.name)[1], path.name)
                    for path in (Path(directory) / "scripts").glob("native_g60_*.py")
                    if PROGRAM_ID.search(path.name)]
        names = [name for _, name in sorted(programs)]
        assert len(names) == 25
        discovery = replay(directory, "unittest-discovery", ordered_logs)
        ordered = [replay(directory, name, ordered_logs) for name in names]

    def run_locked(name):
        with TemporaryDirectory() as directory:
            tree(archive, directory)
            return replay(directory, name, locked_logs)

    with ThreadPoolExecutor(max_workers=3) as workers:
        locked = list(workers.map(run_locked, names))
    with TemporaryDirectory() as directory:
        tree(archive, directory)
        matched_basis = replay(directory, AUDIT_021, output, locked_basis=True)
    write(output / "upstream-replay.json", {
        "audit": "TOM-QM-027.V", "version": "0.1.0", "project46_commit": BASE_COMMIT,
        "scope": "All discovered unittest cases and 25 G60 producers 001 through 023, including lettered subaudits.",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "test_discovery": discovery, "ordered_runs": ordered, "pinned_input_runs": locked,
        "audit_021_pinned_input_with_archived_harmonic_basis": matched_basis,
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
