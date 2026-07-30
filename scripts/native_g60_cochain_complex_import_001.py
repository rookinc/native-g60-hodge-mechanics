from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    ROOT
    / "data"
    / "imported"
    / "project45"
)

MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"

SURFACE_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_orientation_004.json"
)

SYMMETRY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_symmetry_and_kernel_005.json"
)

HOMOLOGY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_integral_homology_006.json"
)

CANONICITY_PATH = (
    SOURCE_ROOT
    / "native_g60_surface_canonicity_census_007.json"
)

B1_PATH = (
    SOURCE_ROOT
    / "native_g60_B1_vertex_edge_004.csv"
)

B2_PATH = (
    SOURCE_ROOT
    / "native_g60_B2_edge_face_004.csv"
)

JSON_OUT = (
    ROOT
    / "artifacts"
    / "json"
    / "native_g60_cochain_complex_import_001.json"
)

CSV_OUT = (
    ROOT
    / "artifacts"
    / "csv"
    / "native_g60_cochain_complex_import_001.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_matrix_csv(
    path: Path,
) -> tuple[list[str], list[int], list[list[int]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise RuntimeError(f"empty matrix file: {path}")

    column_labels = rows[0][1:]
    row_labels = []
    matrix = []

    for row in rows[1:]:
        row_labels.append(int(row[0]))
        matrix.append(
            [int(value) for value in row[1:]]
        )

    return column_labels, row_labels, matrix


def matrix_rank_q(
    matrix: list[list[int]],
) -> int:
    if not matrix:
        return 0

    work = [
        [Fraction(value) for value in row]
        for row in matrix
    ]

    row_count = len(work)
    column_count = len(work[0])

    rank = 0
    pivot_column = 0

    while (
        rank < row_count
        and pivot_column < column_count
    ):
        pivot_row = next(
            (
                row
                for row in range(rank, row_count)
                if work[row][pivot_column] != 0
            ),
            None,
        )

        if pivot_row is None:
            pivot_column += 1
            continue

        work[rank], work[pivot_row] = (
            work[pivot_row],
            work[rank],
        )

        pivot = work[rank][pivot_column]

        work[rank] = [
            value / pivot
            for value in work[rank]
        ]

        for row in range(row_count):
            if row == rank:
                continue

            factor = work[row][pivot_column]

            if factor == 0:
                continue

            work[row] = [
                work[row][column]
                - factor * work[rank][column]
                for column in range(column_count)
            ]

        rank += 1
        pivot_column += 1

    return rank


def matrix_rank_f2(
    matrix: list[list[int]],
) -> int:
    if not matrix:
        return 0

    column_count = len(matrix[0])

    rows = [
        sum(
            ((value & 1) << column)
            for column, value in enumerate(row)
        )
        for row in matrix
    ]

    rank = 0

    for column in range(column_count):
        pivot_row = next(
            (
                row
                for row in range(rank, len(rows))
                if (rows[row] >> column) & 1
            ),
            None,
        )

        if pivot_row is None:
            continue

        rows[rank], rows[pivot_row] = (
            rows[pivot_row],
            rows[rank],
        )

        for row in range(len(rows)):
            if row == rank:
                continue

            if (rows[row] >> column) & 1:
                rows[row] ^= rows[rank]

        rank += 1

    return rank


def transpose(
    matrix: list[list[int]],
) -> list[list[int]]:
    if not matrix:
        return []

    return [
        [
            matrix[row][column]
            for row in range(len(matrix))
        ]
        for column in range(len(matrix[0]))
    ]


def multiply(
    left: list[list[int]],
    right: list[list[int]],
) -> list[list[int]]:
    if not left or not right:
        return []

    if len(left[0]) != len(right):
        raise RuntimeError(
            "matrix dimensions do not align"
        )

    result = [
        [0 for _ in range(len(right[0]))]
        for _ in range(len(left))
    ]

    for i, left_row in enumerate(left):
        for k, value in enumerate(left_row):
            if value == 0:
                continue

            for j, right_value in enumerate(right[k]):
                result[i][j] += value * right_value

    return result


def nonzero_entries(
    matrix: list[list[int]],
) -> list[dict[str, int]]:
    return [
        {
            "row": row,
            "column": column,
            "value": matrix[row][column],
        }
        for row in range(len(matrix))
        for column in range(len(matrix[row]))
        if matrix[row][column] != 0
    ]


def main() -> None:
    JSON_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CSV_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    surface = json.loads(
        SURFACE_PATH.read_text(encoding="utf-8")
    )

    symmetry = json.loads(
        SYMMETRY_PATH.read_text(encoding="utf-8")
    )

    homology = json.loads(
        HOMOLOGY_PATH.read_text(encoding="utf-8")
    )

    canonicity = json.loads(
        CANONICITY_PATH.read_text(encoding="utf-8")
    )

    b1_columns, b1_rows, b1 = read_matrix_csv(
        B1_PATH
    )

    b2_columns, b2_rows, b2 = read_matrix_csv(
        B2_PATH
    )

    hash_rows = []

    for source in manifest["sources"]:
        relative_path = Path(source["file"])

        if relative_path.parts[:3] == (
            "data",
            "imported",
            "project45",
        ):
            path = ROOT / relative_path
        else:
            path = SOURCE_ROOT / relative_path.name

        exists = path.exists()
        actual_hash = sha256(path) if exists else None

        hash_rows.append(
            {
                "file": source["file"],
                "role": source["role"],
                "exists": exists,
                "expected_sha256": source["sha256"],
                "actual_sha256": actual_hash,
                "hash_match": (
                    exists
                    and actual_hash == source["sha256"]
                ),
                "immutable_import": source.get(
                    "immutable_import"
                )
                is True,
            }
        )

    b1_b2 = multiply(b1, b2)
    b1_b2_nonzero = nonzero_entries(b1_b2)

    d0 = transpose(b1)
    d1 = transpose(b2)
    d1_d0 = multiply(d1, d0)
    d1_d0_nonzero = nonzero_entries(d1_d0)

    rank_b1_q = matrix_rank_q(b1)
    rank_b1_f2 = matrix_rank_f2(b1)
    rank_b2_q = matrix_rank_q(b2)
    rank_b2_f2 = matrix_rank_f2(b2)

    b1_column_nonzero_profile = Counter(
        sum(
            1
            for row in range(len(b1))
            if b1[row][column] != 0
        )
        for column in range(len(b1[0]))
    )

    b2_row_nonzero_profile = Counter(
        sum(
            1
            for value in row
            if value != 0
        )
        for row in b2
    )

    b2_row_sum_profile = Counter(
        sum(row)
        for row in b2
    )

    graph_cycle_dimension = (
        len(b1[0]) - rank_b1_q
    )

    h1_dimension_q = (
        len(b1[0])
        - rank_b1_q
        - rank_b2_q
    )

    h1_dimension_f2 = (
        len(b1[0])
        - rank_b1_f2
        - rank_b2_f2
    )

    checks = {
        "manifest_source_count_is_9": (
            manifest["source_count"] == 9
            and len(hash_rows) == 9
        ),
        "all_imported_sources_exist": all(
            row["exists"]
            for row in hash_rows
        ),
        "all_imported_hashes_match": all(
            row["hash_match"]
            for row in hash_rows
        ),
        "all_sources_marked_immutable": all(
            row["immutable_import"]
            for row in hash_rows
        ),
        "surface_input_audit_pass": (
            surface.get("audit_pass") is True
        ),
        "symmetry_input_audit_pass": (
            symmetry.get("audit_pass") is True
        ),
        "homology_input_audit_pass": (
            homology.get("audit_pass") is True
        ),
        "canonicity_input_audit_pass": (
            canonicity.get("audit_pass") is True
        ),
        "b1_shape_is_60_by_120": (
            len(b1) == 60
            and len(b1[0]) == 120
        ),
        "b2_shape_is_120_by_20": (
            len(b2) == 120
            and len(b2[0]) == 20
        ),
        "b1_row_labels_are_0_through_59": (
            b1_rows == list(range(60))
        ),
        "b2_row_labels_are_0_through_119": (
            b2_rows == list(range(120))
        ),
        "b1_column_count_is_120": (
            len(b1_columns) == 120
        ),
        "b2_column_count_is_20": (
            len(b2_columns) == 20
        ),
        "every_edge_has_two_vertex_incidents": (
            dict(b1_column_nonzero_profile)
            == {2: 120}
        ),
        "every_edge_has_two_face_incidents": (
            dict(b2_row_nonzero_profile)
            == {2: 120}
        ),
        "oriented_face_incidents_cancel_per_edge": (
            dict(b2_row_sum_profile)
            == {0: 120}
        ),
        "b1_b2_is_zero": (
            not b1_b2_nonzero
        ),
        "d1_d0_is_zero": (
            not d1_d0_nonzero
        ),
        "rank_b1_q_is_59": (
            rank_b1_q == 59
        ),
        "rank_b1_f2_is_59": (
            rank_b1_f2 == 59
        ),
        "rank_b2_q_is_19": (
            rank_b2_q == 19
        ),
        "rank_b2_f2_is_19": (
            rank_b2_f2 == 19
        ),
        "graph_cycle_dimension_is_61": (
            graph_cycle_dimension == 61
        ),
        "h1_dimension_q_is_42": (
            h1_dimension_q == 42
        ),
        "h1_dimension_f2_is_42": (
            h1_dimension_f2 == 42
        ),
        "import_matches_surface_dimensions": (
            surface["surface"]["vertex_count"] == 60
            and surface["surface"]["edge_count"] == 120
            and surface["surface"]["face_count"] == 20
        ),
        "import_matches_integral_homology": (
            homology["integral_homology"]["h0"]["group"]
            == "Z"
            and homology["integral_homology"]["h1"]["group"]
            == "Z^42"
            and homology["integral_homology"]["h2"]["group"]
            == "Z"
        ),
    }

    audit_pass = all(checks.values())

    payload = {
        "artifact_id": (
            "native_g60_cochain_complex_import_001"
        ),
        "audit_pass": audit_pass,
        "verdict": (
            "project45_native_g60_cochain_complex_received_unchanged"
            if audit_pass
            else "project45_native_g60_cochain_complex_import_failed"
        ),
        "authority": {
            "source_project": (
                "45-native-g60-surface-complex"
            ),
            "consumer_project": (
                "46-native-g60-hodge-mechanics"
            ),
            "source_remains_authority": True,
            "import_is_immutable": True,
            "surface_rederived": False,
            "surface_modified": False,
        },
        "checks": checks,
        "hash_audit": {
            "source_count": len(hash_rows),
            "match_count": sum(
                1
                for row in hash_rows
                if row["hash_match"]
            ),
            "mismatch_count": sum(
                1
                for row in hash_rows
                if not row["hash_match"]
            ),
            "rows": hash_rows,
        },
        "chain_complex": {
            "c0_dimension": 60,
            "c1_dimension": 120,
            "c2_dimension": 20,
            "b1_shape": [60, 120],
            "b2_shape": [120, 20],
            "d0_shape": [120, 60],
            "d1_shape": [20, 120],
            "b1_b2_nonzero_entry_count": (
                len(b1_b2_nonzero)
            ),
            "d1_d0_nonzero_entry_count": (
                len(d1_d0_nonzero)
            ),
        },
        "ranks": {
            "rank_b1_q": rank_b1_q,
            "rank_b1_f2": rank_b1_f2,
            "rank_b2_q": rank_b2_q,
            "rank_b2_f2": rank_b2_f2,
            "graph_cycle_dimension": (
                graph_cycle_dimension
            ),
            "h1_dimension_q": h1_dimension_q,
            "h1_dimension_f2": h1_dimension_f2,
        },
        "incidence_profiles": {
            "b1_edge_vertex_incidence": dict(
                sorted(
                    b1_column_nonzero_profile.items()
                )
            ),
            "b2_edge_face_incidence": dict(
                sorted(
                    b2_row_nonzero_profile.items()
                )
            ),
            "b2_edge_orientation_sum": dict(
                sorted(
                    b2_row_sum_profile.items()
                )
            ),
        },
        "boundary": {
            "cochain_complex_received": audit_pass,
            "hodge_structure_constructed": False,
            "hodge_decomposition_proved": False,
            "dynamics_defined": False,
            "maxwell_claim": False,
            "physical_claim": False,
            "force_claim": False,
            "universe_simulation_claim": False,
            "unification_claim": False,
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

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "file",
                "role",
                "exists",
                "immutable_import",
                "hash_match",
                "expected_sha256",
                "actual_sha256",
            ]
        )

        for row in hash_rows:
            writer.writerow(
                [
                    row["file"],
                    row["role"],
                    row["exists"],
                    row["immutable_import"],
                    row["hash_match"],
                    row["expected_sha256"],
                    row["actual_sha256"],
                ]
            )

    print("audit_pass:", audit_pass)
    print("verdict:", payload["verdict"])
    print(
        "source_hash_matches:",
        payload["hash_audit"]["match_count"],
        "/",
        payload["hash_audit"]["source_count"],
    )
    print("B1_shape:", [len(b1), len(b1[0])])
    print("B2_shape:", [len(b2), len(b2[0])])
    print(
        "B1_B2_nonzero_entry_count:",
        len(b1_b2_nonzero),
    )
    print(
        "d1_d0_nonzero_entry_count:",
        len(d1_d0_nonzero),
    )
    print(
        "rank_B1_Q/F2:",
        rank_b1_q,
        rank_b1_f2,
    )
    print(
        "rank_B2_Q/F2:",
        rank_b2_q,
        rank_b2_f2,
    )
    print(
        "graph_cycle_dimension:",
        graph_cycle_dimension,
    )
    print(
        "H1_dimension_Q/F2:",
        h1_dimension_q,
        h1_dimension_f2,
    )
    print(
        "B1_edge_vertex_profile:",
        dict(
            sorted(
                b1_column_nonzero_profile.items()
            )
        ),
    )
    print(
        "B2_edge_face_profile:",
        dict(
            sorted(
                b2_row_nonzero_profile.items()
            )
        ),
    )
    print(
        "B2_orientation_sum_profile:",
        dict(
            sorted(
                b2_row_sum_profile.items()
            )
        ),
    )
    print("wrote:", JSON_OUT)
    print("wrote:", CSV_OUT)


if __name__ == "__main__":
    main()
