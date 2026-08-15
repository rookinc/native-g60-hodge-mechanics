#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

JSON_PATHS = [
    ROOT / "artifacts/json/native_g60_cross_flux_gap_e5_rational_kernel_chart_051.json",
    ROOT / "artifacts/json/native_g60_cross_flux_gap_e5_rational_face_feasibility_052.json",
]

NPZ_PATHS = [
    ROOT / "artifacts/npy/native_g60_cross_flux_gap_e5_rational_kernel_chart_051.npz",
    ROOT / "artifacts/npy/native_g60_cross_flux_gap_e5_rational_face_feasibility_052.npz",
]


def summarize_json(path: Path) -> None:
    print()
    print("== JSON ==")
    print("path:", path)
    print("exists:", path.exists())

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    print("top_level_type:", type(payload).__name__)

    if isinstance(payload, dict):
        print("top_level_keys:")
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, dict):
                detail = f"dict[{len(value)}]"
            elif isinstance(value, list):
                detail = f"list[{len(value)}]"
            else:
                detail = repr(value)
                if len(detail) > 160:
                    detail = detail[:157] + "..."
            print(" ", key, "->", detail)


def summarize_npz(path: Path) -> None:
    print()
    print("== NPZ ==")
    print("path:", path)
    print("exists:", path.exists())

    if not path.exists():
        return

    with np.load(path, allow_pickle=True) as archive:
        print("keys:")
        for key in sorted(archive.files):
            value = archive[key]
            print(
                " ",
                key,
                "shape=",
                value.shape,
                "dtype=",
                value.dtype,
            )

            if value.ndim == 0:
                try:
                    scalar = value.item()
                    text = repr(scalar)
                    if len(text) > 200:
                        text = text[:197] + "..."
                    print("    scalar:", text)
                except Exception as exc:
                    print("    scalar_read_error:", repr(exc))


def main() -> None:
    print("e5_exact_reconstruction_input_inspection_053a")

    for path in JSON_PATHS:
        summarize_json(path)

    for path in NPZ_PATHS:
        summarize_npz(path)


if __name__ == "__main__":
    main()
