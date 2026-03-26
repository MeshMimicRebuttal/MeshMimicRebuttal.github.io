#!/usr/bin/env python3
"""Convert VideoMimic .npz files to .js for direct browser loading.

Uses prefix __NPZ_VM_ to avoid collision with MeshMimic's __NPZ_ prefix.
"""
import base64
import glob
import os
import sys


def convert(npz_path: str) -> None:
    with open(npz_path, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode("ascii")
    name = os.path.splitext(os.path.basename(npz_path))[0]
    js_path = os.path.join(os.path.dirname(npz_path), name + ".js")

    with open(js_path, "w") as f:
        f.write('window.__NPZ_VM_' + name + '="' + b64 + '";\n')

    in_mb = len(raw) / 1048576
    out_mb = os.path.getsize(js_path) / 1048576
    print(f"  {name}: {in_mb:.1f} MB npz -> {out_mb:.1f} MB js")


def main() -> None:
    npz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videomimic")
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        print("No .npz files found in", npz_dir)
        sys.exit(1)
    print(f"Converting {len(files)} VideoMimic npz files to js ...")
    for f in files:
        convert(f)
    print("Done.")


if __name__ == "__main__":
    main()
