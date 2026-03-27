#!/usr/bin/env python3
"""Standalone script: convert VideoMimic raw data (H5 + OBJ) to NPZ + JS.

Usage:
    python convert_videomimic_standalone.py <raw_dir> <smplx_model_dir>

    raw_dir:          Root directory containing sub-folders, each with
                      background_mesh.obj and gravity_calibrated_megahunter.h5
    smplx_model_dir:  Directory containing SMPLX_NEUTRAL.npz (or .pkl)

Output is saved to <raw_dir>/videomimic_web/ with files like EMDB_20.npz, EMDB_20.js

Example:
    python convert_videomimic_standalone.py  D:/videomimic_results  D:/models/smplx

Dependencies:
    pip install numpy h5py trimesh torch smplx scipy
"""
import argparse
import base64
import os
import re
import sys

import h5py
import numpy as np
import torch
import trimesh

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── SMPL-X helpers ──────────────────────────────────────────────

def _load_smplx_model(model_dir, gender="neutral"):
    import smplx
    gender_map = {"neutral": "SMPLX_NEUTRAL.npz", "male": "SMPLX_MALE.npz", "female": "SMPLX_FEMALE.npz"}
    model_path = os.path.join(model_dir, gender_map[gender])
    if not os.path.exists(model_path):
        model_path = model_dir
    return smplx.create(
        model_path, model_type="smplx", gender=gender,
        use_pca=False, num_betas=10, batch_size=1, flat_hand_mean=True,
    )


def _rotmat_to_aa(rotmats):
    from scipy.spatial.transform import Rotation
    orig_shape = rotmats.shape[:-2]
    flat = rotmats.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return aa.reshape(*orig_shape, 3)


def _smplx_forward(model, betas, body_pose, global_orient, root_transl):
    T = body_pose.shape[0]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    betas_t = torch.tensor(betas, dtype=dtype, device=device)
    if betas_t.ndim == 2 and betas_t.shape[0] == T:
        pass
    else:
        betas_t = betas_t.unsqueeze(0).expand(T, -1)

    bp_aa = _rotmat_to_aa(body_pose[:, :21])
    bp_t = torch.tensor(bp_aa.reshape(T, -1), dtype=dtype, device=device)
    go_aa = _rotmat_to_aa(global_orient.reshape(T, 1, 3, 3))
    go_t = torch.tensor(go_aa.reshape(T, 3), dtype=dtype, device=device)
    transl_t = torch.tensor(root_transl.reshape(T, 3), dtype=dtype, device=device)

    all_verts = []
    for i in range(T):
        kwargs = dict(
            betas=betas_t[i:i+1], body_pose=bp_t[i:i+1],
            global_orient=go_t[i:i+1], transl=transl_t[i:i+1],
            return_verts=True,
        )
        if body_pose.shape[1] > 21:
            jaw_aa = _rotmat_to_aa(body_pose[i:i+1, 21:22])
            kwargs["jaw_pose"] = torch.tensor(jaw_aa.reshape(1, 3), dtype=dtype, device=device)
        if body_pose.shape[1] > 22:
            leye_aa = _rotmat_to_aa(body_pose[i:i+1, 22:23])
            kwargs["leye_pose"] = torch.tensor(leye_aa.reshape(1, 3), dtype=dtype, device=device)
        with torch.no_grad():
            output = model(**kwargs)
        all_verts.append(output.vertices.cpu().numpy().astype(np.float32)[0])

    verts = np.stack(all_verts, axis=0)
    faces = model.faces_tensor.cpu().numpy().astype(np.int32) if hasattr(model, "faces_tensor") else model.faces.astype(np.int32)
    return verts, faces

# ── Scene packaging ─────────────────────────────────────────────

def package_scene(scene_dir, scene_name, output_dir, smplx_model_dir):
    obj_path = os.path.join(scene_dir, "background_mesh.obj")
    h5_path = os.path.join(scene_dir, "gravity_calibrated_megahunter.h5")

    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Missing {obj_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing {h5_path}")

    bg = trimesh.load(obj_path, process=False)
    scene_vertices = np.asarray(bg.vertices, dtype=np.float32)
    scene_faces = np.asarray(bg.faces, dtype=np.int32)
    print(f"  Scene mesh: {scene_vertices.shape[0]} verts, {scene_faces.shape[0]} faces")

    with h5py.File(h5_path, "r") as f:
        smplx_grp = f["our_pred_humans_smplx_params"]
        person_ids = list(smplx_grp.keys())
        if not person_ids:
            raise ValueError("No person found in H5")
        pid = person_ids[0]
        print(f"  Person ID: {pid}")

        betas = np.array(smplx_grp[pid]["betas"], dtype=np.float32)
        body_pose = np.array(smplx_grp[pid]["body_pose"], dtype=np.float32)
        global_orient = np.array(smplx_grp[pid]["global_orient"], dtype=np.float32)
        root_transl = np.array(smplx_grp[pid]["root_transl"], dtype=np.float32)
        print(f"  SMPL-X: betas={betas.shape}, body_pose={body_pose.shape}, "
              f"orient={global_orient.shape}, transl={root_transl.shape}")

        pfl = f["person_frame_info_list"][pid][()]
        human_frame_ids = [s[0].decode() if isinstance(s[0], bytes) else str(s[0]) for s in pfl]
        human_frame_ids_int = [int(fid) for fid in human_frame_ids]
        print(f"  Human frames: {len(human_frame_ids)}")

        cam_grp = f["our_pred_world_cameras_and_structure"]
        c2w_R_list, c2w_t_list = [], []
        for fid_str in human_frame_ids:
            fid_padded = fid_str.zfill(5)
            if fid_padded not in cam_grp:
                print(f"    WARNING: camera frame {fid_padded} not found, skipping")
                continue
            c2w = np.array(cam_grp[fid_padded]["cam2world"], dtype=np.float32)
            c2w_R_list.append(c2w[:3, :3])
            c2w_t_list.append(c2w[:3, 3])

    c2w_R = np.stack(c2w_R_list, axis=0).astype(np.float32)
    c2w_t = np.stack(c2w_t_list, axis=0).astype(np.float32)

    print(f"  Running SMPL-X forward pass ...")
    model = _load_smplx_model(smplx_model_dir)
    human_vertices, human_faces = _smplx_forward(model, betas, body_pose, global_orient, root_transl)
    print(f"  Human mesh: {human_vertices.shape[1]} verts/frame, {human_faces.shape[0]} faces, "
          f"{human_vertices.shape[0]} frames")

    assert human_vertices.shape[0] == c2w_R.shape[0], \
        f"Frame count mismatch: human={human_vertices.shape[0]} vs camera={c2w_R.shape[0]}"

    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{scene_name}.npz")
    np.savez_compressed(
        npz_path, scene_name=scene_name,
        scene_vertices=scene_vertices, scene_faces=scene_faces,
        human_vertices=human_vertices, human_faces=human_faces,
        frame_ids=np.array(human_frame_ids_int, dtype=np.int32),
        c2w_R=c2w_R, c2w_t=c2w_t,
    )
    npz_mb = os.path.getsize(npz_path) / 1048576
    print(f"  Saved NPZ: {npz_path} ({npz_mb:.1f} MB)")

    with open(npz_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    js_path = os.path.join(output_dir, f"{scene_name}.js")
    with open(js_path, "w") as f:
        f.write(f'window.__NPZ_VM_{scene_name}="{b64}";\n')
    js_mb = os.path.getsize(js_path) / 1048576
    print(f"  Saved JS:  {js_path} ({js_mb:.1f} MB)")

    return npz_path

# ── Scene discovery ─────────────────────────────────────────────

NAME_MAP = {
    "emdb_20": "EMDB_20", "emdb_27": "EMDB_27", "emdb_35": "EMDB_35",
    "emdb_36": "EMDB_36", "emdb_40": "EMDB_40", "emdb_48": "EMDB_48",
    "emdb_56": "EMDB_56", "emdb_58": "EMDB_58",
}

LONG_RE = re.compile(
    r"^megahunter_megasam_reconstruction_results_(.+?)_cam\d+_frame_\d+_\d+_subsample_\d+$"
)

def _normalize_name(folder_name):
    """Extract a short scene name from a folder name."""
    m = LONG_RE.match(folder_name)
    raw = m.group(1) if m else folder_name
    return NAME_MAP.get(raw.lower(), raw)


def discover_scenes(raw_dir):
    """Return list of (scene_name, scene_dir_path)."""
    results = []
    for d in sorted(os.listdir(raw_dir)):
        full = os.path.join(raw_dir, d)
        if not os.path.isdir(full):
            continue
        if not os.path.exists(os.path.join(full, "gravity_calibrated_megahunter.h5")):
            continue
        name = _normalize_name(d)
        results.append((name, full))
    return results

# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert VideoMimic raw data (H5+OBJ) to NPZ+JS for web viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("raw_dir", help="Root directory with sub-folders of VideoMimic scenes")
    parser.add_argument("smplx_model_dir", help="Directory containing SMPLX_NEUTRAL.npz")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <raw_dir>/videomimic_web)")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Only process specific scenes (short names, e.g. EMDB_20 parkour)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.raw_dir, "videomimic_web")
    all_scenes = discover_scenes(args.raw_dir)

    if args.scenes:
        wanted = set(args.scenes)
        all_scenes = [(n, d) for n, d in all_scenes if n in wanted]

    if not all_scenes:
        print(f"No scenes found in {args.raw_dir}")
        print("Each sub-folder must contain background_mesh.obj and gravity_calibrated_megahunter.h5")
        sys.exit(1)

    print(f"Found {len(all_scenes)} scenes in {args.raw_dir}")
    print(f"Output directory: {output_dir}\n")

    ok, fail = 0, 0
    for name, scene_dir in all_scenes:
        print(f"=== {name} === ({os.path.basename(scene_dir)})")
        try:
            package_scene(scene_dir, name, output_dir, args.smplx_model_dir)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            fail += 1
        print()

    print(f"Done. {ok} succeeded, {fail} failed.")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
