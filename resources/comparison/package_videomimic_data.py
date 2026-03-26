#!/usr/bin/env python3
"""Package VideoMimic raw data (H5 + OBJ) into .npz files matching MeshMimic format.

Output keys: scene_name, scene_vertices, scene_faces, human_vertices,
             human_faces, frame_ids, c2w_R, c2w_t
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import trimesh

SMPLX_MODEL_DIR = r"C:\Users\A\Desktop\smplx"


def _load_smplx_model(model_dir: str, gender: str = "neutral"):
    """Load SMPL-X model."""
    import smplx
    gender_map = {"neutral": "SMPLX_NEUTRAL.npz", "male": "SMPLX_MALE.npz", "female": "SMPLX_FEMALE.npz"}
    model_path = os.path.join(model_dir, gender_map[gender])
    model = smplx.create(
        model_path,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=10,
        batch_size=1,
        flat_hand_mean=True,
    )
    return model


def _rotmat_to_aa(rotmats: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (..., 3, 3) to axis-angle (..., 3)."""
    from scipy.spatial.transform import Rotation
    orig_shape = rotmats.shape[:-2]
    flat = rotmats.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return aa.reshape(*orig_shape, 3)


def _smplx_forward(model, betas, body_pose, global_orient, root_transl):
    """Run SMPL-X forward pass and return vertices (T, V, 3) and faces (F, 3).

    body_pose may have 21 or 23 joints; we use only the first 21 for SMPL-X body.
    Input rotation matrices are converted to axis-angle for the model.
    """
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
            betas=betas_t[i:i+1],
            body_pose=bp_t[i:i+1],
            global_orient=go_t[i:i+1],
            transl=transl_t[i:i+1],
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


def package_scene(scene_dir: str, scene_name: str, output_dir: str, smplx_model_dir: str) -> str:
    """Convert one VideoMimic scene folder to a MeshMimic-compatible .npz."""
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

        print(f"  SMPL-X params: betas={betas.shape}, body_pose={body_pose.shape}, "
              f"global_orient={global_orient.shape}, root_transl={root_transl.shape}")

        pfl = f["person_frame_info_list"][pid][()]
        human_frame_ids = [s[0].decode() if isinstance(s[0], bytes) else str(s[0])
                           for s in pfl]
        human_frame_ids_int = [int(fid) for fid in human_frame_ids]
        print(f"  Human frames: {len(human_frame_ids)} -> {human_frame_ids}")

        cam_grp = f["our_pred_world_cameras_and_structure"]
        c2w_R_list = []
        c2w_t_list = []
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
    human_vertices, human_faces = _smplx_forward(
        model, betas, body_pose, global_orient, root_transl
    )
    print(f"  Human mesh: {human_vertices.shape[1]} verts/frame, {human_faces.shape[0]} faces, "
          f"{human_vertices.shape[0]} frames")

    assert human_vertices.shape[0] == c2w_R.shape[0], \
        f"Frame count mismatch: human={human_vertices.shape[0]} vs camera={c2w_R.shape[0]}"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scene_name}.npz")
    np.savez_compressed(
        out_path,
        scene_name=scene_name,
        scene_vertices=scene_vertices,
        scene_faces=scene_faces,
        human_vertices=human_vertices,
        human_faces=human_faces,
        frame_ids=np.array(human_frame_ids_int, dtype=np.int32),
        c2w_R=c2w_R,
        c2w_t=c2w_t,
    )
    size_mb = os.path.getsize(out_path) / 1048576
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


import re

def _discover_scenes(raw_dir: str):
    """Auto-discover scene folders. Supports both flat layout (scene_name/)
    and megahunter long-name layout (megahunter_..._SCENE_cam01_...).
    Returns list of (scene_name, scene_dir_path)."""
    results = []
    PREFIX_RE = re.compile(
        r"^megahunter_megasam_reconstruction_results_(.+?)_cam\d+_frame_\d+_\d+_subsample_\d+$"
    )
    for d in sorted(os.listdir(raw_dir)):
        full = os.path.join(raw_dir, d)
        if not os.path.isdir(full):
            continue
        m = PREFIX_RE.match(d)
        name = m.group(1) if m else d
        if os.path.exists(os.path.join(full, "gravity_calibrated_megahunter.h5")):
            results.append((name, full))
    return results


def main():
    parser = argparse.ArgumentParser(description="Package VideoMimic data to npz")
    parser.add_argument("--raw-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "videomimic_raw_data", "output_results"))
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "videomimic"))
    parser.add_argument("--smplx-model-dir", default=SMPLX_MODEL_DIR)
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Specific scene names to process (default: all)")
    args = parser.parse_args()

    all_scenes = _discover_scenes(args.raw_dir)
    if args.scenes:
        wanted = set(args.scenes)
        all_scenes = [(n, d) for n, d in all_scenes if n in wanted]

    print(f"Found {len(all_scenes)} scenes in {args.raw_dir}")
    for name, scene_dir in all_scenes:
        print(f"\n=== {name} === ({os.path.basename(scene_dir)})")
        try:
            package_scene(
                scene_dir=scene_dir,
                scene_name=name,
                output_dir=args.output_dir,
                smplx_model_dir=args.smplx_model_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
