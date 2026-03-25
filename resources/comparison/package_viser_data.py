import argparse
import os
from glob import glob
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import trimesh
from tqdm import tqdm

from lib.utils.utils import load_dict_from_hdf5


def _to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
	R = np.asarray(R, np.float32).reshape(3, 3)
	t = np.asarray(t, np.float32).reshape(-1)[:3]
	T = np.eye(4, dtype=np.float32)
	T[:3, :3] = R
	T[:3, 3] = t
	return T


def _inv_T(T: np.ndarray) -> np.ndarray:
	R = T[:3, :3]
	t = T[:3, 3:4]
	Ti = np.eye(4, dtype=np.float32)
	Ti[:3, :3] = R.T
	Ti[:3, 3:4] = -R.T @ t
	return Ti


def _infer_c2w_batch(
	R_in: np.ndarray,
	t_in: np.ndarray,
	center_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	R_in = np.asarray(R_in, np.float32)
	t_in = np.asarray(t_in, np.float32).reshape(R_in.shape[0], -1)[:, :3]
	center_w = np.asarray(center_w, np.float32).reshape(3)

	z_a = []
	for i in range(R_in.shape[0]):
		pc = R_in[i] @ center_w + t_in[i]
		z_a.append(pc[2])
	z_a = np.median(np.array(z_a))

	z_b = []
	for i in range(R_in.shape[0]):
		T = _to_T(R_in[i], t_in[i])
		w2c = _inv_T(T)
		pc = w2c[:3, :3] @ center_w + w2c[:3, 3]
		z_b.append(pc[2])
	z_b = np.median(np.array(z_b))

	if z_a > z_b:
		c2w_R = []
		c2w_t = []
		for i in range(R_in.shape[0]):
			w2c = _to_T(R_in[i], t_in[i])
			c2w = _inv_T(w2c)
			c2w_R.append(c2w[:3, :3])
			c2w_t.append(c2w[:3, 3])
		return np.stack(c2w_R, axis=0), np.stack(c2w_t, axis=0)

	return R_in, t_in


def _convert_c2w_batch(
	R_in: np.ndarray,
	t_in: np.ndarray,
	center_w: np.ndarray,
	camera_convention: str,
) -> Tuple[np.ndarray, np.ndarray]:
	if camera_convention == "direct":
		return R_in, t_in
	if camera_convention == "inverse":
		c2w_R = []
		c2w_t = []
		for i in range(R_in.shape[0]):
			w2c = _to_T(R_in[i], t_in[i])
			c2w = _inv_T(w2c)
			c2w_R.append(c2w[:3, :3])
			c2w_t.append(c2w[:3, 3])
		return np.stack(c2w_R, axis=0), np.stack(c2w_t, axis=0)
	if camera_convention == "auto":
		return _infer_c2w_batch(R_in=R_in, t_in=t_in, center_w=center_w)
	raise ValueError(f"Unsupported camera_convention: {camera_convention}")


def _discover_scenes(results_root: str, scene_names: Optional[Sequence[str]]) -> List[str]:
	if scene_names:
		return list(scene_names)

	scene_dirs = sorted(glob(os.path.join(results_root, "*")))
	return [os.path.basename(scene_dir) for scene_dir in scene_dirs if os.path.isdir(scene_dir)]


def _required_paths(seq_folder: str) -> dict:
	hps_candidates = sorted(glob(os.path.join(seq_folder, "hps", "hps_track_*.npy")))
	return {
		"background": os.path.join(seq_folder, "pointcloud", "background", "background_planar.obj"),
		"camera": os.path.join(seq_folder, "camera.npy"),
		"faces": os.path.join(seq_folder, "face.npy"),
		"optim": os.path.join(seq_folder, "optims_post.h5"),
		"hps": hps_candidates[0] if hps_candidates else None,
	}


def _validate_scene_inputs(paths: dict) -> List[str]:
	missing = []
	for key, path in paths.items():
		if not path or not os.path.exists(path):
			missing.append(key)
	return missing


def _load_scene_package(seq_folder: str, camera_convention: str = "direct") -> dict:
	paths = _required_paths(seq_folder)
	missing = _validate_scene_inputs(paths)
	if missing:
		raise FileNotFoundError(f"Missing required inputs: {', '.join(missing)}")

	background_obj = trimesh.load(paths["background"])
	hps_dict = np.load(paths["hps"], allow_pickle=True).item()
	cameras = np.load(paths["camera"], allow_pickle=True).item()
	mesh_faces = np.load(paths["faces"]).astype(np.int32)

	with h5py.File(paths["optim"], "r") as f:
		optim_params = load_dict_from_hdf5(f)

	valid_frame_idx = sorted(int(key) for key in hps_dict.keys() if key != "faces")
	if not valid_frame_idx:
		raise ValueError("No human frames found in HPS results.")

	all_smpl_verts = []
	pred_cam_t = []
	scales = []
	trans = []
	c2w_R = []
	c2w_t = []
	for frame_id in valid_frame_idx:
		frame_hps = hps_dict[frame_id][0]
		all_smpl_verts.append(np.asarray(frame_hps["pred_vertices"], dtype=np.float32))
		pred_cam_t.append(np.asarray(optim_params[str(frame_id)]["pred_cam_t"], dtype=np.float32).reshape(3))
		scales.append(np.asarray(optim_params[str(frame_id)]["scale"], dtype=np.float32).reshape(-1)[0])
		trans.append(np.asarray(optim_params[str(frame_id)]["trans"], dtype=np.float32).reshape(3))
		c2w_R.append(np.asarray(cameras["world_cam_R"][frame_id], dtype=np.float32).reshape(3, 3))
		c2w_t.append(np.asarray(cameras["world_cam_T"][frame_id], dtype=np.float32).reshape(3))

	bg_vertices = np.asarray(background_obj.vertices, dtype=np.float32)
	center_w = (bg_vertices.min(axis=0) + bg_vertices.max(axis=0)) * 0.5
	c2w_R, c2w_t = _convert_c2w_batch(
		R_in=np.stack(c2w_R, axis=0),
		t_in=np.stack(c2w_t, axis=0),
		center_w=center_w,
		camera_convention=camera_convention,
	)

	human_vertices_world = []
	for verts, scale, trans_vec, pred_cam_vec, R, t in zip(
		all_smpl_verts,
		np.asarray(scales, dtype=np.float32),
		np.asarray(trans, dtype=np.float32),
		np.asarray(pred_cam_t, dtype=np.float32),
		c2w_R,
		c2w_t,
	):
		verts_cam = verts * scale + trans_vec.reshape(1, 3) + pred_cam_vec.reshape(1, 3)
		verts_world = (R @ verts_cam.T + t.reshape(3, 1)).T
		human_vertices_world.append(verts_world.astype(np.float32))

	return {
		"scene_vertices": bg_vertices,
		"scene_faces": np.asarray(background_obj.faces, dtype=np.int32),
		"human_vertices": np.stack(human_vertices_world, axis=0),
		"human_faces": mesh_faces,
		"frame_ids": np.asarray(valid_frame_idx, dtype=np.int32),
		"c2w_R": c2w_R.astype(np.float32),
		"c2w_t": c2w_t.astype(np.float32),
		"camera_convention": np.asarray(camera_convention),
	}


def _save_scene_package(output_dir: str, scene_name: str, package: dict) -> str:
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, f"{scene_name}.npz")
	np.savez_compressed(output_path, scene_name=scene_name, **package)
	return output_path


def package_scene(
	results_root: str,
	output_dir: str,
	scene_name: str,
	camera_convention: str = "direct",
) -> str:
	seq_folder = os.path.join(results_root, scene_name)
	package = _load_scene_package(seq_folder, camera_convention=camera_convention)
	return _save_scene_package(output_dir, scene_name, package)


def package_all_scenes(
	results_root: str,
	output_dir: str,
	scene_names: Optional[Sequence[str]] = None,
	camera_convention: str = "direct",
) -> Tuple[List[str], List[Tuple[str, str]]]:
	saved = []
	skipped = []
	scenes = _discover_scenes(results_root, scene_names)

	for scene_name in tqdm(scenes, desc="Packaging scenes"):
		try:
			output_path = package_scene(
				results_root,
				output_dir,
				scene_name,
				camera_convention=camera_convention,
			)
			saved.append(output_path)
		except Exception as exc:
			skipped.append((scene_name, str(exc)))

	return saved, skipped


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Package scene mesh and human vertices for Viser demos.",
	)
	parser.add_argument(
		"--results-root",
		type=str,
		default="results",
		help="Root folder that contains per-scene reconstruction outputs.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=os.path.join("paper", "demo", "maskmimic"),
		help="Directory used to store packaged per-scene npz files.",
	)
	parser.add_argument(
		"--scenes",
		nargs="*",
		default=None,
		help="Optional scene names to package. If omitted, package all scenes under results-root.",
	)
	parser.add_argument(
		"--camera-convention",
		type=str,
		choices=["direct", "inverse", "auto"],
		default="direct",
		help=(
			"Interpretation of camera.npy world_cam_R/world_cam_T. "
			"Use direct for c2w (recommended), inverse for w2c, auto for heuristic inference."
		),
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	saved, skipped = package_all_scenes(
		results_root=args.results_root,
		output_dir=args.output_dir,
		scene_names=args.scenes,
		camera_convention=args.camera_convention,
	)

	print(f"Saved {len(saved)} scene packages to {args.output_dir}")
	if saved:
		for path in saved:
			print(f"  [Saved] {path}")
	if skipped:
		print(f"Skipped {len(skipped)} scenes:")
		for scene_name, reason in skipped:
			print(f"  [Skipped] {scene_name}: {reason}")


if __name__ == "__main__":
	main()
