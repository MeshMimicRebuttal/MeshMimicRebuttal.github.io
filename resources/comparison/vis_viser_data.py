import argparse
import os
import time
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import trimesh


SCENE_FACE_COLOR = np.array([230, 230, 210, 255], dtype=np.uint8)
HUMAN_FACE_COLOR = np.array([0, 191, 255, 255], dtype=np.uint8)


def _discover_packages(data_dir: str) -> List[str]:
	return sorted(glob(os.path.join(data_dir, "*.npz")))


def _scene_name_from_path(path: str) -> str:
	return os.path.splitext(os.path.basename(path))[0]


def _resolve_package_path(data_dir: str, scene_name: str = "") -> str:
	package_paths = _discover_packages(data_dir)
	if not package_paths:
		raise FileNotFoundError(f"No packaged scenes found in {data_dir}")

	package_map = {_scene_name_from_path(path): path for path in package_paths}
	if scene_name:
		if scene_name not in package_map:
			available = ", ".join(sorted(package_map.keys())[:20])
			raise FileNotFoundError(f"Scene '{scene_name}' not found in {data_dir}. Sample scenes: {available}")
		return package_map[scene_name]

	if "walk1" in package_map:
		return package_map["walk1"]
	return package_paths[0]


def _colorize_mesh(vertices: np.ndarray, faces: np.ndarray, face_color: np.ndarray) -> trimesh.Trimesh:
	mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
	mesh.visual.face_colors = np.tile(face_color[None, :], (int(faces.shape[0]), 1))
	return mesh


def load_packaged_scene(package_path: str) -> Dict[str, np.ndarray]:
	with np.load(package_path) as data:
		scene_name = data["scene_name"]
		if isinstance(scene_name, np.ndarray):
			scene_name = scene_name.item()
		camera_convention = data["camera_convention"].item() if "camera_convention" in data else "unknown"
		return {
			"scene_name": str(scene_name),
			"scene_vertices": np.asarray(data["scene_vertices"], dtype=np.float32),
			"scene_faces": np.asarray(data["scene_faces"], dtype=np.int32),
			"human_vertices": np.asarray(data["human_vertices"], dtype=np.float32),
			"human_faces": np.asarray(data["human_faces"], dtype=np.int32),
			"frame_ids": np.asarray(data["frame_ids"], dtype=np.int32),
			"camera_convention": str(camera_convention),
		}


def _scene_center(vertices: np.ndarray) -> np.ndarray:
	return (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize packaged scene and human meshes with Viser.")
	parser.add_argument(
		"--data-dir",
		type=str,
		default=os.path.join("paper", "demo", "maskmimic"),
		help="Directory that contains packaged *.npz scene files.",
	)
	parser.add_argument(
		"--scene",
		type=str,
		default="",
		help="Scene name to visualize. Defaults to walk1 if present, otherwise the first packaged scene.",
	)
	parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
	parser.add_argument("--fps", type=float, default=20.0, help="Default playback speed.")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	try:
		import viser  # type: ignore
	except Exception as exc:
		raise ModuleNotFoundError("Viewer dependency missing. Install `viser` to run this script.") from exc

	package_path = _resolve_package_path(args.data_dir, args.scene)
	scene_data = load_packaged_scene(package_path)
	scene_name = str(scene_data["scene_name"])
	scene_vertices = scene_data["scene_vertices"]
	scene_faces = scene_data["scene_faces"]
	human_vertices = scene_data["human_vertices"]
	human_faces = scene_data["human_faces"]
	frame_ids = scene_data["frame_ids"]
	camera_convention = str(scene_data.get("camera_convention", "unknown"))
	num_frames = int(human_vertices.shape[0])

	scene_mesh = _colorize_mesh(scene_vertices, scene_faces, SCENE_FACE_COLOR)
	center = _scene_center(scene_vertices)

	server = viser.ViserServer(port=int(args.port))
	print(f"Viser server started: http://localhost:{int(args.port)}")
	print(f"Loaded scene: {scene_name}")
	print(f"Package: {package_path}")
	print(f"Frames: {num_frames}")
	print(f"Camera convention: {camera_convention}")

	play_button = server.gui.add_button("Play")
	pause_button = server.gui.add_button("Pause")
	prev_button = server.gui.add_button("Prev")
	next_button = server.gui.add_button("Next")
	speed_slider = server.gui.add_slider(
		"Speed",
		min=1,
		max=60,
		step=1,
		initial_value=int(max(1, min(60, round(float(args.fps))))),
	)
	timeline = server.gui.add_slider("t", min=0, max=max(0, num_frames - 1), step=1, initial_value=0)

	is_playing = False
	frame_idx = 0
	last_update = time.time()

	server.scene.add_mesh_trimesh(
		name="/scene_mesh",
		mesh=scene_mesh,
		position=(0.0, 0.0, 0.0),
	)
	server.scene.add_frame(
		"/scene_center",
		position=tuple(center.tolist()),
		show_axes=False,
	)

	def update_scene(local_frame_idx: int) -> None:
		human_mesh = _colorize_mesh(human_vertices[local_frame_idx], human_faces, HUMAN_FACE_COLOR)
		server.scene.add_mesh_trimesh(
			name="/human_mesh",
			mesh=human_mesh,
			position=(0.0, 0.0, 0.0),
		)

	@play_button.on_click
	def _(_evt) -> None:
		nonlocal is_playing
		is_playing = True

	@pause_button.on_click
	def _(_evt) -> None:
		nonlocal is_playing
		is_playing = False

	@prev_button.on_click
	def _(_evt) -> None:
		nonlocal frame_idx
		frame_idx = (frame_idx - 1) % num_frames
		timeline.value = frame_idx
		update_scene(frame_idx)

	@next_button.on_click
	def _(_evt) -> None:
		nonlocal frame_idx
		frame_idx = (frame_idx + 1) % num_frames
		timeline.value = frame_idx
		update_scene(frame_idx)

	update_scene(0)

	try:
		while True:
			time.sleep(0.01)

			if is_playing:
				now = time.time()
				if (now - last_update) > (1.0 / float(speed_slider.value)):
					frame_idx = (frame_idx + 1) % num_frames
					timeline.value = frame_idx
					update_scene(frame_idx)
					last_update = now

			if int(timeline.value) != frame_idx:
				frame_idx = int(timeline.value)
				update_scene(frame_idx)
				frame_label = int(frame_ids[frame_idx]) if frame_idx < len(frame_ids) else frame_idx
				print(f"Frame {frame_idx}/{num_frames - 1} (source frame {frame_label})")

	except KeyboardInterrupt:
		print("Stopping Viser server...")
		server.stop()


if __name__ == "__main__":
	main()
