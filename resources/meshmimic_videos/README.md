## Supplementary Materials Overview

This folder contains the video assets referenced in the paper. The content is organized into two main groups:

- **real2sim2real_videos**: Real-to-sim-to-real results (includes real robot deployment videos).
- **real2sim_videos**: Real-to-sim results (mainly network-data demonstrations).

The **long_parkour*** series additionally provides reconstructed terrain, to make it easier to review the environment context.

---

## Folder Guide

### real2sim2real_videos
Real-to-sim-to-real demonstrations (paired with real deployment clips).

Files:
- climb_box3.mp4
- climb_box3_real2sim.mp4
- climb_box7.mp4
- climb_box7_real2sim.mp4
- climb_box8.mp4
- climb_box8_real2sim.mp4
- jump_box6.mp4
- jump_box6_real2sim.mp4
- jump_box9.mp4
- jump_box9_real2sim.mp4
- odin_jcv3.mp4
- odin_jcv3_real2sim.mp4
- safety_vault10.mp4
- safety_vault10_real2sim.mp4

**Naming convention:**
- `*_real2sim.mp4` indicates the sim counterpart for the same task.
- The paired file without `_real2sim` is the real deployment video.

---

### real2sim_videos
Real-to-sim demonstrations (network-data focused).

Top-level files:
- parkour1.mp4
- parkour2.mp4
- parkour4.mp4
- parkour5.mp4
- tic_tac1.mp4

Subfolders:
- long_parkour1_with_terrain/
- long_parkour2_with_terrain/
- long_parkour3_with_terrain/
- long_parkour4/

**Note:** The `long_parkour*` series contains reconstructed terrain alongside the videos for easier inspection.
