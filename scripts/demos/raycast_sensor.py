"""Raycast sensor demo.

Run with:
  uv run mjpython scripts/demos/raycast_sensor.py [--viewer native|viser]  # macOS
  uv run python scripts/demos/raycast_sensor.py [--viewer native|viser]    # Linux

Examples:
  # Grid pattern (default)
  uv run python scripts/demos/raycast_sensor.py --pattern grid

  # Cluttered scene with walls, columns, and boxes
  uv run python scripts/demos/raycast_sensor.py --scene-variant clutter

  # Closed house-like scene with ceiling and furniture
  uv run python scripts/demos/raycast_sensor.py --scene-variant house

  # Pinhole camera pattern
  uv run python scripts/demos/raycast_sensor.py --pattern pinhole

  # Livox scan-table pattern (Gazebo-style CSV)
  uv run python scripts/demos/raycast_sensor.py --pattern livox \
    --livox-csv-path /path/to/mid360.csv
  # Optional: tilt Livox pattern downward if you want terrain-focused scans
  uv run python scripts/demos/raycast_sensor.py --pattern livox \
    --livox-csv-path /path/to/mid360.csv \
    --livox-zenith-offset-deg 90

  # With yaw alignment (ignores pitch/roll)
  uv run python scripts/demos/raycast_sensor.py --alignment yaw

If using the native viewer, you can launch in interactive mode with:
  uv run mjpython scripts/demos/raycast_sensor.py --viewer native --interactive
"""

from __future__ import annotations

import os
import math
from typing import Literal

import mujoco
import torch
import tyro

import mjlab
import mjlab.terrains as terrain_gen
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  LivoxCsvPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def create_scanner_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "scanner"

  mat = spec.add_material()
  mat.name = "scanner_mat"
  mat.rgba[:] = (1.0, 0.5, 0.0, 0.9)

  scanner = spec.worldbody.add_body(mocap=True)
  scanner.name = "scanner"
  scanner.pos[:] = (0, 0, 2.0)

  geom = scanner.add_geom()
  geom.name = "scanner_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_BOX
  geom.size[:] = (0.15, 0.15, 0.05)
  geom.mass = 1.0
  geom.material = "scanner_mat"

  scanner.add_camera(name="scanner", fovy=58.0, resolution=(16, 12))

  record_cam = scanner.add_camera(name="record_cam")
  record_cam.pos[:] = (2, 0, 2)
  record_cam.fovy = 40.0
  record_cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  record_cam.targetbody = "scanner"

  return spec


def create_clutter_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "clutter"

  floor = spec.worldbody.add_geom()
  floor.name = "floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size[:] = (12.0, 12.0, 0.1)
  floor.rgba[:] = (0.18, 0.18, 0.18, 1.0)

  wall_rgba = (0.55, 0.55, 0.6, 1.0)
  box_rgba = (0.8, 0.45, 0.25, 1.0)
  pillar_rgba = (0.25, 0.65, 0.8, 1.0)
  beam_rgba = (0.55, 0.8, 0.35, 1.0)

  walls = [
    ("wall_north", (0.0, 5.0, 1.0), (5.5, 0.12, 1.0)),
    ("wall_south", (0.0, -5.0, 1.0), (5.5, 0.12, 1.0)),
    ("wall_east", (5.0, 0.0, 1.0), (0.12, 5.5, 1.0)),
    ("wall_west", (-5.0, 0.0, 1.0), (0.12, 5.5, 1.0)),
  ]
  for name, pos, size in walls:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = pos
    geom.size[:] = size
    geom.rgba[:] = wall_rgba

  boxes = [
    ("box_a", (1.6, 1.0, 0.45), (0.55, 0.4, 0.45)),
    ("box_b", (-1.8, 0.5, 0.7), (0.35, 0.7, 0.7)),
    ("box_c", (0.8, -1.9, 0.3), (0.9, 0.3, 0.3)),
    ("box_d", (-2.6, -2.2, 1.0), (0.45, 0.45, 1.0)),
    ("box_e", (2.5, -0.8, 0.6), (0.3, 1.1, 0.6)),
  ]
  for name, pos, size in boxes:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = pos
    geom.size[:] = size
    geom.rgba[:] = box_rgba

  pillars = [
    ("pillar_a", (-0.5, 2.2, 0.9), 0.28, 0.9),
    ("pillar_b", (2.1, 2.4, 1.2), 0.22, 1.2),
    ("pillar_c", (-2.0, -0.4, 0.8), 0.25, 0.8),
    ("pillar_d", (0.2, -3.0, 1.1), 0.18, 1.1),
  ]
  for name, pos, radius, half_height in pillars:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.pos[:] = pos
    geom.size[:] = (radius, half_height, 0.0)
    geom.rgba[:] = pillar_rgba

  beams = [
    ("beam_a", (0.0, 0.0, 2.2), (2.6, 0.12, 0.12)),
    ("beam_b", (-2.0, 2.0, 1.6), (0.12, 1.8, 0.12)),
  ]
  for name, pos, size in beams:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = pos
    geom.size[:] = size
    geom.rgba[:] = beam_rgba

  return spec


def create_house_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "house"

  floor = spec.worldbody.add_geom()
  floor.name = "floor"
  floor.type = mujoco.mjtGeom.mjGEOM_BOX
  floor.pos[:] = (0.0, 0.0, -0.05)
  floor.size[:] = (5.5, 5.5, 0.05)
  floor.rgba[:] = (0.35, 0.3, 0.28, 1.0)

  ceiling = spec.worldbody.add_geom()
  ceiling.name = "ceiling"
  ceiling.type = mujoco.mjtGeom.mjGEOM_BOX
  ceiling.pos[:] = (0.0, 0.0, 2.8)
  ceiling.size[:] = (5.5, 5.5, 0.05)
  ceiling.rgba[:] = (0.82, 0.82, 0.78, 1.0)

  wall_rgba = (0.78, 0.76, 0.72, 1.0)
  furniture_rgba = (0.5, 0.33, 0.2, 1.0)
  accent_rgba = (0.25, 0.55, 0.75, 1.0)

  walls = [
    ("wall_north", (0.0, 5.45, 1.35), (5.5, 0.05, 1.35)),
    ("wall_south_left", (-3.8, -5.45, 1.35), (1.7, 0.05, 1.35)),
    ("wall_south_right", (3.8, -5.45, 1.35), (1.7, 0.05, 1.35)),
    ("wall_west", (-5.45, 0.0, 1.35), (0.05, 5.5, 1.35)),
    ("wall_east", (5.45, 0.0, 1.35), (0.05, 5.5, 1.35)),
    ("inner_wall_a", (-0.8, 1.2, 1.35), (0.05, 4.0, 1.35)),
    ("inner_wall_b", (1.8, -0.8, 1.35), (2.7, 0.05, 1.35)),
    ("inner_wall_c", (2.9, 2.1, 1.35), (0.05, 2.3, 1.35)),
  ]
  for name, pos, size in walls:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = pos
    geom.size[:] = size
    geom.rgba[:] = wall_rgba

  door_headers = [
    ("door_header_main", (0.0, -5.45, 2.3), (2.1, 0.05, 0.45)),
    ("door_header_room", (-0.8, -1.7, 2.3), (0.05, 0.9, 0.45)),
    ("door_header_hall", (1.1, -0.8, 2.3), (0.7, 0.05, 0.45)),
  ]
  for name, pos, size in door_headers:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = pos
    geom.size[:] = size
    geom.rgba[:] = wall_rgba

  furniture = [
    ("table", (2.4, -2.4, 0.42), (0.9, 0.7, 0.42), furniture_rgba),
    ("sofa", (-3.0, 2.7, 0.45), (1.2, 0.45, 0.45), furniture_rgba),
    ("cabinet", (4.4, 3.2, 0.9), (0.35, 1.0, 0.9), furniture_rgba),
    ("kitchen_island", (-3.2, -2.6, 0.5), (0.8, 1.4, 0.5), furniture_rgba),
    ("bed", (3.5, 3.2, 0.32), (1.2, 0.9, 0.32), accent_rgba),
    ("desk", (0.2, 3.8, 0.38), (0.9, 0.45, 0.38), furniture_rgba),
    ("shelf", (-1.8, 4.5, 1.0), (0.25, 0.6, 1.0), furniture_rgba),
  ]
  for name, pos, size, rgba in furniture:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = pos
    geom.size[:] = size
    geom.rgba[:] = rgba

  columns = [
    ("column_a", (-0.8, -4.0, 1.35), 0.12, 1.35),
    ("column_b", (1.8, 1.5, 1.35), 0.12, 1.35),
  ]
  for name, pos, radius, half_height in columns:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.pos[:] = pos
    geom.size[:] = (radius, half_height, 0.0)
    geom.rgba[:] = wall_rgba

  return spec


def create_env_cfg(
  scene_variant: Literal["terrain", "clutter", "house"],
  pattern: Literal["grid", "pinhole", "livox"],
  alignment: Literal["base", "yaw", "world"],
  max_distance: float,
  livox_csv_path: str | None,
  livox_downsample: int,
  livox_num_rays: int | None,
  livox_azimuth_offset_deg: float,
  livox_zenith_offset_deg: float,
  livox_publish_hz: float,
  livox_points_per_cloud: int,
  debug_vis: bool,
  accumulate_hits: bool,
  accumulated_max_points: int,
  accumulated_render_points: int,
) -> ManagerBasedRlEnvCfg:
  terrain_cfg = None
  clutter_entity_cfg = None
  house_entity_cfg = None
  if scene_variant == "terrain":
    custom_terrain_cfg = TerrainGeneratorCfg(
      size=(4.0, 4.0),
      border_width=0.5,
      num_rows=1,
      num_cols=4,
      curriculum=True,
      sub_terrains={
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
          proportion=0.25,
          step_height_range=(0.1, 0.25),
          step_width=0.3,
          platform_width=1.5,
          border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
          proportion=0.25,
          slope_range=(0.6, 1.5),
          platform_width=1.5,
          border_width=0.25,
          inverted=True,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
          proportion=0.25,
          noise_range=(0.05, 0.15),
          noise_step=0.02,
          border_width=0.25,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
          proportion=0.25,
          amplitude_range=(0.15, 0.25),
          num_waves=3,
          border_width=0.25,
        ),
      },
      add_lights=True,
    )

    terrain_cfg = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=custom_terrain_cfg,
      num_envs=1,
    )
  elif scene_variant == "clutter":
    clutter_entity_cfg = EntityCfg(spec_fn=create_clutter_spec)
  else:
    house_entity_cfg = EntityCfg(spec_fn=create_house_spec)

  scanner_entity_cfg = EntityCfg(
    spec_fn=create_scanner_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.65, -0.4, 0.5)),
  )

  sense_dt = 10 * 0.002
  sense_hz = 1.0 / sense_dt
  publish_interval = 1.0 / livox_publish_hz
  auto_livox_num_rays = max(1, math.ceil(livox_points_per_cloud * livox_publish_hz / sense_hz))
  livox_rays_per_sense = livox_num_rays or auto_livox_num_rays

  if pattern == "grid":
    pattern_cfg = GridPatternCfg(
      size=(0.6, 0.6),
      resolution=0.1,
      direction=(0.0, 0.0, -1.0),
    )
  elif pattern == "pinhole":
    pattern_cfg = PinholeCameraPatternCfg.from_mujoco_camera("scanner/scanner")
  else:
    assert pattern == "livox"
    if livox_csv_path is None:
      raise ValueError(
        "--livox-csv-path is required when --pattern livox is selected."
      )
    pattern_cfg = LivoxCsvPatternCfg(
      csv_path=livox_csv_path,
      downsample=livox_downsample,
      num_rays=livox_rays_per_sense,
      azimuth_offset_deg=livox_azimuth_offset_deg,
      zenith_offset_deg=livox_zenith_offset_deg,
    )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="scanner", entity="scanner"),
    pattern=pattern_cfg,
    ray_alignment=alignment,
    max_distance=max_distance,
    exclude_parent_body=True,
    debug_vis=debug_vis,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.0, 1.0, 0.0, 0.9),
      miss_color=(1.0, 0.0, 0.0, 0.5),
      show_rays=(debug_vis and pattern != "livox"),
      show_normals=debug_vis,
      accumulate_hits=accumulate_hits,
      accumulated_max_points=(
        livox_points_per_cloud if pattern == "livox" and accumulate_hits else accumulated_max_points
      ),
      accumulated_render_points=accumulated_render_points,
      publish_interval=(publish_interval if pattern == "livox" and accumulate_hits else None),
      render_published_hits=(pattern == "livox" and accumulate_hits),
    ),
  )

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=0.0,
      extent=6.0,
      terrain=terrain_cfg,
      entities={
        "scanner": scanner_entity_cfg,
        **({"clutter": clutter_entity_cfg} if clutter_entity_cfg is not None else {}),
        **({"house": house_entity_cfg} if house_entity_cfg is not None else {}),
      },
      sensors=(raycast_cfg,),
    ),
  )

  cfg.viewer.body_name = "scanner"
  if scene_variant == "clutter":
    cfg.viewer.distance = 10.0
    cfg.viewer.elevation = -18.0
    cfg.viewer.azimuth = 120.0
  elif scene_variant == "house":
    cfg.viewer.distance = 9.0
    cfg.viewer.elevation = -12.0
    cfg.viewer.azimuth = 145.0
  else:
    cfg.viewer.distance = 12.0
    cfg.viewer.elevation = -25.0
    cfg.viewer.azimuth = 135.0

  return cfg


def main(
  viewer: str = "auto",
  interactive: bool = False,
  scene_variant: Literal["terrain", "clutter", "house"] = "terrain",
  pattern: Literal["grid", "pinhole", "livox"] = "grid",
  alignment: Literal["base", "yaw", "world"] = "base",
  max_distance: float = 30.0,
  livox_csv_path: str | None = None,
  livox_downsample: int = 8,
  livox_num_rays: int | None = None,
  livox_azimuth_offset_deg: float = 0.0,
  livox_zenith_offset_deg: float = 0.0,
  livox_publish_hz: float = 10.0,
  livox_points_per_cloud: int = 12000,
  debug_vis: bool = False,
  accumulate_hits: bool = True,
  accumulated_max_points: int = 12000,
  accumulated_render_points: int = 4000,
) -> None:
  configure_torch_backends()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  sense_hz = 1.0 / (10 * 0.002)
  derived_livox_num_rays = (
    max(1, math.ceil(livox_points_per_cloud * livox_publish_hz / sense_hz))
    if pattern == "livox"
    else None
  )

  print("=" * 60)
  print("Raycast Sensor Demo - 4 Terrain Types")
  print(f"  Scene: {scene_variant}")
  print(f"  Pattern: {pattern}")
  print(f"  Alignment: {alignment}")
  if pattern == "livox":
    print(f"  Livox publish_hz: {livox_publish_hz}")
    print(f"  Livox points_per_cloud: {livox_points_per_cloud}")
    print(f"  Livox rays_per_sense: {livox_num_rays or derived_livox_num_rays}")
  print("=" * 60)
  print()

  effective_debug_vis = debug_vis or pattern == "livox"
  env_cfg = create_env_cfg(
    scene_variant,
    pattern,
    alignment,
    max_distance,
    livox_csv_path,
    livox_downsample,
    livox_num_rays,
    livox_azimuth_offset_deg,
    livox_zenith_offset_deg,
    livox_publish_hz,
    livox_points_per_cloud,
    effective_debug_vis,
    accumulate_hits and pattern == "livox",
    accumulated_max_points,
    accumulated_render_points,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer

  use_auto_scan = (resolved_viewer == "viser") or (not interactive)

  if use_auto_scan:

    class AutoScanPolicy:
      def __init__(self):
        self.step_count = 0

      def __call__(self, obs) -> torch.Tensor:
        del obs
        t = self.step_count * 0.005
        y_period = 1000
        y_normalized = (self.step_count % y_period) / y_period
        y = -8.0 + 16.0 * y_normalized
        x = 1.5 * math.sin(2 * math.pi * t * 0.3)
        z = 1.0
        env.unwrapped.sim.data.mocap_pos[0, 0, :] = torch.tensor(
          [x, y, z], device=device, dtype=torch.float32
        )
        env.unwrapped.sim.data.mocap_quat[0, 0, :] = torch.tensor(
          [1, 0, 0, 0], device=device, dtype=torch.float32
        )
        self.step_count += 1
        return torch.zeros(env.unwrapped.action_space.shape, device=device)

    policy = AutoScanPolicy()
  else:

    class PolicyZero:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(env.unwrapped.action_space.shape, device=device)

    policy = PolicyZero()

  if resolved_viewer == "native":
    print("Launching native viewer...")
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    print("Launching viser viewer...")
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
