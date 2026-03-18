"""Camera sensor demo.

Run with:
  uv run mjpython scripts/demos/camera_sensor.py [--viewer native|viser]  # macOS
  uv run python scripts/demos/camera_sensor.py [--viewer native|viser]    # Linux

Examples:
  # Show both wrapped and created cameras in the viser sidebar
  uv run python scripts/demos/camera_sensor.py --viewer viser

  # Wrap only the existing MuJoCo camera
  uv run python scripts/demos/camera_sensor.py --mode wrap --viewer viser

  # Create only a new world camera sensor
  uv run python scripts/demos/camera_sensor.py --mode create --viewer viser
"""

from __future__ import annotations

import math
import os
from typing import Literal

import mujoco
import torch
import tyro

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def create_demo_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "camera_demo"

  floor_mat = spec.add_material()
  floor_mat.name = "floor_mat"
  floor_mat.rgba[:] = (0.2, 0.22, 0.24, 1.0)

  obj_mat = spec.add_material()
  obj_mat.name = "obj_mat"
  obj_mat.rgba[:] = (0.85, 0.45, 0.2, 1.0)

  floor = spec.worldbody.add_geom()
  floor.name = "floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size[:] = (8.0, 8.0, 0.1)
  floor.material = "floor_mat"

  light = spec.worldbody.add_light()
  light.name = "sun"
  light.pos[:] = (0.0, 0.0, 4.0)
  light.dir[:] = (0.0, 0.0, -1.0)
  light.diffuse[:] = (0.9, 0.9, 0.9)

  box = spec.worldbody.add_body(mocap=True)
  box.name = "box"
  box.pos[:] = (0.0, 0.0, 0.4)
  geom = box.add_geom()
  geom.name = "box_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_BOX
  geom.size[:] = (0.35, 0.25, 0.25)
  geom.material = "obj_mat"

  pillar = spec.worldbody.add_body()
  pillar.name = "pillar"
  pillar.pos[:] = (1.4, -0.6, 0.6)
  pillar_geom = pillar.add_geom()
  pillar_geom.name = "pillar_geom"
  pillar_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
  pillar_geom.size[:] = (0.18, 0.6, 0.0)
  pillar_geom.rgba[:] = (0.25, 0.6, 0.8, 1.0)

  box_static_a = spec.worldbody.add_body()
  box_static_a.name = "box_static_a"
  box_static_a.pos[:] = (-0.9, 0.8, 0.28)
  box_static_a_geom = box_static_a.add_geom()
  box_static_a_geom.name = "box_static_a_geom"
  box_static_a_geom.type = mujoco.mjtGeom.mjGEOM_BOX
  box_static_a_geom.size[:] = (0.22, 0.22, 0.22)
  box_static_a_geom.rgba[:] = (0.25, 0.8, 0.4, 1.0)

  box_static_b = spec.worldbody.add_body()
  box_static_b.name = "box_static_b"
  box_static_b.pos[:] = (0.9, 0.9, 0.2)
  box_static_b_geom = box_static_b.add_geom()
  box_static_b_geom.name = "box_static_b_geom"
  box_static_b_geom.type = mujoco.mjtGeom.mjGEOM_BOX
  box_static_b_geom.size[:] = (0.18, 0.35, 0.18)
  box_static_b_geom.rgba[:] = (0.8, 0.3, 0.7, 1.0)

  sphere = spec.worldbody.add_body()
  sphere.name = "sphere"
  sphere.pos[:] = (-0.2, -0.9, 0.3)
  sphere_geom = sphere.add_geom()
  sphere_geom.name = "sphere_geom"
  sphere_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
  sphere_geom.size[:] = (0.28, 0.0, 0.0)
  sphere_geom.rgba[:] = (0.95, 0.8, 0.2, 1.0)

  cylinder = spec.worldbody.add_body()
  cylinder.name = "cylinder"
  cylinder.pos[:] = (0.6, -1.2, 0.35)
  cylinder_geom = cylinder.add_geom()
  cylinder_geom.name = "cylinder_geom"
  cylinder_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
  cylinder_geom.size[:] = (0.14, 0.35, 0.0)
  cylinder_geom.rgba[:] = (0.2, 0.85, 0.85, 1.0)

  cam_body = spec.worldbody.add_body()
  cam_body.name = "camera_rig"
  cam_body.pos[:] = (1.4, 0.0, 0.7)

  rig_geom = cam_body.add_geom()
  rig_geom.name = "rig_geom"
  rig_geom.type = mujoco.mjtGeom.mjGEOM_BOX
  rig_geom.size[:] = (0.12, 0.12, 0.08)
  rig_geom.rgba[:] = (0.95, 0.9, 0.25, 0.9)

  cam_body.add_camera(
    name="mounted_cam",
    pos=(0.0, 0.0, 0.0),
    quat=(0.5, -0.5, -0.5, 0.5),
    fovy=72.0,
    resolution=(160, 120),
  )

  return spec


def create_env_cfg(mode: Literal["wrap", "create", "both"]) -> ManagerBasedRlEnvCfg:
  scene_entity_cfg = EntityCfg(spec_fn=create_demo_spec)

  sensors: list[CameraSensorCfg] = []
  if mode in ("wrap", "both"):
    sensors.append(
      CameraSensorCfg(
        name="mounted_rgbd",
        camera_name="scene/mounted_cam",
        width=160,
        height=120,
        data_types=("rgb", "depth"),
      )
    )
  if mode in ("create", "both"):
    sensors.append(
      CameraSensorCfg(
        name="overview_cam",
        pos=(0.0, -1, 1.3),
        quat=(0.70710678, 0.70710678, 0.0, 0.0),
        fovy=120.0,
        width=640,
        height=460,
        data_types=("rgb","depth"),
      )
    )

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=0.0,
      extent=4.0,
      entities={"scene": scene_entity_cfg},
      sensors=tuple(sensors),
    ),
  )

  cfg.viewer.body_name = "camera_rig"
  cfg.viewer.distance = 4.5
  cfg.viewer.elevation = -18.0
  cfg.viewer.azimuth = 130.0
  return cfg


def main(
  viewer: Literal["auto", "native", "viser"] = "auto",
  mode: Literal["wrap", "create", "both"] = "both",
) -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = create_env_cfg(mode)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  print("=" * 60)
  print("Camera Sensor Demo")
  print(f"  Mode: {mode}")
  print(f"  Device: {device}")
  print("  Sensors:")
  for sensor in env.unwrapped.scene.sensors.values():
    if isinstance(sensor.cfg, CameraSensorCfg):
      print(f"    - {sensor.cfg.name}: {sensor.cfg.data_types}")
  print("=" * 60)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer

  class PolicyAnimate:
    def __init__(self) -> None:
      self.step_count = 0

    def __call__(self, obs) -> torch.Tensor:
      del obs
      t = self.step_count * 0.02
      x = 0.8 * math.cos(t)
      y = 0.6 * math.sin(0.7 * t)
      z = 0.4 + 0.15 * math.sin(1.3 * t)
      yaw = 0.6 * math.sin(0.5 * t)
      quat = torch.tensor(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        device=device,
        dtype=torch.float32,
      )
      env.unwrapped.sim.data.mocap_pos[0, 0, :] = torch.tensor(
        [x, y, z], device=device, dtype=torch.float32
      )
      env.unwrapped.sim.data.mocap_quat[0, 0, :] = quat

      if self.step_count % 40 == 0:
        for name, sensor in env.unwrapped.scene.sensors.items():
          data = sensor.data
          if getattr(data, "rgb", None) is not None:
            print(f"{name}.rgb shape = {tuple(data.rgb.shape)} dtype = {data.rgb.dtype}")
          if getattr(data, "depth", None) is not None:
            print(
              f"{name}.depth shape = {tuple(data.depth.shape)} dtype = {data.depth.dtype}"
            )

      self.step_count += 1
      return torch.zeros(env.unwrapped.action_space.shape, device=device)

  policy = PolicyAnimate()

  if resolved_viewer == "native":
    print("Launching native viewer...")
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    print("Launching viser viewer...")
    print("Open the camera panels in the sidebar to inspect RGB/depth outputs.")
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
