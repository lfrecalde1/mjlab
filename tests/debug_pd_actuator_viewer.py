"""Interactive angular-dynamics PD demo with mouse disturbances.

This script demonstrates IdealPdActuator/BuiltinPositionActuator on a 3-DoF
rotational model (roll, pitch, yaw hinge chain). You can apply disturbances
with the MuJoCo mouse perturbation tool in the native viewer window.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.01" integrator="RK4"/>
  <worldbody>
    <light name="sun" pos="0 0 3.0" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>

    <body name="yaw_frame" pos="0 0 0.4">
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.05"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
      <body name="pitch_frame">
        <joint name="pitch" type="hinge" axis="0 1 0" damping="0.05"/>
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <body name="roll_frame">
          <joint name="roll" type="hinge" axis="1 0 0" damping="0.05"/>

          <geom name="core" type="box" size="0.06 0.06 0.02" mass="1.0" rgba="1 1 0 1"/>
          <geom name="arm_x" type="box" pos="0.11 0 0" size="0.06 0.006 0.004" rgba="1 0.5 0 1" mass="0.05"/>
          <geom name="arm_y" type="box" pos="0 0.11 0" size="0.006 0.06 0.004" rgba="1 0.5 0 1" mass="0.05"/>
          <site name="imu" pos="0 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _sync_env0_to_mjdata(sim) -> None:
  """Copy env-0 state from mjwarp tensors into CPU MjData for rendering."""
  sim.mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
  sim.mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
  sim.mj_data.ctrl[:] = sim.data.ctrl[0].cpu().numpy()
  sim.mj_data.qfrc_applied[:] = sim.data.qfrc_applied[0].cpu().numpy()

  nmocap = sim.mj_model.nmocap
  if nmocap > 0:
    sim.mj_data.mocap_pos[:nmocap] = sim.data.mocap_pos[0, :nmocap].cpu().numpy()
    sim.mj_data.mocap_quat[:nmocap] = sim.data.mocap_quat[0, :nmocap].cpu().numpy()

  nbody = sim.mj_model.nbody
  sim.mj_data.xfrc_applied[:nbody] = sim.data.xfrc_applied[0, :nbody].cpu().numpy()


def _sync_mouse_perturbation_to_sim(sim, viewer) -> None:
  """Convert native viewer mouse perturbation into qfrc_applied for env 0."""
  pert = viewer.perturb
  if pert.active != 0 and pert.select > 0:
    mujoco.mjv_applyPerturbForce(sim.mj_model, sim.mj_data, pert)

    body_id = pert.select
    force = sim.mj_data.xfrc_applied[body_id, :3].copy()
    torque = sim.mj_data.xfrc_applied[body_id, 3:].copy()
    point = sim.mj_data.xipos[body_id].copy()

    qfrc = np.zeros(sim.mj_model.nv)
    mujoco.mj_applyFT(sim.mj_model, sim.mj_data, force, torque, point, body_id, qfrc)
    sim.data.qfrc_applied[0] = torch.from_numpy(qfrc).to(
      device=sim.data.qfrc_applied.device
    )

    sim.mj_data.xfrc_applied[body_id] = 0.0
  else:
    sim.data.qfrc_applied[0] = 0.0


def _make_actuator_cfg(kind: str, kp: float, kd: float, effort_limit: float):
  targets = ("roll", "pitch", "yaw")
  if kind == "ideal":
    return IdealPdActuatorCfg(
      target_names_expr=targets,
      effort_limit=effort_limit,
      stiffness=kp,
      damping=kd,
    )
  return BuiltinPositionActuatorCfg(
    target_names_expr=targets,
    effort_limit=effort_limit,
    stiffness=kp,
    damping=kd,
  )


def run(args: argparse.Namespace) -> None:
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  actuator_cfg = _make_actuator_cfg(args.actuator, args.kp, args.kd, args.effort_limit)

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(XML),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  # Joint order follows target_names_expr: (roll, pitch, yaw).
  joint_pos = torch.tensor([[args.iroll, args.ipitch, args.iyaw]], device=device)
  joint_vel = torch.tensor([[0.0, 0.0, 0.0]], device=device)
  fixed_pos_target = torch.tensor([[args.roll, args.pitch, args.yaw]], device=device)
  vel_target = torch.tensor([[args.wx, args.wy, args.wz]], device=device)
  eff_target = torch.zeros(1, 3, device=device)

  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  dt = float(sim.mj_model.opt.timestep)
  wall_dt = dt / max(args.realtime_factor, 1e-6)

  with mujoco.viewer.launch_passive(
    sim.mj_model,
    sim.mj_data,
    show_left_ui=False,
    show_right_ui=False,
  ) as viewer:
    for _ in range(args.steps):
      start = time.perf_counter()

      _sync_mouse_perturbation_to_sim(sim, viewer)

      # Rate-tracking mode: no orientation objective (Kp term suppressed).
      if args.rate_only:
        entity.set_joint_position_target(entity.data.joint_pos.clone())
      else:
        entity.set_joint_position_target(fixed_pos_target)
      if args.actuator == "ideal":
        entity.set_joint_velocity_target(vel_target)
        entity.set_joint_effort_target(eff_target)

      entity.write_data_to_sim()
      sim.step()
      entity.update(dt=dt)

      _sync_env0_to_mjdata(sim)
      mujoco.mj_forward(sim.mj_model, sim.mj_data)
      viewer.sync()

      elapsed = time.perf_counter() - start
      sleep_s = wall_dt - elapsed
      if sleep_s > 0:
        time.sleep(sleep_s)

      if not viewer.is_running():
        break

  final_qpos = entity.data.joint_pos[0].detach().cpu().tolist()
  final_qvel = entity.data.joint_vel[0].detach().cpu().tolist()
  final_ctrl = sim.data.ctrl[0].detach().cpu().tolist()
  print("Final joint_pos [roll,pitch,yaw]:", [round(v, 5) for v in final_qpos])
  print("Final joint_vel [wx,wy,wz-like]:", [round(v, 5) for v in final_qvel])
  print("Final ctrl:", [round(v, 5) for v in final_ctrl])


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Interactive Ideal PD demo for angular dynamics (roll, pitch, yaw)."
  )
  parser.add_argument(
    "--actuator",
    choices=("ideal", "builtin"),
    default="ideal",
    help="Actuator implementation.",
  )
  parser.add_argument("--kp", type=float, default=0.0, help="PD stiffness")
  parser.add_argument("--kd", type=float, default=20.5, help="PD damping")
  parser.add_argument(
    "--effort-limit", type=float, default=2.0, help="Torque limit per axis"
  )
  parser.add_argument("--steps", type=int, default=6000, help="Simulation steps")
  parser.add_argument(
    "--realtime-factor",
    type=float,
    default=1.0,
    help="Playback speed. 1.0 = real-time, 2.0 = 2x faster.",
  )

  # Angle references [rad].
  parser.add_argument("--roll", type=float, default=0.0, help="Roll target [rad]")
  parser.add_argument("--pitch", type=float, default=0.0, help="Pitch target [rad]")
  parser.add_argument("--yaw", type=float, default=0.0, help="Yaw target [rad]")

  # Angular-rate references [rad/s].
  parser.add_argument("--wx", type=float, default=0.0, help="Roll rate ref [rad/s]")
  parser.add_argument("--wy", type=float, default=0.0, help="Pitch rate ref [rad/s]")
  parser.add_argument("--wz", type=float, default=10.0, help="Yaw rate ref [rad/s]")
  parser.add_argument(
    "--rate-only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Track only angular velocity (ignore orientation objective).",
  )

  # Initial attitude [rad].
  parser.add_argument("--iroll", type=float, default=0.0, help="Initial roll [rad]")
  parser.add_argument("--ipitch", type=float, default=-0.0, help="Initial pitch [rad]")
  parser.add_argument("--iyaw", type=float, default=0.0, help="Initial yaw [rad]")

  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device override (cpu or cuda). Defaults to auto selection.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  run(_parse_args())
