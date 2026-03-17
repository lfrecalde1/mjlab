"""Interactive quadrotor open-loop actuator demo with mouse disturbances."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

XML = """
<mujoco model="quad_rate_control">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0" integrator="RK4" jacobian="sparse"/>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.9 0.9 0.9 1"/>
    <site name="world_origin" type="sphere" pos="0 0 0.001" size="0.01" rgba="1 1 1 1"/>
    <geom name="world_x_axis" type="capsule" fromto="0 0 0.002 0.4 0 0.002" size="0.006" rgba="1 0 0 1"/>
    <geom name="world_y_axis" type="capsule" fromto="0 0 0.002 0 0.4 0.002" size="0.006" rgba="0 1 0 1"/>
    <geom name="world_z_axis" type="capsule" fromto="0 0 0.002 0 0 0.4" size="0.006" rgba="0 0 1 1"/>

    <body name="drone_0" pos="0 0 0.8" euler="0.0 0.0 0.0">
      <joint name="drone_0" type="free" damping="0.001"/>

      <geom name="core" type="box" size="0.035 0.035 0.015" rgba="1 1 0 1" mass="0.84"/>
      <geom name="arm_x" type="box" pos="0.071 0.071 0.0" size="0.05 0.01 0.0025" quat=".924 0 0 .383" rgba="1 .5 0 1" mass="0.025"/>
      <geom name="arm_y" type="box" pos="0.071 -0.071 0.0" size="0.05 0.01 0.0025" quat=".383 0 0 .924" rgba="1 .5 0 1" mass="0.025"/>
      <geom name="arm_z" type="box" pos="-0.071 -0.071 0.0" size="0.05 0.01 0.0025" quat="-.383 0 0 .924" rgba="1 .5 0 1" mass="0.025"/>
      <geom name="arm_w" type="box" pos="-0.071 0.071 0.0" size="0.05 0.01 0.0025" quat=".924 0 0 -.383" rgba="1 .5 0 1" mass="0.025"/>
      <site name="body_origin" type="sphere" pos="0 0 0" size="0.008" rgba="1 1 1 1"/>
      <geom name="body_x_axis" type="capsule" fromto="0 0 0 0.12 0 0" size="0.004" rgba="1 0 0 1"/>
      <geom name="body_y_axis" type="capsule" fromto="0 0 0 0 0.12 0" size="0.004" rgba="0 1 0 1"/>
      <geom name="body_z_axis" type="capsule" fromto="0 0 0 0 0 0.12" size="0.004" rgba="0 0 1 1"/>

      <site name="thrust" type="sphere" pos="0 0 0" size="0.01" rgba="0 1 1 1"/>
      <site name="rateX" type="sphere" pos="0 0 0" size="0.01" rgba="0 1 1 1"/>
      <site name="rateY" type="sphere" pos="0 0 0" size="0.01" rgba="0 1 1 1"/>
      <site name="rateZ" type="sphere" pos="0 0 0" size="0.01" rgba="0 1 1 1"/>
      <site name="imu" pos="0 0 0"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="body_thrust" site="thrust" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="0 60"/>
    <motor name="x_moment" site="rateX" gear="0 0 0 1 0 0" ctrllimited="true" ctrlrange="-3 3"/>
    <motor name="y_moment" site="rateY" gear="0 0 0 0 1 0" ctrllimited="true" ctrlrange="-3 3"/>
    <motor name="z_moment" site="rateZ" gear="0 0 0 0 0 1" ctrllimited="true" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


def _sync_env0_to_mjdata(sim) -> None:
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


def run(args: argparse.Namespace) -> None:
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  print(device)

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlMotorActuatorCfg(
          target_names_expr=("thrust",), transmission_type=TransmissionType.SITE
        ),
        XmlMotorActuatorCfg(
          target_names_expr=("rateX",), transmission_type=TransmissionType.SITE
        ),
        XmlMotorActuatorCfg(
          target_names_expr=("rateY",), transmission_type=TransmissionType.SITE
        ),
        XmlMotorActuatorCfg(
          target_names_expr=("rateZ",), transmission_type=TransmissionType.SITE
        ),
      )
    ),
  )

  entity = Entity(cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg(
    mujoco=MujocoCfg(
      timestep=0.01,
      gravity=(0.0, 0.0, 0.0),
    )
  )
  sim = Simulation(num_envs=args.num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  print("sim gravity:", sim.mj_model.opt.gravity.tolist())
  body_id = sim.mj_model.body("drone_0").id
  print("drone_0 body id:", body_id)
  print("drone_0 mass:", sim.mj_model.body_mass[body_id])
  print("drone_0 principal inertia:", sim.mj_model.body_inertia[body_id].tolist())
  print("drone_0 inertial pos:", sim.mj_model.body_ipos[body_id].tolist())
  print("drone_0 inertial quat:", sim.mj_model.body_iquat[body_id].tolist())
  print("num_envs:", sim.num_envs)
  if sim.num_envs > 1:
    print("viewer note: only env 0 is shown in the MuJoCo viewer")
  inertia = torch.tensor(
    sim.mj_model.body_inertia[body_id], dtype=torch.float32, device=device
  )
  kom = torch.tensor((20.0, 35.0, 45.0), dtype=torch.float32, device=device)

  site_ids_list, _ = entity.find_sites(
    ("thrust", "rateX", "rateY", "rateZ"), preserve_order=True
  )
  site_ids = torch.tensor(site_ids_list, dtype=torch.long, device=device)

  dt = float(sim.mj_model.opt.timestep)
  wall_dt = dt / max(args.realtime_factor, 1e-6)
  omega_history: list[list[float]] = []
  time_history: list[float] = []

  with mujoco.viewer.launch_passive(
    sim.mj_model,
    sim.mj_data,
    show_left_ui=False,
    show_right_ui=False,
  ) as viewer:
    for i in range(args.steps):
      start = time.perf_counter()

      _sync_mouse_perturbation_to_sim(sim, viewer)

      omega_b = entity.data.root_link_ang_vel_b
      wd = torch.tensor(
        (args.mx, args.my, args.mz), dtype=omega_b.dtype, device=device
      )
      wd = wd.unsqueeze(0).expand(sim.num_envs, -1)
      e_omega = omega_b - wd
      iw = omega_b * inertia.unsqueeze(0)
      gyroscopic = torch.linalg.cross(omega_b, iw, dim=1)
      moments = gyroscopic - inertia.unsqueeze(0) * (kom.unsqueeze(0) * e_omega)

      # [thrust, Mx, My, Mz] in site order above.
      site_eff = torch.zeros((sim.num_envs, 4), device=device)
      site_eff[:, 0] = args.thrust
      site_eff[:, 1:] = moments
      site_eff[:, 1] = torch.clamp(site_eff[:, 1], -args.mx_limit, args.mx_limit)
      site_eff[:, 2] = torch.clamp(site_eff[:, 2], -args.my_limit, args.my_limit)
      site_eff[:, 3] = torch.clamp(site_eff[:, 3], -args.mz_limit, args.mz_limit)

      entity.set_site_effort_target(site_eff, site_ids=site_ids)
      entity.write_data_to_sim()
      sim.step()
      entity.update(dt=dt)
      omega_history.append(entity.data.root_link_ang_vel_b[0].detach().cpu().tolist())
      time_history.append(i * dt)

      _sync_env0_to_mjdata(sim)
      mujoco.mj_forward(sim.mj_model, sim.mj_data)
      viewer.sync()

      if args.print_every > 0 and i % args.print_every == 0:
        w = omega_b[0].detach().cpu().tolist()
        wd_print = wd[0].detach().cpu().tolist()
        m = moments[0].detach().cpu().tolist()
        u = site_eff[0].detach().cpu().tolist()
        print(
          f"step={i:05d} omega_b=[{w[0]: .3f},{w[1]: .3f},{w[2]: .3f}] "
          f"omega_d=[{wd_print[0]: .3f},{wd_print[1]: .3f},{wd_print[2]: .3f}] "
          f"M=[{m[0]: .3f},{m[1]: .3f},{m[2]: .3f}] "
          f"ctrl=[{u[0]: .3f},{u[1]: .3f},{u[2]: .3f},{u[3]: .3f}]"
        )

      elapsed = time.perf_counter() - start
      sleep_s = wall_dt - elapsed
      if sleep_s > 0:
        time.sleep(sleep_s)

      if not viewer.is_running():
        break

  final_omega = entity.data.root_link_ang_vel_b[0].detach().cpu().tolist()
  final_ctrl = sim.data.ctrl[0].detach().cpu().tolist()
  print("Final omega_b [wx,wy,wz]:", [round(v, 5) for v in final_omega])
  print("Final ctrl [thrust,mx,my,mz]:", [round(v, 5) for v in final_ctrl])
  if omega_history:
    omega_arr = np.asarray(omega_history)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time_history, omega_arr[:, 0], color="r", label="wx")
    ax.plot(time_history, omega_arr[:, 1], color="g", label="wy")
    ax.plot(time_history, omega_arr[:, 2], color="b", label="wz")
    ax.set_title("Body Angular Velocity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angular velocity [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Quadrotor body-rate control demo with mouse disturbances."
  )
  parser.add_argument("--thrust", type=float, default=0.0, help="Body z-force command")
  parser.add_argument("--mx", type=float, default=4.0, help="Desired body rate wx")
  parser.add_argument("--my", type=float, default=2.0, help="Desired body rate wy")
  parser.add_argument("--mz", type=float, default=-5.0, help="Desired body rate wz")
  parser.add_argument("--mx-limit", type=float, default=3.0, help="Mx saturation")
  parser.add_argument("--my-limit", type=float, default=3.0, help="My saturation")
  parser.add_argument("--mz-limit", type=float, default=2.0, help="Mz saturation")

  parser.add_argument("--steps", type=int, default=8000, help="Simulation steps")
  parser.add_argument("--num-envs", type=int, default=10, help="Number of batched environments")
  parser.add_argument(
    "--realtime-factor",
    type=float,
    default=2.0,
    help="Playback speed. 1.0 = real-time, 2.0 = 2x faster.",
  )
  parser.add_argument(
    "--print-every",
    type=int,
    default=200,
    help="Print omega/ctrl every N steps (0 disables).",
  )
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device override (cpu or cuda). Defaults to auto selection.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  run(_parse_args())
