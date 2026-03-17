"""Minimal PD-controlled translation-only sphere example.

This is a standalone example (not a registered train/play task).
It builds one sphere with 3 slide joints (x, y, z), no rotational DOF, and
drives it to a target position using a PD actuator.
"""

from __future__ import annotations

import argparse

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg


XML = """
<mujoco>
  <option gravity="0 0 0" timestep="0.01"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="sphere" pos="0 0 0.2">
      <joint name="tx" type="slide" axis="1 0 0" range="-2 2" damping="1"/>
      <joint name="ty" type="slide" axis="0 1 0" range="-2 2" damping="1"/>
      <joint name="tz" type="slide" axis="0 0 1" range="0.05 2" damping="1"/>
      <geom name="sphere_geom" type="sphere" size="0.07" mass="1.0" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run(
  target_xyz: tuple[float, float, float] = (0.8, -0.5, 0.7),
  steps: int = 300,
  kp: float = 150.0,
  kd: float = 30.0,
  effort_limit: float = 200.0,
  device: str | None = None,
) -> list[float]:
  """Run the PD control loop and return final [x, y, z]."""
  chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("tx", "ty", "tz"),
          stiffness=kp,
          damping=kd,
          effort_limit=effort_limit,
        ),
      )
    ),
  )

  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=chosen_device)
  entity.initialize(model, sim.model, sim.data, chosen_device)

  target = torch.tensor([list(target_xyz)], dtype=torch.float32, device=chosen_device)
  zeros = torch.zeros_like(target)

  dt = float(model.opt.timestep)
  for _ in range(steps):
    entity.set_joint_position_target(target)
    entity.set_joint_velocity_target(zeros)
    entity.set_joint_effort_target(zeros)
    entity.write_data_to_sim()
    sim.step()
    entity.update(dt=dt)

  final_xyz = entity.data.joint_pos[0].detach().cpu().tolist()
  return final_xyz


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Move a translation-only sphere to a target using PD control."
  )
  parser.add_argument("--x", type=float, default=0.8, help="Target x")
  parser.add_argument("--y", type=float, default=-0.5, help="Target y")
  parser.add_argument("--z", type=float, default=0.7, help="Target z")
  parser.add_argument("--steps", type=int, default=300, help="Simulation steps")
  parser.add_argument("--kp", type=float, default=150.0, help="PD stiffness")
  parser.add_argument("--kd", type=float, default=30.0, help="PD damping")
  parser.add_argument(
    "--effort-limit", type=float, default=200.0, help="Actuator force limit"
  )
  parser.add_argument(
    "--device", type=str, default=None, help="Device: cuda or cpu (auto if omitted)"
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  final_xyz = run(
    target_xyz=(args.x, args.y, args.z),
    steps=args.steps,
    kp=args.kp,
    kd=args.kd,
    effort_limit=args.effort_limit,
    device=args.device,
  )
  print("Final xyz:", [round(v, 5) for v in final_xyz])


if __name__ == "__main__":
  main()

