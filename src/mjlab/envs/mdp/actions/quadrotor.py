"""Quadrotor body-rate action space with an inner rate controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.action_manager import ActionTerm


@dataclass(kw_only=True)
class QuadrotorRateActionCfg(BaseActionCfg):
  """Command collective thrust and desired body rates.

  The policy outputs ``[thrust_cmd, wx_d, wy_d, wz_d]``. This action term
  applies an inner-loop body-rate controller to compute body moments:

  ``M = ω × (Jω) - J (K (ω - ω_d))``

  The resulting command written to the site actuators is
  ``[thrust_cmd, Mx, My, Mz]``.
  """

  body_name: str = "drone_0"
  rate_gains: tuple[float, float, float] = (20.0, 35.0, 45.0)

  def __post_init__(self):
    self.transmission_type = TransmissionType.SITE

  def build(self, env) -> ActionTerm:
    return QuadrotorRateAction(self, env)


class QuadrotorRateAction(BaseAction):
  """Apply a body-rate control law over site effort actuators."""

  cfg: QuadrotorRateActionCfg

  def __init__(self, cfg: QuadrotorRateActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    if self.action_dim != 4:
      raise ValueError(
        "QuadrotorRateAction expects exactly 4 site targets: "
        "[thrust, rateX, rateY, rateZ]."
      )

    body_ids, _ = self._entity.find_bodies((cfg.body_name,), preserve_order=True)
    if len(body_ids) != 1:
      raise ValueError(
        f"Expected exactly one body matching {cfg.body_name!r}, got {len(body_ids)}."
      )
    body_id = int(self._entity.indexing.body_ids[body_ids[0]].item())
    inertia = torch.tensor(
      env.sim.mj_model.body_inertia[body_id],
      dtype=torch.float32,
      device=self.device,
    )
    self._inertia = inertia
    self._rate_gains = torch.tensor(
      cfg.rate_gains, dtype=torch.float32, device=self.device
    )
    self._computed_efforts = torch.zeros(
      self.num_envs, self.action_dim, device=self.device
    )

  def apply_actions(self) -> None:
    omega_b = self._entity.data.root_link_ang_vel_b
    desired_rates = self._processed_actions[:, 1:4]
    rate_error = omega_b - desired_rates

    inertia = self._inertia.unsqueeze(0)
    gains = self._rate_gains.unsqueeze(0)
    iw = omega_b * inertia
    gyroscopic = torch.linalg.cross(omega_b, iw, dim=1)
    moments = gyroscopic - inertia * (gains * rate_error)

    self._computed_efforts[:, 0] = self._processed_actions[:, 0]
    self._computed_efforts[:, 1:4] = moments
    self._entity.set_site_effort_target(
      self._computed_efforts, site_ids=self._target_ids
    )
