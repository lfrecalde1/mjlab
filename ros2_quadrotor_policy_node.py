#!/usr/bin/env python3
"""ROS2 quadrotor policy node for a trained MjLab checkpoint.

This node keeps the same general ROS2 structure as the user's MPC node, but
replaces the optimizer call with direct neural-policy inference.

The loaded policy is the actor trained for the ``Mjlab-Quadrotor-Hover`` task.
That policy expects the following 19-D observation:

  [ dual-quaternion log pose error (6),
    body linear velocity (3),
    body angular velocity (3),
    projected gravity in body frame (3),
    previous raw action (4) ]

and outputs 4 raw actions which the task interprets as:

  thrust_cmd = m*g + 8.0 * a0
  wd        = [6.0*a1, 6.0*a2, 4.0*a3]

The inner rate controller then computes body moments:

  M = omega x (J omega) - J K (omega - wd)

This node publishes:
  - desired reference path for visualization
  - the learned low-level command as geometry_msgs/Twist:
      linear.z  = thrust command [N]
      angular.x = desired body rate wx [rad/s]
      angular.y = desired body rate wy [rad/s]
      angular.z = desired body rate wz [rad/s]

If your flight stack expects a different message/topic, adapt
``publish_learned_command()``.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


class QuadrotorPolicyNode(Node):
  def __init__(self):
    super().__init__("quadrotor_policy_node")

    # Timing / reference generation.
    self.ts = 0.05
    self.t_horizon = 15.0
    self.transition_hold_time = 1.0
    self.transition_blend_time = 1.5
    self.trajectory_speed_scale = 1.0
    self.start_time: float | None = None

    # Policy/task parameters. These must match the trained task.
    self.policy_task_id = "Mjlab-Quadrotor-Hover"
    self.checkpoint_file = Path(
      "/home/fer/mjlab/logs/rsl_rl/quadrotor_hover/2026-03-16_17-29-35/model_499.pt"
    )
    self.mass = 0.94
    self.gravity = 9.81
    self.rate_scale = np.array([6.0, 6.0, 4.0], dtype=np.float32)
    self.rate_gains = np.diag([20.0, 35.0, 45.0]).astype(np.float32)
    # Replace these if your compiled MuJoCo model prints different values.
    self.inertia = np.diag([0.001, 0.001, 0.001]).astype(np.float32)
    self.dq_eps = 1e-8

    # State.
    self.prev_action = np.zeros(4, dtype=np.float32)
    self.reference_initialized = False
    self.odom_received = False
    self.payload_ref_start = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    self.quad_state = np.zeros(13, dtype=np.float64)

    # ROS interfaces.
    self.subscriber_quad_odom = self.create_subscription(
      Odometry, "/quadrotor1/odom", self.callback_get_odometry_drone_0, 10
    )
    self.publisher_desired_quadrotor = self.create_publisher(
      Path, "/quadrotor1/desired_path", 10
    )
    self.publisher_policy_cmd = self.create_publisher(
      Twist, "/quadrotor1/rl_body_rate_cmd", 10
    )
    self.tf_broadcaster = TransformBroadcaster(self)

    # Load the trained actor once.
    self.device = "cpu"
    self._policy_env = None
    self.policy = self._load_policy()

    self.timer = self.create_timer(self.ts, self.run)

  # Policy loading.

  def _load_policy(self):
    env_cfg = load_env_cfg(self.policy_task_id, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_cfg = load_rl_cfg(self.policy_task_id)
    runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=self.device)
    runner.load(
      str(self.checkpoint_file),
      load_cfg={"actor": True},
      strict=True,
      map_location=self.device,
    )
    self._policy_env = env
    self.get_logger().info(f"Loaded checkpoint: {self.checkpoint_file}")
    return runner.get_inference_policy(device=self.device)

  # Reference generation.

  def _base_lissajous(self, t: float):
    xc, yc, zc = (
      self.payload_ref_start[0],
      self.payload_ref_start[1],
      self.payload_ref_start[2],
    )
    ax, ay, az = 1.2, 0.8, 0.0
    wx, wy, wz = 0.8, 1.4, 0.6
    wx *= self.trajectory_speed_scale
    wy *= self.trajectory_speed_scale
    wz *= self.trajectory_speed_scale
    phix, phiy, phiz = 0.0, np.pi / 3.0, np.pi / 6.0

    xd = np.array(
      [
        xc + ax * (np.sin(wx * t + phix) - np.sin(phix)),
        yc + ay * (np.sin(wy * t + phiy) - np.sin(phiy)),
        zc + az * (np.sin(wz * t + phiz) - np.sin(phiz)),
      ],
      dtype=np.float64,
    )
    vd = np.array(
      [
        ax * wx * np.cos(wx * t + phix),
        ay * wy * np.cos(wy * t + phiy),
        az * wz * np.cos(wz * t + phiz),
      ],
      dtype=np.float64,
    )
    ad = np.array(
      [
        -ax * wx * wx * np.sin(wx * t + phix),
        -ay * wy * wy * np.sin(wy * t + phiy),
        -az * wz * wz * np.sin(wz * t + phiz),
      ],
      dtype=np.float64,
    )
    return xd, vd, ad

  def _min_snap_blend(self, s: float):
    a = 35.0 * s**4 - 84.0 * s**5 + 70.0 * s**6 - 20.0 * s**7
    a_s = 140.0 * s**3 - 420.0 * s**4 + 420.0 * s**5 - 140.0 * s**6
    a_ss = 420.0 * s**2 - 1680.0 * s**3 + 2100.0 * s**4 - 840.0 * s**5
    return a, a_s, a_ss

  def desired_lissajous(self, t: float):
    if t <= self.transition_hold_time:
      zeros = np.zeros(3, dtype=np.float64)
      return self.payload_ref_start.copy(), zeros, zeros

    t_blend = t - self.transition_hold_time
    if t_blend >= self.transition_blend_time:
      return self._base_lissajous(t_blend - self.transition_blend_time)

    s = np.clip(t_blend / self.transition_blend_time, 0.0, 1.0)
    alpha, alpha_s, alpha_ss = self._min_snap_blend(s)
    alpha_dot = alpha_s / self.transition_blend_time
    alpha_ddot = alpha_ss / (self.transition_blend_time**2)

    pd_nom, vd_nom, ad_nom = self._base_lissajous(t_blend)
    p0 = self.payload_ref_start
    delta = pd_nom - p0

    pd = p0 + alpha * delta
    vd = alpha_dot * delta + alpha * vd_nom
    ad = alpha_ddot * delta + 2.0 * alpha_dot * vd_nom + alpha * ad_nom
    return pd, vd, ad

  # Quaternion / dual-quaternion math matching the training task.

  def _h_plus(self, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
      [
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w],
      ],
      dtype=np.float64,
    )

  def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

  def _quat_left_product(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return self._h_plus(q) @ p

  def _pose_to_dual_quat(self, pos_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    trans_quat = np.array([0.0, pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    dual = 0.5 * self._quat_left_product(trans_quat, quat_wxyz)
    return np.hstack((quat_wxyz, dual))

  def _dual_quat_conjugate(self, qd: np.ndarray) -> np.ndarray:
    return np.hstack(
      (self._quat_conjugate(qd[0:4]), self._quat_conjugate(qd[4:8]))
    )

  def _dual_h_plus(self, qd: np.ndarray) -> np.ndarray:
    h_real = self._h_plus(qd[0:4])
    h_dual = self._h_plus(qd[4:8])
    zeros = np.zeros((4, 4), dtype=np.float64)
    return np.block([[h_real, zeros], [h_dual, h_real]])

  def _dual_quat_pose_error_log(
    self, current_dq: np.ndarray, desired_dq: np.ndarray
  ) -> np.ndarray:
    desired_conj = self._dual_quat_conjugate(desired_dq)
    q_error = self._dual_h_plus(desired_conj) @ current_dq

    q_error_real = q_error[0:4]
    q_error_dual = q_error[4:8]
    q_error_real_conj = self._quat_conjugate(q_error_real)

    imag = q_error_real[1:4]
    imag_norm = np.linalg.norm(imag + self.dq_eps)
    angle = np.arctan2(imag_norm, q_error_real[0])
    trans_error = 2.0 * self._quat_left_product(q_error_dual, q_error_real_conj)

    log_quat = 0.5 * angle * imag / imag_norm
    log_trans = 0.5 * trans_error[1:4]
    return np.hstack((log_quat, log_trans)).astype(np.float32)

  # Observation helpers.

  def quat_wxyz_to_rotation(self, q_wxyz: np.ndarray) -> R:
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

  def world_to_body(self, q_wxyz: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    return self.quat_wxyz_to_rotation(q_wxyz).inv().apply(v_w)

  def projected_gravity_body(self, q_wxyz: np.ndarray) -> np.ndarray:
    return self.world_to_body(q_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))

  def build_observation(self, p_des_w: np.ndarray) -> np.ndarray:
    p_w = self.quad_state[0:3]
    v_w = self.quad_state[3:6]
    q_wxyz = self.quad_state[6:10]
    w_b = self.quad_state[10:13]

    current_dq = self._pose_to_dual_quat(p_w, q_wxyz)
    desired_dq = self._pose_to_dual_quat(
      p_des_w, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    dq_pose_error = self._dual_quat_pose_error_log(current_dq, desired_dq)
    v_b = self.world_to_body(q_wxyz, v_w).astype(np.float32)
    g_b = self.projected_gravity_body(q_wxyz).astype(np.float32)

    obs = np.concatenate(
      [
        dq_pose_error,
        v_b,
        w_b.astype(np.float32),
        g_b,
        self.prev_action,
      ],
      axis=0,
    )
    return obs.astype(np.float32)

  # Policy / command conversion.

  def policy_step(self, obs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
      obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
      raw_action = self.policy(obs_t)[0].detach().cpu().numpy().astype(np.float32)
    self.prev_action = raw_action.copy()
    return raw_action

  def action_to_command(
    self, raw_action: np.ndarray, w_b: np.ndarray
  ) -> tuple[float, np.ndarray, np.ndarray]:
    thrust_cmd = float(self.mass * self.gravity + 8.0 * raw_action[0])
    desired_rates = self.rate_scale * raw_action[1:4]
    e_omega = w_b.astype(np.float32) - desired_rates
    iw = self.inertia @ w_b.astype(np.float32)
    gyroscopic = np.cross(w_b.astype(np.float32), iw)
    moments = gyroscopic - self.inertia @ (self.rate_gains @ e_omega)
    return thrust_cmd, desired_rates, moments

  # ROS callbacks / publishing.

  def callback_get_odometry_drone_0(self, msg: Odometry):
    x = np.zeros(13, dtype=np.float64)
    x[0] = msg.pose.pose.position.x
    x[1] = msg.pose.pose.position.y
    x[2] = msg.pose.pose.position.z
    x[3] = msg.twist.twist.linear.x
    x[4] = msg.twist.twist.linear.y
    x[5] = msg.twist.twist.linear.z
    x[6] = msg.pose.pose.orientation.w
    x[7] = msg.pose.pose.orientation.x
    x[8] = msg.pose.pose.orientation.y
    x[9] = msg.pose.pose.orientation.z
    x[10] = msg.twist.twist.angular.x
    x[11] = msg.twist.twist.angular.y
    x[12] = msg.twist.twist.angular.z
    self.quad_state = x
    self.odom_received = True
    self.try_initialize_reference()

  def try_initialize_reference(self):
    if self.reference_initialized or not self.odom_received:
      return
    self.payload_ref_start = self.quad_state[0:3].copy()
    self.start_time = time.time()
    self.reference_initialized = True
    arr_str = np.array2string(
      self.payload_ref_start, precision=3, separator=", ", suppress_small=True
    )
    self.get_logger().info(f"Initialized desired path at measured position {arr_str}")

  def publish_desired_path(self):
    if self.start_time is None:
      return
    now = self.get_clock().now().to_msg()
    quad_path = Path()
    quad_path.header.stamp = now
    quad_path.header.frame_id = "world"
    for k in range(int(self.t_horizon / self.ts) + 1):
      tk = (time.time() - self.start_time) + k * self.ts
      pd, _, _ = self.desired_lissajous(tk)
      pose = PoseStamped()
      pose.header = quad_path.header
      pose.pose.position.x = float(pd[0])
      pose.pose.position.y = float(pd[1])
      pose.pose.position.z = float(pd[2])
      pose.pose.orientation.w = 1.0
      quad_path.poses.append(pose)
    self.publisher_desired_quadrotor.publish(quad_path)

  def publish_transforms(self):
    tf_world_quad = TransformStamped()
    tf_world_quad.header.stamp = self.get_clock().now().to_msg()
    tf_world_quad.header.frame_id = "world"
    tf_world_quad.child_frame_id = "quadrotor"
    tf_world_quad.transform.translation.x = float(self.quad_state[0])
    tf_world_quad.transform.translation.y = float(self.quad_state[1])
    tf_world_quad.transform.translation.z = float(self.quad_state[2])
    tf_world_quad.transform.rotation.w = float(self.quad_state[6])
    tf_world_quad.transform.rotation.x = float(self.quad_state[7])
    tf_world_quad.transform.rotation.y = float(self.quad_state[8])
    tf_world_quad.transform.rotation.z = float(self.quad_state[9])
    self.tf_broadcaster.sendTransform(tf_world_quad)

  def publish_learned_command(
    self, thrust_cmd: float, desired_rates: np.ndarray, moments: np.ndarray
  ):
    msg = Twist()
    msg.linear.x = float(moments[0])
    msg.linear.y = float(moments[1])
    msg.linear.z = float(thrust_cmd)
    msg.angular.x = float(desired_rates[0])
    msg.angular.y = float(desired_rates[1])
    msg.angular.z = float(desired_rates[2])
    self.publisher_policy_cmd.publish(msg)

  # Main loop.

  def run(self):
    if not self.reference_initialized or self.start_time is None:
      return

    t_now = time.time() - self.start_time
    pd, _, _ = self.desired_lissajous(t_now)
    obs = self.build_observation(pd)
    raw_action = self.policy_step(obs)
    thrust_cmd, desired_rates, moments = self.action_to_command(
      raw_action, self.quad_state[10:13]
    )

    self.publish_desired_path()
    self.publish_learned_command(thrust_cmd, desired_rates, moments)
    self.publish_transforms()

  def destroy_node(self):
    if self._policy_env is not None:
      self._policy_env.close()
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node = QuadrotorPolicyNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    node.get_logger().info("Policy node stopped manually.")
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
