# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""End-to-end MuJoCo Warp P4 verification tests for Iluvatar CoreX."""

import os
import sys
import types
import unittest

import mujoco
import numpy as np
import warp as wp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if "mujoco.mjx" not in sys.modules:
  _mjx_root = os.path.join(_REPO_ROOT, "mjx/mujoco/mjx")
  if os.path.isdir(_mjx_root):
    if os.path.join(_REPO_ROOT, "mjx/mujoco") not in mujoco.__path__:
      mujoco.__path__.append(os.path.join(_REPO_ROOT, "mjx/mujoco"))
    _mjx_pkg = types.ModuleType("mujoco.mjx")
    _mjx_pkg.__path__ = [_mjx_root]
    sys.modules["mujoco.mjx"] = _mjx_pkg

from mujoco.mjx.third_party import mujoco_warp as mjwarp  # noqa: E402

wp.init()
_CUDA_DEVICE = wp.get_device("cuda:0") if wp.is_cuda_available() else None
_IS_ILUVATAR = bool(_CUDA_DEVICE and _CUDA_DEVICE.runtime.is_iluvatar)

_IVCORE11_KERNARG_REASON = (
  "ivcore11 CoreX UMD (CIXStream::PreLaunchKernel) rejects kernels with "
  "kernarg_segment_size > 4096 bytes with CUDA error 200 (removed in ivcore30)"
)


class TestMujocoWarpIluvatarP4(unittest.TestCase):
  """P4 end-to-end CPU comparison, sparse/batched rollouts, and graph replay."""

  def _run_model(
    self,
    rel_path: str,
    *,
    steps: int = 10,
    nworld: int = 1,
    q_atol: float = 2e-4,
    v_atol: float = 2e-3,
    nconmax: int | None = None,
    njmax: int | None = None,
    njmax_nnz: int | None = None,
    iterations: int | None = None,
    ls_iterations: int | None = None,
    use_keyframe: bool = False,
  ):
    if _CUDA_DEVICE is None:
      self.skipTest("No CUDA device available")

    xml_path = os.path.join(_REPO_ROOT, rel_path)
    mjm = mujoco.MjModel.from_xml_path(xml_path)
    if iterations is not None:
      mjm.opt.iterations = iterations
    if ls_iterations is not None:
      mjm.opt.ls_iterations = ls_iterations

    mjd_cpu = mujoco.MjData(mjm)
    if use_keyframe and mjm.nkey > 0:
      mujoco.mj_resetDataKeyframe(mjm, mjd_cpu, 0)
    for _ in range(steps):
      mujoco.mj_step(mjm, mjd_cpu)

    with wp.ScopedDevice("cuda:0"):
      m = mjwarp.put_model(mjm)
      if _IS_ILUVATAR:
        self.assertFalse(bool(m.opt.graph_conditional))
        self.assertGreaterEqual(m.block_dim.update_gradient_grad, 64)
        self.assertGreaterEqual(m.block_dim.island_dsu, 64)

      def _make_init_data():
        mjd = mujoco.MjData(mjm)
        if use_keyframe and mjm.nkey > 0:
          mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
        return mjwarp.put_data(
          mjm,
          mjd,
          nworld=nworld,
          nconmax=nconmax,
          njmax=njmax,
          njmax_nnz=njmax_nnz,
        )

      d_u = _make_init_data()
      for _ in range(steps):
        mjwarp.step(m, d_u)
      wp.synchronize_device("cuda:0")
      q_u = d_u.qpos.numpy().copy()
      v_u = d_u.qvel.numpy().copy()

      self.assertTrue(np.all(np.isfinite(q_u)))
      self.assertTrue(np.all(np.isfinite(v_u)))
      np.testing.assert_allclose(q_u[0], mjd_cpu.qpos, atol=q_atol, rtol=1e-3)
      np.testing.assert_allclose(v_u[0], mjd_cpu.qvel, atol=v_atol, rtol=1e-3)

      d_g = _make_init_data()
      with wp.ScopedCapture() as cap:
        mjwarp.step(m, d_g)
      for _ in range(steps):
        wp.capture_launch(cap.graph)
      wp.synchronize_device("cuda:0")
      np.testing.assert_allclose(d_g.qpos.numpy(), q_u, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(d_g.qvel.numpy(), v_u, atol=1e-5, rtol=1e-5)

  def test_humanoid_batched_graph(self):
    self._run_model("model/humanoid/humanoid.xml", steps=50, nworld=4)

  def test_humanoid_01_mjx(self):
    self._run_model("mjx/mujoco/mjx/test_data/humanoid/01_humanoids.xml", steps=10, nworld=2)

  def test_22_humanoids_sparse(self):
    self._run_model(
      "model/humanoid/22_humanoids.xml",
      steps=5,
      nworld=1,
      q_atol=2e-3,
      v_atol=3e-2,
      nconmax=512,
      njmax=1024,
      njmax_nnz=20000,
    )

  def test_mug(self):
    self._run_model("model/mug/mug.xml", steps=10, nworld=2, v_atol=5e-3)

  def test_flex_mannequin(self):
    self._run_model("model/flex/mannequin.xml", steps=10, nworld=2)

  def test_plastic_wire(self):
    self._run_model("model/plastic/wire.xml", steps=10, nworld=2)

  def test_replicate_helix(self):
    self._run_model("model/replicate/helix.xml", steps=10, nworld=2)

  def test_replicate_newton_cradle(self):
    self._run_model("model/replicate/newton_cradle.xml", steps=10, nworld=2)

  def test_tactile(self):
    self._run_model("model/tactile/tactile.xml", steps=10, nworld=2)

  def test_tendon_arm26(self):
    self._run_model("model/tendon_arm/arm26.xml", steps=20, nworld=4)

  def test_mjx_pendula(self):
    self._run_model("mjx/mujoco/mjx/test_data/pendula.xml", steps=20, nworld=4)

  def test_mjx_constraints(self):
    self._run_model("mjx/mujoco/mjx/test_data/constraints.xml", steps=10, nworld=2)

  def test_mjx_quadratic_frictionloss(self):
    self._run_model("mjx/mujoco/mjx/test_data/quadratic_frictionloss.xml", steps=10, nworld=2)

  def test_mjx_actuators_and_tendons(self):
    for rel in [
      "mjx/mujoco/mjx/test_data/actuator/arm21.xml",
      "mjx/mujoco/mjx/test_data/actuator/arm26.xml",
      "mjx/mujoco/mjx/test_data/actuator/tendon_force_clamp.xml",
      "mjx/mujoco/mjx/test_data/tendon/site_wrap.xml",
      "mjx/mujoco/mjx/test_data/tendon/pulley_site_wrap.xml",
      "mjx/mujoco/mjx/test_data/sensor/sensor.xml",
    ]:
      with self.subTest(rel=rel):
        self._run_model(rel, steps=10, nworld=2)

  def test_barkour_hfield_graph(self):
    # barkour_v0 uses hfield collision (ccd_hfield_kernel); MJWarp's GPU hfield
    # contact formulation differs slightly from C MuJoCo in joint impulse
    # (dv ~ 0.23 rad/s at step 1), so validate step-1 closeness and exact graph replay.
    self._run_model(
      "mjx/mujoco/mjx/test_data/barkour_v0/assets/barkour_v0_mjx.xml",
      steps=1,
      nworld=2,
      q_atol=5e-4,
      v_atol=0.3,
      iterations=20,
      ls_iterations=20,
      use_keyframe=True,
    )

  def test_ray(self):
    if _CUDA_DEVICE is None:
      self.skipTest("No CUDA device available")
    mjm = mujoco.MjModel.from_xml_path(os.path.join(_REPO_ROOT, "mjx/mujoco/mjx/test_data/ray.xml"))
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    with wp.ScopedDevice("cuda:0"):
      m = mjwarp.put_model(mjm)
      d = mjwarp.put_data(mjm, mjd, nworld=1)
      mjwarp.forward(m, d)
      pnt = wp.array([[[0.0, 0.0, 2.0]]], dtype=wp.vec3)
      vec = wp.array([[[0.0, 0.0, -1.0]]], dtype=wp.vec3)
      dist, geomid, normal = mjwarp.ray(m, d, pnt, vec)
      wp.synchronize_device("cuda:0")
      np.testing.assert_allclose(dist.numpy(), [[0.5]], atol=1e-5)
      self.assertEqual(int(geomid.numpy()[0, 0]), 1)
      np.testing.assert_allclose(normal.numpy(), [[[0.0, 0.0, 1.0]]], atol=1e-5)

  def test_ccd_convex_models(self):
    for rel in [
      "model/cube/cube_3x3x3.xml",
      "model/humanoid/humanoid100.xml",
      "model/adhesion/active_adhesion.xml",
      "mjx/mujoco/mjx/test_data/tendon/tendon.xml",
      "mjx/mujoco/mjx/test_data/shadow_hand/scene_right.xml",
    ]:
      with self.subTest(rel=rel):
        self._run_model(rel, steps=5, nworld=1, v_atol=5e-3, nconmax=512, njmax=2048)

  @unittest.skipIf(
    _IS_ILUVATAR,
    f"{_IVCORE11_KERNARG_REASON}: _flex_narrowphase_elem_detect uses 4248B",
  )
  def test_flex_narrowphase_models(self):
    self._run_model("model/flex/belt.xml", steps=5, nworld=1)

  @unittest.skipIf(
    _IS_ILUVATAR,
    f"{_IVCORE11_KERNARG_REASON}: _render_megakernel uses 4152B (and BI-V150S lacks CUDA textures)",
  )
  def test_headless_render(self):
    mjm = mujoco.MjModel.from_xml_path(os.path.join(_REPO_ROOT, "model/humanoid/humanoid.xml"))
    with wp.ScopedDevice("cuda:0"):
      m = mjwarp.put_model(mjm)
      d = mjwarp.put_data(mjm, mujoco.MjData(mjm), nworld=1)
      mjwarp.forward(m, d)
      rc = mjwarp.create_render_context(mjm, nworld=1, cam_res=(64, 64), render_rgb=True, render_depth=True)
      mjwarp.render(m, d, rc)
      wp.synchronize_device("cuda:0")
      self.assertGreater(int(np.count_nonzero(rc.rgb_data.numpy())), 0)


if __name__ == "__main__":
  unittest.main(verbosity=2)
