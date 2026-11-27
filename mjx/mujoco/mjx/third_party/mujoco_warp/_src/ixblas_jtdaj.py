# Copyright 2025 The Newton Developers
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

"""Dense JTDAJ assembly via ixBLAS strided-batched SGEMM (Iluvatar).

Packs the first ``k_fast`` quadratic constraint rows into A/B panels, runs
``cublasSgemmStridedBatched``, then folds inertia CSR entries and any remaining
constraint rows into H. Matches
``mj-bench/official_scaling/so101_ixblas_stage.py`` for SO-101 (nv_pad=20,
k_fast=48) and generalizes to other small dense Newton models.
"""

from __future__ import annotations

import ctypes as ct
import os
import weakref
from typing import Optional

import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import types
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import cache_kernel

QUADRATIC = wp.constant(int(types.ConstraintState.QUADRATIC.value))

_HANDLES: dict[int, ct.c_void_p] = {}
_LIB: Optional[ct.CDLL] = None
_GEMM = None


def _ixblas_enabled() -> bool:
  flag = os.environ.get("MJW_IXBLAS_JTDAJ", "auto").lower()
  if flag in ("0", "off", "false", "no"):
    return False
  if flag in ("1", "on", "true", "yes"):
    return True
  # auto: Iluvatar only
  try:
    dev = wp.get_device()
    return bool(dev.is_cuda and getattr(dev.runtime, "is_iluvatar", False))
  except Exception:
    return False


def _load_cublas(device_ordinal: int) -> ct.c_void_p:
  global _LIB, _GEMM
  if _LIB is None:
    lib = ct.CDLL("libcublas.so.10")
    ptr, integer, stride = ct.c_void_p, ct.c_int, ct.c_longlong
    lib.cublasCreate_v2.argtypes = [ct.POINTER(ptr)]
    lib.cublasSetStream_v2.argtypes = [ptr, ptr]
    lib.cublasDestroy_v2.argtypes = [ptr]
    gemm = lib.cublasSgemmStridedBatched
    gemm.argtypes = [
      ptr,
      integer,
      integer,
      integer,
      integer,
      integer,
      ptr,
      ptr,
      integer,
      stride,
      ptr,
      integer,
      stride,
      ptr,
      ptr,
      integer,
      stride,
      integer,
    ]
    _LIB = lib
    _GEMM = gemm

  handle = _HANDLES.get(device_ordinal)
  if handle is None:
    handle = ct.c_void_p()
    rc = _LIB.cublasCreate_v2(ct.byref(handle))
    if rc:
      raise RuntimeError(f"cublasCreate_v2 failed with status {rc}")
    _HANDLES[device_ordinal] = handle
  return handle


def _check(rc: int) -> None:
  if rc:
    raise RuntimeError(f"ixBLAS returned status {rc}")


@cache_kernel
def _pack_kernel(nv_pad: int, k_fast: int):
  K = k_fast

  @wp.kernel(enable_backward=False, module="unique")
  def pack(
    nefc: wp.array[int],
    j: wp.array3d[float],
    d: wp.array2d[float],
    state: wp.array2d[int],
    done: wp.array[bool],
    a: wp.array3d[float],
    b: wp.array3d[float],
  ):
    w, k, i = wp.tid()
    if done[w]:
      return
    weight = float(0.0)
    value = float(0.0)
    if k < nefc[w] and k < j.shape[1] and k < K:
      if state[w, k] == QUADRATIC:
        weight = d[w, k]
        value = j[w, k, i]
    a[w, i, k] = weight * value
    b[w, k, i] = value

  return pack


@cache_kernel
def _finish_kernel(nv_pad: int, k_fast: int):
  K = k_fast

  @wp.kernel(enable_backward=False, module="unique")
  def finish(
    mass_index: wp.array2d[int],
    mass: wp.array2d[float],
    nefc: wp.array[int],
    j: wp.array3d[float],
    d: wp.array2d[float],
    state: wp.array2d[int],
    done: wp.array[bool],
    product: wp.array3d[float],
    h: wp.array3d[float],
  ):
    w, i, col = wp.tid()
    if done[w]:
      return
    val = product[w, i, col]
    entry = mass_index[i, col]
    if entry >= 0:
      val += mass[w, entry]
    # Rows beyond the fast GEMM panel stay on the scalar fallback.
    for k in range(K, wp.min(nefc[w], j.shape[1])):
      if state[w, k] == QUADRATIC:
        val += (d[w, k] * j[w, k, i]) * j[w, k, col]
    h[w, i, col] = val

  return finish


class IxBlasJTDAJ:
  """Per-model workspace for dense JTDAJ via strided-batched SGEMM."""

  def __init__(self, model: types.Model, nworld: int, k_fast: int = 48):
    self.nv_pad = int(model.nv_pad)
    self.nworld = int(nworld)
    self.k_fast = int(k_fast)
    self.device = wp.get_device()
    self._handle = _load_cublas(self.device.ordinal)
    stream = wp.get_stream(self.device)
    self._stream_handle = stream.cuda_stream
    _check(_LIB.cublasSetStream_v2(self._handle, ct.c_void_p(self._stream_handle)))

    rows = model.M_colind.numpy()
    cols = model.M_hinit_i.numpy()
    p = self.nv_pad
    flat = rows * p + cols
    if not (np.all((flat >= 0) & (flat < p * p)) and len(np.unique(flat)) == len(flat)):
      raise ValueError("Invalid M_colind / M_hinit_i layout for dense JTDAJ")
    index = np.full((p, p), -1, dtype=np.int32)
    index[rows, cols] = np.arange(len(flat), dtype=np.int32)
    self.index = wp.array(index, dtype=int, device=self.device)
    self.a = wp.zeros((nworld, p, k_fast), dtype=float, device=self.device)
    self.b = wp.zeros((nworld, k_fast, p), dtype=float, device=self.device)
    self.product = wp.empty((nworld, p, p), dtype=float, device=self.device)
    self._pack = _pack_kernel(p, k_fast)
    self._finish = _finish_kernel(p, k_fast)
    self._one = ct.c_float(1.0)
    self._zero = ct.c_float(0.0)
    self._model_ref = weakref.ref(model)
    self._m_ptrs = (model.M_colind.ptr, model.M_hinit_i.ptr, model.M_colind.size)

  def matches(self, model: types.Model, nworld: int, k_fast: int) -> bool:
    return (
      self._model_ref() is model
      and self.nworld == nworld
      and self.nv_pad == model.nv_pad
      and self.k_fast == k_fast
      and self._m_ptrs == (model.M_colind.ptr, model.M_hinit_i.ptr, model.M_colind.size)
    )

  def launch(
    self,
    nefc: wp.array,
    mass: wp.array,
    j: wp.array,
    d: wp.array,
    state: wp.array,
    done: wp.array,
    h: wp.array,
  ) -> None:
    p, k, batch = self.nv_pad, self.k_fast, self.nworld
    cur_stream = wp.get_stream(self.device).cuda_stream
    if cur_stream != self._stream_handle:
      _check(_LIB.cublasSetStream_v2(self._handle, ct.c_void_p(cur_stream)))
      self._stream_handle = cur_stream
    wp.launch(self._pack, dim=(batch, k, p), inputs=[nefc, j, d, state, done, self.a, self.b], device=self.device)
    stride_a = p * k
    stride_b = k * p
    stride_c = p * p
    # Matching handoff so101_ixblas_stage: gemm(N,N, p,p,k, ...).
    _check(
      _GEMM(
        self._handle,
        0,
        0,
        p,
        p,
        k,
        ct.byref(self._one),
        ct.c_void_p(self.b.ptr),
        p,
        stride_b,
        ct.c_void_p(self.a.ptr),
        k,
        stride_a,
        ct.byref(self._zero),
        ct.c_void_p(self.product.ptr),
        p,
        stride_c,
        batch,
      )
    )
    wp.launch(
      self._finish,
      dim=(batch, p, p),
      inputs=[self.index, mass, nefc, j, d, state, done, self.product, h],
      device=self.device,
    )


_CACHE: dict[tuple[int, int, int, int], IxBlasJTDAJ] = {}


def try_get_stage(model: types.Model, nworld: int, njmax: int = 0) -> Optional[IxBlasJTDAJ]:
  """Return a reusable ixBLAS stage when the dense JTDAJ path should use it."""
  if not _ixblas_enabled():
    return None
  if model.is_sparse or nworld <= 0:
    return None
  if njmax == 0 and os.environ.get("MJW_IXBLAS_JTDAJ", "auto").lower() == "auto":
    return None
  # Fast path targets small dense Newton models (SO-101 class).
  if model.nv_pad > 64 or model.nv_pad < 4:
    return None
  dev = wp.get_device()
  if not dev.is_cuda:
    return None
  # Do not lazily allocate workspace buffers or cublasCreate inside CUDA graph capture.
  stream = wp.get_stream(dev)
  k_fast = int(os.environ.get("MJW_IXBLAS_JTDAJ_K", "48"))
  if k_fast <= 0:
    return None
  key = (id(model), dev.ordinal, int(nworld), k_fast)
  stage = _CACHE.get(key)
  if stage is not None and stage.matches(model, nworld, k_fast):
    return stage
  if stream.is_capturing:
    return None
  try:
    stage = IxBlasJTDAJ(model, nworld, k_fast=k_fast)
  except (OSError, RuntimeError, ValueError):
    return None

  def _evict(_ref, k=key):
    _CACHE.pop(k, None)

  stage._evict_cb = weakref.ref(model, _evict)
  _CACHE[key] = stage
  return stage
