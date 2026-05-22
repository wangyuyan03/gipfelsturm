# AOT ID: ['8_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/ge/cgedg34yc4lvhl52bthwmdvy4tvye3i2c6mce636onbuqhg3zs3s.py
# Topologically Sorted Source Nodes: [unsqueeze, vocab_parallel_logits, predicted_logits_1d, arange_1d, masked_target, lt, ge, target_mask, setitem, predicted_logits, setitem_1], Original ATen: [aten.unsqueeze, aten.sub, aten.view, aten.arange, aten.lt, aten.ge, aten.bitwise_or, aten.lift_fresh, aten.index_put, aten.index]
# Source node to ATen node mapping:
#   arange_1d => iota
#   ge => ge
#   lt => lt
#   masked_target => sub_1
#   predicted_logits => view_4
#   predicted_logits_1d => index, view_2, view_3
#   setitem => full_default, index_put
#   setitem_1 => full_default_1, index_put_1
#   target_mask => bitwise_or
#   unsqueeze => unsqueeze
#   vocab_parallel_logits => sub
# Graph fragment:
#   %arg2_1 : Tensor "i64[4096, 16][16, 1]cuda:2" = PlaceHolder[target=arg2_1]
#   %bitwise_or : Tensor "b8[4096, 16][16, 1]cuda:2" = PlaceHolder[target=bitwise_or]
#   %index_put : Tensor "i64[4096, 16][16, 1]cuda:2" = PlaceHolder[target=index_put]
#   %copy_ : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2" = PlaceHolder[target=copy_]
#   %arg1_1 : Tensor "f32[4096, 16][16, 1]cuda:2" = PlaceHolder[target=arg1_1]
#   %unsqueeze : Tensor "f32[4096, 16, 1][16, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, -1), kwargs = {})
#   %sub : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %unsqueeze), kwargs = {})
#   %view_2 : Tensor "f32[65536, 50304][50304, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [-1, 50304]), kwargs = {})
#   %iota : Tensor "i64[65536][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (65536,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   %sub_1 : Tensor "i64[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, 0), kwargs = {})
#   %lt : Tensor "b8[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg2_1, 0), kwargs = {})
#   %ge : Tensor "b8[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg2_1, 50304), kwargs = {})
#   %bitwise_or : Tensor "b8[4096, 16][16, 1]cuda:2"[num_users=3] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%lt, %ge), kwargs = {})
#   %full_default : Tensor "i64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : Tensor "i64[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%sub_1, [%bitwise_or], %full_default), kwargs = {})
#   %view_3 : Tensor "i64[65536][1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%index_put, [-1]), kwargs = {})
#   %index : Tensor "f32[65536][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_2, [%iota, %view_3]), kwargs = {})
#   %view_4 : Tensor "f32[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [4096, 16]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_1 : Tensor "f32[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_4, [%bitwise_or], %full_default_1), kwargs = {})
#   return %bitwise_or,%index_put,%index,%buf3
triton_poi_fused_arange_bitwise_or_ge_index_index_put_lift_fresh_lt_sub_unsqueeze_view_0 = async_compile.triton('triton_poi_fused_arange_bitwise_or_ge_index_index_put_lift_fresh_lt_sub_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i64', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_or_ge_index_index_put_lift_fresh_lt_sub_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 4, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_or_ge_index_index_put_lift_fresh_lt_sub_unsqueeze_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp14 = tl.load(in_ptr2 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 50304, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = tmp0 - tmp1
    tmp7 = tl.where(tmp5, tmp1, tmp6)
    tmp8 = tl.full([XBLOCK], 50304, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 50304), "index out of bounds: 0 <= tmp11 < 50304")
    tmp13 = tl.load(in_ptr1 + (tmp11 + 50304*x0), None, eviction_policy='evict_last')
    tmp15 = tmp13 - tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp5, tmp16, tmp15)
    tl.store(out_ptr0 + (x0), tmp5, None)
    tl.store(out_ptr1 + (x0), tmp7, None)
    tl.store(out_ptr2 + (x0), tmp15, None)
    tl.store(out_ptr3 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/hm/chmvybxhmblomxdmcfx6sjbrg6zcijowfbnmy2iokcr3h2ssc4wa.py
# Topologically Sorted Source Nodes: [unsqueeze, vocab_parallel_logits, predicted_logits_1d, arange_1d, predicted_logits, setitem_1, predicted_logits_sum_exp_logits], Original ATen: [aten.unsqueeze, aten.sub, aten.view, aten.arange, aten.index, aten.lift_fresh, aten.index_put, aten.cat]
# Source node to ATen node mapping:
#   arange_1d => iota
#   predicted_logits => view_4
#   predicted_logits_1d => index, view_2, view_3
#   predicted_logits_sum_exp_logits => cat
#   setitem_1 => full_default_1, index_put_1
#   unsqueeze => unsqueeze
#   vocab_parallel_logits => sub
# Graph fragment:
#   %buf3 : Tensor "f32[4096, 16][16, 1]cuda:2" = PlaceHolder[target=buf3]
#   %index : Tensor "f32[65536][1]cuda:2" = PlaceHolder[target=index]
#   %buf4 : Tensor "f32[4096, 16][1]cuda:2" = PlaceHolder[target=buf4]
#   %unsqueeze : Tensor "f32[4096, 16, 1][16, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, -1), kwargs = {})
#   %sub : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %unsqueeze), kwargs = {})
#   %view_2 : Tensor "f32[65536, 50304][50304, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [-1, 50304]), kwargs = {})
#   %iota : Tensor "i64[65536][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (65536,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   %view_3 : Tensor "i64[65536][1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%index_put, [-1]), kwargs = {})
#   %index : Tensor "f32[65536][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_2, [%iota, %view_3]), kwargs = {})
#   %view_4 : Tensor "f32[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [4096, 16]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_1 : Tensor "f32[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_4, [%bitwise_or], %full_default_1), kwargs = {})
#   %cat : Tensor "f32[8192, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%index_put_1, %sum_1],), kwargs = {})
#   return %buf4,%buf6
triton_poi_fused_arange_cat_index_index_put_lift_fresh_sub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused_arange_cat_index_index_put_lift_fresh_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_cat_index_index_put_lift_fresh_sub_unsqueeze_view_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_cat_index_index_put_lift_fresh_sub_unsqueeze_view_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/3l/c3lnzkcgodxgcsobp5mb22tfqpqen22qckiyl3p7k5z7zxalages.py
# Topologically Sorted Source Nodes: [unsqueeze, vocab_parallel_logits, exp, sum_exp_logits], Original ATen: [aten.unsqueeze, aten.sub, aten.exp, aten.sum, aten.copy_]
# Source node to ATen node mapping:
#   exp => exp
#   sum_exp_logits => sum_1
#   unsqueeze => unsqueeze
#   vocab_parallel_logits => sub
# Graph fragment:
#   %copy_ : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2" = PlaceHolder[target=copy_]
#   %arg1_1 : Tensor "f32[4096, 16][16, 1]cuda:2" = PlaceHolder[target=arg1_1]
#   %exp : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2" = PlaceHolder[target=exp]
#   %buf4 : Tensor "f32[4096, 16][1]cuda:2" = PlaceHolder[target=buf4]
#   %buf3 : Tensor "f32[4096, 16][16, 1]cuda:2" = PlaceHolder[target=buf3]
#   %sum_1 : Tensor "f32[4096, 16][16, 1]cuda:2" = PlaceHolder[target=sum_1]
#   %unsqueeze : Tensor "f32[4096, 16, 1][16, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, -1), kwargs = {})
#   %sub : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %unsqueeze), kwargs = {})
#   %exp : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[4096, 16][16, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %copy_ : Tensor "f32[4096, 16, 50304][804864, 50304, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %exp), kwargs = {})
#   return %sum_1,%exp,%buf10
triton_red_fused_copy__exp_sub_sum_unsqueeze_2 = async_compile.triton('triton_red_fused_copy__exp_sub_sum_unsqueeze_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_copy__exp_sub_sum_unsqueeze_2', 'mutated_arg_names': ['in_ptr0', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432, 'r0_': 39560675328}}
)
@triton.jit
def triton_red_fused_copy__exp_sub_sum_unsqueeze_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 50304
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 50304*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = libdevice.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tl.store(out_ptr1 + (r0_1 + 50304*x0), tmp3, r0_mask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, None)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(out_ptr1 + (r0_1 + 50304*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tl.store(out_ptr2 + (r0_1 + 50304*x0), tmp7, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4096, 16, 50304), (804864, 50304, 1))
        assert_size_stride(arg1_1, (4096, 16), (16, 1))
        assert_size_stride(arg2_1, (4096, 16), (16, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((4096, 16), (16, 1), torch.bool)
            buf1 = empty_strided_cuda((4096, 16), (16, 1), torch.int64)
            buf2 = empty_strided_cuda((65536, ), (1, ), torch.float32)
            buf3 = empty_strided_cuda((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze, vocab_parallel_logits, predicted_logits_1d, arange_1d, masked_target, lt, ge, target_mask, setitem, predicted_logits, setitem_1], Original ATen: [aten.unsqueeze, aten.sub, aten.view, aten.arange, aten.lt, aten.ge, aten.bitwise_or, aten.lift_fresh, aten.index_put, aten.index]
            stream2 = get_raw_stream(2)
            triton_poi_fused_arange_bitwise_or_ge_index_index_put_lift_fresh_lt_sub_unsqueeze_view_0.run(arg2_1, arg0_1, arg1_1, buf0, buf1, buf2, buf3, 65536, stream=stream2)
            del arg2_1
            buf7 = empty_strided_cuda((8192, 16), (16, 1), torch.float32)
            buf6 = reinterpret_tensor(buf7, (4096, 16), (16, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [unsqueeze, vocab_parallel_logits, predicted_logits_1d, arange_1d, predicted_logits, setitem_1, predicted_logits_sum_exp_logits], Original ATen: [aten.unsqueeze, aten.sub, aten.view, aten.arange, aten.index, aten.lift_fresh, aten.index_put, aten.cat]
            stream2 = get_raw_stream(2)
            triton_poi_fused_arange_cat_index_index_put_lift_fresh_sub_unsqueeze_view_1.run(buf3, buf2, buf6, 65536, stream=stream2)
            buf5 = reinterpret_tensor(buf7, (4096, 16), (16, 1), 65536)  # alias
            buf9 = empty_strided_cuda((4096, 16, 50304), (804864, 50304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze, vocab_parallel_logits, exp, sum_exp_logits], Original ATen: [aten.unsqueeze, aten.sub, aten.exp, aten.sum, aten.copy_]
            stream2 = get_raw_stream(2)
            triton_red_fused_copy__exp_sub_sum_unsqueeze_2.run(arg0_1, arg1_1, buf5, buf9, arg0_1, 65536, 50304, stream=stream2)
            del arg1_1
            del buf2
            del buf3
            del buf9
        return (buf0, reinterpret_tensor(buf1, (65536, ), (1, ), 0), buf7, arg0_1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, 16, 50304), (804864, 50304, 1), device='cuda:2', dtype=torch.float32)
    arg1_1 = rand_strided((4096, 16), (16, 1), device='cuda:2', dtype=torch.float32)
    arg2_1 = rand_strided((4096, 16), (16, 1), device='cuda:2', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
