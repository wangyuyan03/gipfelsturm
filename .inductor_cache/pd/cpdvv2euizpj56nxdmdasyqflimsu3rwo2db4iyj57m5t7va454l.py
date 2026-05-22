# AOT ID: ['11_inference']
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


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/fg/cfghlwswivdknrvsrhazwgjemkavlixithqzndbzpda6uy26osiu.py
# Topologically Sorted Source Nodes: [chunk, sigmoid, mul, sigmoid_1, sub, mul_1, add, mul_2, mul_3, silu, mul_4, cat], Original ATen: [aten.split, aten.sigmoid, aten.mul, aten.rsub, aten.add, aten.silu, aten.cat]
# Source node to ATen node mapping:
#   add => add
#   cat => cat
#   chunk => split
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_5
#   sigmoid => sigmoid
#   sigmoid_1 => sigmoid_1
#   silu => convert_element_type, convert_element_type_1, mul_4, sigmoid_2
#   sub => sub
# Graph fragment:
#   %arg1_1 : Tensor "bf16[16384, 4352][4352, 1]cuda:3" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "bf16[16384, 8704][8704, 1]cuda:3" = PlaceHolder[target=arg0_1]
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg0_1, 4352, -1), kwargs = {})
#   %sigmoid : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem,), kwargs = {})
#   %mul : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %sigmoid), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem,), kwargs = {})
#   %sub : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_1), kwargs = {})
#   %mul_1 : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem, %sub), kwargs = {})
#   %add : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 1), kwargs = {})
#   %mul_2 : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add), kwargs = {})
#   %mul_3 : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %getitem_1), kwargs = {})
#   %convert_element_type : Tensor "f32[16384, 4352][4352, 1]cuda:3"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.float32), kwargs = {})
#   %sigmoid_2 : Tensor "f32[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type,), kwargs = {})
#   %mul_4 : Tensor "f32[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %sigmoid_2), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_5 : Tensor "bf16[16384, 4352][4352, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %convert_element_type_1), kwargs = {})
#   %cat : Tensor "bf16[16384, 8704][8704, 1]cuda:3"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_3, %mul_5], -1), kwargs = {})
#   return %cat
triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0 = async_compile.triton('triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=3, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1711276032}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 142606336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 8704)
    x1 = xindex // 8704
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4352, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4352*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (8704*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp7
    tmp11 = tmp6 * tmp10
    tmp12 = tmp11 + tmp9
    tmp13 = tmp8 * tmp12
    tmp14 = tl.load(in_ptr1 + (4352 + 8704*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 8704, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr0 + (4352*x1 + ((-4352) + x0)), tmp18, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (8704*x1 + ((-4352) + x0)), tmp18, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp21 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp18, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp17, tmp29)
    tl.store(out_ptr0 + (x2), tmp30, None)
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
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (16384, 8704), (8704, 1))
        assert_size_stride(arg1_1, (16384, 4352), (4352, 1))
        with torch.cuda._DeviceGuard(3):
            torch.cuda.set_device(3)
            buf0 = empty_strided_cuda((16384, 8704), (8704, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, sigmoid, mul, sigmoid_1, sub, mul_1, add, mul_2, mul_3, silu, mul_4, cat], Original ATen: [aten.split, aten.sigmoid, aten.mul, aten.rsub, aten.add, aten.silu, aten.cat]
            stream3 = get_raw_stream(3)
            triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0.run(arg1_1, arg0_1, buf0, 142606336, stream=stream3)
            del arg0_1
            del arg1_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16384, 8704), (8704, 1), device='cuda:3', dtype=torch.bfloat16)
    arg1_1 = rand_strided((16384, 4352), (4352, 1), device='cuda:3', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
