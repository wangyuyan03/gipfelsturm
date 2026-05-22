# AOT ID: ['0_forward']
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


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/ub/cub2kndg4qhr25ep7nye6eopjgnii32ekohmyvwlkptrwok7tyvp.py
# Topologically Sorted Source Nodes: [y, chunk, silu, mul], Original ATen: [aten.add, aten.split, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   chunk => split
#   mul => mul_1
#   silu => convert_element_type, convert_element_type_1, mul, sigmoid
#   y => add
# Graph fragment:
#   %primals_1 : Tensor "bf16[4096, 4, 4352][17408, 4352, 1]cuda:2" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[4352][1]cuda:2" = PlaceHolder[target=primals_2]
#   %add : Tensor "bf16[4096, 4, 4352][17408, 4352, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %primals_2), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%add, 2176, -1), kwargs = {})
#   %convert_element_type : Tensor "f32[4096, 4, 2176][8704, 2176, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[4096, 4, 2176][8704, 2176, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type,), kwargs = {})
#   %mul : Tensor "f32[4096, 4, 2176][8704, 2176, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %sigmoid), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4096, 4, 2176][8704, 2176, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[4096, 4, 2176][8704, 2176, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %getitem_1), kwargs = {})
#   return %mul_1
triton_poi_fused_add_mul_silu_split_0 = async_compile.triton('triton_poi_fused_add_mul_silu_split_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_split_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 285221376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_silu_split_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 35651584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2176)
    x1 = xindex // 2176
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4352*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (2176 + x0 + 4352*x1), None).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (2176 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
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
        primals_1, primals_2 = args
        args.clear()
        assert_size_stride(primals_1, (4096, 4, 4352), (17408, 4352, 1))
        assert_size_stride(primals_2, (4352, ), (1, ))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((4096, 4, 2176), (8704, 2176, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y, chunk, silu, mul], Original ATen: [aten.add, aten.split, aten.silu, aten.mul]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_mul_silu_split_0.run(primals_1, primals_2, buf0, 35651584, stream=stream2)
        return (buf0, primals_1, primals_2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4096, 4, 4352), (17408, 4352, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_2 = rand_strided((4352, ), (1, ), device='cuda:2', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
