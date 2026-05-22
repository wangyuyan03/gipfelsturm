# AOT ID: ['2_forward']
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


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/gj/cgjgkjcvipcdvdtctofoveg44qm5qsjw74mljmrqwfu3mqjs2ge7.py
# Topologically Sorted Source Nodes: [x, out, out_1], Original ATen: [aten.add, aten.native_dropout]
# Source node to ATen node mapping:
#   out => convert_element_type_default, gt, inductor_lookup_seed_default, inductor_random_default, mul, mul_1
#   out_1 => add_1
#   x => add
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[1][1]cuda:2" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default : Tensor "f32[4096, 4, 1536][6144, 1536, 1]cuda:2" = PlaceHolder[target=inductor_random_default]
#   %primals_1 : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2" = PlaceHolder[target=primals_1]
#   %gt : Tensor "b8[4096, 4, 1536][6144, 1536, 1]cuda:2" = PlaceHolder[target=gt]
#   %primals_2 : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[4096, 4, 1536][0, 0, 1]cuda:2" = PlaceHolder[target=primals_3]
#   %add : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %primals_3), kwargs = {})
#   %inductor_lookup_seed_default : Tensor "i64[][]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default : Tensor "f32[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([4096, 4, 1536], %inductor_lookup_seed_default, rand), kwargs = {})
#   %convert_element_type_default : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default, torch.bfloat16), kwargs = {})
#   %gt : Tensor "b8[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default, 0.1), kwargs = {})
#   %mul : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %add), kwargs = {})
#   %mul_1 : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.1111111111111112), kwargs = {})
#   %add_1 : Tensor "bf16[4096, 4, 1536][6144, 1536, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %mul_1), kwargs = {})
#   return %inductor_random_default,%gt,%add_1
triton_poi_fused_add_native_dropout_0 = async_compile.triton('triton_poi_fused_add_native_dropout_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*i1', 'out_ptr2': '*bf16', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=2, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 251661312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_dropout_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 1536)
    tmp6 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp7 = tmp5.to(tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 + tmp13
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp14, None)
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
        primals_1, primals_2, primals_3 = args
        args.clear()
        assert_size_stride(primals_1, (4096, 4, 1536), (6144, 1536, 1))
        assert_size_stride(primals_2, (4096, 4, 1536), (6144, 1536, 1))
        assert_size_stride(primals_3, (4096, 4, 1536), (0, 0, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((1, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: []
            aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf0)
            buf2 = empty_strided_cuda((4096, 4, 1536), (6144, 1536, 1), torch.bool)
            buf3 = empty_strided_cuda((4096, 4, 1536), (6144, 1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, out, out_1], Original ATen: [aten.add, aten.native_dropout]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_native_dropout_0.run(buf0, primals_1, primals_2, primals_3, buf2, buf3, 0, 25165824, stream=stream2)
            del buf0
            del primals_1
            del primals_2
            del primals_3
        return (buf3, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4096, 4, 1536), (6144, 1536, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_2 = rand_strided((4096, 4, 1536), (6144, 1536, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_3 = rand_strided((4096, 4, 1536), (0, 0, 1), device='cuda:2', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
