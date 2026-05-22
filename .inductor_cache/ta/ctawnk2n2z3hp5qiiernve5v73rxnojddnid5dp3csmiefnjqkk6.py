# AOT ID: ['9_inference']
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


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/zi/czi4xykxi2o2wdy7k2u6yxxgb5uzxgw3ach4xbwwd6jyts3ioung.py
# Topologically Sorted Source Nodes: [split, log, loss], Original ATen: [aten.split, aten.log, aten.sub]
# Source node to ATen node mapping:
#   log => log
#   loss => sub
#   split => split
# Graph fragment:
#   %arg0_1 : Tensor "f32[8192, 8][8, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg0_1, 4096), kwargs = {})
#   %log : Tensor "f32[4096, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub : Tensor "f32[4096, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%log, %getitem), kwargs = {})
#   return %sub
triton_poi_fused_log_split_sub_0 = async_compile.triton('triton_poi_fused_log_split_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_log_split_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_log_split_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (32768 + x0), None)
    tmp2 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl_math.log(tmp0)
    tmp3 = tmp1 - tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/24/c24bxiokgjrileihg2ejp7apzg6k3ab4ie4ctrdyd5azwu2jwzcz.py
# Topologically Sorted Source Nodes: [split, unsqueeze, div_], Original ATen: [aten.split, aten.unsqueeze, aten.div, aten.copy_]
# Source node to ATen node mapping:
#   div_ => div
#   split => split
#   unsqueeze => unsqueeze
# Graph fragment:
#   %copy_ : Tensor "f32[4096, 8, 50304][402432, 50304, 1]cuda:0" = PlaceHolder[target=copy_]
#   %arg0_1 : Tensor "f32[8192, 8][8, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %div : Tensor "f32[4096, 8, 50304][402432, 50304, 1]cuda:0" = PlaceHolder[target=div]
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg0_1, 4096), kwargs = {})
#   %unsqueeze : Tensor "f32[4096, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_1, -1), kwargs = {})
#   %div : Tensor "f32[4096, 8, 50304][402432, 50304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg1_1, %unsqueeze), kwargs = {})
#   %copy_ : Tensor "f32[4096, 8, 50304][402432, 50304, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %div), kwargs = {})
#   return %div,%buf2
triton_poi_fused_copy__div_split_unsqueeze_1 = async_compile.triton('triton_poi_fused_copy__div_split_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy__div_split_unsqueeze_1', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 19780337664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy__div_split_unsqueeze_1(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1648361472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 50304
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (32768 + x1), None, eviction_policy='evict_last')
    tmp2 = (tmp0 / tmp1)
    tl.store(out_ptr1 + (x2), tmp2, None)
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
        assert_size_stride(arg0_1, (8192, 8), (8, 1))
        assert_size_stride(arg1_1, (4096, 8, 50304), (402432, 50304, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((4096, 8), (8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [split, log, loss], Original ATen: [aten.split, aten.log, aten.sub]
            stream0 = get_raw_stream(0)
            triton_poi_fused_log_split_sub_0.run(arg0_1, buf0, 32768, stream=stream0)
            # Topologically Sorted Source Nodes: [split, unsqueeze, div_], Original ATen: [aten.split, aten.unsqueeze, aten.div, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy__div_split_unsqueeze_1.run(arg1_1, arg0_1, arg1_1, 1648361472, stream=stream0)
            del arg0_1
            del arg1_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8192, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4096, 8, 50304), (402432, 50304, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
