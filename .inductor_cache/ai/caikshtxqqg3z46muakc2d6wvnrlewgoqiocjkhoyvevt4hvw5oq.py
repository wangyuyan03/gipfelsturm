# AOT ID: ['10_inference']
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


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/dk/cdkvqtdav7cp73uggtim4r5ecy4ahdg7vkyauj7ju7wby7slg33u.py
# Topologically Sorted Source Nodes: [grad_2d, arange_1d, getitem, view_1, float_1, softmax_update, isub, setitem], Original ATen: [aten.view, aten.arange, aten.index, aten._to_copy, aten.rsub, aten.sub, aten.index_put]
# Source node to ATen node mapping:
#   arange_1d => iota
#   float_1 => convert_element_type
#   getitem => index
#   grad_2d => view
#   isub => sub_1
#   setitem => index_put
#   softmax_update => sub
#   view_1 => view_1
# Graph fragment:
#   %copy_ : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0" = PlaceHolder[target=copy_]
#   %view : Tensor "f32[16384, 50304][50304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 50304]), kwargs = {})
#   %iota : Tensor "i64[16384][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (16384,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %index : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view, [%iota, %arg2_1]), kwargs = {})
#   %view_1 : Tensor "b8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [-1]), kwargs = {})
#   %convert_element_type : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %sub_1 : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index, %sub), kwargs = {})
#   %index_put : Tensor "f32[16384, 50304][50304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%view, [%iota, %arg2_1], %sub_1), kwargs = {})
#   return %index_put
triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_0 = async_compile.triton('triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9890168832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 824180736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/aa/caampm4wpqbbda3msqsvzk6s6pcq54er57xx3jdddqmgkhzjetcg.py
# Topologically Sorted Source Nodes: [grad_2d, arange_1d, getitem, view_1, float_1, softmax_update, isub, setitem], Original ATen: [aten.view, aten.arange, aten.index, aten._to_copy, aten.rsub, aten.sub, aten.index_put]
# Source node to ATen node mapping:
#   arange_1d => iota
#   float_1 => convert_element_type
#   getitem => index
#   grad_2d => view
#   isub => sub_1
#   setitem => index_put
#   softmax_update => sub
#   view_1 => view_1
# Graph fragment:
#   %arg2_1 : Tensor "i64[16384][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %copy_ : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0" = PlaceHolder[target=copy_]
#   %arg1_1 : Tensor "b8[4096, 4][4, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %index_put : Tensor "f32[16384, 50304][50304, 1]cuda:0" = PlaceHolder[target=index_put]
#   %view : Tensor "f32[16384, 50304][50304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 50304]), kwargs = {})
#   %iota : Tensor "i64[16384][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (16384,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %index : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view, [%iota, %arg2_1]), kwargs = {})
#   %view_1 : Tensor "b8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [-1]), kwargs = {})
#   %convert_element_type : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %sub_1 : Tensor "f32[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index, %sub), kwargs = {})
#   %index_put : Tensor "f32[16384, 50304][50304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%view, [%iota, %arg2_1], %sub_1), kwargs = {})
#   return %buf1
triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_1 = async_compile.triton('triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.int1)
    tmp1 = tl.full([XBLOCK], 50304, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 50304), "index out of bounds: 0 <= tmp4 < 50304")
    tmp6 = tl.load(in_ptr1 + (tmp4 + 50304*x0), None, eviction_policy='evict_last')
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp6 - tmp10
    tl.store(out_ptr0 + (tmp4 + 50304*x0), tmp11, None)
''', device_str='cuda')


# kernel path: /iopsstor/scratch/cscs/course_00313/gipfelsturm/.inductor_cache/wm/cwmwf6s2irsomrdp23fmyh3vxrw7f3nwvy25tjkxegpqijb3qo7c.py
# Topologically Sorted Source Nodes: [setitem, unsqueeze, mul_, grad_input], Original ATen: [aten.view, aten.unsqueeze, aten.mul, aten._to_copy, aten.copy_]
# Source node to ATen node mapping:
#   grad_input => convert_element_type_1
#   mul_ => mul
#   setitem => view_2
#   unsqueeze => unsqueeze
# Graph fragment:
#   %buf1 : Tensor "f32[16384, 50304][50304, 1]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "f32[4096, 4][1, 4096]cuda:0" = PlaceHolder[target=arg3_1]
#   %mul : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0" = PlaceHolder[target=mul]
#   %copy_ : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0" = PlaceHolder[target=copy_]
#   %view_2 : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_put, [4096, 4, 50304]), kwargs = {})
#   %unsqueeze : Tensor "f32[4096, 4, 1][1, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %mul : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %unsqueeze), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4096, 4, 50304][201216, 50304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %copy_ : Tensor "f32[4096, 4, 50304][201216, 50304, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %mul), kwargs = {})
#   return %convert_element_type_1,%mul,%buf6
triton_poi_fused__to_copy_copy__mul_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__to_copy_copy__mul_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy__mul_unsqueeze_view_2', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '502235316AAA7116816FC5334150DAA728287DFB16834348BBED12BB2F6DAC50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9890168832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_copy__mul_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 824180736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 50304) % 4)
    x2 = xindex // 201216
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x2 + 4096*x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp2, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4096, 4, 50304), (201216, 50304, 1))
        assert_size_stride(arg1_1, (4096, 4), (4, 1))
        assert_size_stride(arg2_1, (16384, ), (1, ))
        assert_size_stride(arg3_1, (4096, 4), (1, 4096))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16384, 50304), (50304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [grad_2d, arange_1d, getitem, view_1, float_1, softmax_update, isub, setitem], Original ATen: [aten.view, aten.arange, aten.index, aten._to_copy, aten.rsub, aten.sub, aten.index_put]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_0.run(arg0_1, buf0, 824180736, stream=stream0)
            # Topologically Sorted Source Nodes: [grad_2d, arange_1d, getitem, view_1, float_1, softmax_update, isub, setitem], Original ATen: [aten.view, aten.arange, aten.index, aten._to_copy, aten.rsub, aten.sub, aten.index_put]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_index_index_put_rsub_sub_view_1.run(arg2_1, arg0_1, arg1_1, buf0, 16384, stream=stream0)
            del arg1_1
            del arg2_1
            buf2 = empty_strided_cuda((4096, 4, 50304), (201216, 50304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [setitem, unsqueeze, mul_, grad_input], Original ATen: [aten.view, aten.unsqueeze, aten.mul, aten._to_copy, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_copy__mul_unsqueeze_view_2.run(buf0, arg3_1, buf2, arg0_1, 824180736, stream=stream0)
            del arg0_1
            del arg3_1
            del buf0
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, 4, 50304), (201216, 50304, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4096, 4), (4, 1), device='cuda:0', dtype=torch.bool)
    arg2_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((4096, 4), (1, 4096), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
