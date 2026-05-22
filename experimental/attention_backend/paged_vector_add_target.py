# Copyright 2026 The Spyre-Inference Authors.
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

import os

os.environ["SPYRE_INDUCTOR_LOG"] = "1"
os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "DEBUG"
os.environ["TORCH_SENDNN_LOG"] = "DEBUG"
# os.environ["DT_DEEPRT_VERBOSE"] = "1"
# os.environ["DTLOG_LEVEL"] = "debug"

import torch
from torch.spyre import SpyreTensorLayout, get_device_dtype

#########################################
# debugging

import torch._logging

# Enable graph code printing
# torch._logging.set_logs(graph_code=True)

# to enable profiling
# from torch.profiler import profile, ProfilerActivity

# debug = True
debug = False

if debug:
    import debugpy

    host_addr = os.environ.get("TORCH_SPYRE_DEBUG_ADDR", "0.0.0.0")
    pdb_port = int(os.environ.get("TORCH_SPYRE_DEBUG_PORT", "5679"))
    debugpy.listen((host_addr, pdb_port))
    print(f"[debugpy] listening at {host_addr}:{pdb_port}; wait for client...\n")
    debugpy.wait_for_client()
    print("[debugpy] connected")


#########################################
# set up paged memory

DEVICE = torch.device("spyre")

# vllm uses as KV cache tensor ONE big tensor with the shape
#  [num_blocks, 2, block_size, num_kv_heads, head_size]
# I guess we can use two tensors on Spyre

NUMBER_OF_PAGES = 512  # some value
PAGE_SIZE = 16  # some value, but must be a multiple of 16
HEAD_SIZE = 128  # like granite, llama
NUM_HEADS = 8  #

key_cache = torch.full([NUMBER_OF_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 1.0, dtype=torch.float16)
value_cache = torch.full(
    [NUMBER_OF_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 2.0, dtype=torch.float16
)

cache_stl = SpyreTensorLayout(
    # one approach to stickification, could also be improved
    # and e.g. NUM_HEADS or PAGE_SIZE stickified
    device_size=[NUMBER_OF_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE // 64, 64],
    dim_map=[0, 1, 2, 3, 3],  # Maps device→host dims
    stride_map=[
        PAGE_SIZE * NUM_HEADS * HEAD_SIZE,
        NUM_HEADS * HEAD_SIZE,
        HEAD_SIZE,
        64,
        1,
    ],  # Device memory strides
    device_dtype=get_device_dtype(torch.float16),
)

# manipulate single pages so that we can test the addressing
key_cache[42, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 40.0, dtype=torch.float16)
value_cache[312, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 31.0, dtype=torch.float16)

# Transfer to device with layout
key_cache_dev = key_cache.to("spyre", device_layout=cache_stl)
value_cache_dev = value_cache.to("spyre", device_layout=cache_stl)

print("paged memory created & transferred.")

# check layout
actual_layout = key_cache_dev.device_tensor_layout()
print("actual layout:", actual_layout)


#########################################
# prepare computation, calculate absolute page table

base_addr_k = key_cache_dev.data_ptr()
base_addr_v = value_cache_dev.data_ptr()
print(f"base address of key cache: {base_addr_k}")
print(f"base address of value cache: {base_addr_v}")
# -> apparently not the correct addresses right now...

page_table_list = [13, 312, 42, 14, 15]
page_table_cpu = torch.tensor(page_table_list, dtype=torch.int64, device="cpu")
print(f"page table on CPU with indexes: {page_table_cpu}")


abs_page_table_K = torch.empty(len(page_table_cpu), dtype=torch.int64, device="cpu")
abs_page_table_V = torch.empty(len(page_table_cpu), dtype=torch.int64, device="cpu")

page_stride = cache_stl.stride_map[0]
abs_page_table_K = page_table_cpu * page_stride + base_addr_k
abs_page_table_V = page_table_cpu * page_stride + base_addr_v

print(f"page table for K with addr: {abs_page_table_K}")
print(f"page table for V with addr: {abs_page_table_V}")

# transfer to device
#  -> implicit cast to int32 right now...

abs_page_table_K_dev = abs_page_table_K.to(DEVICE)
abs_page_table_V_dev = abs_page_table_V.to(DEVICE)

# NOTE: we have TWO page tables here, due to separate tensors for K and V
#  -> addresses for pages in K and V are different.
#  but I think this is not a problem for vLLM
#  alternatively, we could have the same approach with one tensor and then
#   using separate indexes/addresses (would still be two page table tensors)


#########################################
# computation
#  (and yes, paged vector add is here just a placeholder for attention
#   ...but we want to separate the infrastructure for paged access from
#   attention computation ops enablement....)

print("prepare computation...")


def paged_vector_add(
    a_pages, b_pages, a_page_table, b_page_table, max_page_table_length, out_pages
):
    # output pages are starting from 0, in this case
    for i in range(max_page_table_length):
        out_page = out_pages[i]
        # here we have two indirect accesses per computation
        #   for paged attention, one indirect access per computation
        #   COULD be enough, if they are then views, i.e. NOT realized
        #   as a new tensor in DRAM
        sum = a_pages[a_page_table[i]] + b_pages[b_page_table[i]]
        # NOTE: this syntax is _not_ correct on CPU/GPU with absolute addresses
        # NOTE: in the future, we would also like to do SLICING with indirect access...
        #   (see comment below)
        # also, the format of the indirect views needs to be compatile with
        #   attention computation, esp. torch.matmul/torch.bmm
        # for paged attention, there would be THREE indirect access tensors
        #   K and V (as simulated here)
        #   but also Q, due to the varlen representation
        out_page.copy_(sum)
        # copy into right output format
        # (also, masking and padding would be required for this type
        #   of "compiled loop" computation, but this can be done in
        #   the attention metadata creation on CPU)


# comment for SLICING and indirect_access:
# Based on our experience, we need to have different "Tile sizes" within the
# paged attention computation that are _different_ from the "page sizes" of vLLM,
# for e.g. apply different optimizations for prefill/decode.
# One way to implement this would be via Slicing. We imagine smth like this:
#   - in case TILE_SIZE > PAGE_SIZE
#       a_pages[a_page_table[i:i+2]]
#   - in case TILE_SIZE < PAGE_SIZE
#       a_pages[a_page_table[i],:8]
# (This is of course a more complex discussion, I just added the comment here
#  to make clear that also slicing would be critical in the future.)


def create_compilable_paged_vector_add(page_table_length):
    def paged_vector_add_with_fixed_length(a_page_table, b_page_table, out_pages):
        assert len(a_page_table) == page_table_length
        assert len(b_page_table) == page_table_length
        return paged_vector_add(
            # we can also "freeze" the K and V tensors here, since they remain always the same...
            key_cache_dev,
            value_cache_dev,
            a_page_table,
            b_page_table,
            page_table_length,
            out_pages,
        )

    return paged_vector_add_with_fixed_length


list_of_compiled_functions_per_request_length = {}

# yes, we will maintain a list of compiled functions per padded request length
#  (very similar to CUDA graph handling of vLLM)
page_table_length = len(page_table_cpu)
list_of_compiled_functions_per_request_length[page_table_length] = torch.compile(
    create_compilable_paged_vector_add(page_table_length)
)
#  -> cannot be compiled right now...

# do the computation (that is where it breaks right now)
print("computing paged vector add...")

out_pages_dev = torch.empty(page_table_length, device=DEVICE, dtype=torch.float16)

list_of_compiled_functions_per_request_length[page_table_length](
    abs_page_table_K_dev, abs_page_table_V_dev, out_pages_dev
)

print("result...")

# print result, should have the following contents:
#  0: 3.0
#  1: 32.0
#  2: 42.0
#  3: 3.0
#  4: 3.0
print(out_pages_dev.cpu())
