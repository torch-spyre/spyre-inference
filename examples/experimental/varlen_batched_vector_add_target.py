# Copyright 2026 The Torch-Spyre Authors.
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


demo_on_cpu = True


if demo_on_cpu:
    DEVICE = torch.device("cpu")
else:
    from torch.spyre import SpyreTensorLayout, get_device_dtype
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

if not demo_on_cpu:
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

# set pages to 0 to be used as NOP for padding
key_cache[0, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 0.0, dtype=torch.float16)
value_cache[0, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 0.0, dtype=torch.float16)

# manipulate single pages so that we can test the addressing
key_cache[42, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 40.0, dtype=torch.float16)
key_cache[41, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 41.0, dtype=torch.float16)
value_cache[312, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 31.0, dtype=torch.float16)
value_cache[471, :, :, :] = torch.full([PAGE_SIZE, NUM_HEADS, HEAD_SIZE], 63.0, dtype=torch.float16)

# Transfer to device with layout
if DEVICE.type == "spyre":
    key_cache_dev = key_cache.to(DEVICE, device_layout=cache_stl)
    value_cache_dev = value_cache.to(DEVICE, device_layout=cache_stl)
else:
    key_cache_dev = key_cache
    value_cache_dev = value_cache

print("paged memory created & transfered.")

# check layout
if not demo_on_cpu:
    actual_layout = key_cache_dev.device_tensor_layout()
    print("actual layout:", actual_layout)


#########################################
# prepare computation, calculate absolute page table

# base_addr_k = key_cache_dev.data_ptr()
# base_addr_v = value_cache_dev.data_ptr()
# print(f"base address of key cache: {base_addr_k}")
# print(f"base address of value cache: {base_addr_v}")
# -> apparently not the correct addresses right now...


# Now, we have an actual page _table_. Meaning, different requests with
#  different page indices. 
page_table_list = [
    [13, 312, 42, 0, 0],
    [32, 312, 42, 41, 471]
]
page_table_cpu = torch.tensor(page_table_list, dtype=torch.int64, device="cpu")
print(f"page table on CPU with indexes: {page_table_cpu}")


# abs_page_table_K = torch.empty(len(page_table_cpu), dtype=torch.int64, device="cpu")
# abs_page_table_V = torch.empty(len(page_table_cpu), dtype=torch.int64, device="cpu")
# 
# page_stride = cache_stl.stride_map[0]
# abs_page_table_K = page_table_cpu * page_stride + base_addr_k
# abs_page_table_V = page_table_cpu * page_stride + base_addr_v
# 
# print(f"page table for K with addr: {abs_page_table_K}")
# print(f"page table for V with addr: {abs_page_table_V}")
# 
# # transfer to device
# #  -> implicit cast to int32 right now...
# 
# abs_page_table_K_dev = abs_page_table_K.to(DEVICE)
# abs_page_table_V_dev = abs_page_table_V.to(DEVICE)


#########################################
# computation
#  (and yes, paged vector add is here just a placeholder for attention
#   ...but we want to seperate the infrastructure for paged access from
#   attention computation ops enablement....)

print("prepare computation...")


def paged_varlen_vector_add(
    a_pages, b_pages, a_page_table, b_page_table, 
    max_sequence_length, num_sequences, 
    out_pages
):
    for seq_idx in range(num_sequences):
        for i in range(max_sequence_length):
            # TODO: can we skip parts of the graph? 
            #  e.g. if we know that a_page_table[seq_idx][i] == 0?
            #  This would NOT be knowable at compile time, so can we have
            #  jumps in the graph?
            a_data_view = a_pages[a_page_table[seq_idx][i]]
            b_data_view = b_pages[b_page_table[seq_idx][i]]
            sum = a_data_view + b_data_view

            # output pages are starting from 0, in this case
            out_page = out_pages[seq_idx][i]
            out_page.copy_(sum)




def create_compilable_paged_vector_add(compiled_max_sequence_length, compiled_num_sequences):
    def compileable_paged_vector_add(a_pages, b_pages, a_page_table, b_page_table, 
                   out_pages):
        assert len(a_page_table) == compiled_num_sequences
        assert len(b_page_table) == compiled_num_sequences
        assert len(a_page_table[0]) == compiled_max_sequence_length
        assert len(b_page_table[0]) == compiled_max_sequence_length

        return paged_varlen_vector_add(
            a_pages, b_pages, a_page_table, b_page_table,
            compiled_max_sequence_length, compiled_num_sequences,
            out_pages
        )

    return compileable_paged_vector_add 


list_of_compiled_functions_per_request_length = {}

# yes, we will maintain a list of compiled functions per padded request length
#  (very similar to CUDA graph handling of vLLM)
page_table_length = len(page_table_cpu[0])
max_num_sequences = len(page_table_cpu)
fn_key = f"ql{page_table_length}_bs{max_num_sequences}"
list_of_compiled_functions_per_request_length[fn_key] = torch.compile(
    create_compilable_paged_vector_add(page_table_length, max_num_sequences)
)
#  -> cannot be compiled right now...

# do the computation (that is where it breaks right now)
print("computing varlen paged attention...")

out_pages_dev = torch.empty([max_num_sequences, page_table_length, PAGE_SIZE, NUM_HEADS, HEAD_SIZE], device=DEVICE, dtype=torch.float16)

# list_of_compiled_functions_per_request_length[page_table_length](
#     abs_page_table_K_dev, abs_page_table_V_dev, out_pages_dev
# )
list_of_compiled_functions_per_request_length[fn_key](
    key_cache, value_cache, page_table_cpu, page_table_cpu,
    out_pages_dev
)


print("result...")

# print result, should have the following contents:
#  0: 3.0
#  1: 32.0
#  2: 42.0
#  3: 3.0
#  4: 3.0
print(out_pages_dev.cpu())
