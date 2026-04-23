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

import torch
import os
import debugpy

# debug = True
debug = False

if debug:
    host_addr = os.environ.get("TORCH_SPYRE_DEBUG_ADDR", "0.0.0.0")
    pdb_port = int(os.environ.get("TORCH_SPYRE_DEBUG_PORT", "5679"))
    debugpy.listen((host_addr, pdb_port))
    print(f"[debugpy] listening at {host_addr}:{pdb_port}; wait for client...\n")
    debugpy.wait_for_client()
    print("[debugpy] connected")

DEVICE = torch.device("spyre")

def create_paged_memory(
    page_size: int, number_of_pages: int, fill_value: float = 0.0, dtype=torch.float16
):
    list_of_pages = []
    for i in range(number_of_pages):
        new_page = torch.full([page_size], fill_value, dtype=dtype)
        new_page_device = new_page.to(DEVICE)
        list_of_pages.append(new_page_device)
    return list_of_pages


print("create paged memory...")
PAGE_SIZE = 16
# create paged memory 
a_pages = create_paged_memory(PAGE_SIZE, 512, 1.0)
b_pages = create_paged_memory(PAGE_SIZE, 512, 2.0)

# test output also as paged
out_pages = create_paged_memory(PAGE_SIZE, 256)


# manipulate individual pages 

a_pages[42].fill_(40.0)
b_pages[312].fill_(31.0)


print("prepare computation...")
# create page table 
page_table = [13, 312, 42, 14, 15]

def paged_vector_add(a_pages, b_pages, page_table, out_pages):
    # output pages are starting from 0, in this case
    for i, page_index in enumerate(page_table):
        a_page = a_pages[page_index]
        b_page = b_pages[page_index]
        out_page = out_pages[i]
        sum = a_page + b_page
        out_page.copy_(sum)


def create_compilable_paged_vector_add(page_table):
    def paged_vector_add_with_fixed_table(a_pages, b_pages, out_pages):
        return paged_vector_add(a_pages, b_pages, page_table, out_pages)
    return paged_vector_add_with_fixed_table


list_of_compiled_functions_per_request_length = {}

list_of_compiled_functions_per_request_length[5] = torch.compile(
    create_compilable_paged_vector_add(page_table)
)


# do the computation 
print("computing paged vector add...")

list_of_compiled_functions_per_request_length[5](a_pages, b_pages, out_pages)

print("result...")

# print result, should have the following pages
#  0: 3.0
#  1: 32.0
#  2: 42.0
#  3: 3.0
#  4: 3.0

out_pages_cpu = [p.cpu() for p in out_pages]
for i in range(len(page_table)):
    print(f"out page {i}: {out_pages_cpu[i].tolist()}")
    # print(
    #     f"  dtype: {out_pages_cpu[i].dtype}, shape: {out_pages_cpu[i].shape}, "
    #     f"device: {out_pages_cpu[i].device}"
    # )


