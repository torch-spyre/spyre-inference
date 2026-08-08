
import torch
import os

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

DEVICE = torch.device("spyre")

paged_memory = torch.ones([512, 16], dtype=torch.float16)
paged_memory[42] = torch.full([16], 2.0, dtype=torch.float16)
paged_memory[321] = torch.full([16], 3.0, dtype=torch.float16)

print(f"paged_memory strides: {paged_memory.stride()}")

paged_memory_device = paged_memory.to(DEVICE)

base_addr = paged_memory_device.data_ptr()
print(f"base_addr: {base_addr}")


# local_block_table = torch.tensor([42, 321], dtype=torch.int32, device="cpu")
# local_block_table = torch.tensor([42, 321], dtype=torch.int64)
local_block_table = torch.tensor([42, 321], dtype=torch.int64, device="cpu")
# local_block_table = torch.tensor([42, 321], dtype=torch.float16)
print(f"local_block_table: {local_block_table}")


################################################
# TRYING to compute absolute block table on spyre
# not working for now 
#
# device_ltb = local_block_table.to(DEVICE)
# 
# absolute_block_table = torch.zeros_like(local_block_table, dtype=torch.float16, device=DEVICE)
# base_addr_as_vector = torch.full([2], float(base_addr), dtype=torch.float16, device=DEVICE)
# 
# # create_block_table_on_device = torch.compile(lambda bt, ba: bt + ba)
# 
# def create_block_table_helper(res, bt, ba):
#     res += bt + ba
# 
# create_block_table_on_device = torch.compile(create_block_table_helper)

# absolute_block_table = create_block_table_on_device(device_ltb, base_addr)
# absolute_block_table = create_block_table_on_device(device_ltb, base_addr_as_vector)
# create_block_table_on_device(absolute_block_table, device_ltb, base_addr_as_vector)


# instead, calculating on CPU and then move 
################################################

abs_bt = torch.full([2], 0.0, dtype=torch.int64)
abs_bt = local_block_table * paged_memory.stride()[0] + base_addr
print(f"abs_bt: {abs_bt}")
absolute_block_table = abs_bt.to(torch.int64).to(DEVICE)

print("absolute_block_table: ")
print(absolute_block_table.cpu())
# 
# def paged_add(addr_block_table, data):
#     res = torch.zeors([16], dtype=torch.float16, device=DEVICE)
#     for page_idx, page_addr in enumerate(addr_block_table):
#         res += torch.add(data, paged_memory_device[page_addr])
#     return res
# 
# paged_add_compield = torch.compile(paged_add)
# 
# result_spyre = paged_add_compield(absolute_block_table, paged_memory_device)
# 
# print(result_spyre.cpu())

print("done")
