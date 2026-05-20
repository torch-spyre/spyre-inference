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

"""
pastamachine.transpile
~~~~~~~~~~~~~~~~~~~~~~

Convert a Helion kernel into a Spyre-compiled callable.

Public API:
    compile_helion_to_spyre(helion_kernel, example_spyre_inputs, do_verify_run_on_cpu=False)
"""

import inspect
import logging
import math
import os

import torch
from torch._ops import OpOverload

from pastamachine._logging import getLogger
from pastamachine.util import TranspileMeta, _capture_sdsc_paths
from pastamachine.verify import verify_on_cpu

log = getLogger("transpile")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_fx_graph_and_extract_nodes(graph, graph_name="FX Graph", detailed=True,
                                     level=logging.INFO):
    """Print FX graph information in a structured format and return node list."""
    log.log(level, "%s", "=" * 60)
    log.log(level, "%s", graph_name)
    log.log(level, "%s", "=" * 60)

    nodes_list = list(graph.nodes)
    log.log(level, "Number of nodes: %d", len(nodes_list))

    if detailed:
        log.log(level, "Node Details:")
        for i, node in enumerate(nodes_list):
            log.log(level, "  [%d] %s", i, node.name)
            log.log(level, "      Op: %s", node.op)
            log.log(level, "      Target: %s", node.target)
            log.log(level, "      Target type: %s", type(node.target))
            if node.args:
                log.log(level, "      Args: %s", node.args)
            if node.kwargs:
                log.log(level, "      Kwargs: %s", node.kwargs)
            log.log(level, "      Users: %d", len(node.users))

    log.log(level, "Graph Structure:")
    log.log(level, "%s", graph)
    log.log(level, "%s", "=" * 60)

    return nodes_list


# ---------------------------------------------------------------------------
# Helpers for Helion IR → aten conversion
# ---------------------------------------------------------------------------

def _resolve_constant_symnode(node, kernel_params):
    """Resolve a non-block-size ``_get_symnode`` node to its concrete value.

    Tries FX node metadata first (only concrete numerics, not SymInt/SymFloat),
    then matches the debug name to a kernel parameter, and finally falls back
    to the first non-tensor parameter with a default value.
    """
    _log = getLogger("transpile")
    debug_name = str(node.args[0]) if node.args else ""
    _log.debug("  _resolve_constant_symnode: node=%s, debug_name=%r, "
               "meta keys=%s", node.name, debug_name, list(node.meta.keys()))

    # 1. Try FX node metadata — only accept concrete numeric values
    #    (SymInt / SymFloat are symbolic and print as variable names)
    for key in ('val', 'example_value'):
        val = node.meta.get(key)
        _log.debug("    meta[%r] = %r (type=%s)", key, val, type(val).__name__)
        if isinstance(val, (int, float)):
            _log.debug("    -> resolved from meta[%r]: %s", key, val)
            return val

    # 2. Try matching the debug_name to a kernel parameter name
    if debug_name in kernel_params:
        param = kernel_params[debug_name]
        _log.debug("    debug_name %r matches param: annotation=%s, default=%r",
                   debug_name, param.annotation, param.default)
        if param.default is not inspect.Parameter.empty:
            _log.debug("    -> resolved from param default: %s", param.default)
            return param.default

    # 3. Fallback: first non-tensor parameter with a default value
    for pname, param in kernel_params.items():
        ann = param.annotation
        _log.debug("    fallback check param %r: annotation=%s, default=%r",
                   pname, ann, param.default)
        if ann is not inspect.Parameter.empty and ann is not torch.Tensor:
            if param.default is not inspect.Parameter.empty:
                _log.debug("    -> resolved from fallback param %r: %s",
                           pname, param.default)
                return param.default

    _log.warning("  _resolve_constant_symnode: could not resolve %s", node.name)
    return None


def _subscript_to_unsqueeze_dim(node):
    """Determine the unsqueeze dimension from a Helion ``subscript`` node.

    A subscript like ``tensor[:, None]`` is represented with a tuple arg
    containing ``None`` at the position of the new axis.
    """
    for arg in node.args:
        if isinstance(arg, (tuple, list)):
            for i, elem in enumerate(arg):
                if elem is None:
                    return i
    return -1  # conservative default


# ---------------------------------------------------------------------------
# Graph transformations
# ---------------------------------------------------------------------------

def _get_node_dtype(node):
    """Extract the tensor dtype from a node's metadata."""
    meta = node.meta.get('val') or node.meta.get('example_value')
    if meta is not None and hasattr(meta, 'dtype'):
        return meta.dtype
    tm = node.meta.get('tensor_meta')
    if tm is not None and hasattr(tm, 'dtype'):
        return tm.dtype
    return None


# Reduction ops whose inductor lowering respects the dtype kwarg.
# Pinning dtype to the input tensor's dtype prevents _make_reduction_inner
# from promoting to float32.
_PINNABLE_REDUCTION_OPS = {
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
    torch.ops.aten.prod.dim_int,
}


def prevent_reduction_upcasts(graph_module):
    """Prevent inductor from creating float32 Reduction IR nodes for float16 inputs.

    **Pin dtype** on reduction ops.  The inductor's ``_make_reduction_inner``
    respects an explicit ``dtype`` kwarg — by setting it to the input
    tensor's dtype we prevent automatic promotion.

    Note: ``aten.mean.dim`` is **not** decomposed here — the Spyre-native
    ``lower_mean`` lowering (activated by ``enable_spyre_context``) handles
    it correctly without upcasting.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The FX graph to transform (modified in-place).

    Returns
    -------
    bool
        Whether any nodes were modified.
    """
    _log = getLogger("transpile")
    graph = graph_module.graph
    modified = False

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        # Pin dtype on reduction ops to prevent automatic float32 promotion
        if node.target in _PINNABLE_REDUCTION_OPS:
            if node.kwargs.get('dtype') is not None:
                continue  # already pinned

            # Determine the input tensor's dtype
            x_node = node.args[0]
            input_dtype = _get_node_dtype(x_node)
            if input_dtype is None:
                _log.debug("Cannot pin dtype for %s: missing dtype metadata", node.name)
                continue

            node.kwargs = {**node.kwargs, 'dtype': input_dtype}
            modified = True
            _log.info("Pinned reduction %s (%s) to dtype=%s",
                      node.name, node.target, input_dtype)

    if modified:
        graph.lint()
        graph_module.recompile()

    return modified


# .Tensor ops that may have a raw Python scalar where a tensor is expected.
# Convert these to .Scalar overloads which the Spyre backend handles natively.
# (full_like / constant tensors are NOT supported by the Spyre SDSC compiler.)
_TENSOR_TO_SCALAR_OPS = {
    torch.ops.aten.add.Tensor: torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor: torch.ops.aten.sub.Scalar,
    torch.ops.aten.mul.Tensor: torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor: torch.ops.aten.div.Scalar,
}


def fix_scalar_args_for_spyre(graph_module):
    """Convert ``.Tensor`` ops with raw scalar arguments to ``.Scalar`` ops.

    The transpiler or upstream passes sometimes emit ``.Tensor`` overloads
    with a raw Python number as the second argument (e.g.
    ``aten.add.Tensor(tensor, 1e-5)``).  The Spyre backend cannot handle
    these — it expects either proper tensor arguments or ``.Scalar``
    overloads.  This pass switches to the ``.Scalar`` overload, which the
    Spyre backend handles natively.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The FX graph to transform (modified in-place).

    Returns
    -------
    bool
        Whether any nodes were modified.
    """
    _log = getLogger("transpile")
    graph = graph_module.graph
    modified = False

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _TENSOR_TO_SCALAR_OPS:
            continue

        # Only convert if the second arg is a raw scalar, not an FX Node
        if len(node.args) < 2:
            continue
        scalar_arg = node.args[1]
        if isinstance(scalar_arg, torch.fx.Node):
            continue  # both args are tensors — leave as .Tensor op
        if not isinstance(scalar_arg, (int, float)) or isinstance(scalar_arg, bool):
            continue

        scalar_target = _TENSOR_TO_SCALAR_OPS[node.target]
        old_name = node.name
        old_target = node.target
        node.target = scalar_target
        modified = True
        _log.info("Converted %s: %s → %s (scalar arg: %s)",
                  old_name, old_target, scalar_target, scalar_arg)

    if modified:
        graph.lint()
        graph_module.recompile()

    return modified


# ---------------------------------------------------------------------------
# Step 1: Helion kernel  →  FX GraphModule
# ---------------------------------------------------------------------------

def transpile_fx_graphs(helion_kernel, example_cpu_inputs):
    """Transpiles a helion FX graph to a spyre compatible FX graph.

    This happens via binding a *helion_kernel* with CPU tensors and convert its device IR into a
    plain ``torch.fx.GraphModule`` containing only ``aten`` ops while taking care of spyre-specific dtypes and operations.

    Parameters
    ----------
    helion_kernel : helion.kernel
        The decorated Helion kernel function.
    example_cpu_inputs : tuple[torch.Tensor, ...]
        CPU tensors whose shapes / dtypes match the kernel signature.

    Returns
    -------
    tuple[torch.fx.GraphModule, list[torch.fx.Node]]
        A new GraphModule with aten ops and ``aten.copy_`` write-backs,
        and a list of FX nodes representing Helion block-size parameters.
    """
    # Bind to trigger Helion compilation
    bound = helion_kernel.bind(example_cpu_inputs)
    log.info("Helion kernel bound successfully")

    device_ir = bound.host_function.device_ir

    log.info("=== Helion Device IR Analysis ===")
    log.info("Number of graphs in device_ir: %d", len(device_ir.graphs))
    log.info("Root graph IDs: %s", device_ir.root_ids)

    root_graph_id = device_ir.root_ids[0]

    helion_nodes_lists = {}
    for gid, ginfo in enumerate(device_ir.graphs):
        graph_name = f"Helion Graph {gid}: {ginfo.name} ({type(ginfo).__name__})"
        level = logging.INFO if gid == root_graph_id else logging.DEBUG
        helion_nodes_lists[gid] = print_fx_graph_and_extract_nodes(
            ginfo.graph, graph_name, detailed=True, level=level
        )
    root_graph_info = device_ir.graphs[root_graph_id]
    helion_root_nodes = helion_nodes_lists[root_graph_id]

    log.info("=== Selected Helion Root Graph (ID: %d) ===", root_graph_id)
    log.info("Graph name: %s", root_graph_info.name)
    log.info("Number of nodes: %d", len(helion_root_nodes))

    # -- Build new Spyre-compatible FX graph --------------------------------
    log.info("=== Build Spyre-Compatible FX Graph ===")

    new_graph = torch.fx.Graph()

    node_mapping = {}                # old node → new node
    arg_nodes_needing_placeholders = set()
    block_size_nodes = []
    store_ops = []                   # (dst_node, value_node)
    host_tensor_to_param = {}        # _host_tensor node → param name
    constant_symnode_values = {}     # non-block-size _get_symnode node → value
    extra_lowering_inputs = {}       # _inductor_lowering_extra node → input node
    view_op_nodes = {}               # subscript/view node → input tensor node
    kernel_params = inspect.signature(helion_kernel).parameters

    # First pass: identify OverloadOp nodes, stores, and arguments
    log.info("=== First Pass: Identify OverloadOp nodes, stores, and arguments ===")
    overload_nodes = []
    for i, node in enumerate(helion_root_nodes):
        if node.op == "call_function" and isinstance(node.target, OpOverload):
            overload_nodes.append(node)
            log.info("  Found OverloadOp [%d]: %s -> %s", i, node.name, node.target)

            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    arg_nodes_needing_placeholders.add(arg)
                    log.info("    Arg node: %s", arg.name)

            for v in node.kwargs.values():
                if isinstance(v, torch.fx.Node):
                    arg_nodes_needing_placeholders.add(v)
                    log.info("    Kwarg node: %s", v.name)

        elif node.op == "call_function":
            target_name = getattr(node.target, '__qualname__',
                                  getattr(node.target, '__name__', ''))
            target_module = getattr(node.target, '__module__', '')
            target_repr = str(node.target)

            is_store = ('store' in target_name or 'store' in target_repr)
            is_helion = ('helion' in target_module or 'helion' in target_repr)

            if '_host_tensor' in target_name or '_host_tensor' in target_repr:
                host_tensor_to_param[node] = node.name
                log.info("  Found _host_tensor [%d]: %s -> param '%s'", i, node.name, node.name)

            elif is_store and is_helion:
                dst_node = node.args[0] if len(node.args) > 0 else None
                val_node = node.args[2] if len(node.args) > 2 else None
                if isinstance(dst_node, torch.fx.Node) and isinstance(val_node, torch.fx.Node):
                    store_ops.append((dst_node, val_node))
                    log.info("  Found store [%d]: %s -> store(%s, ..., %s, ...)",
                             i, node.name, dst_node.name, val_node.name)
                else:
                    log.info("  Found store [%d] but args not Nodes: args=%s",
                             i, [type(a).__name__ for a in node.args])

            elif is_helion:
                if '_get_symnode' in target_name:
                    debug_name = node.args[0] if node.args else ""
                    if isinstance(debug_name, str) and debug_name.startswith("block_size_"):
                        log.info("  Found block size node [%d]: %s -> %s (debug=%s)",
                                 i, node.name, target_name, debug_name)
                    else:
                        # Non-block-size symnode (e.g. eps) → resolve as constant
                        val = _resolve_constant_symnode(node, kernel_params)
                        constant_symnode_values[node] = val
                        log.info("  Found constant symnode [%d]: %s = %s (debug=%s)",
                                 i, node.name, val, debug_name)

                elif '_inductor_lowering_extra' in target_name:
                    # Reduction helper — input is the most recent OverloadOp
                    # (Helion IR uses positional convention, not explicit args)
                    preceding_overload = overload_nodes[-1] if overload_nodes else None
                    extra_lowering_inputs[node] = preceding_overload
                    log.info("  Found _inductor_lowering_extra [%d]: %s -> input=%s",
                             i, node.name,
                             preceding_overload.name if preceding_overload else "None")

                elif 'subscript' in target_name:
                    # View op (e.g. tensor[:, None]) → will convert to aten.unsqueeze
                    input_node = None
                    for a in node.args:
                        if isinstance(a, torch.fx.Node):
                            input_node = a
                            break
                    view_op_nodes[node] = input_node
                    log.info("  Found view op [%d]: %s -> input=%s, args=%s",
                             i, node.name,
                             input_node.name if input_node else "None",
                             [str(a) if isinstance(a, torch.fx.Node) else repr(a)
                              for a in node.args])

                else:
                    log.info("  Skipped helion node [%d]: %s -> name=%r module=%r",
                             i, node.name, target_name, target_module)
            else:
                log.info("  Skipped unknown call_function [%d]: %s -> type=%s, repr=%s",
                         i, node.name, type(node.target).__name__, target_repr[:80])

    log.info("Found %d OverloadOp nodes", len(overload_nodes))
    log.info("Found %d store operations", len(store_ops))
    log.info("Found %d unique argument nodes needing placeholders", len(arg_nodes_needing_placeholders))

    # Filter pass: Remove nested placeholders (outputs of other OverloadOp nodes)
    log.info("=== Filter Pass: Remove nested placeholders ===")
    overload_node_set = set(overload_nodes)
    filtered_arg_nodes = set()

    for arg_node in arg_nodes_needing_placeholders:
        if arg_node in overload_node_set:
            log.info("  Removing nested placeholder: %s (is output of OverloadOp)", arg_node.name)
        elif arg_node in constant_symnode_values:
            log.info("  Removing constant symnode: %s (value=%s)",
                     arg_node.name, constant_symnode_values[arg_node])
        elif arg_node in view_op_nodes:
            log.info("  Removing view op: %s (will be converted to aten)", arg_node.name)
        else:
            filtered_arg_nodes.add(arg_node)
            log.info("  Keeping placeholder: %s", arg_node.name)

    log.info("Filtered from %d to %d placeholder nodes",
             len(arg_nodes_needing_placeholders), len(filtered_arg_nodes))
    arg_nodes_needing_placeholders = filtered_arg_nodes

    # Second pass: Create placeholders ordered by kernel signature
    log.info("=== Second Pass: Create placeholders for argument nodes ===")

    kernel_param_names = list(inspect.signature(helion_kernel).parameters.keys())
    param_order = {name: i for i, name in enumerate(kernel_param_names)}
    log.info("Kernel param order: %s", kernel_param_names)

    load_to_param = {}  # load node → param name

    def _param_index(load_node):
        if load_node.op == 'call_function' and len(load_node.args) > 0:
            host_tensor = load_node.args[0]
            if isinstance(host_tensor, torch.fx.Node) and host_tensor in host_tensor_to_param:
                pname = host_tensor_to_param[host_tensor]
                load_to_param[load_node] = pname
                idx = param_order.get(pname, len(kernel_param_names))
                log.info("  %s loads from _host_tensor '%s' (index %d)", load_node.name, pname, idx)
                return idx
        return len(kernel_param_names)

    arg_nodes_list = sorted(arg_nodes_needing_placeholders, key=_param_index)
    for arg_node in arg_nodes_list:
        placeholder_name = f"L_{arg_node.name}_"
        new_placeholder = new_graph.placeholder(placeholder_name)
        node_mapping[arg_node] = new_placeholder
        log.info("  Created placeholder: %s for %s", placeholder_name, arg_node.name)

    # Third pass: Create nodes in topological order (OverloadOps + view ops)
    log.info("=== Third Pass: Create nodes with mapped arguments ===")
    view_op_set = set(view_op_nodes.keys())

    for node in helion_root_nodes:
        # --- View ops (subscript → aten.unsqueeze) ---
        if node in view_op_set:
            input_node = view_op_nodes[node]
            if input_node is not None and input_node in node_mapping:
                unsqueeze_dim = _subscript_to_unsqueeze_dim(node)
                new_node = new_graph.call_function(
                    torch.ops.aten.unsqueeze.default,
                    args=(node_mapping[input_node], unsqueeze_dim),
                )
                node_mapping[node] = new_node
                log.info("  Created unsqueeze for %s: input=%s, dim=%d -> %s",
                         node.name, input_node.name, unsqueeze_dim, new_node.name)
            else:
                log.warning("  Could not convert view op %s: input %s not in mapping",
                            node.name,
                            input_node.name if input_node else "None")
            continue

        # --- OverloadOp nodes ---
        if not (node.op == "call_function" and isinstance(node.target, OpOverload)):
            continue

        log.info("  Processing OverloadOp: %s", node.name)
        log.info("    Target: %s", node.target)

        # Resolve _extra_args → actual input tensor (for reductions like mean)
        extra_input_mapped = None
        extra_nodes = node.kwargs.get('_extra_args', [])
        if isinstance(extra_nodes, (list, tuple)):
            for en in extra_nodes:
                if isinstance(en, torch.fx.Node) and en in extra_lowering_inputs:
                    src = extra_lowering_inputs[en]
                    if src is not None and src in node_mapping:
                        extra_input_mapped = node_mapping[src]
                        log.info("    Resolved _extra_args: %s -> %s -> %s",
                                 en.name, src.name, extra_input_mapped.name)
                        break

        new_args = []
        for j, arg in enumerate(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg in node_mapping:
                    new_args.append(node_mapping[arg])
                    log.info("      Mapped arg %s -> %s", arg.name, node_mapping[arg].name)
                elif arg in constant_symnode_values:
                    new_args.append(constant_symnode_values[arg])
                    log.info("      Resolved constant arg %s -> %s",
                             arg.name, constant_symnode_values[arg])
                else:
                    log.warning("      Arg %s not in mapping, using as-is", arg.name)
                    new_args.append(arg)
            elif arg is None and j == 0 and extra_input_mapped is not None:
                # Reduction op: replace None first arg with _extra_args input
                new_args.append(extra_input_mapped)
                log.info("      Replaced None arg[0] with _extra_args input: %s",
                         extra_input_mapped.name)
            elif arg is None and new_args:
                # Binary self-op (e.g. x * x): None means "same as first operand"
                first_tensor = None
                for prev in new_args:
                    if isinstance(prev, torch.fx.Node):
                        first_tensor = prev
                        break
                if first_tensor is not None:
                    new_args.append(first_tensor)
                    log.info("      Replaced None arg[%d] with first tensor arg: %s",
                             j, first_tensor.name)
                else:
                    new_args.append(arg)
                    log.warning("      Unresolved None arg[%d]", j)
            else:
                new_args.append(arg)
                log.info("      Kept constant arg: %s", arg)

        new_kwargs = {}
        for k, v in node.kwargs.items():
            if k == '_extra_args':
                log.info("      Dropped Helion-internal kwarg: %s", k)
                continue
            if isinstance(v, torch.fx.Node):
                if v in node_mapping:
                    new_kwargs[k] = node_mapping[v]
                    log.info("      Mapped kwarg %s: %s -> %s", k, v.name, node_mapping[v].name)
                else:
                    log.warning("      Kwarg %s node %s not in mapping", k, v.name)
                    new_kwargs[k] = v
            else:
                new_kwargs[k] = v
                log.info("      Kept constant kwarg %s: %s", k, v)

        new_node = new_graph.call_function(
            node.target,
            args=tuple(new_args),
            kwargs=new_kwargs
        )
        node_mapping[node] = new_node
        log.info("    -> Created node: %s with target %s", new_node.name, new_node.target)

    # Fourth pass: Add inplace write-backs from store ops and create output node
    log.info("=== Fourth Pass: Add store write-backs and create output ===")

    param_to_placeholder = {}
    for load_node, pname in load_to_param.items():
        if load_node in node_mapping:
            param_to_placeholder[pname] = node_mapping[load_node]
    log.info("  Param -> placeholder: {%s}",
             ", ".join(f"{k!r}: {v.name}" for k, v in param_to_placeholder.items()))

    inplace_stores = []   # successful write-backs to input buffers
    output_values = []    # store values whose dst is output-only (no placeholder)

    for store_dst, store_val in store_ops:
        dst_param = host_tensor_to_param.get(store_dst)
        dst_new = param_to_placeholder.get(dst_param) if dst_param else None
        val_new = node_mapping.get(store_val)
        if dst_new is not None and val_new is not None:
            new_graph.call_function(
                torch.ops.aten.copy_.default,
                args=(dst_new, val_new),
            )
            inplace_stores.append(dst_param)
            log.info("  Added write-back: copy_(%s, %s)  [param '%s']",
                     dst_new.name, val_new.name, dst_param)
        elif val_new is not None:
            # Store to an output-only tensor (e.g. out = empty_like(x))
            # — this is a return value, not an inplace write-back.
            output_values.append(val_new)
            log.info("  Store to output-only tensor '%s': will return %s",
                     dst_param, val_new.name)
        else:
            log.warning("  Could not map store(%s, ..., %s)"
                        " -- dst_param=%r, dst=%s, val=%s",
                        store_dst.name, store_val.name, dst_param,
                        "found" if dst_new else "MISSING",
                        "found" if val_new else "MISSING")

    if output_values:
        # Kernel creates and returns new tensor(s) via stores to output-only
        # buffers.  Return the last such value.
        new_graph.output(output_values[-1])
        log.info("  Created output node returning %s (non-inplace kernel)",
                 output_values[-1].name)
    elif inplace_stores:
        # Pure inplace kernel: the copy_ nodes are side-effects that write
        # directly into the caller's buffers; no return value needed.
        new_graph.output(None)
        log.info("  Created output node (None -- inplace kernel)")
    else:
        # No stores at all: return the last computed OverloadOp result.
        last_value = node_mapping[overload_nodes[-1]]
        new_graph.output(last_value)
        log.info("  Created output node returning %s", last_value.name)

    # -- Track block size nodes and which ops they affect ----------------------
    log.info("=== Tracking block size nodes ===")

    # 1. Find _get_symnode nodes and extract block_size index from debug_name.
    #    E.g. _get_symnode("block_size_0") → index 0.
    block_size_by_index = {}   # block_size_index → helion FX node
    for node in helion_root_nodes:
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, '__name__', '')
        if '_get_symnode' not in target_name:
            continue
        debug_name = node.args[0] if node.args else ""
        if isinstance(debug_name, str) and debug_name.startswith("block_size_"):
            try:
                idx = int(debug_name.split("_")[-1])
            except ValueError:
                continue
            block_size_by_index[idx] = node
            block_size_nodes.append(node)
            log.info("  Tracked %s → block_size index %d", node.name, idx)

    # 2. For each overload (aten) op, walk its helion-graph args to find
    #    loads that reference block_size nodes.  A load's args look like:
    #        load(%tensor, [%block_size_0, …], None, None)
    #    The position in the list tells us the host dimension.
    #
    #    Result: affected_ops maps  transpiled-node-name →
    #                               {host_dim_index: block_size_index}
    affected_ops: dict[str, dict[int, int]] = {}
    for overload_node in overload_nodes:
        if overload_node not in node_mapping:
            continue
        transpiled_name = node_mapping[overload_node].name

        dims_for_node: dict[int, int] = {}
        visited: set[torch.fx.Node] = set()
        stack = [a for a in overload_node.args if isinstance(a, torch.fx.Node)]
        while stack:
            arg = stack.pop()
            if arg in visited:
                continue
            visited.add(arg)
            # Check if this is a load with block_size references
            arg_target = getattr(arg.target, '__name__', '') if arg.op == "call_function" else ''
            if 'load' in arg_target and len(arg.args) > 1:
                bs_list = arg.args[1]
                if isinstance(bs_list, (list, tuple)):
                    for dim, bs_node in enumerate(bs_list):
                        if isinstance(bs_node, torch.fx.Node) and bs_node in block_size_by_index.values():
                            # Reverse-lookup the index for this node
                            for bs_idx, bs_n in block_size_by_index.items():
                                if bs_n is bs_node:
                                    dims_for_node[dim] = bs_idx
                                    break
            # Keep walking transitive inputs
            for a in arg.args:
                if isinstance(a, torch.fx.Node):
                    stack.append(a)

        if dims_for_node:
            affected_ops[transpiled_name] = dims_for_node
            log.info("  %s affected by block_sizes: %s", transpiled_name, dims_for_node)

    graph_module = torch.fx.GraphModule({}, new_graph)

    return graph_module, block_size_nodes, affected_ops


# ---------------------------------------------------------------------------
# Config-aware core division
# ---------------------------------------------------------------------------

def _make_config_aware_core_division(node_block_sizes):
    """Return a replacement for ``core_division_planning`` that uses
    block sizes from a Helion config to drive core division for affected nodes.

    Parameters
    ----------
    node_block_sizes : dict[str, dict[int, int]]
        Merged mapping built by the caller:
        ``{transpiled_fx_node_name: {host_dim: block_size_value}}``.
        Only nodes present in this dict get config-driven division;
        all other nodes fall through to the original core_division logic.

    The returned function replaces ``core_division_planning`` and sets
    ``num_cores = host_dim_size / block_size`` (capped at *max_cores*)
    for each tiled dimension of affected nodes.
    """
    from torch._inductor.ir import (
        ComputedBuffer,
        FallbackKernel,
        MultiOutput,
        Pointwise,
        Reduction,
    )
    from torch._inductor.scheduler import (
        BaseSchedulerNode,
        ExternKernelSchedulerNode,
        NopKernelSchedulerNode,
        SchedulerNode,
    )

    from torch_spyre._inductor.ir import FixedTiledLayout
    from torch_spyre._inductor.pass_utils import get_mem_deps
    from torch_spyre._inductor.errors import Unsupported
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP
    
    # Helper functions that were removed from torch_spyre - implement locally
    def no_division(args, output):
        """Initialize core division structure with no splits (all 1s)."""
        num_args = len(args)
        # Return list of dicts, one per arg + output
        return [{} for _ in range(num_args + 1)]
    
    def map_host_dim_to_device_dim(layout: FixedTiledLayout, host_dim: int) -> int:
        """Map host dimension index to device dimension index.
        
        For FixedTiledLayout, the device_layout contains the mapping.
        This is a simplified implementation that assumes 1:1 mapping.
        """
        # In the new structure, we need to map through the device_layout
        # For now, use a simple 1:1 mapping as a fallback
        return host_dim
    
    def divide_pointwise_op(op, args, max_cores, pass_fn=None):
        """Wrapper for pointwise op division."""
        if pass_fn is not None:
            pass_fn(op, args, max_cores)
        else:
            # Default behavior: no division
            op.spyre_core_division = no_division(args, op.node.get_layout())
            op.n_cores_used = 1
    
    def divide_reduction_op(op, args, max_cores, pass_fn=None):
        """Wrapper for reduction op division."""
        if pass_fn is not None:
            pass_fn(op, args, max_cores)
        else:
            # Default behavior: no division
            op.spyre_core_division = no_division(args, op.node.get_layout())
            op.n_cores_used = 1

    max_cores = int(os.getenv("SENCORES", "32"))
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")

    log.info("node_block_sizes: %s", node_block_sizes)

    def _get_dim_to_bs(n) -> dict[int, int] | None:
        """Check if scheduler node *n* is affected by any block_size.

        Returns the ``{host_dim: block_size_value}`` dict if affected,
        or ``None`` if the node should use the default division.
        """
        origins = n.node.get_origins() if hasattr(n.node, 'get_origins') else set()
        for origin_fx_node in origins:
            bs = node_block_sizes.get(origin_fx_node.name)
            if bs is not None:
                return bs
        log.debug("scheduler node not matched to any affected op (origins: %s)",
                  [o.name for o in origins])
        return None

    def _compute_split(host_dim_size, block_size, remaining_cores):
        """Compute cores for one dimension: host_dim_size / block_size."""
        if block_size <= 0 or host_dim_size % block_size != 0:
            log.warning(
                "block_size %d does not evenly divide host dim size %d; "
                "skipping this dimension",
                block_size, host_dim_size,
            )
            return 1
        return min(host_dim_size // block_size, remaining_cores)

    def _divide_pointwise_with_config(n, args, max_cores, dim_to_bs):
        """Pointwise core division driven by helion block_sizes."""
        output: FixedTiledLayout = n.node.get_layout()
        n.spyre_core_division = no_division(args, output)
        n.n_cores_used = 1

        if max_cores == 1:
            return

        if len(n.node.get_outputs()) > 2:
            return

        for a in args:
            if a.layout.size != output.size:
                return

        host_sizes = output.size

        # Compute splits for each tiled dimension
        splits = {}  # host_dim_idx → num_cores
        remaining_cores = max_cores
        for host_dim, bs in dim_to_bs.items():
            if host_dim >= len(host_sizes):
                continue
            host_dim_size = int(host_sizes[host_dim])
            cores = _compute_split(host_dim_size, bs, remaining_cores)
            if cores > 1:
                splits[host_dim] = cores
                remaining_cores //= cores

        if not splits:
            return

        n.n_cores_used = math.prod(splits.values())

        for host_dim, cores in splits.items():
            dev_dim = map_host_dim_to_device_dim(output, host_dim)
            for cd in n.spyre_core_division:
                cd[dev_dim] = cores

    def _divide_reduction_with_config(n, args, max_cores, dim_to_bs):
        """Reduction core division driven by helion block_sizes."""
        red: Reduction = n.node.data
        output = n.node.get_layout()
        n.spyre_core_division = no_division(args, output)
        n.n_cores_used = 1

        if max_cores == 1:
            return

        if red.reduction_type == BATCH_MATMUL_OP:
            assert len(args) == 2, "matmul has exactly 2 input args"
            num_dims = len(args[0].layout.size)

            if num_dims == 2:
                M = args[0].layout.size[0]
                N = args[1].layout.size[1]

                op_dims = [("M", M), ("N", N)]
                splits = {}
                remaining_cores = max_cores
                for host_dim, (name, dim_size) in enumerate(op_dims):
                    bs = dim_to_bs.get(host_dim)
                    if bs is None:
                        continue
                    cores = _compute_split(dim_size, bs, remaining_cores)
                    if cores > 1:
                        splits[name] = cores
                        remaining_cores //= cores

                n.n_cores_used = math.prod(splits.values()) if splits else 1

                if splits.get("M", 1) > 1:
                    n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 0)] = splits["M"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 0)] = splits["M"]
                if splits.get("N", 1) > 1:
                    n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 1)] = splits["N"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 1)] = splits["N"]

            elif num_dims == 3:
                B = args[0].layout.size[0]
                M = args[0].layout.size[1]
                N = args[1].layout.size[2]
                op_dims = [("B", B), ("M", M), ("N", N)]

                splits = {}
                remaining_cores = max_cores
                for host_dim, (name, dim_size) in enumerate(op_dims):
                    bs = dim_to_bs.get(host_dim)
                    if bs is None:
                        continue
                    cores = _compute_split(dim_size, bs, remaining_cores)
                    if cores > 1:
                        splits[name] = cores
                        remaining_cores //= cores

                n.n_cores_used = math.prod(splits.values()) if splits else 1

                if splits.get("B", 1) > 1:
                    n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 0)] = splits["B"]
                    n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 0)] = splits["B"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 0)] = splits["B"]
                if splits.get("M", 1) > 1:
                    n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 1)] = splits["M"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 1)] = splits["M"]
                if splits.get("N", 1) > 1:
                    n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 2)] = splits["N"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 2)] = splits["N"]

            elif num_dims == 4:
                B1 = args[0].layout.size[0]
                B2 = args[0].layout.size[1]
                M = args[0].layout.size[2]
                N = args[1].layout.size[3]
                op_dims = [("B1", B1), ("B2", B2), ("M", M), ("N", N)]

                splits = {}
                remaining_cores = max_cores
                for host_dim, (name, dim_size) in enumerate(op_dims):
                    bs = dim_to_bs.get(host_dim)
                    if bs is None:
                        continue
                    cores = _compute_split(dim_size, bs, remaining_cores)
                    if cores > 1:
                        splits[name] = cores
                        remaining_cores //= cores

                n.n_cores_used = math.prod(splits.values()) if splits else 1

                if splits.get("B1", 1) > 1:
                    n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 0)] = splits["B1"]
                    n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 0)] = splits["B1"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 0)] = splits["B1"]
                if splits.get("B2", 1) > 1:
                    n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 1)] = splits["B2"]
                    n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 1)] = splits["B2"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 1)] = splits["B2"]
                if splits.get("M", 1) > 1:
                    n.spyre_core_division[0][map_host_dim_to_device_dim(args[0].layout, 2)] = splits["M"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 2)] = splits["M"]
                if splits.get("N", 1) > 1:
                    n.spyre_core_division[1][map_host_dim_to_device_dim(args[1].layout, 3)] = splits["N"]
                    n.spyre_core_division[2][map_host_dim_to_device_dim(output, 3)] = splits["N"]

            else:
                raise RuntimeError(f"Unsupported matmul dimension count: {num_dims}")
        else:
            # Unknown reduction type — fall back to original logic
            divide_reduction_op(n, args, max_cores)

    def config_aware_core_division(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        it = iter(nodes)
        for n in it:
            if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
                args = get_mem_deps(n)
                dim_to_bs = _get_dim_to_bs(n)

                if isinstance(n.node.data, Pointwise):
                    if dim_to_bs is not None:
                        _divide_pointwise_with_config(n, args, max_cores, dim_to_bs)
                    else:
                        divide_pointwise_op(n, args, max_cores)
                elif isinstance(n.node.data, Reduction):
                    if dim_to_bs is not None:
                        _divide_reduction_with_config(n, args, max_cores, dim_to_bs)
                    else:
                        divide_reduction_op(n, args, max_cores)
                else:
                    pass
            elif isinstance(n, ExternKernelSchedulerNode):
                if isinstance(n.node, FallbackKernel):
                    n = next(it, None)
                    if not (
                        isinstance(n, ExternKernelSchedulerNode)
                        and isinstance(n.node, MultiOutput)
                    ):
                        raise RuntimeError("FallbackKernel must be followed by MultiOutput")
                    pass
                else:
                    log.warning("unhandled node type %s", type(n.node))
            elif isinstance(n, NopKernelSchedulerNode):
                pass
            else:
                log.warning("unhandled scheduler node type %s", type(n))
        return nodes

    return config_aware_core_division


# ---------------------------------------------------------------------------
# Step 2: Compile for Spyre
# ---------------------------------------------------------------------------

def lower_to_spyre(graph_module, example_spyre_inputs):
    """Propagate Spyre FakeTensors through *graph_module* and compile it via
    the Spyre inductor stack.

    Returns the compiled callable.
    """
    from torch._inductor import compile_fx as cfx_module
    from torch_spyre._inductor.spyre_kernel import SpyreOpFuncs
    from torch_spyre._inductor.patches import enable_spyre_context

    # Ensure the Spyre backend is fully registered (device interface,
    # codegen backend, etc.).  In normal torch_spyre usage this happens at
    # import time, but our transpile path may bypass the top-level __init__
    # autoloading.
    #
    # NOTE: We intentionally do NOT call enable_spyre_compile_fx_wrapper().
    # That wrapper makes compile_fx enter enable_spyre_context a *second*
    # time (nested), which is redundant because we manage the context
    # ourselves below.  By calling the original compile_fx directly we
    # avoid the double-nesting and ensure the Spyre lowerings dict is
    # the one actually consulted during GraphLowering.run().
    from torch_spyre._inductor import _autoload
    _autoload()
    # Also ensure the Spyre C++ runtime is initialized (the compile_fx
    # wrapper normally does this, but we're bypassing it).
    import torch.spyre
    torch.spyre._impl._lazy_init()

    # Grab the unwrapped compile_fx.  If the Spyre wrapper was already
    # installed (e.g. by an earlier import), extract the original via
    # __wrapped__ (set by @wraps).
    _already_wrapped = getattr(cfx_module, '_spyre_wrapped', False)
    if _already_wrapped:
        _orig_compile_fx = cfx_module.compile_fx.__wrapped__
        log.info("[DEBUG] compile_fx was already Spyre-wrapped; extracted __wrapped__")
    else:
        _orig_compile_fx = cfx_module.compile_fx
    log.info("[DEBUG] Using original compile_fx: %s (was wrapped? %s)",
             _orig_compile_fx, _already_wrapped)

    log.info("=== Continue compilation with Spyre Stack ===")

    # Build a fresh GraphModule from the same graph (CPU validation may have
    # mutated internal state of the original).
    try:
        graph_module_spyre = torch.fx.GraphModule({}, graph_module.graph)
        log.info("Created fresh GraphModule")
    except Exception as e:
        log.error(f"Error creating GraphModule: {e}")
        raise

    # Ensure all Spyre input tensors have device layouts attached
    log.info("Ensuring Spyre tensors have device layouts...")
    try:
        from torch_spyre._C import SpyreTensorLayout
        log.info("Imported SpyreTensorLayout successfully")
    except Exception as e:
        log.error(f"Error importing SpyreTensorLayout: {e}")
        raise
    from torch_spyre._C import spyre_empty_with_layout
    processed_inputs = []
    for i, t in enumerate(example_spyre_inputs):
        log.info(f"  Processing input {i}: device={t.device}, shape={t.shape}, dtype={t.dtype}")
        if t.device.type == "spyre":
            # Check if tensor already has a layout
            try:
                existing_layout = t.device_tensor_layout()
                log.info(f"    Existing layout: {existing_layout}")
                if existing_layout is not None:
                    log.info(f"    Tensor {i} already has layout, keeping it")
                    processed_inputs.append(t)
                    continue
            except Exception as e:
                log.info(f"    Error checking existing layout: {e}")
            
            # Create tensor with default layout using spyre_empty_with_layout
            log.info(f"    Creating tensor with default layout for tensor {i}")
            layout = SpyreTensorLayout(list(t.shape), t.dtype)
            log.info(f"    Created layout: {layout}")
            
            # Create new tensor with layout
            t_cpu = t.cpu()
            log.info(f"    Moved to CPU: {t_cpu.shape}")
            
            # Use spyre_empty_with_layout to create tensor with layout
            # Compute stride - for contiguous tensor, stride[i] = product of sizes[i+1:]
            stride = []
            s = 1
            for size in reversed(list(t.shape)):
                stride.insert(0, s)
                s *= size
            log.info(f"    Computed stride: {stride}")
            
            t_with_layout = spyre_empty_with_layout(list(t.shape), stride, t.dtype, layout)
            log.info(f"    Created empty tensor with layout: device={t_with_layout.device}")
            
            # Copy data from CPU to Spyre using .copy_() which internally uses _C.copy_tensor
            t_with_layout.copy_(t_cpu)
            log.info(f"    Copied data to Spyre tensor")
            
            # Verify the layout was attached
            try:
                verify_layout = t_with_layout.device_tensor_layout()
                log.info(f"    Verified layout on new tensor: {verify_layout}")
            except Exception as e:
                log.error(f"    ERROR: Failed to verify layout: {e}")
            processed_inputs.append(t_with_layout)
        else:
            log.info(f"    Tensor {i} is not on spyre device, keeping as-is")
            processed_inputs.append(t)
    
    log.info(f"Processed {len(processed_inputs)} inputs")
    example_spyre_inputs = processed_inputs

    # Debug patch: confirm SuperDSCScheduling is instantiated
    from torch_spyre._inductor.scheduler import SuperDSCScheduling
    _orig_sdsc_init = SuperDSCScheduling.__init__
    def _debug_sdsc_init(self, *args, **kwargs):
        log.debug("[SPYRE DEBUG] SuperDSCScheduling.__init__ called -- Spyre codegen is active!")
        _orig_sdsc_init(self, *args, **kwargs)
    SuperDSCScheduling.__init__ = _debug_sdsc_init

    # Debug patch: print what devices compile_fx_inner sees
    import torch._inductor.compile_fx as _cfx_mod
    _orig_cfx_inner = _cfx_mod.compile_fx_inner
    def _debug_cfx_inner(gm, example_inputs, *args, **kwargs):
        devices = [getattr(getattr(t, 'device', None), 'type', 'N/A')
                   for t in example_inputs if isinstance(t, torch.Tensor)]
        log.debug("[SPYRE DEBUG] compile_fx_inner: input devices=%s", devices)
        return _orig_cfx_inner(gm, example_inputs, *args, **kwargs)
    _cfx_mod.compile_fx_inner = _debug_cfx_inner

    # Strip to_dtype ops at the Spyre codegen level.
    #
    # The inductor inserts ops.to_dtype(x, float32) / ops.to_dtype(y, float16)
    # around math ops (rsqrt, sqrt, etc.) for "OPMATH" type promotion.  The
    # Spyre backend cannot handle nested PointwiseOp trees that result from
    # these casts.  By making to_dtype an identity operation, all computation
    # stays in the original dtype (float16) which Spyre handles natively.
    _orig_to_dtype = SpyreOpFuncs.to_dtype
    SpyreOpFuncs.to_dtype = staticmethod(lambda x, dtype, src_dtype=None: x)
    log.info("Patched SpyreOpFuncs.to_dtype → identity (strip dtype casts)")

    # -----------------------------------------------------------------------
    # DEBUG instrumentation: trace whether Spyre lowerings are activated
    # -----------------------------------------------------------------------
    import torch._inductor.lowering as _ind_lowering
    from torch_spyre._inductor.lowering import (
        enable_spyre_lowerings,
        spyre_lowerings,
        lower_mean,
    )
    from torch_spyre._inductor.decompositions import (
        enable_spyre_decompositions,
        spyre_decompositions,
    )
    from torch_spyre._inductor import lowering as _spyre_low_mod

    # 1) Dump what's in spyre_lowerings before entering context
    log.info("[DEBUG] spyre_lowerings dict has %d entries:", len(spyre_lowerings))
    _mean_dim = torch.ops.aten.mean.dim
    log.info("[DEBUG] Reference: aten.mean.dim = %s (id=%d, type=%s)",
             _mean_dim, id(_mean_dim), type(_mean_dim).__name__)
    _found_mean_in_spyre = False
    for op_key in spyre_lowerings:
        is_mean = (op_key is _mean_dim)
        eq_mean = (op_key == _mean_dim)
        marker = ""
        if is_mean:
            marker = " <-- MEAN (identity match)"
            _found_mean_in_spyre = True
        elif eq_mean:
            marker = " <-- MEAN (equality match, DIFFERENT object!)"
            _found_mean_in_spyre = True
        log.info("[DEBUG]   %s  (id=%d, type=%s)%s", op_key, id(op_key), type(op_key).__name__, marker)
    if not _found_mean_in_spyre:
        log.warning("[DEBUG] !!! aten.mean.dim NOT FOUND in spyre_lowerings !!!")
        log.warning("[DEBUG]     The Spyre lower_mean was never registered.")
        for op_key in spyre_lowerings:
            if 'mean' in str(op_key):
                log.warning("[DEBUG]     But found mean-like key: %s (id=%d)", op_key, id(op_key))

    # 2) Check if aten.mean.dim is already in the main lowerings dict
    log.info("[DEBUG] aten.mean.dim (id=%d) in lowerings BEFORE context? %s",
             id(_mean_dim), _mean_dim in _ind_lowering.lowerings)
    if _mean_dim in _ind_lowering.lowerings:
        log.info("[DEBUG]   current lowering: %s", _ind_lowering.lowerings[_mean_dim])
    log.info("[DEBUG] All mean-related keys in main lowerings dict:")
    for k in _ind_lowering.lowerings:
        if 'mean' in str(k).lower():
            log.info("[DEBUG]   %s (id=%d) → %s", k, id(k), _ind_lowering.lowerings[k])

    # 3) Monkey-patch enable_spyre_lowerings to log entry/exit
    _orig_enable_spyre_lowerings = _spyre_low_mod.enable_spyre_lowerings

    from contextlib import contextmanager as _cm
    @_cm
    def _debug_enable_spyre_lowerings():
        log.info("[DEBUG] >>> ENTERING enable_spyre_lowerings (nesting=%d)",
                 _spyre_low_mod._lowerings_nesting)
        with _orig_enable_spyre_lowerings():
            log.info("[DEBUG] enable_spyre_lowerings entered (nesting=%d)",
                     _spyre_low_mod._lowerings_nesting)
            log.info("[DEBUG] aten.mean.dim in lowerings AFTER enter? %s",
                     _mean_dim in _ind_lowering.lowerings)
            if _mean_dim in _ind_lowering.lowerings:
                fn = _ind_lowering.lowerings[_mean_dim]
                log.info("[DEBUG]   lowerings[aten.mean.dim] = %s (module: %s)",
                         fn, getattr(fn, '__module__', '?'))
                wrapped = getattr(fn, '__wrapped__', None)
                log.info("[DEBUG]   __wrapped__ = %s", wrapped)
            yield
        log.info("[DEBUG] <<< EXITED enable_spyre_lowerings (nesting=%d)",
                 _spyre_low_mod._lowerings_nesting)

    _spyre_low_mod.enable_spyre_lowerings = _debug_enable_spyre_lowerings

    # 4) Monkey-patch the standard inductor mean lowering to log when called
    _mean_in_lowerings = _ind_lowering.lowerings.get(_mean_dim)
    _debug_mean_lowering = None
    if _mean_in_lowerings is not None:
        import functools
        @functools.wraps(_mean_in_lowerings)
        def _debug_mean_lowering(*args, **kwargs):
            import traceback
            log.warning("[DEBUG] !!! STANDARD INDUCTOR mean lowering called !!!")
            log.warning("[DEBUG]     This means the Spyre lower_mean was NOT used.")
            log.warning("[DEBUG]     Traceback:\n%s", "".join(traceback.format_stack()[-5:]))
            return _mean_in_lowerings(*args, **kwargs)
    else:
        log.info("[DEBUG] aten.mean.dim NOT in lowerings dict (no standard lowering found)")

    # 5) Monkey-patch the Spyre lower_mean to log when called.
    from torch_spyre._inductor.ir import SpyreReduction as _SpyreReduction
    _orig_lower_mean = _spyre_low_mod.lower_mean

    _spyre_mean_wrapped = spyre_lowerings.get(_mean_dim)
    if _spyre_mean_wrapped is not None:
        def _debug_spyre_mean_wrapped(*args, **kwargs):
            import traceback
            log.info("[DEBUG] +++ SPYRE lower_mean (wrapped) called! +++")
            log.info("[DEBUG]     args types: %s", [type(a).__name__ for a in args])
            if args and hasattr(args[0], 'get_dtype'):
                log.info("[DEBUG]     x.get_dtype() = %s", args[0].get_dtype())
            try:
                result = _spyre_mean_wrapped(*args, **kwargs)
                log.info("[DEBUG]     lower_mean returned: %s (type: %s)", result, type(result).__name__)
                if hasattr(result, 'data') and hasattr(result.data, 'data'):
                    inner = result.data.data
                    log.info("[DEBUG]     inner IR node: %s (type: %s)", type(inner).__name__, type(inner))
                    log.info("[DEBUG]     is SpyreReduction? %s", isinstance(inner, _SpyreReduction))
                    if hasattr(inner, 'dtype'):
                        log.info("[DEBUG]     inner.dtype = %s", inner.dtype)
                    if hasattr(inner, 'reduction_type'):
                        log.info("[DEBUG]     inner.reduction_type = %s", inner.reduction_type)
                return result
            except Exception as e:
                log.error("[DEBUG]     !!! lower_mean RAISED: %s: %s", type(e).__name__, e)
                log.error("[DEBUG]     traceback:\n%s", "".join(traceback.format_exc()))
                raise
        spyre_lowerings[_mean_dim] = _debug_spyre_mean_wrapped
        log.info("[DEBUG] Installed debug wrapper on spyre_lowerings[aten.mean.dim]")
    else:
        log.warning("[DEBUG] !!! aten.mean.dim NOT in spyre_lowerings — cannot patch!")

    # Also patch SpyreReduction.create to log when called
    _orig_spyre_reduction_create = _SpyreReduction.create
    @classmethod
    def _debug_spyre_reduction_create(cls, **kwargs):
        log.info("[DEBUG] SpyreReduction.create called!")
        log.info("[DEBUG]   reduction_type=%s, dst_dtype=%s, src_dtype=%s",
                 kwargs.get('reduction_type'), kwargs.get('dst_dtype'), kwargs.get('src_dtype'))
        return _orig_spyre_reduction_create.__func__(cls, **kwargs)
    _SpyreReduction.create = _debug_spyre_reduction_create

    # Also patch standard Reduction.create to log when called with mean origins
    from torch._inductor.ir import Reduction as _StdReduction
    _orig_std_reduction_create = _StdReduction.create
    @classmethod
    def _debug_std_reduction_create(cls, reduction_type=None, **kwargs):
        log.info("[DEBUG] Reduction.create called: reduction_type=%s, dst_dtype=%s",
                 reduction_type, kwargs.get('dst_dtype'))
        if reduction_type == 'sum':
            import traceback
            log.info("[DEBUG]   (sum reduction) traceback:\n%s",
                     "".join(traceback.format_stack()[-8:]))
        return _orig_std_reduction_create.__func__(cls, reduction_type=reduction_type, **kwargs)
    _StdReduction.create = _debug_std_reduction_create

    # 6) Monkey-patch the lowering dispatch (graph.py call_function) to trace mean ops
    import torch._inductor.graph as _graph_mod
    _OrigGraphLowering = _graph_mod.GraphLowering
    _orig_call_function = _OrigGraphLowering.call_function
    def _debug_call_function(self, target, args, kwargs):
        if target is _mean_dim or (hasattr(target, 'name') and 'mean' in str(target)):
            log.info("[DEBUG] GraphLowering.call_function: target=%s (id=%d)", target, id(target))
            log.info("[DEBUG]   target in lowerings? %s", target in _ind_lowering.lowerings)
            if target in _ind_lowering.lowerings:
                fn = _ind_lowering.lowerings[target]
                log.info("[DEBUG]   will dispatch to: %s (module: %s)",
                         fn, getattr(fn, '__module__', '?'))
                wrapped = getattr(fn, '__wrapped__', None)
                log.info("[DEBUG]   __wrapped__ = %s", wrapped)
        return _orig_call_function(self, target, args, kwargs)
    _OrigGraphLowering.call_function = _debug_call_function

    # 7) Monkey-patch spyre_data_types to confirm it's entered
    from torch_spyre._inductor import patches as _patches
    _orig_spyre_data_types = _patches.spyre_data_types
    @_cm
    def _debug_spyre_data_types():
        import torch._prims_common as _pc
        log.info("[DEBUG] >>> ENTERING spyre_data_types")
        log.info("[DEBUG]   _computation_dtype_map BEFORE: %s", _pc._computation_dtype_map)
        with _orig_spyre_data_types():
            log.info("[DEBUG]   _computation_dtype_map AFTER: %s", _pc._computation_dtype_map)
            yield
        log.info("[DEBUG] <<< EXITED spyre_data_types")
    _patches.spyre_data_types = _debug_spyre_data_types

    # 8) Monkey-patch enable_spyre_decompositions to log entry
    from torch_spyre._inductor import decompositions as _dec_mod
    _orig_enable_spyre_decompositions = _dec_mod.enable_spyre_decompositions
    @_cm
    def _debug_enable_spyre_decompositions(decomps=None):
        log.info("[DEBUG] >>> ENTERING enable_spyre_decompositions")
        with _orig_enable_spyre_decompositions(decomps=decomps):
            log.info("[DEBUG] enable_spyre_decompositions entered")
            yield
        log.info("[DEBUG] <<< EXITED enable_spyre_decompositions")
    _dec_mod.enable_spyre_decompositions = _debug_enable_spyre_decompositions

    # 9) Install the debug mean lowering trap on the CURRENT lowerings dict.
    if _debug_mean_lowering is not None:
        _ind_lowering.lowerings[_mean_dim] = _debug_mean_lowering
        log.info("[DEBUG] Installed debug trap on standard mean lowering")

    # 10) Check if aten.mean.dim has a decomposition that might fire before lowering
    from torch._inductor.decomposition import decompositions as _decomp_table
    log.info("[DEBUG] aten.mean.dim in decompositions table? %s",
             _mean_dim in _decomp_table)
    if _mean_dim in _decomp_table:
        log.info("[DEBUG]   decomposition: %s", _decomp_table[_mean_dim])
    _mean_pkg = torch.ops.aten.mean
    for _ov_name in _mean_pkg.overloads():
        _ov = getattr(_mean_pkg, _ov_name)
        if _ov in _decomp_table:
            log.info("[DEBUG]   decomp for aten.mean.%s: %s", _ov_name, _decomp_table[_ov])

    # -----------------------------------------------------------------------
    # End DEBUG instrumentation
    # -----------------------------------------------------------------------

    # Activate the full Spyre inductor context: custom lowerings (including
    # lower_mean which avoids the float32 upcast), Spyre decompositions,
    # data-type overrides, and config tweaks.
    #
    # We call the ORIGINAL compile_fx (not the Spyre-wrapped version) to
    # avoid double-nesting of enable_spyre_context.
    log.info("Calling compile_fx inside enable_spyre_context...")
    try:
        with enable_spyre_context(example_spyre_inputs):
            # Post-context-enter checks
            log.info("[DEBUG] Inside enable_spyre_context. Checking lowerings state:")
            log.info("[DEBUG]   aten.mean.dim in lowerings? %s",
                     _mean_dim in _ind_lowering.lowerings)
            if _mean_dim in _ind_lowering.lowerings:
                fn = _ind_lowering.lowerings[_mean_dim]
                log.info("[DEBUG]   lowerings[aten.mean.dim] = %s", fn)
                log.info("[DEBUG]   __wrapped__ = %s", getattr(fn, '__wrapped__', None))
                log.info("[DEBUG]   is debug_mean? %s", fn is _debug_mean_lowering)
            log.info("[DEBUG]   lowerings nesting: %d", _spyre_low_mod._lowerings_nesting)

            from torch._inductor.decomposition import decompositions
            compiled_fn = _orig_compile_fx(
                graph_module_spyre,
                example_spyre_inputs,
                decompositions=decompositions,
            )
    finally:
        SpyreOpFuncs.to_dtype = _orig_to_dtype
        # Restore debug patches
        _spyre_low_mod.enable_spyre_lowerings = _orig_enable_spyre_lowerings
        _OrigGraphLowering.call_function = _orig_call_function
        _patches.spyre_data_types = _orig_spyre_data_types
        _dec_mod.enable_spyre_decompositions = _orig_enable_spyre_decompositions
        if _spyre_mean_wrapped is not None:
            spyre_lowerings[_mean_dim] = _spyre_mean_wrapped
        _SpyreReduction.create = _orig_spyre_reduction_create
        _StdReduction.create = _orig_std_reduction_create
        if _mean_in_lowerings is not None:
            _ind_lowering.lowerings[_mean_dim] = _mean_in_lowerings
    log.debug("compile_fx returned: %s", type(compiled_fn))

    return compiled_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_helion_to_spyre(
    helion_kernel,
    example_spyre_inputs,
    do_verify_run_on_cpu=False,
    return_meta=False,
    config=None,
):
    """Compiles a Helion kernel to a Spyre-compiled callable.

    The compilation pipeline first uses Helion to lower the decorated kernel
    into an FX graph, then transpiles that graph into a form suitable for
    the Spyre backend. The transpiled graph is analyzed to determine
    core-division and tiling strategies before being handed off to the Spyre
    compiler, which produces a callable that executes directly on Spyre
    hardware.

    Parameters
    ----------
    helion_kernel : helion.kernel
        The decorated Helion kernel function.
    example_spyre_inputs : tuple[torch.Tensor, ...]
        Spyre-device tensors matching the kernel signature.
    do_verify_run_on_cpu : bool
        If True, run the intermediate FX graph on CPU first to validate
        correctness before compiling for Spyre.
    return_meta : bool
        If True, return a ``(compiled_fn, TranspileMeta)`` tuple instead
        of just the compiled callable.  The meta object contains paths to
        the generated SDSC files and can be passed to the ``analyze``
        module for inspection.
    config : helion.Config | None
        If provided, use ``config.block_sizes`` to drive core division
        planning instead of the default divisor-based heuristic.
        ``block_sizes[i]`` maps to the i-th ``hl.tile`` dimension.

    Returns
    -------
    callable | tuple[callable, TranspileMeta]
        A compiled function that can be invoked with Spyre tensors.
        When *return_meta* is ``True``, returns ``(compiled_fn, meta)``.
    """
    # Derive CPU dummy inputs (same shapes / dtypes, on CPU)
    cpu_inputs = tuple(
        torch.zeros_like(t, device="cpu") if t.device.type != "cpu"
        else t.clone()
        for t in example_spyre_inputs
    )

    # Step 1: Helion kernel → FX GraphModule
    graph_module, block_size_nodes, affected_ops = transpile_fx_graphs(
        helion_kernel, cpu_inputs
    )

    # Propagate shapes through the graph so node metadata is available
    # for subsequent passes (e.g. prevent_reduction_upcasts needs dtype info).
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(graph_module).propagate(*cpu_inputs)

    # Step 1b: Spyre-specific FX graph transformations
    prevent_reduction_upcasts(graph_module)
    fix_scalar_args_for_spyre(graph_module)

    # Log the final graph that will be verified / compiled
    log.info("=== Final Spyre-Compatible Graph ===")
    log.info("Total nodes: %d", len(list(graph_module.graph.nodes)))
    log.info("Placeholder nodes: %d",
             len([n for n in graph_module.graph.nodes if n.op == "placeholder"]))
    log.info("Call nodes: %d",
             len([n for n in graph_module.graph.nodes if n.op == "call_function"]))
    log.info("Block size nodes tracked: %d", len(block_size_nodes))
    log.info("Affected ops: %s", affected_ops)
    log.info("Graph structure:\n%s", graph_module.graph)

    # Step 2 (optional): validate on CPU
    if do_verify_run_on_cpu:
        verify_on_cpu(graph_module, example_spyre_inputs)

    # Step 3: compile for Spyre (with optional config-aware core division)
    def _compile(graph_module, example_spyre_inputs):
        if config is not None:
            import torch_spyre._inductor.passes as spyre_passes

            block_sizes_cfg = config.block_sizes

            # Merge: affected_ops has {node_name: {dim: bs_index}},
            # config has block_sizes_cfg[bs_index] = value.
            # Result: {node_name: {dim: block_size_value}}.
            node_block_sizes: dict[str, dict[int, int]] = {}
            for node_name, dim_to_idx in affected_ops.items():
                resolved = {}
                for dim, bs_idx in dim_to_idx.items():
                    if bs_idx < len(block_sizes_cfg):
                        resolved[dim] = block_sizes_cfg[bs_idx]
                    else:
                        log.warning("block_size index %d out of range for "
                                    "config.block_sizes (len=%d), skipping "
                                    "dim %d of %s",
                                    bs_idx, len(block_sizes_cfg), dim, node_name)
                if resolved:
                    node_block_sizes[node_name] = resolved

            log.info("Using config-aware core division: block_sizes=%s, "
                     "affected nodes=%s", block_sizes_cfg, node_block_sizes)

            patched_fn = _make_config_aware_core_division(node_block_sizes)
            original_fn = spyre_passes.core_division_planning
            spyre_passes.core_division_planning = patched_fn
            try:
                return lower_to_spyre(graph_module, example_spyre_inputs)
            finally:
                spyre_passes.core_division_planning = original_fn
        else:
            return lower_to_spyre(graph_module, example_spyre_inputs)

    if return_meta:
        with _capture_sdsc_paths() as meta:
            compiled_fn = _compile(graph_module, example_spyre_inputs)
        meta.graph_module = graph_module
        meta.block_size_nodes = block_size_nodes
        meta.tile_dim_to_block_index = affected_ops
        # Store the merged form (with resolved block_size values) for analysis
        if config is not None:
            block_sizes_cfg = config.block_sizes
            meta.tile_dim_to_block_value = {
                name: {dim: block_sizes_cfg[idx]
                       for dim, idx in dim_to_idx.items()
                       if idx < len(block_sizes_cfg)}
                for name, dim_to_idx in affected_ops.items()
            }
        else:
            meta.tile_dim_to_block_value = {}
        log.info("Captured %d SDSC path(s), %d block-size node(s), "
                 "%d affected op(s)",
                 len(meta.sdsc_paths), len(meta.block_size_nodes),
                 len(meta.tile_dim_to_block_index))
        return compiled_fn, meta

    compiled_fn = _compile(graph_module, example_spyre_inputs)
    return compiled_fn
