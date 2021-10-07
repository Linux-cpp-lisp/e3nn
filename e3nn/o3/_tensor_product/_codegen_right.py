from math import sqrt
from typing import List, Tuple

from opt_einsum_fx import jitable, optimize_einsums_full
import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction
from ._codegen_default import _sum_tensors


# TODO: just right
def codegen_default_tensor_product_right(
    irreps_in1: o3.Irreps,
    in1_var: List[float],
    irreps_in2: o3.Irreps,
    in2_var: List[float],
    irreps_out: o3.Irreps,
    out_var: List[float],
    instructions: List[Instruction],
    normalization: str = 'component',
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> fx.GraphModule:
    graph_right = fx.Graph()

    # = Function definitions =
    x2s_right = fx.Proxy(graph_right.placeholder('x2', torch.Tensor))
    ws_right = fx.Proxy(graph_right.placeholder('w', torch.Tensor))

    empty_right = fx.Proxy(graph_right.call_function(torch.empty, ((),), dict(device='cpu')))
    if shared_weights:
        size_right = x2s_right.shape[:-1]
    else:
        size_right = torch.broadcast_tensors(empty_right.expand(x2s_right.shape[:-1]), empty_right.expand(ws_right.shape[:-1]))[0].shape

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        out_right = x2s_right.new_zeros(size_right + (irreps_in1.dim, irreps_out.dim,))

        graph_right.output(out_right.node, torch.Tensor)
        # Short circut
        return fx.GraphModule({}, graph_right, "tp_right")

    # = Broadcast inputs =
    if not shared_weights:
        x2s_right, ws_right = x2s_right.broadcast_to(size_right + (-1,)), ws_right.broadcast_to(size_right + (-1,))

    outsize_right = size_right + (irreps_in1.dim, irreps_out.dim,)
    x2s_right = x2s_right.reshape(-1, irreps_in2.dim)
    batch_right = x2s_right.shape[0]

    # = Determine number of weights and reshape weights ==
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        ws_right = ws_right.reshape(-1, weight_numel)
    del weight_numel

    # = book-keeping for wigners =
    w3j = []
    w3j_dict_right = dict()

    # = extract individual input irreps =
    x2_list_right = []
    # If only one input irrep, can avoid creating a view
    if len(irreps_in2) == 1:
        x2_list_right.append(
            x2s_right.reshape(batch_right, irreps_in2[0].mul, irreps_in2[0].ir.dim)
        )
    else:
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list_right.append(
                x2s_right[:, i].reshape(batch_right, mul_ir.mul, mul_ir.ir.dim)
            )

    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Current index in the flat weight tensor
    flat_weight_index = 0

    out_list_right = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        # Open the profiler block
        # - PROFILER - name = f"{mul_ir_in1} x {mul_ir_in2} = {mul_ir_out} {ins.connection_mode} {ins.has_weight}"
        # - PROFILER - handle_right = graph_right.call_function(torch.ops.profiler._record_function_enter, (name,))

        x2_right = x2_list_right[ins.i_in2]

        e1_right = fx.Proxy(graph_right.call_function(torch.eye, (mul_ir_in1.mul,), dict(dtype=x2s_right.dtype.node, device=x2s_right.device.node)))
        e2_right = fx.Proxy(graph_right.call_function(torch.eye, (mul_ir_in2.mul,), dict(dtype=x2s_right.dtype.node, device=x2s_right.device.node)))
        i1_right = fx.Proxy(graph_right.call_function(torch.eye, (mul_ir_in1.ir.dim,), dict(dtype=x2s_right.dtype.node, device=x2s_right.device.node)))

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

        alpha = ins.path_weight * out_var[ins.i_out] / sum(
            in1_var[i.i_in1] * in2_var[i.i_in2] * {
                'uvw': (irreps_in1[i.i_in1].mul * irreps_in2[i.i_in2].mul),
                'uvu': irreps_in2[i.i_in2].mul,
                'uvv': irreps_in1[i.i_in1].mul,
                'uuw': irreps_in1[i.i_in1].mul,
                'uuu': 1,
                'uvuv': 1,
            }[i.connection_mode]
            for i in instructions if i.i_out == ins.i_out
        )
        alpha = sqrt(alpha)

        if ins.has_weight:
            # Extract the weight from the flattened weight tensor
            w_right = ws_right[:, flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape((() if shared_weights else (-1,)) + tuple(ins.path_shape))
            flat_weight_index += prod(ins.path_shape)

        # Create a proxy & request for the relevant wigner w3j
        # If not used (because of specialized code), will get removed later.
        key = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if key not in w3j:
            w3j_dict_right[key] = fx.Proxy(graph_right.get_attr(f"_w3j_{key[0]}_{key[1]}_{key[2]}"))
            w3j.append(key)
        w3j_right = w3j_dict_right[key]

        exp = {'component': 1, 'norm': -1}[normalization]

        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            if specialized_code and key == (0, 0, 0):
                ein_right = torch.einsum(f"{z}uvw,zv->zuw", w_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
            elif specialized_code and mul_ir_in1.ir.l == 0:
                ein_right = torch.einsum(f"{z}uvw,zvi->zuwi", w_right, x2_right)
            elif specialized_code and mul_ir_in2.ir.l == 0:
                ein_right = torch.einsum(f"{z}uvw,ij,zv->zuiwj", w_right, i1_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
            elif specialized_code and mul_ir_out.ir.l == 0:
                ein_right = torch.einsum(f"{z}uvw,zvi->zuiw", w_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
            else:
                ein_right = torch.einsum(f"{z}uvw,ijk,zvj->zuiwk", w_right, w3j_right, x2_right)
        if ins.connection_mode == 'uvu':
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_right = torch.einsum(f"{z}uv,uw,zv->zuw", w_right, e1_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_right = torch.einsum(f"{z}uv,uw,zvi->zuwi", w_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_right = torch.einsum(f"{z}uv,ij,uw,zv->zuiwj", w_right, i1_right, e1_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_right = torch.einsum(f"{z}uv,uw,zvi->zuiw", w_right, e1_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_right = torch.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                # not so useful operation because v is summed
                ein_right = torch.einsum("ijk,uw,zvj->zuiwk", w3j_right, e1_right, x2_right)
        if ins.connection_mode == 'uvv':
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_right = torch.einsum(f"{z}uv,vw,zv->zuw", w_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_right = torch.einsum(f"{z}uv,vw,zvi->zuwi", w_right, e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_right = torch.einsum(f"{z}uv,ij,vw,zv->zuiwj", w_right, i1_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_right = torch.einsum(f"{z}uv,vw,zvi->zuiw", w_right, e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_right = torch.einsum(f"{z}uv,ijk,zvj->zuivk", w_right, w3j_right, x2_right)
            else:
                # not so useful operation because u is summed
                # only specialize out for this path
                s2ones = fx.Proxy(graph_right.call_function(torch.ones, (mul_ir_in1.mul,), dict(device=x2_right.device.node, dtype=x2_right.dtype.node)))
                ein_right = torch.einsum("u,ijk,zvj->zuivk", s2ones, w3j_right, x2_right)
        if ins.connection_mode == 'uuw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                # TODO: specialized code for right()
                ein_right = torch.einsum(f"{z}uw,ijk,zuj->zuiwk", w_right, w3j_right, x2_right)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                ein_right = torch.einsum("ijk,zuj->zuik", w3j_right, x2_right)
        if ins.connection_mode == 'uuu':
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_right = torch.einsum(f"{z}u,uw,zu->zuw", w_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_right = torch.einsum(f"{z}u,uw,zui->zuwi", w_right, e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_right = torch.einsum(f"{z}u,ij,uw,zu->zuiwj", w_right, i1_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_right = torch.einsum(f"{z}u,uw,zui->zuiw", w_right, e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_right = torch.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                if specialized_code and key == (0, 0, 0):
                    ein_right = torch.einsum("uw,zu->zuw", e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_right = torch.einsum("uw,zui->zuwi", e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_right = torch.einsum("ij,uw,zu->zuiwj", i1_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_right = torch.einsum("uw,zui->zuiw", e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_right = torch.einsum("ijk,uw,zuj->zuiwk", w3j_right, e1_right, x2_right)
        if ins.connection_mode == 'uvuv':
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                ein_right = torch.einsum(f"{z}uv,ijk,uw,zvj->zuiwvk", w_right, w3j_right, e1_right, x2_right)
            else:
                # TODO implement specialized code
                ein_right = torch.einsum("ijk,uw,zvj->zuiwvk", w3j_right, e1_right, x2_right)

        ein_right = alpha * ein_right

        out_list_right += [ein_right.reshape(batch_right, mul_ir_in1.dim, mul_ir_out.dim)]

        # Close the profiler block
        # - PROFILER - graph_right.call_function(torch.ops.profiler._record_function_exit, (handle_right,))

        # Remove unused w3js:
        if len(w3j_right.node.users) == 0:
            del w3j[-1]
            # The w3j nodes are reshapes, so we have to remove them from the graph
            # Although they are dead code, they try to reshape to dimensions that don't exist
            # (since the corresponding w3js are not in w3j)
            # so they screw up the shape propagation, even though they would be removed later as dead code by TorchScript.
            graph_right.erase_node(w3j_dict_right.pop(key).node)

    # = Return the result =
    out_right = [
        torch.cat([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list_right) if (ins.i_in1, ins.i_out) == (i_in1, i_out)],
                shape=(batch_right, mul_ir_in1.dim, mul_ir_out.dim),
                like=x2s_right
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
            if mul_ir_out.mul > 0
        ], dim=2)
        for i_in1, mul_ir_in1 in enumerate(irreps_in1)
        if mul_ir_in1.mul > 0
    ]
    if len(out_right) > 1:
        out_right = torch.cat(out_right, dim=1)
    else:
        out_right = out_right[0]

    out_right = out_right.reshape(outsize_right)
    graph_right.output(out_right.node, torch.Tensor)

    # check graphs
    graph_right.lint()

    # Make GraphModules
    wigner_mats = {}
    for l_1, l_2, l_out in w3j:
        wig = o3.wigner_3j(l_1, l_2, l_out)

        if normalization == 'component':
            wig *= (2 * l_out + 1) ** 0.5
        if normalization == 'norm':
            wig *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

        wigner_mats[f"_w3j_{l_1}_{l_2}_{l_out}"] = wig

    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    for wkey, wmat in wigner_mats.items():
        constants_root.register_buffer(wkey, wmat)
    graphmod_right = fx.GraphModule(constants_root, graph_right, class_name="tp_right")

    return graphmod_right
