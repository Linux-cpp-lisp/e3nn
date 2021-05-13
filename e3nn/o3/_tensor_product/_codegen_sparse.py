from typing import List, Tuple
from math import sqrt

import numpy as np
import torch
from torch import fx
import torch_scatter  # noqa

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction


def codegen_tensor_product(
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
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    graph_out = fx.Graph()
    graph_right = fx.Graph()

    # = Function definitions =
    x1s_out = fx.Proxy(graph_out.placeholder('x1', torch.Tensor))
    x2s_out = fx.Proxy(graph_out.placeholder('x2', torch.Tensor))
    ws_out = fx.Proxy(graph_out.placeholder('w', torch.Tensor))

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        empty_out = fx.Proxy(graph_out.call_function(torch.empty, ((),), dict(device='cpu')))
        if shared_weights:
            size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]))[0].shape
        else:
            size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]), empty_out.expand(ws_out.shape[:-1]))[0].shape
        out_out = x1s_out.new_zeros(size_out + (irreps_out.dim,))
        graph_out.output(out_out.node, torch.Tensor)
        # Short circut
        return (
            fx.GraphModule({}, graph_out, "tp_forward"),
            fx.GraphModule({}, graph_right, "tp_right")
        )

    x1_indexes = []
    x2_indexes = []
    w_indexes = []
    w3j_values = []
    out_indexes = []

    assert shared_weights  # TODO

    flat_weight_index = 0

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']
        alpha = sqrt(alpha / {
            'uvw': (mul_ir_in1.mul * mul_ir_in2.mul),
            'uvu': mul_ir_in2.mul,
            'uvv': mul_ir_in1.mul,
            'uuw': mul_ir_in1.mul,
            'uuu': 1,
            'uvuv': 1,
        }[ins.connection_mode])

        w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if normalization == 'component':
            w3j *= (2 * mul_ir_out.ir.l + 1) ** 0.5
        if normalization == 'norm':
            w3j *= (2 * mul_ir_in1.ir.l + 1) ** 0.5 * (2 * mul_ir_in2.ir.l + 1) ** 0.5

        this_indexes = w3j.nonzero()
        this_w3j_value = w3j[this_indexes[:, 0], this_indexes[:, 1], this_indexes[:, 2]]
        this_w3j_value *= alpha
        this_x1_index = this_indexes[:, 0] + irreps_in1[:ins.i_in1].dim
        this_x2_index = this_indexes[:, 1] + irreps_in2[:ins.i_in2].dim
        this_out_index = this_indexes[:, 2] + irreps_out[:ins.i_out].dim
        assert this_w3j_value.ndim == 1
        assert len(this_w3j_value) == len(this_x1_index)
        assert len(this_w3j_value) == len(this_x2_index)
        assert len(this_w3j_value) == len(this_out_index)

        assert ins.has_weight  # TODO

        # TODO
        assert ins.connection_mode == "uvw"
        # we need u*v*w copies of the w3j
        path_size = prod(ins.path_shape)
        # each with a different index
        weight_indexes = [
            o.flatten().repeat_interleave(len(this_w3j_value))
            for o in torch.meshgrid(
                torch.arange(0, mul_ir_in1.mul),
                torch.arange(0, mul_ir_in2.mul),
                torch.arange(0, mul_ir_out.mul)
            )
        ]
        offset_x1 = (weight_indexes[0] * mul_ir_in1.ir.dim)
        offset_x2 = (weight_indexes[1] * mul_ir_in2.ir.dim)
        offset_out = (weight_indexes[2] * mul_ir_out.ir.dim)

        this_w_index = torch.as_tensor(np.ravel_multi_index(weight_indexes, ins.path_shape))
        this_w_index += flat_weight_index
        flat_weight_index += path_size
        this_w3j_value = this_w3j_value.repeat(path_size)
        this_x1_index = this_x1_index.repeat(path_size) + offset_x1
        this_x2_index = this_x2_index.repeat(path_size) + offset_x2
        this_out_index = this_out_index.repeat(path_size) + offset_out

        x1_indexes.append(this_x1_index)
        x2_indexes.append(this_x2_index)
        w_indexes.append(this_w_index)
        w3j_values.append(this_w3j_value)
        out_indexes.append(this_out_index)

    out_indexes = torch.cat(out_indexes, dim=0)
    # Since we'll scatter over out_indexes, we want it to be efficient
    sort_idex = out_indexes.argsort()
    out_indexes = out_indexes[sort_idex]
    x1_indexes = torch.cat(x1_indexes, dim=0)[sort_idex]
    x2_indexes = torch.cat(x2_indexes, dim=0)[sort_idex]
    w_indexes = torch.cat(w_indexes, dim=0)[sort_idex]
    w3j_values = torch.cat(w3j_values, dim=0)[sort_idex]

    # Now generate the actual code
    x1_indexes_proxy = fx.Proxy(graph_out.get_attr("_x1_indexes"))
    x2_indexes_proxy = fx.Proxy(graph_out.get_attr("_x2_indexes"))
    out_ptr_proxy = fx.Proxy(graph_out.get_attr("_out_ptr"))
    w_indexes_proxy = fx.Proxy(graph_out.get_attr("_w_indexes"))
    w3j_values_proxy = fx.Proxy(graph_out.get_attr("_w3j_values"))
    out = (
        (x1s_out.index_select(-1, x1_indexes_proxy) * x2s_out.index_select(-1, x2_indexes_proxy)) * (ws_out.index_select(-1, w_indexes_proxy) * w3j_values_proxy)
    )

    # We have to build the pointer for out sum
    # works because its sorted
    out_ptr = torch.cat(
        [
            torch.LongTensor([0]),
            torch.cumsum(torch.bincount(out_indexes, minlength=irreps_out.dim), 0),
        ],
        dim=0
    ).unsqueeze(0)

    out = graph_out.call_function(
        torch.ops.torch_scatter.segment_sum_csr,
        (out.node, out_ptr_proxy.node, None),
        type_expr=torch.Tensor
    )
    graph_out.output(out, torch.Tensor)

    # Generate modules
    graphmod_out = fx.GraphModule(
        {
            "_x1_indexes": x1_indexes,
            "_x2_indexes": x2_indexes,
            "_out_ptr": out_ptr,
            "_w_indexes": w_indexes,
            "_w3j_values": w3j_values.unsqueeze(0)
        },
        graph_out,
        class_name="tp_forward"
    )
    # TODO: right
    graphmod_right = fx.GraphModule({}, graph_right, class_name="tp_right")

    return graphmod_out, graphmod_right
