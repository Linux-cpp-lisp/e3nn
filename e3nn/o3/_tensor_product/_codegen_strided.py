from math import sqrt
from typing import List, Optional

import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod
from e3nn.util.strided import StridedLayout

from ._instruction import Instruction


def codegen_strided_tensor_product_forward(
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
) -> Optional[fx.GraphModule]:
    """Returns None if strided doesn't make sense for this TP."""
    # Check if irreps can be strided
    try:
        layout_in1 = StridedLayout(irreps_in1)
        layout_in2 = StridedLayout(irreps_in2)
        layout_out = StridedLayout(irreps_out)
    except ValueError:
        # one cannot be strided
        return None

    # check the instructions
    connection_mode = instructions[0].connection_mode
    if not all(ins.connection_mode == connection_mode for ins in instructions):
        return None
    has_weight = instructions[0].has_weight
    if not all(ins.has_weight == has_weight for ins in instructions):
        return None
    if not has_weight:
        # TODO: unweighted is not yet supported
        return None

    # Make the big w3j
    w3j_index = []
    w3j_values = []

    for ins_i, ins in enumerate(instructions):
        mul_ir_in1 = layout_in1.base_irreps[ins[0]]
        mul_ir_in2 = layout_in2.base_irreps[ins[1]]
        mul_ir_out = layout_out.base_irreps[ins[2]]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        this_w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        this_w3j_index = this_w3j.nonzero()
        w3j_values.append(
            this_w3j[
                this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]
            ]
        )

        # Normalize the path through its w3j entries
        # TODO: path_weight
        # TODO: in and out var
        if normalization == 'component':
            w3j_norm_term = (2 * mul_ir_out.ir.l + 1)
        if normalization == 'norm':
            w3j_norm_term = (2 * mul_ir_in1.ir.l + 1) * (2 * mul_ir_in2.ir.l + 1)
        alpha = sqrt(
            ins.path_weight  # per-path weight
            * out_var[ins.i_out]  # enforce output variance
            * w3j_norm_term
            / sum(
                in1_var[i.i_in1] * in2_var[i.i_in2] * {
                    'uvw': (layout_in1.mul * layout_in2.mul),
                    'uvu': layout_in2.mul,
                    'uvv': layout_in1.mul,
                    'uuw': layout_in1.mul,
                    'uuu': 1,
                    'uvuv': 1,
                }[i.connection_mode]
                for i in instructions if i.i_out == ins.i_out
            )
        )
        w3j_values[-1].mul_(alpha)

        this_w3j_index[:, 0] += layout_in1.base_irreps[: ins[0]].dim
        this_w3j_index[:, 1] += layout_in2.base_irreps[: ins[1]].dim
        this_w3j_index[:, 2] += layout_out.base_irreps[: ins[2]].dim
        # Now need to flatten the index to be for [pk][ij]
        w3j_index.append(
            torch.cat(
                (
                    ins_i * layout_out.base_dim + this_w3j_index[:, 2].unsqueeze(-1),
                    this_w3j_index[:, 0].unsqueeze(-1) * layout_in2.base_dim
                    + this_w3j_index[:, 1].unsqueeze(-1),
                ),
                dim=1,
            )
        )

    w3j = torch.sparse_coo_tensor(
        indices=torch.cat(w3j_index, dim=0).t(),
        values=torch.cat(w3j_values, dim=0),
        size=(len(instructions) * layout_out.base_dim, layout_in1.base_dim * layout_in2.base_dim),
    ).to_dense()
    # TODO: support use of sparse w3j
    # for now, in dense, must shape it:
    w3j = w3j.reshape(len(instructions), layout_out.base_dim, layout_in1.base_dim, layout_in2.base_dim).contiguous()

    # Generate the mixer
    u, v, w = connection_mode

    weight_label = {"uvw": "puvw", "uuu": "pu", "uvv": "puv"}[connection_mode]
    z = '' if shared_weights else 'z'
    einstr = f"{z}{weight_label},z{u}i,z{v}j,pkij->z{w}k"

    weight_shape = {
        "uvw": (layout_in1.mul, layout_in2.mul, layout_out.mul),
        "uuu": (layout_in1.mul,),
        "uvv": (layout_in1.mul, layout_in2.mul),
    }[connection_mode]
    weight_shape = (len(instructions),) + weight_shape
    if not shared_weights:
        weight_shape = (-1,) + weight_shape

    # generate actual code
    graph_out = fx.Graph()

    # = Function definitions =
    x1s_out = fx.Proxy(graph_out.placeholder('x1', torch.Tensor))
    x2s_out = fx.Proxy(graph_out.placeholder('x2', torch.Tensor))
    ws_out = fx.Proxy(graph_out.placeholder('w', torch.Tensor))
    w3j_proxy = fx.Proxy(graph_out.get_attr("_big_w3j"))
    in1_convert = fx.Proxy(graph_out.get_attr("_in1_convert"))
    in2_convert = fx.Proxy(graph_out.get_attr("_in2_convert"))
    out_convert = fx.Proxy(graph_out.get_attr("_out_convert"))

    # flatten batch dims
    batch_shape = x1s_out.shape[:-1]

    # convert to strided
    x1s_out = x1s_out.view(-1, irreps_in1.dim)[:, in1_convert].reshape(-1, layout_in1.mul, layout_in1.base_dim)
    x2s_out = x2s_out.view(-1, irreps_in2.dim)[:, in2_convert].reshape(-1, layout_in2.mul, layout_in2.base_dim)

    ws_out = ws_out.reshape(weight_shape)

    # do the einsum
    # has shape zwk
    out = torch.einsum(einstr, ws_out, x1s_out, x2s_out, w3j_proxy)
    out = out.reshape(-1, layout_out.dim)
    out = out[:, out_convert]
    # bring back batch shape
    out = out.reshape(batch_shape + (irreps_out.dim,))

    graph_out.output(out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    constants_root.register_buffer("_big_w3j", w3j)
    constants_root.register_buffer("_in1_convert", layout_in1.indexes_to_strided)
    constants_root.register_buffer("_in2_convert", layout_in2.indexes_to_strided)
    constants_root.register_buffer("_out_convert", layout_out.indexes_to_catted)
    graphmod_out = fx.GraphModule(constants_root, graph_out, class_name="tp_forward")

    return graphmod_out
