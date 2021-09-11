from math import sqrt
import itertools
from collections import defaultdict
from typing import List, Tuple, NamedTuple, Optional

from opt_einsum_fx import jitable, optimize_einsums_full
import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction
from ._codegen import _sum_tensors


class FusedInstruction(NamedTuple):
    i_in1_start: int
    i_in1_end: int  # inclusive
    i_in2_start: int
    i_in2_end: int  # inclusive
    is_out: List[List[int]]
    connection_mode: str
    has_weight: bool
    path_weight: List[List[float]]
    base_path_shape: tuple

    @property
    def i_in1_num(self) -> int:
        return 1 + self.i_in1_end - self.i_in1_start

    @property
    def i_in2_num(self) -> int:
        return 1 + self.i_in2_end - self.i_in2_start

    @property
    def path_shape(self) -> tuple:
        out = []
        if self.i_in1_num > 1:
            out.append(self.i_in1_num)
        if self.i_in2_num > 1:
            out.append(self.i_in2_num)
        return tuple(out) + self.base_path_shape
 
# to fuse everything must be the same except i_in1, i_in2, i_out, and parity
def fuse_instructions(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction]
) -> List[FusedInstruction]:
    print(f"Initially {len(instructions)} instructions")
    # To fuse, it must be that:
    # - their i_in1s are consecutive
    # - their i_in2s are consecutive
    # - their connection mode, weightedness, and path_shape are the same
    old_inst = instructions
    instructions = list(instructions)  # copy
    fused_instructions: List[FusedInstruction] = []

    # Find ranges of same L and mul in the irreps that could be fused
    def _irrep_range(irreps: o3.Irreps) -> List[Tuple[int, int]]:
        if len(irreps) == 0:
            return []
        out = []
        cur_start = 0
        cur_l: int = irreps[0].ir.l
        cur_mul: int = irreps[0].mul
        for i, (mul, ir) in enumerate(irreps):
            if ir.l != cur_l or mul != cur_mul:
                # break the stretch
                out.append((cur_start, i - 1))
                cur_start = i
                cur_l = ir.l
                cur_mul = mul
        # The final one is always a break
        out.append((cur_start, i))
        return out

    in1_ranges = _irrep_range(irreps_in1)
    in2_ranges = _irrep_range(irreps_in2)

    # Now, we want to figure out if any "full rectangular" products
    # between these ranges are represented in `instructions`
    # TODO: this currently misses "partial" rectangular products that could still be accelerated
    def _fuse_key(ins: Instruction) -> tuple:
        # By checking irrep ranges we already know that path shapes are compat
        return (ins.connection_mode, ins.has_weight, irreps_out[ins.i_out].ir.l)
    have_products = defaultdict(set)
    for ins in instructions:
        have_products[_fuse_key(ins)].add((ins.i_in1, ins.i_in2))

    for in1_range, in2_range in itertools.product(in1_ranges, in2_ranges):
        # iterate through all paths that would be needed to make this happen
        # and check if they are present
        want_products = set(itertools.product(
            range(in1_range[0], in1_range[1] + 1),
            range(in2_range[0], in2_range[1] + 1)
        ))
        # Now check each set of fusable instructions
        for this_fuse_key, these_have_products in have_products.items():
            if not want_products.issubset(these_have_products):
                # We don't have what we need, ignore this one
                continue
            # We have all the products we need
            # So now lets make a fusion group
            group: List[Instruction] = []
            for i1, i2 in want_products:
                to_fuse = next(
                    ins
                    for ins in instructions
                    if ins.i_in1 == i1 and ins.i_in2 == i2 and _fuse_key(ins) == this_fuse_key
                )
                group.append(to_fuse)
                # Remove it from those left to process
                instructions.remove(to_fuse)
            assert len(group) == len(want_products)
            # Now fuse them
            fused_instructions.append(FusedInstruction(
                i_in1_start=in1_range[0], i_in1_end=in1_range[1],
                i_in2_start=in2_range[0], i_in2_end=in2_range[1],
                is_out=[
                    [
                        next(
                            ins.i_out
                            for ins in group
                            if ins.i_in1 == i1 and ins.i_in2 == i2
                        )
                        for i2 in range(in2_range[0], in2_range[1] + 1)
                    ]
                    for i1 in range(in1_range[0], in1_range[1] + 1)
                ],
                path_weight=[
                    [
                        next(
                            ins.path_weight
                            for ins in group
                            if ins.i_in1 == i1 and ins.i_in2 == i2
                        )
                        for i2 in range(in2_range[0], in2_range[1] + 1)
                    ]
                    for i1 in range(in1_range[0], in1_range[1] + 1)
                ],
                connection_mode=to_fuse.connection_mode,
                has_weight=to_fuse.has_weight,
                base_path_shape=to_fuse.path_shape
            ))
    
    # Finally, we just conver the remaining single instructions to "fused"
    # ones of a single instruction:
    for ins in instructions:
        fused_instructions.append(FusedInstruction(
            i_in1_start=ins.i_in1, i_in1_end=ins.i_in1,
            i_in2_start=ins.i_in2, i_in2_end=ins.i_in2,
            is_out=[[ins.i_out]],
            path_weight=[[ins.path_weight]],
            connection_mode=ins.connection_mode,
            has_weight=ins.has_weight,
            base_path_shape=ins.path_shape
        ))

    print(f"After fusion, {len(fused_instructions)} instructions")

    return fused_instructions



def codegen_tensor_product_fused(
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
    # FUSE!!
    instructions = fuse_instructions(irreps_in1, irreps_in2, irreps_out, instructions)

    graph_out = fx.Graph()

    # Build a dummy right() that throws an error
    graph_right = fx.Graph()
    graph_right.placeholder("y")
    graph_right.placeholder("w")
    graph_right.call_function(torch._assert, (False, "Fused doesn't support right()"))

    # = Function definitions =
    x1s_out = fx.Proxy(graph_out.placeholder('x1', torch.Tensor))
    x2s_out = fx.Proxy(graph_out.placeholder('x2', torch.Tensor))
    ws_out = fx.Proxy(graph_out.placeholder('w', torch.Tensor))

    empty_out = fx.Proxy(graph_out.call_function(torch.empty, ((),), dict(device='cpu')))
    if shared_weights:
        size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]))[0].shape
    else:
        size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]), empty_out.expand(ws_out.shape[:-1]))[0].shape

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        out_out = x1s_out.new_zeros(size_out + (irreps_out.dim,))

        graph_out.output(out_out.node, torch.Tensor)
        # Short circut
        return (
            fx.GraphModule({}, graph_out, "tp_forward"),
            fx.GraphModule({}, graph_right, "tp_right")
        )

    # = Broadcast inputs =
    if shared_weights:
        x1s_out, x2s_out = x1s_out.broadcast_to(size_out + (-1,)), x2s_out.broadcast_to(size_out + (-1,))
    else:
        x1s_out, x2s_out, ws_out = x1s_out.broadcast_to(size_out + (-1,)), x2s_out.broadcast_to(size_out + (-1,)), ws_out.broadcast_to(size_out + (-1,))

    outsize_out = size_out + (irreps_out.dim,)

    x1s_out = x1s_out.reshape(-1, irreps_in1.dim)
    x2s_out = x2s_out.reshape(-1, irreps_in2.dim)

    batch_out = x1s_out.shape[0]

    # = Determine number of weights and reshape weights ==
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        ws_out = ws_out.reshape(-1, weight_numel)
    del weight_numel

    # = book-keeping for wigners =
    w3j = []
    w3j_dict_out = dict()
    w3j_dict_right = dict()

    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Current index in the flat weight tensor
    flat_weight_index = 0

    out_list_out = [list() for _ in irreps_out]

    for ins in instructions:
        # Check it
        for fuse_i1, fuse_i2 in itertools.product(range(ins.i_in1_num), range(ins.i_in2_num)):
            mul_ir_in1 = irreps_in1[ins.i_in1_start + fuse_i1]
            mul_ir_in2 = irreps_in2[ins.i_in2_start + fuse_i2]
            mul_ir_out = irreps_out[ins.is_out[fuse_i1][fuse_i2]]
            # Check valid tp
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            # Check allowed for fuse because all same L1 x L2 -> L_out
            assert mul_ir_in1.ir.l == irreps_in1[ins.i_in1_start].ir.l
            assert mul_ir_in2.ir.l == irreps_in2[ins.i_in2_start].ir.l
            assert mul_ir_out.ir.l == irreps_out[ins.is_out[0][0]].ir.l
            assert mul_ir_in1.mul == irreps_in1[ins.i_in1_start].mul
            assert mul_ir_in2.mul == irreps_in2[ins.i_in2_start].mul
            assert mul_ir_out.mul == irreps_out[ins.is_out[0][0]].mul

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

        # Open the profiler block
        name = f"{mul_ir_in1} x {mul_ir_in2} = {mul_ir_out} {ins.connection_mode} {ins.has_weight}"
        handle_out = graph_out.call_function(torch.ops.profiler._record_function_enter, (name,))

        # Take the full fused set of Xs
        x1_out = x1s_out.narrow(
            -1,
            irreps_in1[:ins.i_in1_start].dim,
            irreps_in1[ins.i_in1_start:ins.i_in1_end + 1].dim
        )
        if ins.i_in1_num > 1:
            x1_out = x1_out.reshape(-1, ins.i_in1_num, mul_ir_in1.mul, mul_ir_in1.ir.dim)
        else:
            x1_out = x1_out.reshape(-1, mul_ir_in1.mul, mul_ir_in1.ir.dim)
        x2_out = x2s_out.narrow(
            -1,
            irreps_in2[:ins.i_in2_start].dim,
            irreps_in2[ins.i_in2_start:ins.i_in2_end + 1].dim
        )
        if ins.i_in2_num > 1:
            x2_out = x2_out.reshape(-1, ins.i_in2_num, mul_ir_in2.mul, mul_ir_in2.ir.dim)
        else:
            x2_out = x2_out.reshape(-1, mul_ir_in2.mul, mul_ir_in2.ir.dim)

        if ins.has_weight:
            # Extract the weight from the flattened weight tensor
            w_out = ws_out[:, flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape(
                (() if shared_weights else (-1,)) + tuple(ins.path_shape)
            )
            flat_weight_index += prod(ins.path_shape)

        p1 = "p" if ins.i_in1_num > 1 else ""
        p2 = "q" if ins.i_in2_num > 1 else ""

        if ins.connection_mode == "uvuv":
            u = "u"
            v = "v"
            w = "uv"
        else:
            u, v, w = ins.connection_mode
        w_subs = {
            "uvw": "uvw",
            "uvv": "uv",
            "uuu": "u",
            "uvu": "uv",
            "uuw": "uw",
            "uvuv": "uv"
        }[ins.connection_mode]

        # Construct the general xx in case this instruction isn't specialized
        # If this isn't used, the dead code will get removed
        key = (ins.i_in1_start, ins.i_in2_end, ins.i_in2_start, ins.i_in2_end, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == 'uv':
                xx_dict[key] = torch.einsum(f'z{p1}ui,z{p2}vj->z{p1}{p2}uvij', x1_out, x2_out)
            if ins.connection_mode[:2] == 'uu':
                xx_dict[key] = torch.einsum(f'z{p1}ui,z{p2}uj->z{p1}{p2}uij', x1_out, x2_out)
        xx = xx_dict[key]
        xx_label = {
            "uv": "uv",
            "uu": "u"
        }[ins.connection_mode[:2]]

        # Create a proxy & request for the relevant wigner w3j
        # If not used (because of specialized code), will get removed later.
        key = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if key not in w3j:
            w3j_dict_out[key] = fx.Proxy(graph_out.get_attr(f"_w3j_{key[0]}_{key[1]}_{key[2]}"))
            w3j.append(key)
        w3j_out = w3j_dict_out[key]

        exp = {'component': -1, 'norm': 1}[normalization]
        
        if ins.has_weight:
            if specialized_code and key == (0, 0, 0):
                ein_out = torch.einsum(
                    f"{z}{p1}{p2}{w_subs},z{p1}{u},z{p2}{v}->z{p1}{p2}{w}",
                    w_out,
                    x1_out.squeeze(-1),
                    x2_out.squeeze(-1)
                )
            elif specialized_code and mul_ir_in1.ir.l == 0:
                ein_out = torch.einsum(
                    f"{z}{p1}{p2}{w_subs},z{p1}{u},z{p2}{v}j->z{p1}{p2}{w}j",
                    w_out,
                    x1_out.squeeze(-1),
                    x2_out
                )
            elif specialized_code and mul_ir_in2.ir.l == 0:
                ein_out = torch.einsum(
                f"{z}{p1}{p2}{w_subs},z{p1}{u}i,z{p2}{v}->z{p1}{p2}{w}i",
                    w_out,
                    x1_out,
                    x2_out.squeeze(-1)
                )
            elif specialized_code and mul_ir_out.ir.l == 0:
                ein_out = torch.einsum(
                    f"{z}{p1}{p2}{w_subs},z{p1}{u}i,z{p2}{v}i->z{p1}{p2}{w}",
                    w_out,
                    x1_out,
                    x2_out
                ) * sqrt(mul_ir_in1.ir.dim)**exp
            else:
                ein_out = torch.einsum(
                    f"{z}{p1}{p2}{w_subs},ijk,z{p1}{p2}{xx_label}ij->z{p1}{p2}{w}k",
                    w_out,
                    w3j_out,
                    xx
                )
        else:
            if specialized_code and key == (0, 0, 0):
                ein_out = torch.einsum(
                    f"z{p1}{u},z{p2}{v}->z{p1}{p2}{w}",
                    x1_out.squeeze(-1),
                    x2_out.squeeze(-1)
                )
            elif specialized_code and mul_ir_in1.ir.l == 0:
                ein_out = torch.einsum(
                    f"z{p1}{u},z{p2}{v}j->z{p1}{p2}{w}j",
                    x1_out.squeeze(-1),
                    x2_out
                )
            elif specialized_code and mul_ir_in2.ir.l == 0:
                ein_out = torch.einsum(
                f"z{p1}{u}i,z{p2}{v}->z{p1}{p2}{w}i",
                    x1_out,
                    x2_out.squeeze(-1)
                )
            elif specialized_code and mul_ir_out.ir.l == 0:
                ein_out = torch.einsum(
                    f"z{p1}{u}i,z{p2}{v}i->z{p1}{p2}{w}",
                    x1_out,
                    x2_out
                ) * sqrt(mul_ir_in1.ir.dim)**exp
            else:
                ein_out = torch.einsum(
                    f"ijk,z{p1}{p2}{xx_label}ij->z{p1}{p2}{w}k",
                    w3j_out,
                    xx
                )

        # Extract individual paths and normalize them
        # TODO: check if pqz or zpq is better perf, must change above too
        ein_out = ein_out.reshape(batch_out, ins.i_in1_num, ins.i_in2_num, mul_ir_out.dim)
        for fuse_i1, fuse_i2 in itertools.product(range(ins.i_in1_num), range(ins.i_in2_num)):
            # Compute the normalization
            i_out = ins.is_out[fuse_i1][fuse_i2]
            alpha = ins.path_weight[fuse_i1][fuse_i2] * out_var[i_out] / sum(
                in1_var[i.i_in1_start + i1] * in2_var[i.i_in2_start + i2]
                for i in instructions
                for i1, i2 in itertools.product(range(i.i_in1_num), range(i.i_in2_num))
                if i_out == i.is_out[i1][i2]
            )
            alpha = sqrt(alpha / {
                'uvw': (mul_ir_in1.mul * mul_ir_in2.mul),
                'uvu': mul_ir_in2.mul,
                'uvv': mul_ir_in1.mul,
                'uuw': mul_ir_in1.mul,
                'uuu': 1,
                'uvuv': 1,
            }[ins.connection_mode])
            # Extract the path:
            out_list_out[i_out].append(alpha * ein_out[:, fuse_i1, fuse_i2])

        # Close the profiler block
        graph_out.call_function(torch.ops.profiler._record_function_exit, (handle_out,))

        # Remove unused w3js:
        if len(w3j_out.node.users) == 0:
            del w3j[-1]
            # The w3j nodes are reshapes, so we have to remove them from the graph
            # Although they are dead code, they try to reshape to dimensions that don't exist
            # (since the corresponding w3js are not in w3j)
            # so they screw up the shape propagation, even though they would be removed later as dead code by TorchScript.
            graph_out.erase_node(w3j_dict_out.pop(key).node)

    # = Return the result =
    out_out = [
        _sum_tensors(
            ol,
            shape=(batch_out, mul_ir_out.dim),
            like=x1s_out
        )
        for ol, mul_ir_out in zip(out_list_out, irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(out_out) > 1:
        out_out = torch.cat(out_out, dim=1)
    else:
        # Avoid an unnecessary copy in a size one torch.cat
        out_out = out_out[0]

    out_out = out_out.reshape(outsize_out)
    graph_out.output(out_out.node, torch.Tensor)

    # check graphs
    graph_out.lint()

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
    graphmod_out = fx.GraphModule(constants_root, graph_out, class_name="tp_forward")
    graphmod_right = fx.GraphModule({}, graph_right, class_name="tp_right")

    # == Optimize ==
    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, irreps_in1.dim)),
            torch.zeros((batchdim, irreps_in2.dim)),
            torch.zeros(
                1 if shared_weights else batchdim,
                flat_weight_index,
            ),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    return graphmod_out, graphmod_right
