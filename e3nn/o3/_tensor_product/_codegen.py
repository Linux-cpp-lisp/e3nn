from math import sqrt
from typing import List, Tuple

from opt_einsum_fx import jitable, optimize_einsums_full
import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction
from ._codegen_default import codegen_default_tensor_product_forward
from ._codegen_right import codegen_default_tensor_product_right
from ._codegen_strided import codegen_strided_tensor_product_forward


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
    compile_right: bool = True,
    specialized_code: bool = True,
    optimize_einsums: bool = True,  # TODO: generic optimization options dict!
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """Central codegen method that dispatches to specific implementations based on options."""
    # preprocess commands
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    # make dict
    kwargs = dict(
        irreps_in1=irreps_in1,
        in1_var=in1_var,
        irreps_in2=irreps_in2,
        in2_var=in2_var,
        irreps_out=irreps_out,
        out_var=out_var,
        instructions=instructions,
        normalization=normalization,
        shared_weights=shared_weights,
        specialized_code=specialized_code
    )
    # codegen:
    graphmod_out = codegen_strided_tensor_product_forward(**kwargs)
    if graphmod_out is None:
        # fallback to default
        graphmod_out = codegen_default_tensor_product_forward(**kwargs)
    if compile_right:
        graphmod_right = codegen_default_tensor_product_right(**kwargs)
    else:
        # TODO: make some fake graph that throws an error
        graphmod_right = None
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
                sum(prod(ins.path_shape) for ins in instructions if ins.has_weight),
            ),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))
        if compile_right:
            graphmod_right = jitable(optimize_einsums_full(graphmod_right, example_inputs[1:]))

    return graphmod_out, graphmod_right
