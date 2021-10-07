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
    return