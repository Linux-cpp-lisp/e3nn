import pytest

from e3nn.util.test import random_irreps

from e3nn.util.strided import StridedLayout

@pytest.mark.parametrize("irreps", random_irreps(n=10))
@pytest.mark.parametrize("padding", [1, 4])
def test_make_strided(irreps, padding):
    if StridedLayout.can_be_strided(irreps):
        layout = StridedLayout(irreps, pad_to_multiple=padding)