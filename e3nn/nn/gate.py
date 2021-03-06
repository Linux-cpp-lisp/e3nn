import torch

from e3nn import o3
from e3nn.nn import Extract
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode


@compile_mode('trace')
class Activation(torch.nn.Module):
    r"""Scalar activation function

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------

    >>> a = Activation("256x0o", [torch.abs])
    >>> a.irreps_out
    256x0e

    >>> a = Activation("256x0o+16x1e", [None, None])
    >>> a.irreps_out
    256x0o+16x1e
    """
    def __init__(self, irreps_in, acts):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError("Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.")
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.acts = torch.nn.ModuleList(acts)

    def forward(self, features, dim=-1):
        '''evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(...)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape the same shape as the input
        '''
        with torch.autograd.profiler.record_function(repr(self)):
            output = []
            index = 0
            for (mul, ir), act in zip(self.irreps_in, self.acts):
                if act is not None:
                    output.append(act(features.narrow(dim, index, mul)))
                else:
                    output.append(features.narrow(dim, index, mul * ir.dim))
                index += mul * ir.dim

            if output:
                return torch.cat(output, dim=dim)
            else:
                return torch.zeros_like(features)


@compile_mode('script')
class _Sortcut(torch.nn.Module):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(o3.Irreps(irreps).simplify() for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, o3.Irreps([]))

        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions += [tuple(range(i, i + len(irreps_out)))]
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)

        irreps_in, p, _ = irreps_in.sort()
        instructions = [tuple(p[i] for i in x) for x in instructions]

        self.cut = Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in.simplify()

    def forward(self, x):
        return self.cut(x)


@compile_mode('script')
class Gate(torch.nn.Module):
    r"""Gate activation function.

    The gate activation is a direct sum of two sets of irreps. The first set
    of irreps is ``irreps_scalars`` passed through activation functions
    ``act_scalars``. The second set of irreps is ``irreps_gated`` multiplied
    by the scalars ``irreps_gates`` passed through activation functions
    ``act_gates``. Mathematically, this can be written as:

    .. math::
        \left(\bigoplus_i \phi_i(x_i) \right) \oplus \left(\bigoplus_j \phi_j(g_j) y_j \right)

    where :math:`x_i` and :math:`\phi_i` are from ``irreps_scalars`` and
    ``act_scalars``, and :math:`g_j`, :math:`\phi_j`, and :math:`y_j` are
    from ``irreps_gates``, ``act_gates``, and ``irreps_gated``.

    The parameters passed in should adhere to two conditions:

    1. ``len(irreps_scalars) == len(act_scalars)``.
    2. ``len(irreps_gates) == len(act_gates)``.
    3. ``irreps_gates.num_irreps == irreps_gated.num_irreps``.

    Parameters
    ----------
    irreps_scalars : `Irreps`
        Representation of the scalars that will be passed through the
        activation functions ``act_scalars``.

    act_scalars : list of function or None
        Activation functions acting on the scalars.

    irreps_gates : `Irreps`
        Representation of the scalars that will be passed through the
        activation functions ``act_gates`` and multiplied by the
        ``irreps_gated``.

    act_gates : list of function or None
        Activation functions acting on the gates. The number of functions in
        the list should match the number of irrep groups in ``irreps_gates``.

    irreps_gated : `Irreps`
        Representation of the gated tensors.
        ``irreps_gates.num_irreps == irreps_gated.num_irreps``

    Examples
    --------

    >>> g = Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")
    >>> g.irreps_out
    16x0o+16x1o+16x1e
    """
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)
        self.irreps_scalars, self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
        self._irreps_in = self.sc.irreps_in

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        """Evaluate the gated activation function.

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        with torch.autograd.profiler.record_function('Gate'):
            scalars, gates, gated = self.sc(features)

            scalars = self.act_scalars(scalars)
            if gates.shape[-1]:
                gates = self.act_gates(gates)
                gated = self.mul(gated, gates)
                features = torch.cat([scalars, gated], dim=-1)
            else:
                features = scalars
            return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out
