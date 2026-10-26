"""Indexed real interpolation and Galerkin actions without assembled matrices.

For triples (src_fine, dst_coarse, weight), P[src,dst] += weight.  ``prolong``
is P and ``restrict`` is its real transpose P.T, both by gather/index_add.
Weights must be real float64; vectors may be float64 or complex128.  This keeps
the same transpose transfer for a future complex-symmetric harmonic operator.
No coarse inverse is stored: the coarsest level must also use graph iteration.
These are conventional transfer primitives, not a learned multigrid method.
"""

import torch


class IndexedTransfer:
    def __init__(self, src, dst, weights, n_fine=None, n_coarse=None):
        if (
            not isinstance(weights, torch.Tensor)
            or weights.ndim != 1
            or weights.layout != torch.strided
        ):
            raise TypeError("weights must be a dense one-dimensional tensor")
        if weights.dtype != torch.float64 or not bool(torch.isfinite(weights).all().item()):
            raise TypeError("weights must be finite real float64")
        source = torch.as_tensor(src, device=weights.device)
        destination = torch.as_tensor(dst, device=weights.device)
        integer_types = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
        if source.dtype not in integer_types or destination.dtype not in integer_types:
            raise TypeError("src/dst must contain integer indices")
        if (
            source.ndim != 1
            or destination.ndim != 1
            or source.shape != weights.shape
            or destination.shape != weights.shape
        ):
            raise ValueError("src,dst,weights must have equal one-dimensional shapes")
        if source.numel() and (int(source.min().item()) < 0 or int(destination.min().item()) < 0):
            raise ValueError("Indices must be nonnegative")
        inferred_fine = int(source.max().item()) + 1 if source.numel() else 0
        inferred_coarse = int(destination.max().item()) + 1 if destination.numel() else 0
        self.n_fine = inferred_fine if n_fine is None else n_fine
        self.n_coarse = inferred_coarse if n_coarse is None else n_coarse
        for name, size, minimum in (
            ("n_fine", self.n_fine, inferred_fine),
            ("n_coarse", self.n_coarse, inferred_coarse),
        ):
            if isinstance(size, bool) or not isinstance(size, int) or size < minimum:
                raise ValueError("%s must be an integer covering all indices" % name)
        self.src = source.to(torch.int64).detach().clone()
        self.dst = destination.to(torch.int64).detach().clone()
        self.weights = weights.detach().clone()

    def _check(self, vector, size):
        if (
            not isinstance(vector, torch.Tensor)
            or vector.layout != torch.strided
            or vector.ndim != 1
            or vector.numel() != size
        ):
            raise ValueError("Wrong dense transfer-vector size")
        if (
            vector.dtype not in (torch.float64, torch.complex128)
            or vector.device != self.weights.device
        ):
            raise TypeError("Transfer vector must be double/complex-double on the weights device")
        return vector

    def prolong(self, coarse):
        coarse = self._check(coarse, self.n_coarse)
        fine = torch.zeros(self.n_fine, dtype=coarse.dtype, device=coarse.device)
        return fine.index_add(0, self.src, self.weights * coarse[self.dst])

    def restrict(self, fine):
        fine = self._check(fine, self.n_fine)
        coarse = torch.zeros(self.n_coarse, dtype=fine.dtype, device=fine.device)
        return coarse.index_add(0, self.dst, self.weights * fine[self.src])


class GalerkinCoarseAction:
    """Callable c -> P.T A(P c); no fine/coarse matrix or factorization."""

    def __init__(self, fine_operator, transfer):
        if not callable(fine_operator) or not isinstance(transfer, IndexedTransfer):
            raise TypeError("Need a callable fine operator and IndexedTransfer")
        self.fine_operator = fine_operator
        self.transfer = transfer
        self.calls = 0

    def __call__(self, coarse):
        self.calls += 1
        fine_result = self.fine_operator(self.transfer.prolong(coarse))
        return self.transfer.restrict(fine_result)


def coarse_action(fine_operator, transfer):
    return GalerkinCoarseAction(fine_operator, transfer)
