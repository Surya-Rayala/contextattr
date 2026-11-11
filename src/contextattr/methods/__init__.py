"""Which masks to evaluate and which fit (docs/methods.md). A method is named or given as an object."""

from .base import Method, Surrogate, all_masks
from .indices import banzhaf, shapley, shapley_taylor, terms
from .interactions import SPEX, FaithBanzhaf, FaithShap, ProxySPEX, ShapiqMethod
from .kernelshap import KernelSHAP
from .linear import LIME, Banzhaf, ContextCite
from .loo import LeaveOneOut

METHODS = {"loo": LeaveOneOut, "banzhaf": Banzhaf, "lime": LIME, "contextcite": ContextCite,
           "kernelshap": KernelSHAP, "faithbanzhaf": FaithBanzhaf, "faithshap": FaithShap, "spex": SPEX,
           "proxyspex": ProxySPEX}


def get_method(method) -> Method:
    """A Method from its name (loo, banzhaf, lime, contextcite, kernelshap, faithbanzhaf, faithshap, spex,
    proxyspex) or the object itself."""
    if isinstance(method, Method):
        return method
    if isinstance(method, str) and method in METHODS:
        return METHODS[method]()
    raise ValueError(f"method must be one of {', '.join(METHODS)} or a Method, not {method!r}")


__all__ = ["LIME", "METHODS", "SPEX", "Banzhaf", "ContextCite", "FaithBanzhaf", "FaithShap", "KernelSHAP",
           "LeaveOneOut", "Method", "ProxySPEX", "ShapiqMethod", "Surrogate", "all_masks", "banzhaf", "get_method",
           "shapley", "shapley_taylor", "terms"]
