import torch


def unravel_index(indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Converts a tensor of flat indices into a tensor of coordinate vectors.
    This is a `torch` implementation of `numpy.unravel_index`.

    Parameters
    ----------
    indices : torch.Tensor
        Tensor of flat indices. (*,)
    shape : torch.Size
        Target shape.

    Returns
    -------
    torch.Tensor
        The unraveled coordinates, (*, D).

    Notes
    -------
    See: https://github.com/pytorch/pytorch/issues/35674
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape[:-1]


def log2cosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable version of log(2*cosh(x)).

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        log(2*cosh(x))
    """
    return torch.abs(x) + torch.log1p(torch.exp(-2 * torch.abs(x)))
