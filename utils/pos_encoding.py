import torch
import torch.nn as nn



def to_log_freq(x, N_freqs, dim):
    """
    Logarithmic frequency embedding of the input tensor.

    Args:
        x (torch.Tensor): Input tensor to be embedded.
        N_freqs (int): Number of frequency bands.
        dim (int): Dimension along which to concatenate the embedded frequencies.

    Returns:
        torch.Tensor: Embedded tensor containing the frequencies.
    """
    include_input = True

    max_freq = N_freqs - 1
    log_sampling = True
    periodic_fns = [torch.sin, torch.cos]

    embed_fns = []
    d = x.shape[dim]
    out_dim = 0
    if include_input:
        embed_fns.append(lambda x: x)
        out_dim += d

    if log_sampling:
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
    else:
        freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

    for freq in freq_bands:
        for p_fn in periodic_fns:
            embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            out_dim += d

    embed_fns = embed_fns
    out_dim = out_dim

    return torch.cat([fn(x) for fn in embed_fns], -1)