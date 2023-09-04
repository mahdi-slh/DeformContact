import torch
import torch.nn as nn

# def to_freqs(x, magn, num_freqs, dim):
#     """
#     Embeds the input tensor into a set of sinusoidal frequencies with varying magnitudes.

#     Args:
#         x (torch.Tensor): Input tensor to be embedded.
#         magn (float): Initial magnitude value.
#         num_freqs (int): Number of frequencies to embed.
#         dim (int): Dimension along which to concatenate the frequencies.

#     Returns:
#         torch.Tensor: Embedded tensor containing the sinusoidal frequencies.
#     """
#     r = x.clone()
#     f = []
#     for _ in range(num_freqs):
#         f.append(r / magn)
#         r -= r // magn
#         magn = magn / 2
#     return torch.cat(f, dim)

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

def from_log_freqs(embedded_tensor, original_shape, N_freqs, dim):
    """
    Inverse operation to convert embedded frequencies back to the original values.

    Args:
        embedded_tensor (torch.Tensor): Embedded tensor containing frequencies.
        original_shape (tuple): Original shape of the input tensor.
        N_freqs (int): Number of frequency bands.
        dim (int): Dimension along which frequencies were concatenated.

    Returns:
        torch.Tensor: Restored tensor with original values.
    """
    include_input = True

    max_freq = N_freqs - 1
    log_sampling = True
    periodic_fns = [torch.sin, torch.cos]

    d = original_shape[dim]
    out_dim = 0
    if include_input:
        out_dim += d

    if log_sampling:
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
    else:
        freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

    num_funcs = len(periodic_fns)
    num_freqs = len(freq_bands)

    input_fns = []
    for func_idx in range(num_funcs):
        for freq_idx in range(num_freqs):
            input_fns.append(
                lambda y, func_idx=func_idx, freq_idx=freq_idx: periodic_fns[func_idx](y / freq_bands[freq_idx])
            )

    restored_values = []
    start_idx = 0
    for input_fn in input_fns:
        end_idx = start_idx + d
        restored_values.append(input_fn(embedded_tensor[..., start_idx:end_idx]))
        start_idx = end_idx

    return torch.cat(restored_values, dim=dim)
