import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')


@triton.jit
def rms_norm_kernel(x_ptr, scale_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):

    # Pointer stuff
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    offsets_x = row_idx * n_cols + col_offsets
    offsets_scale = col_offsets

    # Load data into registers
    x = tl.load(x_ptr + offsets_x, mask=mask, other=0.0)
    # No shift (bias) needed in RMSNorm -- unlike LayerNorm which subtracts the mean
    # and needs shift to restore it, RMSNorm never removes the mean so there's nothing to add back.
    scale = tl.load(scale_ptr + offsets_scale, mask=mask, other=0.0)

    # RMS norm
    square = x * x
    _sum = tl.sum(square)
    mean = _sum / n_cols
    root = tl.sqrt(mean + 1e-6)
    rms = x / root
    output = rms * scale

    # Back to HBM
    tl.store(out_ptr + offsets_x, output, mask=mask)


def rms_norm(x: torch.Tensor, scale: torch.Tensor):
    output = torch.zeros_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)

    block_size = triton.next_power_of_2(n_cols)

    rms_norm_kernel[grid](
        x,
        scale,
        output,
        n_cols,
        block_size
    )

    return output
