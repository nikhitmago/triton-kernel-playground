import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')

# NOTE: this is the case where n_cols <= BLOCK_SIZE, i.e. the whole row fits in
# one block/SRAM and is loaded once (BLOCK_SIZE = next_power_of_2(n_cols)).
# TODO: handle the reverse case (n_cols > BLOCK_SIZE) where the row does not fit
#       in a single block -- loop over the row in chunks and go multi-pass
#       (accumulate mean, re-read for variance, re-read to normalize).


@triton.jit
def layernorm_fwd_fused_kernel(x_ptr, out_ptr, scale_ptr, shift_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):

    # Pointers stuff
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    offsets_x = row_idx * n_cols + col_offsets
    offsets_params = col_offsets

    # Read data into SRAM
    row = tl.load(x_ptr + offsets_x, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets_params, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + offsets_params, mask=mask, other=0.0)

    # Compute layer norm

    # mean: padding lanes loaded as 0 (other=0.0), which is the identity for
        # sum — they add nothing, and we divide by the true n_cols. So mean is correct.
    mean = tl.sum(row) / n_cols

     # var: CAREFUL — after subtracting mean, padding lanes become (0 - mean) = -mean,
        # which is NOT 0. Squaring gives +mean² per padding lane, polluting the sum.
        # So re-mask the deviations to 0 before squaring (tl.where), else variance is
        # inflated whenever n_cols is not a power of 2.
    diff = row - mean
    diff = tl.where(mask, diff, 0.0)
    var = tl.sum(diff * diff) / n_cols

    layer_norm = (row - mean) / tl.sqrt(var + eps)

    # Scale and shift
    out = layer_norm * scale + shift
    tl.store(out_ptr + offsets_x, out, mask=mask)


def layernorm_fwd_fused(x: torch.tensor, scale: torch.tensor, shift: torch.tensor, eps: float):
    """
    x is a 2d tensor
    scale and shift are 1d tensors of shape (n_cols,) where each value represents parameter for each feature
        which gives each feature it's own learned mean and variance that is learned and diveregd from 1 and 0
    """
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    layernorm_fwd_fused_kernel[grid](
        x,
        out,
        scale,
        shift,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
