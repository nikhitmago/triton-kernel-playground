import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')


@triton.jit
def fused_softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):

    # Pointer stuff where grid is (n_rows,)
    row_idx = tl.program_id(axis=0)

    # Offset is BLOCK_SIZE (n_cols and rest masked because BLOCK_SIZE needs to be a power of 2)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Strided offset
    offsets = row_idx * n_cols + col_offsets

    # Fused softmax
    row = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))  # masked values are -inf as it works for all 3 (max, exp and sum)
    m = tl.max(row)  # in max case, -inf gets replaced with max val, so all good
    numerator = tl.exp(row - m)  # in exp case, exp(-inf - m) => exp(-inf) => 1/exp(inf) => 1/inf => 0
    denominator = tl.sum(numerator)  # in sum case, we just sum 0's from above

    out = numerator / denominator

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_softmax(x: torch.tensor):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    fused_softmax_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
