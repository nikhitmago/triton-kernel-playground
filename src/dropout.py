import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')

# Seeded-mask dropout: the keep-mask is passed in as a tensor (x_keep), so the
# kernel is deterministic given that mask. Kept elements are scaled by 1/(1-p)
# (inverted dropout) so the expected activation magnitude is preserved and no
# scaling is needed at inference time.


@triton.jit
def dropout_kernel(x_ptr, x_keep_ptr, out_ptr, n, p, BLOCK_SIZE: tl.constexpr):
    # Pointer stuff
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Loading tensors on SRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    # Dropping out and back to HBM
    x_dropout = tl.where(x_keep, x / (1-p), 0.0)
    tl.store(out_ptr + offsets, x_dropout, mask=mask)


def dropout(x, x_keep, p):
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    dropout_kernel[grid](
        x,
        x_keep,
        out,
        n,
        p,
        BLOCK_SIZE=4
    )

    return out
