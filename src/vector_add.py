import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def add(x: torch.tensor, y: torch.tensor):
    output = torch.empty_like(x)
    n = output.numel()

    assert output.device == DEVICE
    assert x.device == DEVICE
    assert y.device == DEVICE

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=4)

    return output
