import pytest
import torch

from vector_add import add, DEVICE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)


@requires_cuda
@pytest.mark.parametrize("n", [1, 4, 7, 128, 1000, 100_003])
def test_add_matches_torch(n):
    x = torch.randn(n, device=DEVICE)
    y = torch.randn(n, device=DEVICE)
    out = add(x, y)
    torch.testing.assert_close(out, x + y)


@requires_cuda
def test_add_non_multiple_of_block_size():
    # n not divisible by BLOCK_SIZE (=4) exercises the mask on the last block
    x = torch.randn(103, device=DEVICE)
    y = torch.randn(103, device=DEVICE)
    torch.testing.assert_close(add(x, y), x + y)


@requires_cuda
def test_add_preserves_shape_and_dtype():
    x = torch.randn(50, device=DEVICE)
    y = torch.randn(50, device=DEVICE)
    out = add(x, y)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
