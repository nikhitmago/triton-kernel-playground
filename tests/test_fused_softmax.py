import pytest
import torch

from fused_softmax import fused_softmax, DEVICE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)


@requires_cuda
@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(1, 1), (3, 5), (8, 128), (1823, 781), (128, 1024)],
)
def test_softmax_matches_torch(n_rows, n_cols):
    x = torch.randn(n_rows, n_cols, device=DEVICE)
    out = fused_softmax(x)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)


@requires_cuda
def test_softmax_rows_sum_to_one():
    x = torch.randn(64, 781, device=DEVICE)
    out = fused_softmax(x)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


@requires_cuda
def test_softmax_numerically_stable_large_values():
    # Large logits would overflow exp() without the max-subtraction trick
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], device=DEVICE)
    out = fused_softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), atol=1e-6, rtol=1e-5)


@requires_cuda
def test_softmax_non_power_of_two_cols():
    # n_cols=781 -> BLOCK_SIZE padded to 1024; exercises the mask
    x = torch.randn(10, 781, device=DEVICE)
    torch.testing.assert_close(
        fused_softmax(x), torch.softmax(x, dim=-1), atol=1e-6, rtol=1e-5
    )
