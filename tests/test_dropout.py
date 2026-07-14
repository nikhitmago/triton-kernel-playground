import pytest
import torch

from dropout import dropout, DEVICE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)


@requires_cuda
@pytest.mark.parametrize("n", [1, 4, 7, 128, 1000, 100_003])
@pytest.mark.parametrize("p", [0.0, 0.5, 0.9])
def test_dropout_matches_reference(n, p):
    x = torch.randn(n, device=DEVICE)
    x_keep = torch.rand(n, device=DEVICE) > p
    out = dropout(x, x_keep, p)
    # reference: kept elements scaled by 1/(1-p), dropped elements zeroed
    ref = torch.where(x_keep, x / (1 - p), torch.zeros_like(x))
    torch.testing.assert_close(out, ref)


@requires_cuda
def test_dropout_kept_elements_are_scaled():
    x = torch.randn(1000, device=DEVICE)
    p = 0.5
    x_keep = torch.rand(1000, device=DEVICE) > p
    out = dropout(x, x_keep, p)
    # kept positions -> x / (1-p); dropped positions -> exactly 0
    torch.testing.assert_close(out[x_keep], x[x_keep] / (1 - p))
    assert torch.all(out[~x_keep] == 0.0)


@requires_cuda
def test_dropout_p_zero_is_identity_up_to_scale():
    # with p=0 and keep-all mask, output equals input (1/(1-0) = 1)
    x = torch.randn(256, device=DEVICE)
    x_keep = torch.ones(256, device=DEVICE, dtype=torch.bool)
    torch.testing.assert_close(dropout(x, x_keep, 0.0), x)


@requires_cuda
def test_dropout_non_multiple_of_block_size():
    # n not divisible by BLOCK_SIZE (=4) exercises the mask on the last block
    n = 103
    x = torch.randn(n, device=DEVICE)
    x_keep = torch.rand(n, device=DEVICE) > 0.3
    ref = torch.where(x_keep, x / (1 - 0.3), torch.zeros_like(x))
    torch.testing.assert_close(dropout(x, x_keep, 0.3), ref)
