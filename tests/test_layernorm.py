import pytest
import torch
import torch.nn.functional as F

from layernorm import layernorm_fwd_fused, DEVICE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)

EPS = 1e-5


@requires_cuda
@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(1, 1), (4, 5), (64, 512), (128, 781), (1823, 1000), (8, 3)],
)
def test_layernorm_matches_torch(n_rows, n_cols):
    x = torch.randn(n_rows, n_cols, device=DEVICE)
    scale = torch.randn(n_cols, device=DEVICE)
    shift = torch.randn(n_cols, device=DEVICE)
    out = layernorm_fwd_fused(x, scale, shift, EPS)
    ref = F.layer_norm(x, (n_cols,), weight=scale, bias=shift, eps=EPS)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@requires_cuda
def test_layernorm_non_power_of_two_cols():
    # n_cols=781 -> BLOCK_SIZE padded to 1024; exercises the variance-masking
    # fix (padding lanes must not pollute the variance sum).
    x = torch.randn(32, 781, device=DEVICE)
    scale = torch.randn(781, device=DEVICE)
    shift = torch.randn(781, device=DEVICE)
    torch.testing.assert_close(
        layernorm_fwd_fused(x, scale, shift, EPS),
        F.layer_norm(x, (781,), weight=scale, bias=shift, eps=EPS),
        atol=1e-5,
        rtol=1e-4,
    )


@requires_cuda
def test_layernorm_identity_affine_normalizes():
    # With scale=1, shift=0, each row should be ~mean 0, ~var 1.
    x = torch.randn(16, 256, device=DEVICE)
    scale = torch.ones(256, device=DEVICE)
    shift = torch.zeros(256, device=DEVICE)
    out = layernorm_fwd_fused(x, scale, shift, EPS)
    means = out.mean(dim=-1)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-4, rtol=0)
    # biased variance (matches LayerNorm's 1/N) should be ~1
    var = out.var(dim=-1, unbiased=False)
    torch.testing.assert_close(var, torch.ones_like(var), atol=1e-2, rtol=1e-2)
