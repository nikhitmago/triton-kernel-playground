# triton-kernel-playground

Learning and playing with Triton GPU kernels for fun.

Each kernel is written from scratch and validated against its PyTorch equivalent.

## Kernels

| File | Kernel | Concepts |
|------|--------|----------|
| `src/vector_add.py` | element-wise add | program_id, offsets, mask, load/store |
| `src/fused_softmax.py` | row-wise fused softmax | 2D addressing, reductions, numerical stability (`-inf` trick) |
| `src/layernorm.py` | row-wise fused LayerNorm | mean/variance reductions, per-column affine, variance-masking fix |
| `src/dropout.py` | seeded-mask dropout | `tl.where`, inverted-dropout scaling `1/(1-p)` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU (Triton compiles to GPU).

## Running tests

Run from the repository root:

```bash
pytest -v
```

Tests compare each kernel against the PyTorch reference across a range of shapes,
including non-power-of-two sizes that exercise the masking on the last block.
Tests auto-skip if no CUDA GPU is available.
