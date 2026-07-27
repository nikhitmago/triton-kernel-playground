import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    '''
    A tiles (M, K):      B tiles (K, N):      C (acc) tiles (M, M):
    ┌──────┬──────┐      ┌──────┬──────┐      ┌──────┬──────┐
    │ A00  │ A01  │      │ B00  │ B01  │      │ C00  │ C01  │
    │[1,2] │[3,4] │      │[1,0] │[1,0] │      │[0,0] │[0,0] │
    │[5,6] │[7,8] │      │[0,1] │[0,1] │      │[0,0] │[0,0]│
    ├──────┼──────┤      ├──────┼──────┤      ├──────┼──────┤
    │ A10  │ A11  │      │ B10  │ B11  │      │ C10  │ C11  │
    │[9,10]│[11,12]│     │[1,1] │[0,0] │      │[0,0] │[0,0] │
    │[13,14]│[15,16]│    │[0,0] │[1,1] │      │[0,0] │[0,0] │
    └──────┴──────┘      └──────┴──────┘      └──────┴──────┘

    Objective:
    C00 = (A00 * B00) + (A01 * B10)
    C01 = ....
    C10 = ....
    C11 = ....

    Strides (row-major, N=4):
    stride_am = 4 (jump 4 to go down one row in A)
    stride_ak = 1 (jump 1 to go right one col in A)
    stride_bk = 4 (jump 4 to go down one row in B)
    stride_bn = 1 (jump 1 to go right one col in B)
    stride_cm = 4
    stride_cn = 1

    Memory layout of A (flat):
    index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
    value: 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16

    A[row][col] lives at: A_ptr + row * stride_am + col * stride_ak
    A[1][2] = A_ptr + 1*4 + 2*1 = A_ptr + 6 → value 7
    '''
    # Pointer stuff
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offset stuff
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 0 + [0,1] = [0,1], 2 + [0,1] = [2,3]
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [0,1], [2,3]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [2,2]

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)  # [0,1], [2,3]

        # Load A: (M, K)
        # row_offsets[:, None] makes [0,1] => [[0], [1]]
        a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(
            A_ptr + (row_offsets[:, None] * stride_am) + (k_offsets[None, :] * stride_ak), # broadcasting
            mask=a_mask,
            other=0.0
        )

        # Load B: (K, N)
        b_mask = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)
        b_tile = tl.load(
            B_ptr + (k_offsets[:, None] * stride_bk) + (col_offsets[None, :] * stride_bn),  # broadcasting
            mask=b_mask,
            other=0.0
        )

        # Local matmul
        c_tile = tl.dot(a_tile, b_tile)  # in triton .dot does matmul
        acc += c_tile

    # Store results back in C
    c_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(
        C_ptr + (row_offsets[:, None] * stride_cm) + (col_offsets[None, :] * stride_cn),  # broadcasting
        acc,
        mask=c_mask
    )


def matmul(A, B, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device)

    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return C


if __name__ == '__main__':
    M, K = 64, 64
    K, N = 64, 64

    A = torch.rand((M, K), device=DEVICE)
    B = torch.rand((K, N), device=DEVICE)

    C = matmul(A, B)
    ref = A @ B

    max_diff = (C - ref).abs().max().item()
    print(f'Max difference: {max_diff:.6f}')
    assert max_diff < 0.05, f'Mismatch! Max diff: {max_diff}'
    print('PASSED')
