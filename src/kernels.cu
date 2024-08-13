#include <cuda_runtime.h>
#include <stdint.h>

// Assuming 256-bit field elements
typedef uint32_t fe_t[8];

__device__ int fe_cmp(const fe_t a, const fe_t b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ void fe_sub(fe_t c, const fe_t a, const fe_t b, const fe_t modulus);

__device__ void fe_add(fe_t c, const fe_t a, const fe_t b, const fe_t modulus) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t tmp = (uint64_t)a[i] + b[i] + carry;
        c[i] = (uint32_t)tmp;
        carry = tmp >> 32;
    }
    if (carry || fe_cmp(c, modulus) >= 0) {
        fe_sub(c, c, modulus, modulus);
    }
}

__device__ void fe_sub(fe_t c, const fe_t a, const fe_t b, const fe_t modulus) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t tmp = (int64_t)a[i] - b[i] - borrow;
        c[i] = (uint32_t)tmp;
        borrow = (tmp >> 32) & 1;
    }
    if (borrow) {
        fe_add(c, c, modulus, modulus);
    }
}

__device__ void fe_mul(fe_t c, const fe_t a, const fe_t b, const fe_t modulus) {
    uint32_t t[16] = {0};
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t tmp = (uint64_t)a[i] * b[j] + t[i+j] + carry;
            t[i+j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        t[i+8] = carry;
    }
    // Reduction modulo p (assuming modulus is prime)
    for (int i = 15; i >= 8; i--) {
        for (int j = 0; j < 8; j++) {
            uint64_t product = (uint64_t)t[i] * modulus[j];
            int k = i + j - 8;
            if (k >= 0) {
                uint64_t tmp = t[k] - (product & 0xFFFFFFFF);
                t[k] = (uint32_t)tmp;
                if (k + 1 < 8) {
                    t[k+1] -= (product >> 32) + (tmp >> 32);
                }
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        c[i] = t[i];
    }
    if (fe_cmp(c, modulus) >= 0) {
        fe_sub(c, c, modulus, modulus);
    }
}

extern "C" {

    __global__ void evaluate_polynomial_kernel(fe_t* coeffs, uint32_t degree, const fe_t x, fe_t* result, const fe_t modulus) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int i = 0; i < 8; i++) {
                result[0][i] = coeffs[0][i];
            }
        }
    }

__global__ void calculate_quotient_polynomial_kernel(fe_t* p, uint32_t degree, const fe_t b, const fe_t a, fe_t* q, const fe_t modulus) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    fe_t a_neg;
    fe_sub(a_neg, modulus, a, modulus); // -a = p - a

    for (uint32_t i = idx; i < degree; i += stride) {
        fe_t term;
        if (i == 0) {
            fe_sub(term, p[i], b, modulus);
        } else {
            for (int j = 0; j < 8; j++) term[j] = p[i][j];
        }

        fe_t divisor = {1, 0, 0, 0, 0, 0, 0, 0}; // Start with 1
        for (uint32_t j = 0; j < i; j++) {
            fe_mul(divisor, divisor, a_neg, modulus);
        }

        // Compute the inverse of divisor
        fe_t divisor_inv;
        // TODO: Implement modular inversion 
        for (int j = 0; j < 8; j++) divisor_inv[j] = divisor[j];

        fe_mul(q[i], term, divisor_inv, modulus);
    }
}

__global__ void perform_msm_kernel(fe_t* points, fe_t* scalars, uint32_t size, fe_t* result, const fe_t modulus) {
    __shared__ fe_t partial_sums[32][2]; // For x and y coordinates
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    uint32_t idx = blockIdx.x * blockDim.x + tid;

    fe_t local_sum[2] = {{0}, {0}}; // For x and y coordinates

    for (uint32_t i = idx; i < size; i += stride) {
        fe_t temp[2];
        fe_mul(temp[0], points[i * 2], scalars[i], modulus);     // x coordinate
        fe_mul(temp[1], points[i * 2 + 1], scalars[i], modulus); // y coordinate
        
        fe_add(local_sum[0], local_sum[0], temp[0], modulus);
        fe_add(local_sum[1], local_sum[1], temp[1], modulus);
    }

    for (int i = 0; i < 8; i++) {
        partial_sums[tid][0][i] = local_sum[0][i];
        partial_sums[tid][1][i] = local_sum[1][i];
    }
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            fe_add(partial_sums[tid][0], partial_sums[tid][0], partial_sums[tid + s][0], modulus);
            fe_add(partial_sums[tid][1], partial_sums[tid][1], partial_sums[tid + s][1], modulus);
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < 8; i++) {
            result[0][i] = partial_sums[0][0][i];
            result[1][i] = partial_sums[0][1][i];
        }
    }
}

} // extern "C"