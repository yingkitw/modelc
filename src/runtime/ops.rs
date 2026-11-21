use crate::runtime::tensor::Tensor;

#[cfg(target_os = "macos")]
use crate::metal::MetalBackend;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Try GPU acceleration on macOS
    #[cfg(target_os = "macos")]
    {
        if let Some(result) = MetalBackend::new().and_then(|mb| mb.matmul_gpu(a, b)) {
            return result;
        }
    }

    // Fallback to optimized CPU implementation
    matmul_cpu(a, b)
}

fn matmul_cpu(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = b.shape[1];
    assert_eq!(a.shape[1], b.shape[0]);

    let mut out = vec![0.0f32; m * n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe {
                matmul_avx(a, b, &mut out);
            }
            return Tensor::from_vec(out, vec![m, n]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                matmul_neon(a, b, &mut out);
            }
            return Tensor::from_vec(out, vec![m, n]);
        }
    }

    // Parallel fallback for reasonably large matrices
    if m >= 8 {
        use rayon::prelude::*;
        out.par_chunks_exact_mut(n)
            .enumerate()
            .for_each(|(i, row_out)| matmul_row(a, b, i, n, k, row_out));
    } else {
        for (i, row_out) in out.chunks_exact_mut(n).enumerate() {
            matmul_row(a, b, i, n, k, row_out);
        }
    }
    Tensor::from_vec(out, vec![m, n])
}

/// Compute one output row of `a @ b`: `out[j] = Σ_p a[i,p]·b[p,j]`.
/// Shared by the rayon parallel path and the small-matrix serial fallback.
fn matmul_row(a: &Tensor, b: &Tensor, i: usize, n: usize, k: usize, out: &mut [f32]) {
    for (j, out_cell) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += a.data[i * k + p] * b.data[p * n + j];
        }
        *out_cell = sum;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn matmul_avx(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = b.shape[1];

    for i in 0..m {
        let mut j = 0;
        while j + 8 <= n {
            let mut sum = _mm256_setzero_ps();
            for p in 0..k {
                let a_val = _mm256_set1_ps(a.data[i * k + p]);
                let b_vec = _mm256_loadu_ps(b.data.as_ptr().add(p * n + j));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, b_vec));
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(i * n + j), sum);
            j += 8;
        }
        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn matmul_neon(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = b.shape[1];

    for i in 0..m {
        let mut j = 0;
        while j + 4 <= n {
            let mut sum = unsafe { vdupq_n_f32(0.0) };
            for p in 0..k {
                let a_val = unsafe { vdupq_n_f32(a.data[i * k + p]) };
                let b_vec = unsafe { vld1q_f32(b.data.as_ptr().add(p * n + j)) };
                sum = unsafe { vfmaq_f32(sum, a_val, b_vec) };
            }
            unsafe { vst1q_f32(out.as_mut_ptr().add(i * n + j), sum) };
            j += 4;
        }
        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
            j += 1;
        }
    }
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    #[cfg(target_os = "macos")]
    {
        if let Some(result) = MetalBackend::new().and_then(|mb| mb.add_gpu(a, b)) {
            return result;
        }
    }
    add_cpu(a, b)
}

fn add_cpu(a: &Tensor, b: &Tensor) -> Tensor {
    if a.shape == b.shape {
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        Tensor::from_vec(data, a.shape.clone())
    } else if a.shape.len() == 2 && b.shape.len() == 1 && a.shape[1] == b.shape[0] {
        let _rows = a.shape[0];
        let cols = a.shape[1];
        let data: Vec<f32> = a
            .data
            .chunks_exact(cols)
            .flat_map(|row| row.iter().zip(&b.data).map(|(x, y)| x + y))
            .collect();
        Tensor::from_vec(data, a.shape.clone())
    } else if a.shape.len() == 1 && b.shape.len() == 2 && a.shape[0] == b.shape[1] {
        add_cpu(b, a)
    } else {
        panic!("add: incompatible shapes {:?} and {:?}", a.shape, b.shape);
    }
}

pub fn mul_scalar(a: &Tensor, s: f32) -> Tensor {
    #[cfg(target_os = "macos")]
    {
        if let Some(result) = MetalBackend::new().and_then(|mb| mb.mul_scalar_gpu(a, s)) {
            return result;
        }
    }
    mul_scalar_cpu(a, s)
}

fn mul_scalar_cpu(a: &Tensor, s: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|x| x * s).collect();
    Tensor::from_vec(data, a.shape.clone())
}

pub fn softmax(a: &Tensor, axis: usize) -> Tensor {
    #[cfg(target_os = "macos")]
    {
        if let Some(result) = MetalBackend::new().and_then(|mb| mb.softmax_gpu(a, axis)) {
            return result;
        }
    }
    softmax_cpu(a, axis)
}

fn softmax_cpu(a: &Tensor, axis: usize) -> Tensor {
    let rank = a.shape.len();
    assert!(axis < rank);

    let dim = a.shape[axis];
    let outer: usize = a.shape[..axis].iter().product();
    let inner: usize = a.shape[axis + 1..].iter().product();
    let stride = dim * inner;

    let mut data = Vec::with_capacity(a.data.len());

    for o in 0..outer {
        for i in 0..inner {
            let base = o * stride + i;

            // Single pass: find max, compute exp, accumulate sum
            let mut max = a.data[base + i];
            for d in 0..dim {
                let val = a.data[base + d * inner];
                if val > max {
                    max = val;
                }
            }

            let mut sum = 0.0f32;
            let temp_offset = data.len();
            for d in 0..dim {
                let val = (a.data[base + d * inner] - max).exp();
                data.push(val);
                sum += val;
            }

            // Normalize the values we just added
            for d in 0..dim {
                data[temp_offset + d] /= sum;
            }
        }
    }
    Tensor::from_vec(data, a.shape.clone())
}

pub fn relu(a: &Tensor) -> Tensor {
    #[cfg(target_os = "macos")]
    {
        if let Some(result) = MetalBackend::new().and_then(|mb| mb.relu_gpu(a)) {
            return result;
        }
    }
    relu_cpu(a)
}

fn relu_cpu(a: &Tensor) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|x| x.max(0.0)).collect();
    Tensor::from_vec(data, a.shape.clone())
}

pub fn layer_norm(a: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    #[cfg(target_os = "macos")]
    {
        if let Some(result) =
            MetalBackend::new().and_then(|mb| mb.layer_norm_gpu(a, weight, bias, eps))
        {
            return result;
        }
    }
    layer_norm_cpu(a, weight, bias, eps)
}

fn layer_norm_cpu(a: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    assert!(!a.shape.is_empty());
    let last_dim = *a.shape.last().unwrap();
    let n_elements = a.data.len();
    let n_vectors = n_elements / last_dim;

    let mut data = Vec::with_capacity(n_elements);

    for i in 0..n_vectors {
        let base = i * last_dim;
        let slice = &a.data[base..base + last_dim];

        // Single-pass: compute mean
        let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;

        // Second pass: compute variance and apply normalization
        let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Apply normalization in single loop
        for (j, &val) in slice.iter().enumerate() {
            data.push((val - mean) * inv_std * weight.data[j] + bias.data[j]);
        }
    }
    Tensor::from_vec(data, a.shape.clone())
}

pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let mut out = matmul(x, weight);
    if let Some(b) = bias {
        out = add(&out, b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape.to_vec())
    }

    /// Reference result for `[[1,2,3],[4,5,6]] @ [[7,8,9,10],[11,12,13,14],[15,16,17,18]]`
    /// = [[74,80,86,92],[173,188,203,218]].
    fn matmul_fixture() -> (Tensor, Tensor, Vec<f32>) {
        (
            t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            t(
                &[
                    7.0, 8.0, 9.0, 10.0, //
                    11.0, 12.0, 13.0, 14.0, //
                    15.0, 16.0, 17.0, 18.0,
                ],
                &[3, 4],
            ),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0],
        )
    }

    #[test]
    fn matmul_cpu_matches_reference() {
        let (a, b, expected) = matmul_fixture();
        let out = matmul_cpu(&a, &b);
        assert_eq!(out.data, expected);
    }

    #[test]
    fn matmul_row_kernel_matches_reference() {
        let (a, b, expected) = matmul_fixture();
        let mut out = vec![0.0f32; 4];
        matmul_row(&a, &b, 1, 4, 3, &mut out);
        assert_eq!(out, expected[4..]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmul_neon_matches_reference() {
        let (a, b, expected) = matmul_fixture();
        let mut out = vec![0.0f32; 8];
        unsafe { matmul_neon(&a, &b, &mut out) };
        assert_eq!(out, expected);
    }

    #[test]
    fn add_cpu_same_shape() {
        let out = add_cpu(&t(&[1.0, 2.0, 3.0], &[3]), &t(&[10.0, 20.0, 30.0], &[3]));
        assert_eq!(out.data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_cpu_broadcasts_1d_over_2d() {
        // [1,2,3] broadcast over rows of [[1,1,1],[2,2,2]].
        let out = add_cpu(
            &t(&[1.0, 2.0, 3.0], &[3]),
            &t(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]),
        );
        assert_eq!(out.data, vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    fn add_cpu_broadcasts_row_vector_over_2d() {
        // [[1,1,1],[2,2,2]] + [1,2,3] (the 2D+1D arm).
        let out = add_cpu(
            &t(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]),
            &t(&[1.0, 2.0, 3.0], &[3]),
        );
        assert_eq!(out.data, vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn add_cpu_rejects_incompatible_shapes() {
        let _ = add_cpu(&t(&[1.0; 4], &[4]), &t(&[1.0; 6], &[2, 3]));
    }

    #[test]
    fn mul_scalar_cpu_scales() {
        let out = mul_scalar_cpu(&t(&[1.0, -2.0, 0.5], &[3]), 4.0);
        assert_eq!(out.data, vec![4.0, -8.0, 2.0]);
    }

    #[test]
    fn softmax_cpu_last_axis_sums_to_one() {
        let out = softmax_cpu(&t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]), 1);
        let sums: Vec<f32> = out.data.chunks(3).map(|r| r.iter().sum()).collect();
        assert!(sums.iter().all(|s| (*s - 1.0).abs() < 1e-5));
    }

    #[test]
    fn softmax_cpu_first_axis_normalizes_columns() {
        // axis=0 on a 2×3 tensor exercises the outer/inner/stride layout.
        // Output interleaves per-column softmax pairs: [c0d0, c0d1, c1d0, c1d1, …].
        let out = softmax_cpu(&t(&[1.0, 2.0, 4.0, 3.0, 1.0, 0.0], &[2, 3]), 0);
        for col in 0..3 {
            let s = out.data[col * 2] + out.data[col * 2 + 1];
            assert!((s - 1.0).abs() < 1e-5, "column {col} sum {s}");
        }
    }

    #[test]
    fn relu_cpu_clamps_negatives() {
        let out = relu_cpu(&t(&[-1.0, 0.0, 2.0, -0.5], &[4]));
        assert_eq!(out.data, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn layer_norm_cpu_normalizes_each_vector() {
        let out = layer_norm_cpu(&t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), &t(&[1.0; 2], &[2]), &t(&[0.0; 2], &[2]), 1e-5);
        for row in out.data.chunks(2) {
            let mean = row.iter().sum::<f32>() / 2.0;
            assert!(mean.abs() < 1e-5, "row mean {mean}");
        }
    }
}
