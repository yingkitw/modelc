use crate::runtime::tensor::Tensor;

#[cfg(target_os = "macos")]
use crate::metal::MetalBackend;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Try GPU acceleration on macOS
    #[cfg(target_os = "macos")]
    {
        if let Some(backend) = MetalBackend::new() {
            if let Some(result) = backend.matmul_gpu(a, b) {
                return result;
            }
        }
    }

    // Fallback to optimized CPU implementation
    matmul_cpu(a, b)
}

fn matmul_cpu(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = b.shape[1];
    assert_eq!(a.shape[1], b.shape[0]);

    // Original implementation for correctness
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Tensor::from_vec(out, vec![m, n])
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
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
        add(b, a)
    } else {
        panic!("add: incompatible shapes {:?} and {:?}", a.shape, b.shape);
    }
}

pub fn mul_scalar(a: &Tensor, s: f32) -> Tensor {
    let data: Vec<f32> = a.data.iter().map(|x| x * s).collect();
    Tensor::from_vec(data, a.shape.clone())
}

pub fn softmax(a: &Tensor, axis: usize) -> Tensor {
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
    let data: Vec<f32> = a.data.iter().map(|x| x.max(0.0)).collect();
    Tensor::from_vec(data, a.shape.clone())
}

pub fn layer_norm(a: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
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
