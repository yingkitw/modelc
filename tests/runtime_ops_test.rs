use modelc::runtime::ops;
use modelc::runtime::tensor::Tensor;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn vec_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(approx_eq(*x, *y, tol), "at index {i}: {x} != {y}");
    }
}

#[test]
fn test_matmul_identity() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let result = ops::matmul(&a, &eye);
    assert_eq!(result.shape, vec![2, 2]);
    vec_approx_eq(&result.data, &a.data, 1e-6);
}

#[test]
fn test_matmul_basic() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = ops::matmul(&a, &b);
    assert_eq!(result.shape, vec![2, 2]);
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    vec_approx_eq(&result.data, &[19.0, 22.0, 43.0, 50.0], 1e-6);
}

#[test]
fn test_matmul_non_square() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let result = ops::matmul(&a, &b);
    assert_eq!(result.shape, vec![2, 2]);
    // row0: [1*1+2*3+5*5, 1*2+2*4+5*6] -- wait, let me recalculate
    // a = [[1,2,3],[4,5,6]], b = [[1,2],[3,4],[5,6]]
    // c[0,0] = 1*1 + 2*3 + 3*5 = 1+6+15 = 22
    // c[0,1] = 1*2 + 2*4 + 3*6 = 2+8+18 = 28
    // c[1,0] = 4*1 + 5*3 + 6*5 = 4+15+30 = 49
    // c[1,1] = 4*2 + 5*4 + 6*6 = 8+20+36 = 64
    vec_approx_eq(&result.data, &[22.0, 28.0, 49.0, 64.0], 1e-5);
}

#[test]
fn test_matmul_vector() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3, 1]);
    let result = ops::matmul(&a, &b);
    assert_eq!(result.shape, vec![1, 1]);
    vec_approx_eq(&result.data, &[32.0], 1e-6);
}

#[test]
#[should_panic]
fn test_matmul_dimension_mismatch() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);
    ops::matmul(&a, &b);
}

#[test]
fn test_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
    let result = ops::add(&a, &b);
    assert_eq!(result.shape, vec![3]);
    vec_approx_eq(&result.data, &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn test_add_2d() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let result = ops::add(&a, &b);
    vec_approx_eq(&result.data, &[11.0, 22.0, 33.0, 44.0], 1e-6);
}

#[test]
fn test_add_negatives() {
    let a = Tensor::from_vec(vec![-1.0, -2.0], vec![2]);
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let result = ops::add(&a, &b);
    vec_approx_eq(&result.data, &[0.0, 0.0], 1e-6);
}

#[test]
fn test_mul_scalar() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let result = ops::mul_scalar(&a, 2.0);
    vec_approx_eq(&result.data, &[2.0, 4.0, 6.0], 1e-6);
}

#[test]
fn test_mul_scalar_zero() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let result = ops::mul_scalar(&a, 0.0);
    vec_approx_eq(&result.data, &[0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn test_mul_scalar_negative() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
    let result = ops::mul_scalar(&a, -1.0);
    vec_approx_eq(&result.data, &[-1.0, 2.0, -3.0], 1e-6);
}

#[test]
fn test_softmax_1d() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let result = ops::softmax(&a, 0);
    let sum: f32 = result.data.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-5));
    assert!(result.data[2] > result.data[1]);
    assert!(result.data[1] > result.data[0]);
}

#[test]
fn test_softmax_uniform() {
    let a = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let result = ops::softmax(&a, 0);
    vec_approx_eq(&result.data, &[0.25, 0.25, 0.25, 0.25], 1e-6);
}

#[test]
fn test_softmax_extreme_values() {
    let a = Tensor::from_vec(vec![1000.0, 1000.0], vec![2]);
    let result = ops::softmax(&a, 0);
    vec_approx_eq(&result.data, &[0.5, 0.5], 1e-4);
}

#[test]
fn test_softmax_2d_axis0() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = ops::softmax(&a, 0);
    assert_eq!(result.shape, vec![2, 3]);
    // Each column should sum to 1
    for j in 0..3 {
        let sum = result.data[j] + result.data[3 + j];
        assert!(approx_eq(sum, 1.0, 1e-5));
    }
}

#[test]
fn test_softmax_2d_axis1() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = ops::softmax(&a, 1);
    assert_eq!(result.shape, vec![2, 3]);
    // Each row should sum to 1
    let row0_sum: f32 = result.data[0..3].iter().sum();
    let row1_sum: f32 = result.data[3..6].iter().sum();
    assert!(approx_eq(row0_sum, 1.0, 1e-5));
    assert!(approx_eq(row1_sum, 1.0, 1e-5));
}

#[test]
fn test_relu_positive() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let result = ops::relu(&a);
    vec_approx_eq(&result.data, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn test_relu_negative() {
    let a = Tensor::from_vec(vec![-1.0, -2.0, -3.0], vec![3]);
    let result = ops::relu(&a);
    vec_approx_eq(&result.data, &[0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn test_relu_mixed() {
    let a = Tensor::from_vec(vec![-2.0, 0.0, 3.0, -0.5, 1.5, -100.0], vec![2, 3]);
    let result = ops::relu(&a);
    vec_approx_eq(&result.data, &[0.0, 0.0, 3.0, 0.0, 1.5, 0.0], 1e-6);
}

#[test]
fn test_layer_norm_no_transform() {
    let a = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let result = ops::layer_norm(&a, &w, &b, 1e-5);
    vec_approx_eq(&result.data, &[0.0, 0.0, 0.0, 0.0], 1e-5);
}

#[test]
fn test_layer_norm_standardize() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let result = ops::layer_norm(&a, &w, &b, 1e-5);
    let mean: f32 = result.data.iter().sum::<f32>() / 4.0;
    assert!(approx_eq(mean, 0.0, 1e-4));
}

#[test]
fn test_layer_norm_with_weight_bias() {
    let a = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
    let w = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    let b = Tensor::from_vec(vec![0.5, -0.5], vec![2]);
    let result = ops::layer_norm(&a, &w, &b, 1e-5);
    // mean=1.0, var=0, but with eps it'll be ~0. so inv_std = 1/sqrt(eps)
    // result = (1-1) * inv_std * w + b = b
    // Hmm, var=0 so this is (data - mean) * inv_std = 0 * inv_std = 0
    // So result = 0 * w + b = b
    vec_approx_eq(&result.data, &[0.5, -0.5], 1e-4);
}

#[test]
fn test_layer_norm_2d_batch() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], vec![2, 4]);
    let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let b = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let result = ops::layer_norm(&a, &w, &b, 1e-5);
    // Each row is normalized independently
    let mean0: f32 = result.data[0..4].iter().sum::<f32>() / 4.0;
    let mean1: f32 = result.data[4..8].iter().sum::<f32>() / 4.0;
    assert!(approx_eq(mean0, 0.0, 1e-4));
    assert!(approx_eq(mean1, 0.0, 1e-3));
}

#[test]
fn test_linear_no_bias() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let w = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    let result = ops::linear(&x, &w, None);
    assert_eq!(result.shape, vec![1, 2]);
    // [1*3+2*5, 1*4+2*6] = [13, 16]
    vec_approx_eq(&result.data, &[13.0, 16.0], 1e-5);
}

#[test]
fn test_linear_with_bias() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]);
    let result = ops::linear(&x, &w, Some(&b));
    assert_eq!(result.shape, vec![1, 2]);
    // identity matmul: [1, 2] + [10, 20] = [11, 22]
    vec_approx_eq(&result.data, &[11.0, 22.0], 1e-5);
}

#[test]
fn test_linear_batched() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
    let result = ops::linear(&x, &w, Some(&b));
    assert_eq!(result.shape, vec![2, 2]);
    // [1+2, 1+2] = [3, 3]
    // [3+4, 3+4] = [7, 7]
    vec_approx_eq(&result.data, &[3.0, 3.0, 7.0, 7.0], 1e-5);
}
