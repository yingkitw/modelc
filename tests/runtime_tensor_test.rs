use modelc::runtime::tensor::Tensor;

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(vec![3, 4]);
    assert_eq!(t.shape, vec![3, 4]);
    assert_eq!(t.data.len(), 12);
    assert!(t.data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_zeros_1d() {
    let t = Tensor::zeros(vec![5]);
    assert_eq!(t.shape, vec![5]);
    assert_eq!(t.data.len(), 5);
}

#[test]
fn test_tensor_zeros_scalar() {
    let t = Tensor::zeros(vec![1]);
    assert_eq!(t.shape, vec![1]);
    assert_eq!(t.data.len(), 1);
}

#[test]
fn test_tensor_from_vec() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_len() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(t.len(), 3);
}

#[test]
fn test_tensor_is_empty() {
    let t = Tensor::from_vec(vec![], vec![0]);
    assert!(t.is_empty());
    let t2 = Tensor::from_vec(vec![1.0], vec![1]);
    assert!(!t2.is_empty());
}

#[test]
fn test_tensor_reshape_same_size() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.shape, vec![3, 2]);
    assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_reshape_1d() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let r = t.reshape(vec![4]);
    assert_eq!(r.shape, vec![4]);
}

#[test]
#[should_panic]
fn test_tensor_reshape_mismatch() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    t.reshape(vec![2, 2]);
}

#[test]
fn test_tensor_preserves_data() {
    let data = vec![42.0, -1.5, 0.0, 100.0, f32::MAX, f32::MIN];
    let t = Tensor::from_vec(data.clone(), vec![2, 3]);
    assert_eq!(t.data, data);
}
