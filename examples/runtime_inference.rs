use std::collections::HashMap;

use modelc::model::{DataType, TensorData};
use modelc::runtime::ops;
use modelc::runtime::serve::Runtime;
use modelc::runtime::tensor::Tensor;

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() {
    let mut raw_tensors = HashMap::new();

    let input_dim = 4usize;
    let hidden_dim = 8usize;
    let output_dim = 2usize;

    let fc1_weight: Vec<f32> = (0..input_dim * hidden_dim)
        .map(|i| (i as f32 / (input_dim * hidden_dim) as f32) - 0.5)
        .collect();
    raw_tensors.insert(
        "fc1.weight".to_string(),
        TensorData {
            shape: vec![input_dim, hidden_dim],
            dtype: DataType::F32,
            data: f32_to_bytes(&fc1_weight),
        },
    );

    let fc1_bias: Vec<f32> = vec![0.1; hidden_dim];
    raw_tensors.insert(
        "fc1.bias".to_string(),
        TensorData {
            shape: vec![hidden_dim],
            dtype: DataType::F32,
            data: f32_to_bytes(&fc1_bias),
        },
    );

    let fc2_weight: Vec<f32> = (0..hidden_dim * output_dim)
        .map(|i| (i as f32 / (hidden_dim * output_dim) as f32) - 0.5)
        .collect();
    raw_tensors.insert(
        "fc2.weight".to_string(),
        TensorData {
            shape: vec![hidden_dim, output_dim],
            dtype: DataType::F32,
            data: f32_to_bytes(&fc2_weight),
        },
    );

    let fc2_bias: Vec<f32> = vec![0.0; output_dim];
    raw_tensors.insert(
        "fc2.bias".to_string(),
        TensorData {
            shape: vec![output_dim],
            dtype: DataType::F32,
            data: f32_to_bytes(&fc2_bias),
        },
    );

    let runtime = Runtime::from_raw(&raw_tensors);
    eprintln!("Loaded {} tensors:", runtime.tensor_names().len());
    for name in runtime.tensor_names() {
        let t = runtime.get(name).unwrap();
        eprintln!("  {} {:?}", name, t.shape);
    }

    let input = Tensor::from_vec(vec![1.0, 0.5, -0.3, 0.8], vec![1, input_dim]);

    let fc1_w = runtime.get("fc1.weight").unwrap();
    let fc1_b = runtime.get("fc1.bias").unwrap();
    let hidden = ops::linear(&input, fc1_w, Some(fc1_b));
    eprintln!("\nAfter fc1 (pre-activation): {:?}", hidden.data);

    let activated = ops::relu(&hidden);
    eprintln!("After ReLU: {:?}", activated.data);

    let fc2_w = runtime.get("fc2.weight").unwrap();
    let fc2_b = runtime.get("fc2.bias").unwrap();
    let output = ops::linear(&activated, fc2_w, Some(fc2_b));
    eprintln!("After fc2 (output): {:?}", output.data);

    let probs = ops::softmax(&output, 1);
    eprintln!("Softmax probabilities: {:?}", probs.data);

    let total: f32 = probs.data.iter().sum();
    eprintln!("Probability sum: {}", total);
}
