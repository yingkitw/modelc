mod common;

use modelc::parsers::WeightParser;
use modelc::parsers::safetensors::SafetensorsParser;

#[test]
fn test_parse_single_f32_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single.safetensors");

    let data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
    common::create_safetensors_file(&path, vec![("weights", "F32", vec![2, 2], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.name, "single");
    assert_eq!(model.tensors.len(), 1);
    let tensor = &model.tensors["weights"];
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.dtype, modelc::model::DataType::F32);
    assert_eq!(tensor.data.len(), 16);
}

#[test]
fn test_parse_multiple_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.safetensors");

    let w_data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_data = common::f32_to_bytes(&[0.1, 0.2, 0.3]);

    common::create_safetensors_file(
        &path,
        vec![
            ("bias", "F32", vec![3], b_data),
            ("weight", "F32", vec![2, 3], w_data),
        ],
    );

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors.len(), 2);
    assert!(model.tensors.contains_key("weight"));
    assert!(model.tensors.contains_key("bias"));
    assert_eq!(model.tensors["weight"].shape, vec![2, 3]);
    assert_eq!(model.tensors["bias"].shape, vec![3]);
}

#[test]
fn test_parse_f16_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f16.safetensors");

    let data = vec![0u8; 8]; // 4 elements * 2 bytes each
    common::create_safetensors_file(&path, vec![("data", "F16", vec![4], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors["data"].dtype, modelc::model::DataType::F16);
    assert_eq!(model.tensors["data"].shape, vec![4]);
}

#[test]
fn test_parse_preserves_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.safetensors");

    let original_data = common::f32_to_bytes(&[42.0, -1.5, 0.0, 100.0]);
    common::create_safetensors_file(
        &path,
        vec![("tensor", "F32", vec![2, 2], original_data.clone())],
    );

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors["tensor"].data, original_data);
}

#[test]
fn test_parse_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.safetensors");

    let data = common::f32_to_bytes(&[1.0]);
    common::create_safetensors_file(&path, vec![("x", "F32", vec![1], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.metadata["format"], "safetensors");
    assert_eq!(model.metadata["source"], "test.safetensors");
}

#[test]
fn test_parse_1d_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vec.safetensors");

    let data = common::f32_to_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    common::create_safetensors_file(&path, vec![("vec", "F32", vec![5], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors["vec"].shape, vec![5]);
    assert_eq!(model.tensors["vec"].element_count(), 5);
}

#[test]
fn test_parse_scalar_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scalar.safetensors");

    let data = common::f32_to_bytes(&[42.0]);
    common::create_safetensors_file(&path, vec![("value", "F32", vec![1], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors["value"].shape, vec![1]);
    assert_eq!(model.tensors["value"].element_count(), 1);
    assert_eq!(model.tensors["value"].data.len(), 4);
}

#[test]
fn test_parse_3d_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cube.safetensors");

    let data = common::f32_to_bytes(&[0.0; 24]);
    common::create_safetensors_file(&path, vec![("cube", "F32", vec![2, 3, 4], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors["cube"].shape, vec![2, 3, 4]);
    assert_eq!(model.tensors["cube"].element_count(), 24);
}

#[test]
fn test_format_name() {
    let parser = SafetensorsParser;
    assert_eq!(parser.format_name(), "safetensors");
}

#[test]
fn test_parse_nonexistent_file() {
    let parser = SafetensorsParser;
    let result = parser.parse(std::path::Path::new("/nonexistent/file.safetensors"));
    assert!(result.is_err());
}

#[test]
fn test_parse_invalid_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.safetensors");
    std::fs::write(&path, b"not a safetensors file").unwrap();

    let parser = SafetensorsParser;
    let result = parser.parse(&path);
    assert!(result.is_err());
}

#[test]
fn test_parse_empty_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.safetensors");

    common::create_safetensors_file(&path, vec![]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors.len(), 0);
    assert_eq!(model.total_params(), 0);
}

#[test]
fn test_parse_i32_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int.safetensors");

    let data: Vec<u8> = [1i32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    common::create_safetensors_file(&path, vec![("indices", "I32", vec![4], data)]);

    let parser = SafetensorsParser;
    let model = parser.parse(&path).unwrap();

    assert_eq!(model.tensors["indices"].dtype, modelc::model::DataType::I32);
    assert_eq!(model.tensors["indices"].shape, vec![4]);
}
