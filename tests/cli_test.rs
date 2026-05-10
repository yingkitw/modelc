use std::path::PathBuf;

use modelc::cli::WeightFormat;

#[test]
fn test_detect_safetensors() {
    let path = PathBuf::from("model.safetensors");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Safetensors));
}

#[test]
fn test_detect_safetensors_with_path() {
    let path = PathBuf::from("/some/deep/path/model.safetensors");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Safetensors));
}

#[test]
fn test_detect_gguf() {
    let path = PathBuf::from("model.gguf");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Gguf));
}

#[test]
fn test_detect_onnx() {
    let path = PathBuf::from("model.onnx");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Onnx));
}

#[test]
fn test_detect_pytorch_pt() {
    let path = PathBuf::from("model.pt");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Pytorch));
}

#[test]
fn test_detect_pytorch_pth() {
    let path = PathBuf::from("model.pth");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Pytorch));
}

#[test]
fn test_detect_pytorch_bin() {
    let path = PathBuf::from("pytorch_model.bin");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Pytorch));
}

#[test]
fn test_detect_unknown() {
    let path = PathBuf::from("model.h5");
    assert_eq!(WeightFormat::detect(&path), None);
}

#[test]
fn test_detect_no_extension() {
    let path = PathBuf::from("model");
    assert_eq!(WeightFormat::detect(&path), None);
}

#[test]
fn test_detect_case_insensitive() {
    let path = PathBuf::from("model.SafeTensors");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Safetensors));
}

#[test]
fn test_detect_gguf_case_insensitive() {
    let path = PathBuf::from("MODEL.GGUF");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Gguf));
}

#[test]
fn test_detect_onnx_case_insensitive() {
    let path = PathBuf::from("model.Onnx");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Onnx));
}

#[test]
fn test_detect_pt_case_insensitive() {
    let path = PathBuf::from("model.PT");
    assert_eq!(WeightFormat::detect(&path), Some(WeightFormat::Pytorch));
}
