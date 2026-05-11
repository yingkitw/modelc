//! Torch-style ZIP checkpoints that bundle nested `*.safetensors` files.

mod common;

use std::fs::{File, read};
use std::io::Write;

use tempfile::tempdir;

use zip::CompressionMethod;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use modelc::parsers::WeightParser;
use modelc::parsers::pytorch::PytorchParser;

#[test]
fn parse_zip_with_single_nested_safetensors() {
    let dir = tempdir().unwrap();
    let inner_path = dir.path().join("inner.safetensors");
    let data = common::f32_to_bytes(&[9.0, -1.0, 2.5]);
    common::create_safetensors_file(&inner_path, vec![("vec", "F32", vec![3], data.clone())]);

    let zip_path = dir.path().join("checkpoint.pt");
    let file = File::create(&zip_path).unwrap();
    let mut zip = ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("model_data.safetensors", opts).unwrap();
    zip.write_all(&read(&inner_path).unwrap()).unwrap();
    zip.finish().unwrap();

    let model = PytorchParser.parse(&zip_path).unwrap();
    assert_eq!(model.tensors.len(), 1);
    let t = model.tensors.get("vec").expect("tensor vec");
    assert_eq!(t.shape, vec![3usize]);
    assert_eq!(t.data, data);
    assert_eq!(
        model.metadata.get("format").map(String::as_str),
        Some("pytorch_zip_safe")
    );
}

#[test]
fn parse_zip_with_two_safetensors_prefixes_keys() {
    let dir = tempdir().unwrap();
    let a_path = dir.path().join("a.safetensors");
    let b_path = dir.path().join("b.safetensors");
    common::create_safetensors_file(
        &a_path,
        vec![("w", "F32", vec![1], common::f32_to_bytes(&[1.0]))],
    );
    common::create_safetensors_file(
        &b_path,
        vec![("w", "F32", vec![1], common::f32_to_bytes(&[2.0]))],
    );

    let zip_path = dir.path().join("two.zip");
    let file = File::create(&zip_path).unwrap();
    let mut zip = ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("shard_a.safetensors", opts).unwrap();
    zip.write_all(&read(&a_path).unwrap()).unwrap();
    zip.start_file("shard_b.safetensors", opts).unwrap();
    zip.write_all(&read(&b_path).unwrap()).unwrap();
    zip.finish().unwrap();

    let model = PytorchParser.parse(&zip_path).unwrap();
    assert_eq!(model.tensors.len(), 2);
    assert!(model.tensors.contains_key("shard_a.w"));
    assert!(model.tensors.contains_key("shard_b.w"));
}

#[test]
fn plain_safetensors_file_still_parsed_as_pytorch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("weights.pt");
    let data = common::f32_to_bytes(&[1.0, 2.0]);
    common::create_safetensors_file(&path, vec![("x", "F32", vec![2], data.clone())]);

    let model = PytorchParser.parse(&path).unwrap();
    assert_eq!(model.tensors["x"].data, data);
}
