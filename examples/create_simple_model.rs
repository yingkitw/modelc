use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use modelc::model::{DataType, Model, TensorData};

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn create_safetensors(path: &std::path::Path, tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>) {
    let mut sorted: Vec<_> = tensors.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_string(),
        serde_json::Value::Object(serde_json::Map::new()),
    );

    let mut offset = 0usize;
    let mut entries = Vec::new();
    for (name, dtype, shape, data) in sorted {
        let end = offset + data.len();
        entries.push((
            name.to_string(),
            dtype.to_string(),
            shape,
            data,
            offset,
            end,
        ));
        offset = end;
    }

    for (name, dtype, shape, _, start, end) in &entries {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "dtype".to_string(),
            serde_json::Value::String(dtype.clone()),
        );
        obj.insert("shape".to_string(), serde_json::json!(shape));
        obj.insert(
            "data_offsets".to_string(),
            serde_json::json!([*start, *end]),
        );
        header.insert(name.clone(), serde_json::Value::Object(obj));
    }

    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();
    for (_, _, _, data, _, _) in &entries {
        file.write_all(data).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("simple_model.safetensors")
    };

    let hidden = 8;

    let layer1_weight = f32_to_bytes(
        &(0..hidden * hidden)
            .map(|i| (i as f32) / (hidden * hidden) as f32)
            .collect::<Vec<_>>(),
    );
    let layer1_bias = f32_to_bytes(&vec![0.1; hidden]);
    let layer2_weight = f32_to_bytes(
        &(0..hidden)
            .map(|i| (i as f32) / hidden as f32)
            .collect::<Vec<_>>(),
    );
    let layer2_bias = f32_to_bytes(&[0.0]);

    create_safetensors(
        &output_path,
        vec![
            ("layer1.bias", "F32", vec![hidden], layer1_bias),
            (
                "layer1.weight",
                "F32",
                vec![hidden, hidden],
                layer1_weight.clone(),
            ),
            ("layer2.bias", "F32", vec![1], layer2_bias),
            ("layer2.weight", "F32", vec![1, hidden], layer2_weight),
        ],
    );

    let model = Model {
        name: "simple_mlp".to_string(),
        architecture: "mlp".to_string(),
        tensors: {
            let mut m = HashMap::new();
            m.insert(
                "layer1.weight".to_string(),
                TensorData {
                    shape: vec![hidden, hidden],
                    dtype: DataType::F32,
                    data: layer1_weight,
                },
            );
            m
        },
        metadata: HashMap::new(),
    };

    eprintln!("Created {} with {} tensors", output_path.display(), 4,);
    eprintln!(
        "Model: {} ({} params, {:.2} KB)",
        model.name,
        hidden * hidden + hidden + hidden + 1,
        (hidden * hidden * 4 + hidden * 4 + hidden * 4 + 4) as f64 / 1024.0,
    );
}
