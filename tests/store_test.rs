mod common;

use anyhow::Result;

#[test]
fn test_search_models_empty() -> Result<()> {
    let results = modelc::store::search_models("nonexistent_xyz")?;
    assert!(results.is_empty());
    Ok(())
}

#[test]
fn test_list_models_does_not_panic() -> Result<()> {
    let _models = modelc::store::list_models()?;
    Ok(())
}

#[test]
fn test_resolve_model_path_rejects_empty() {
    let result = modelc::store::resolve_model_path("");
    assert!(result.is_err());
}

#[test]
fn test_remove_model_deletes_file() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let name = "__modelc_test_rm__";
    let store_path = modelc::store::install(tmp.path(), name)?;
    assert!(store_path.is_file());

    modelc::store::remove_model(name, false, false)?;
    assert!(!store_path.is_file());

    Ok(())
}

#[test]
fn test_remove_model_all_deletes_versions() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let name = "__modelc_test_rm_all__";
    let store_path = modelc::store::install(tmp.path(), name)?;
    let dir = store_path.parent().unwrap();
    let v1 = dir.join(format!("{}.v1.modelc", name));
    let v2 = dir.join(format!("{}.v2.modelc", name));
    std::fs::copy(&store_path, &v1)?;
    std::fs::copy(&store_path, &v2)?;

    modelc::store::remove_model(name, true, false)?;
    assert!(!store_path.is_file());
    assert!(!v1.is_file());
    assert!(!v2.is_file());

    Ok(())
}

#[test]
fn test_remove_model_guard_rejects_without_force_or_all() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let name = "__modelc_test_rm_guard__";
    let store_path = modelc::store::install(tmp.path(), name)?;
    let dir = store_path.parent().unwrap();
    let v1 = dir.join(format!("{}.v1.modelc", name));
    std::fs::copy(&store_path, &v1)?;

    // Without --force or --all, deletion should fail because versioned copies exist.
    let result = modelc::store::remove_model(name, false, false);
    assert!(result.is_err());
    assert!(store_path.is_file());
    assert!(v1.is_file());

    // --force should allow deletion even with versioned copies present.
    modelc::store::remove_model(name, false, true)?;
    assert!(!store_path.is_file());
    assert!(v1.is_file());

    // Clean up the orphaned version copy.
    std::fs::remove_file(&v1)?;

    Ok(())
}

#[test]
fn test_copy_model_creates_copy() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let src_name = "__modelc_test_cp_src__";
    let dest_name = "__modelc_test_cp_dest__";
    modelc::store::install(tmp.path(), src_name)?;

    let dest_path = modelc::store::copy_model(src_name, dest_name)?;
    assert!(dest_path.is_file());

    // Verify the copy is identical to the source.
    let src_path = modelc::store::resolve_model_path(src_name)?;
    let src_bytes = std::fs::read(&src_path)?;
    let dest_bytes = std::fs::read(&dest_path)?;
    assert_eq!(src_bytes, dest_bytes);

    // Clean up.
    modelc::store::remove_model(src_name, false, true)?;
    modelc::store::remove_model(dest_name, false, true)?;

    Ok(())
}

#[test]
fn test_copy_model_rejects_existing_dest() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let name = "__modelc_test_cp_exists__";
    modelc::store::install(tmp.path(), name)?;

    // Copying to the same name should fail.
    let result = modelc::store::copy_model(name, name);
    assert!(result.is_err());

    // Clean up.
    modelc::store::remove_model(name, false, true)?;

    Ok(())
}

#[test]
fn test_copy_model_rejects_missing_source() -> Result<()> {
    let result = modelc::store::copy_model("__nonexistent_model__", "__dest__");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_list_versions_finds_versioned_copies() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let name = "__modelc_test_versions__";
    let store_path = modelc::store::install(tmp.path(), name)?;
    let dir = store_path.parent().unwrap();
    let v1 = dir.join(format!("{}.v1.modelc", name));
    let v3 = dir.join(format!("{}.v3.modelc", name));
    let v2 = dir.join(format!("{}.v2.modelc", name));
    std::fs::copy(&store_path, &v2)?;
    std::fs::copy(&store_path, &v1)?;
    std::fs::copy(&store_path, &v3)?;

    let mut versions = modelc::store::list_versions(name)?;
    versions.sort_by_key(|(v, _)| *v);
    let vers: Vec<u32> = versions.iter().map(|(v, _)| *v).collect();
    assert_eq!(vers, vec![1, 2, 3], "versions sorted ascending: {vers:?}");

    // Clean up.
    for p in [&v1, &v2, &v3] {
        std::fs::remove_file(p)?;
    }
    modelc::store::remove_model(name, false, true)?;
    Ok(())
}

#[test]
fn test_list_versions_empty_for_unknown_model() -> Result<()> {
    let versions = modelc::store::list_versions("__modelc_no_versions__")?;
    assert!(versions.is_empty());
    Ok(())
}

#[test]
fn test_switch_version_activates_requested_version() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;

    let name = "__modelc_test_switch__";
    let store_path = modelc::store::install(tmp.path(), name)?;
    let dir = store_path.parent().unwrap();
    let v2 = dir.join(format!("{}.v2.modelc", name));
    std::fs::copy(&store_path, &v2)?;

    // Record v2's content, then overwrite the main artifact with junk so we
    // can prove the switch actually replaced it.
    let v2_bytes = std::fs::read(&v2)?;
    std::fs::write(&store_path, b"junk-not-a-model")?;

    let dest = modelc::store::switch_version(name, 2)?;
    assert_eq!(dest, store_path);
    let restored = std::fs::read(&store_path)?;
    assert_eq!(
        restored, v2_bytes,
        "switch should copy the version over the main artifact"
    );

    // Clean up.
    modelc::store::remove_model(name, true, true)?;
    Ok(())
}

#[test]
fn test_switch_version_missing_version_errors() -> Result<()> {
    let result = modelc::store::switch_version("__modelc_no_such__", 9);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_list_models_reports_architecture_params_and_compression() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;

    // Compressed artifact → the compressed flag should be observable.
    modelc::pack::pack(&model, tmp.path(), true)?;
    let name = "__modelc_test_list_meta__";
    modelc::store::install(tmp.path(), name)?;

    let models = modelc::store::list_models()?;
    let found = models
        .iter()
        .find(|m| m.name == name)
        .expect("installed model should be listed");
    assert_eq!(found.architecture.as_deref(), Some("mlp"));
    assert!(
        found.params.unwrap_or(0) > 0,
        "params should be parsed from header"
    );
    assert!(found.compressed, "artifact was packed with --compress");
    assert!(found.size_bytes > 0);

    // Clean up.
    modelc::store::remove_model(name, false, true)?;
    Ok(())
}

/// Minimal single-request HTTP server for `http_get` testing: accepts one
/// connection, writes a canned response, and closes.
fn serve_one_http_response(response: &'static [u8]) -> std::net::SocketAddr {
    use std::io::Write;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = std::io::Read::read(&mut stream, &mut buf);
            let _ = stream.write_all(response);
            let _ = stream.flush();
        }
    });
    addr
}

#[test]
fn test_download_via_local_http_server() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;
    let payload = std::fs::read(tmp.path())?;

    // Build a canned HTTP response containing the artifact bytes.
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    let mut response = header.into_bytes();
    response.extend_from_slice(&payload);
    // Leak a 'static copy for the server thread.
    let response: &'static [u8] = Box::leak(response.into_boxed_slice());
    let addr = serve_one_http_response(response);

    let name = "__modelc_test_download__";
    let dest = modelc::store::download(&format!("http://{addr}/model.modelc"), name)?;
    let downloaded = std::fs::read(&dest)?;
    assert_eq!(
        downloaded, payload,
        "downloaded bytes should match the served payload"
    );

    // Clean up.
    modelc::store::remove_model(name, false, true)?;
    Ok(())
}

#[test]
fn test_download_http_error_status_propagates() -> Result<()> {
    let response: &'static [u8] =
        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    let addr = serve_one_http_response(response);
    let result = modelc::store::download(&format!("http://{addr}/missing.modelc"), "__x__");
    assert!(result.is_err(), "404 should surface as an error");
    Ok(())
}

#[test]
fn test_download_https_unreachable_errors() -> Result<()> {
    // https is now delegated to the system curl — a refused port fails fast
    // whether curl exists (connection refused) or not (curl-not-found error).
    let result = modelc::store::download("https://127.0.0.1:1/x.modelc", "__x__");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("curl") || err.contains("https") || err.contains("refused"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// The modelc-web store sends `X-Modelc-Digest: sha256:<hex>` — a correct
/// digest verifies, a wrong one must fail the download.
#[test]
fn test_download_verifies_x_modelc_digest() -> Result<()> {
    let model = common::create_test_model();
    let tmp = common::tempfile()?;
    modelc::pack::pack(&model, tmp.path(), false)?;
    let payload = std::fs::read(tmp.path())?;

    let good = modelc::digest::digest_bytes(&payload);

    // Correct digest → download succeeds.
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nX-Modelc-Digest: {good}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    let mut response = header.into_bytes();
    response.extend_from_slice(&payload);
    let response: &'static [u8] = Box::leak(response.into_boxed_slice());
    let addr = serve_one_http_response(response);
    let name = "__modelc_test_digest__";
    let dest = modelc::store::download(&format!("http://{addr}/m.modelc"), name)?;
    assert_eq!(std::fs::read(&dest)?, payload);
    modelc::store::remove_model(name, false, true)?;

    // Wrong digest → error, nothing left in the store.
    let bad = modelc::digest::digest_bytes(b"different-bytes");
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nX-Modelc-Digest: {bad}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    let mut response = header.into_bytes();
    response.extend_from_slice(&payload);
    let response: &'static [u8] = Box::leak(response.into_boxed_slice());
    let addr = serve_one_http_response(response);
    let result = modelc::store::download(&format!("http://{addr}/m.modelc"), "__x_bad__");
    assert!(result.is_err(), "digest mismatch must fail the download");
    Ok(())
}

/// `download_auth` must send `Authorization: Bearer <token>` so the CLI can
/// pull from token-gated endpoints like modelc-web.
#[test]
fn test_download_sends_bearer_token() -> Result<()> {
    use std::io::Read;

    let payload = b"artifact-body".to_vec();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    let mut response = header.into_bytes();
    response.extend_from_slice(&payload);
    let response: &'static [u8] = Box::leak(response.into_boxed_slice());

    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let captured2 = captured.clone();
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            if let Ok(n) = stream.read(&mut buf) {
                captured2.lock().unwrap().extend_from_slice(&buf[..n]);
            }
            let _ = std::io::Write::write_all(&mut stream, response);
        }
    });

    let name = "__modelc_test_auth__";
    let dest = modelc::store::download_auth(
        &format!("http://{addr}/m.modelc"),
        name,
        Some("tok123"),
    )?;
    assert_eq!(std::fs::read(&dest)?, payload);
    modelc::store::remove_model(name, false, true)?;

    let req = String::from_utf8_lossy(&captured.lock().unwrap()).to_string();
    assert!(
        req.to_lowercase().contains("authorization: bearer tok123"),
        "request missing bearer header:\n{req}"
    );
    Ok(())
}
