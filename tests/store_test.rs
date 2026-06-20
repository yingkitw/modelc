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
    let tmp = tempfile::NamedTempFile::new()?;
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
    let tmp = tempfile::NamedTempFile::new()?;
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
    let tmp = tempfile::NamedTempFile::new()?;
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
