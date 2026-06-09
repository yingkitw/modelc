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
