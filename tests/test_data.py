from recommenditos import config


def test_data_dirs_are_nested_under_project_root():
    assert config.RAW_DATA_DIR == config.DATA_DIR / "raw"
    assert config.INTERIM_DATA_DIR == config.DATA_DIR / "interim"
    assert config.PROCESSED_DATA_DIR == config.DATA_DIR / "processed"
    assert config.EXTERNAL_DATA_DIR == config.DATA_DIR / "external"
    assert config.DATA_DIR == config.PROJ_ROOT / "data"
