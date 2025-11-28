from accusleepy.fileio import AccuSleePyConfig, load_config


def test_load_config():
    """Test the configuration file loads successfully"""
    config = load_config()
    assert isinstance(config, AccuSleePyConfig)
    assert type(config.epochs_to_show) is int
