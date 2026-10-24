import pytest

from pipeline import config

UUID = "0a1b2c3d-4e5f-6789-abcd-ef0123456789"


def test_env_overrides_data_dir(reload_config, tmp_path):
    config = reload_config(RS_DATA_DIR=tmp_path)
    assert config.DATA_DIR == tmp_path
    assert config.META_DIR == tmp_path / "metadata"
    assert config.DETECTOR_DIR == tmp_path / "detector"


def test_require_image_root_exits_when_missing(reload_config, tmp_path):
    config = reload_config(RS_IMAGE_ROOT=tmp_path)
    with pytest.raises(SystemExit, match="Mount the drive or set RS_IMAGE_ROOT"):
        config.require_image_root()
    (tmp_path / "00").mkdir()
    assert config.require_image_root() == tmp_path


def test_image_path_uses_two_char_prefix(reload_config, tmp_path):
    config = reload_config(RS_IMAGE_ROOT=tmp_path)
    assert config.image_path(UUID) == tmp_path / "0a" / f"{UUID}.jpg"


@pytest.mark.parametrize("bad", ["../../etc/passwd", UUID.upper(), UUID + "\n", UUID + "/x", UUID[:-1], ""])
def test_is_uuid_rejects_non_canonical_ids(bad):
    assert config.is_uuid(UUID)
    assert not config.is_uuid(bad)
