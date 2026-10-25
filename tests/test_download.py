import io
import tarfile

import pytest

from skynet.data.download import download_mini, extract_needed, find_nuscenes, find_or_download


@pytest.fixture
def make_archive(tmp_path):
    def _make(prefix=""):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, data in (
                ("v1.0-mini/sample.json", b"[]"),
                ("v1.0-mini/scene.json", b"[]"),
                ("samples/LIDAR_TOP/a.pcd.bin", b"\x00" * 20),
                ("samples/CAM_FRONT/a.jpg", b"jpg"),
                ("sweeps/LIDAR_TOP/b.pcd.bin", b"\x00" * 20),
                ("maps/m.png", b"png"),
            ):
                info = tarfile.TarInfo(prefix + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        path = tmp_path / "v1.0-mini.tgz"
        path.write_bytes(buf.getvalue())
        return path

    return _make


def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_extract_keeps_only_tables(make_archive, tmp_path):
    root = tmp_path / "out"

    with make_archive().open("rb") as f:
        extract_needed(f, root)

    assert _files(root) == ["v1.0-mini/sample.json", "v1.0-mini/scene.json"]


def test_extract_with_lidar_adds_keyframe_point_clouds_only(make_archive, tmp_path):
    root = tmp_path / "out"

    with make_archive().open("rb") as f:
        extract_needed(f, root, with_lidar=True)

    assert _files(root) == [
        "samples/LIDAR_TOP/a.pcd.bin",
        "v1.0-mini/sample.json",
        "v1.0-mini/scene.json",
    ]


def test_extract_strips_archive_prefix(make_archive, tmp_path):
    root = tmp_path / "out"

    with make_archive(prefix="mini/").open("rb") as f:
        extract_needed(f, root)

    assert _files(root) == ["v1.0-mini/sample.json", "v1.0-mini/scene.json"]


def test_download_mini_streams_from_url(make_archive, tmp_path):
    root = download_mini(tmp_path / "out", url=make_archive().as_uri())

    assert _files(root) == ["v1.0-mini/sample.json", "v1.0-mini/scene.json"]


def test_find_nuscenes_uses_environment_variable(nuscenes_root, monkeypatch, tmp_path):
    root = nuscenes_root
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NUSCENES_ROOT", str(root))

    assert find_nuscenes() == root


def test_find_nuscenes_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NUSCENES_ROOT", raising=False)

    assert find_nuscenes([str(tmp_path / "nowhere")]) is None


def test_find_or_download_prefers_existing_folder(nuscenes_root):
    assert find_or_download(nuscenes_root) == nuscenes_root
