"""形象库 —— 存取、类型识别、路径安全。

这一层最容易出的两类问题都不报错：
  · 按扩展名认类型 → 一个伪装成 jpg 的东西让模型在加载时抛看不懂的错
  · 房间名直接拼路径 → `../` 写到任何地方
所以两条都单独钉。
"""
import json
import pathlib

import pytest

from closecrab_avatar.persona import MAX_BYTES, PersonaError, PersonaStore, sniff

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64


@pytest.fixture
def store(tmp_path) -> PersonaStore:
    return PersonaStore(str(tmp_path / "personas"))


# ── 类型识别 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("data,ctype", [
    (JPEG, "image/jpeg"), (PNG, "image/png"), (WEBP, "image/webp")])
def test_sniff_by_content(data, ctype):
    assert sniff(data)[0] == ctype


def test_sniff_rejects_non_image():
    """⭐ 按**内容**判，不按扩展名 —— 改个名字骗不过去。

    放过去的后果是模型在加载参考图时抛一个跟图片八竿子打不着的错，
    查半天才发现是上传的东西根本不是图。
    """
    with pytest.raises(PersonaError, match="认不出"):
        sniff(b"<!DOCTYPE html><html>not an image</html>")


def test_sniff_rejects_riff_that_is_not_webp():
    """RIFF 不等于 WebP —— wav 也是 RIFF 开头。"""
    with pytest.raises(PersonaError):
        sniff(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 64)


def test_sniff_rejects_oversize():
    with pytest.raises(PersonaError, match="太大"):
        sniff(JPEG + b"\x00" * MAX_BYTES)


# ── ⭐ 路径安全 ───────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    "../etc/passwd", "a/b", "..", ".hidden", "", "x" * 65, "room name", "r;m"])
def test_room_name_traversal_is_refused(store, bad):
    """⭐ 房间名直接当文件名是危险的。

    **不做静默替换** —— 替换之后两个不同房间可能落到同一个文件上，
    那比报错难查得多。
    """
    with pytest.raises(PersonaError):
        store.put(bad, JPEG)


@pytest.mark.parametrize("ok", ["bunny", "jarvis", "room-1", "a_b", "v1.2"])
def test_normal_room_names_pass(store, ok):
    assert store.put(ok, JPEG).room == ok


# ── 存取 ──────────────────────────────────────────────────────────

def test_put_then_get_roundtrip(store):
    p = store.put("bunny", PNG, note="兔子在机房")
    assert (p.content_type, p.bytes_len, p.note) == ("image/png", len(PNG), "兔子在机房")

    got = store.get("bunny")
    assert got == p
    data, ctype = store.read_image("bunny")
    assert data == PNG and ctype == "image/png"


def test_missing_room_reads_as_none(store):
    """「没设过」要能跟「设过」区分开 —— 客户端据此决定显示默认脸。"""
    assert store.get("nobody") is None
    assert store.read_image("nobody") is None


def test_version_is_content_addressed(store):
    """⭐ 同一张图重传，版本号不能变。

    worker 靠版本号判断「要不要重开生成循环」。用时间戳的话每次上传都
    触发一次重启，而重启中间画面会断 —— 用户看到的是「我什么都没改，
    它自己抖了一下」。
    """
    v1 = store.put("bunny", JPEG).version
    v2 = store.put("bunny", JPEG).version
    assert v1 == v2
    v3 = store.put("bunny", PNG).version
    assert v3 != v1


def test_changing_format_removes_the_old_file(store):
    """换格式之后目录里不能留两张图 —— 留着就说不清谁是当前的。"""
    store.put("bunny", JPEG)
    store.put("bunny", PNG)
    root = pathlib.Path(store._root)                      # noqa: SLF001
    imgs = sorted(p.name for p in root.glob("bunny.*") if p.suffix != ".json")
    assert imgs == ["bunny.png"], f"目录里还剩：{imgs}"


def test_meta_without_image_reports_none_not_stale(store):
    """⭐ 元数据在、图没了 → 当作没有。

    静默返回旧内容或半截文件的话，worker 会用一张坏图去跑，
    而你以为自己换过了。
    """
    store.put("bunny", JPEG)
    (pathlib.Path(store._root) / "bunny.jpg").unlink()    # noqa: SLF001
    assert store.read_image("bunny") is None


def test_write_is_atomic_no_tmp_left_behind(store):
    store.put("bunny", JPEG)
    leftovers = list(pathlib.Path(store._root).glob(".tmp-*"))   # noqa: SLF001
    assert not leftovers, f"留下了临时文件：{leftovers}"


def test_delete_and_list(store):
    store.put("bunny", JPEG)
    store.put("jarvis", PNG)
    assert store.rooms() == ["bunny", "jarvis"]
    assert store.delete("bunny") is True
    assert store.delete("bunny") is False
    assert store.rooms() == ["jarvis"]
    assert not list(pathlib.Path(store._root).glob("bunny.*"))   # noqa: SLF001


def test_meta_json_is_readable_by_humans(store):
    """元数据是要用肉眼排障的 —— 别写成 base64 或 pickle。"""
    store.put("bunny", JPEG, note="矮人铁匠")
    d = json.loads((pathlib.Path(store._root) / "bunny.json")   # noqa: SLF001
                   .read_text(encoding="utf-8"))
    assert d["note"] == "矮人铁匠" and d["ext"] == ".jpg"
