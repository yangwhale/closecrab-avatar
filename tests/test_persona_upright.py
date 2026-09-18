"""EXIF 摆正 —— 手机照片几乎都是横着存、靠标记说明「其实是竖的」。"""
import io
import pytest
from PIL import Image
from worker.imageutil import upright as _upright


def jpeg(size, orientation=None):
    im = Image.new("RGB", size, (120, 60, 30))
    out = io.BytesIO()
    if orientation is None:
        im.save(out, format="JPEG")
    else:
        ex = im.getexif(); ex[274] = orientation
        im.save(out, format="JPEG", exif=ex)
    return out.getvalue()


def test_rotates_when_exif_says_so():
    """⭐ Orientation=6 是顺时针 90°，横存竖图 —— Chris 那张自拍就是这个。"""
    got = _upright(jpeg((800, 600), orientation=6))
    assert got is not None, "带旋转标记的图没被摆正"
    assert Image.open(io.BytesIO(got)).size == (600, 800)


def test_leaves_upright_images_alone():
    """本来就是正的就别动 —— 重编码一遍是白白掉画质，而参考图的清晰度
    直接决定生成出来那张脸的清晰度。"""
    assert _upright(jpeg((800, 600), orientation=1)) is None
    assert _upright(jpeg((800, 600))) is None


@pytest.mark.parametrize("o", [3, 6, 8])
def test_handles_the_three_common_phone_orientations(o):
    assert _upright(jpeg((800, 600), orientation=o)) is not None


def test_garbage_does_not_raise():
    """取不到图/坏图**不能让会话起不来** —— 旧脸总比没脸好，
    所以这里只能返回 None 让调用方回退，绝不抛。"""
    assert _upright(b"not an image at all") is None
    assert _upright(b"") is None
