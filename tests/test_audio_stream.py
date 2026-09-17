"""块几何。就三个数，但拿错任何一个都是静默错误。"""
import pytest

from worker.audio_stream import BlockGeometry

G = BlockGeometry()


def test_production_values():
    # fps 是 25 不是 16 —— s2v-14B 走 wan_2_2/configs/，不是 wan_base 那棵树。
    # 曾经拿错过一次，产物 mp4（443 帧 / 17.72 s = 25 fps）才把它揪出来。
    assert (G.fps, G.frames_per_block, G.sample_rate) == (25, 12, 16000)


def test_block_is_point_four_eight_seconds():
    assert G.block_seconds == 0.48


def test_block_samples_uses_audio_rate_not_frame_rate():
    """⭐ 三个率都像「帧率」：视频 25 fps、音频嵌入 30 Hz、PCM 16000 Hz。

    拿错差几个数量级，而现象只是「怎么一直不出帧」—— 不会报错。
    """
    assert G.block_samples == 7680 == int(16000 * 0.48)
    assert G.block_samples != G.frames_per_block
    assert G.block_samples != int(30 * 0.48)


@pytest.mark.parametrize("kw", [{"fps": 0}, {"frames_per_block": 0}, {"sample_rate": 0}])
def test_bad_values_raise(kw):
    # 错了只会让口型跟声音差一截，不会报错。宁可起不来。
    with pytest.raises(ValueError):
        BlockGeometry(**kw)


def test_scales_with_config():
    assert BlockGeometry(fps=30, frames_per_block=15).block_seconds == 0.5
    assert BlockGeometry(fps=30, frames_per_block=15).block_samples == 8000
