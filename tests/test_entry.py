"""入口的密钥解析。

这一层看着无聊，但它决定**鉴权按什么配置跑** —— 配额是安全边界，
解析错了不会抛、只会让某个 key 拿到比它该有的更多并发。
所以这里的每条断言都盯「错了会不会静默」。
"""
import pytest

from liveavatar_gateway.auth import ApiKey
from liveavatar_gateway.entry import ENV_KEYS, load_keyring


def test_parses_a_normal_ring():
    ring = load_keyring('[{"key_id":"ios","secret":"s1","max_concurrency":4},'
                        ' {"key_id":"web","secret":"s2"}]')
    assert ring.get("ios") == ApiKey(key_id="ios", secret="s1", max_concurrency=4)
    assert ring.get("web").max_concurrency == ApiKey.max_concurrency  # 用默认
    assert ring.get("nope") is None


def test_secret_may_contain_delimiters():
    """⭐ 这条就是不用 `id:secret:n` 拼接的理由。

    secret 是随机串，含 `:` `,` 完全正常。分隔符方案会把它切碎，
    而切碎的结果**看起来还挺像回事**（少一段就落到默认值），不会报错。
    """
    nasty = "a:b,c:d,,::"
    ring = load_keyring('[{"key_id":"k","secret":"%s","max_concurrency":2}]' % nasty)
    assert ring.get("k").secret == nasty
    assert ring.get("k").max_concurrency == 2


def test_missing_env_refuses_to_start(monkeypatch):
    """⭐ 没有 key 的网关能起来、能过 /healthz，但每个请求都 401 ——
    看起来像客户端配错了。宁可起不来。
    """
    monkeypatch.delenv(ENV_KEYS, raising=False)
    with pytest.raises(RuntimeError, match=ENV_KEYS):
        load_keyring()


@pytest.mark.parametrize("raw,why", [
    ("", "空串"),
    ("   ", "只有空白"),
    ("{", "不是合法 JSON"),
    ("{}", "对象不是数组"),
    ("[]", "空数组"),
    ('["ios"]', "元素不是对象"),
    ('[{"secret":"s"}]', "缺 key_id"),
    ('[{"key_id":"k"}]', "缺 secret"),
    ('[{"key_id":"k","secret":""}]', "secret 是空串"),
])
def test_bad_shapes_raise(raw, why):
    with pytest.raises(RuntimeError):
        load_keyring(raw)


def test_duplicate_key_id_raises():
    """⭐ 重复 key_id 用 dict 装会**静默覆盖**，连同它的并发上限一起。

    真实后果：运维给某个 key 临时调大并发、复制粘贴多留了一行旧的，
    结果生效的是旧那条 —— 配额悄悄变回去，没有任何日志。
    """
    with pytest.raises(RuntimeError, match="重复"):
        load_keyring('[{"key_id":"k","secret":"a","max_concurrency":1},'
                     ' {"key_id":"k","secret":"b","max_concurrency":99}]')


@pytest.mark.parametrize("n", [0, -1, "8", 1.5, None])
def test_bad_concurrency_raises(n):
    import json
    raw = json.dumps([{"key_id": "k", "secret": "s", "max_concurrency": n}])
    with pytest.raises(RuntimeError, match="max_concurrency"):
        load_keyring(raw)


def test_true_is_not_a_valid_concurrency():
    """⭐ `bool` 是 `int` 的子类 —— `True` 会通过 isinstance(int) 然后被当成 1。

    JSON 里写 `true` 多半是手滑，而「并发上限 1」是个能跑的值，
    症状是那个 key 莫名其妙只能开一路。单独挡掉。
    """
    with pytest.raises(RuntimeError, match="max_concurrency"):
        load_keyring('[{"key_id":"k","secret":"s","max_concurrency":true}]')


def test_reads_from_env_when_no_arg(monkeypatch):
    monkeypatch.setenv(ENV_KEYS, '[{"key_id":"e","secret":"s"}]')
    assert load_keyring().get("e") is not None
