#!/usr/bin/env python3
"""端到端自证：**真音频进去，真视频帧出来。**

`smoke.py` 只走控制面契约（不碰 GPU、不连 LiveKit）；这个脚本走完整条链：

    建会话 → 控制面派给某个 worker → worker 进房 → 本脚本扮演 agent 推音频
           → worker 的模型出帧 → 本脚本订阅数字人的视频轨、逐帧收下来

部署完跑一次，能回答四个「装完到底能不能用」的问题：

    有没有出帧            没有 = 模型没跑起来，看 worker 日志里的退回原因
    首帧多久              这是用户感觉到的「反应快不快」
    分辨率对不对          模型按 64 的网格取整，跟你请求的多半不一样
    帧率稳不稳            低于请求帧率说明生成跟不上，得降分辨率或加卡

用法：

    scripts/e2e-check.py --gateway http://127.0.0.1:8080 \\
        --livekit-url ws://10.0.0.1:7880 --audio 一段.wav --out /tmp/e2e

⚠️ 要 `livekit-agents`（控制面的 venv 里没有，那边只装了铸票用的
`livekit-api`）。用跑 agent 那个环境的 python 来执行。
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from livekit import api, rtc                                     # noqa: E402
from livekit.agents.voice.avatar import DataStreamAudioOutput    # noqa: E402

from closecrab_avatar.auth import sign_client_token              # noqa: E402

SAMPLE_RATE = 16000
"""推流采样率。**启动时按输入 wav 改写**（见 `_read_wav_mono`）。

不写死的理由是它本身就是个被怀疑对象：线上 CloseCrab 推 48 kHz，
这个脚本原来只推 16 kHz —— 拿 16 k 量出来的口型偏移不能直接安到线上，
两者中间隔着一次重采样。要比就得能两边都推。"""

_ALLOWED_RATES = (16000, 24000, 48000)


def _read_wav_mono(path: str) -> np.ndarray:
    """读单声道 int16，**采样率以文件为准**。**不做重采样** —— 不对就报错。

    悄悄重采样会让「口型对不上」这种问题多一个嫌疑人；宁可让调用方先转好。
    """
    global SAMPLE_RATE
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        if rate not in _ALLOWED_RATES or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(
                f"{path} 是 {rate} Hz / {w.getnchannels()} 声道 / "
                f"{w.getsampwidth() * 8} bit，需要 {_ALLOWED_RATES} 之一、单声道 16 bit。\n"
                f"  ffmpeg -i {path} -ar 48000 -ac 1 -c:a pcm_s16le 转好的.wav")
        SAMPLE_RATE = rate
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def load_env_file(path: str) -> None:
    """读 systemd 的 `EnvironmentFile`，塞进 `os.environ`。**不打印任何值。**

    ⚠️ **不要用 `. /etc/closecrab-avatar/env` 去 source 它。** systemd 的
    EnvironmentFile **不是 shell 脚本**：它按字面取值，引号是值的一部分的
    反面 —— bash 会把引号吃掉。`LA_API_KEYS=[{"key_id":"x"}]` 被 source
    之后变成 `[{key_id:x}]`，然后 `json.loads` 报
    「Expecting property name enclosed in double quotes」。

    症状看起来像「配置文件写坏了」，其实文件是对的，错的是读法。
    """
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v)


def _gateway_key() -> tuple[str, str]:
    """从 `LA_API_KEYS` 取第一把钥匙。**不打印 secret。**"""
    raw = os.environ.get("LA_API_KEYS")
    if not raw:
        raise SystemExit(
            "没有 LA_API_KEYS。要么 --env-file /etc/closecrab-avatar/env，"
            "要么自己把 LA_API_KEYS / LIVEKIT_API_KEY / LIVEKIT_API_SECRET 导进环境。")
    k = json.loads(raw)[0]
    return k["key_id"], k["secret"]


class _Collector:
    """订阅数字人的视频轨，逐帧记账。"""

    def __init__(self, avatar_identity: str, out_dir: pathlib.Path | None,
                 every: int = 25):
        self.identity = avatar_identity
        self.out = out_dir
        self.every = max(1, every)
        self.count = 0
        self.first_at: float | None = None
        self.last_at: float | None = None
        self.size: tuple[int, int] | None = None
        self._task: asyncio.Task | None = None
        self._dump_warned = False

    def attach(self, room: rtc.Room) -> None:
        # ⚠️ **显式订阅，不要只靠 auto_subscribe。**
        #    worker 那边 `_publish_track()` 里有一句
        #    `await self._audio_publication.wait_for_subscription()` ——
        #    没人订阅音频轨，它就**一直停在那**，视频轨永远发不出来。
        #    表现是「数字人进房了但什么都没有」，看起来像模型挂了。
        #    自动订阅在某些 SDK 版本 / 房间配置下不生效，而这一句代价为零。
        @room.on("track_published")
        def _(pub, participant):
            if participant.identity == self.identity:
                pub.set_subscribed(True)

        @room.on("track_subscribed")
        def _(track, pub, participant):
            if (track.kind == rtc.TrackKind.KIND_VIDEO
                    and participant.identity == self.identity):
                self._task = asyncio.create_task(self._pump(track))

    async def _pump(self, track: rtc.Track) -> None:
        stream = rtc.VideoStream(track)
        async for ev in stream:
            f = ev.frame
            now = time.monotonic()
            if self.first_at is None:
                self.first_at = now
                self.size = (f.width, f.height)
            self.last_at = now
            self.count += 1
            if self.out is not None and self.count % self.every == 1 % max(self.every, 1):
                # 存图只是方便人眼看一下，**炸了不能把收帧带走** ——
                # 否则一个 PIL 没装就让整次测量报「没收到帧」，
                # 结论跟真的模型没跑一模一样。
                try:
                    self._dump(f)
                except Exception as e:      # noqa: BLE001
                    if not self._dump_warned:
                        print(f"  （抽帧存不下来，不影响测量：{e}）")
                        self._dump_warned = True

    def _dump(self, frame: rtc.VideoFrame) -> None:
        from PIL import Image
        rgba = frame.convert(rtc.VideoBufferType.RGBA)
        img = np.frombuffer(rgba.data, dtype=np.uint8).reshape(
            rgba.height, rgba.width, 4)
        Image.fromarray(img, "RGBA").convert("RGB").save(
            self.out / f"frame-{self.count:05d}.jpg", quality=92)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


async def run(a) -> int:
    import aiohttp

    key_id, secret = _gateway_key()
    lk_key = os.environ["LIVEKIT_API_KEY"]
    lk_secret = os.environ["LIVEKIT_API_SECRET"]

    pcm = _read_wav_mono(a.audio)          # ← 顺带把 SAMPLE_RATE 定下来
    secs = len(pcm) / SAMPLE_RATE
    print(f"音频 {a.audio}：{secs:.1f} s / {len(pcm)} 采样 @ {SAMPLE_RATE} Hz")

    room_name = a.room
    avatar_identity, agent_identity = "e2e-avatar", "e2e-agent"
    out_dir = pathlib.Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 建会话 —— 控制面挑一个 worker 派过去
    headers = {"Authorization": f"Bearer {sign_client_token(key_id, secret)}"}
    # ⚠️ **`sample_rate` 不能省。** worker 拿它建数字人自己那条音轨的
    #    `AudioSource`（`AvatarOptions.audio_sample_rate`，缺省 16000），
    #    而进来的音频是**原样转发**进那个 source 的。两边不一致时：
    #
    #      Exception: InvalidState - sample_rate and num_channels don't match
    #
    #    这句异常抛在 `_forward_video` 里，**整条推帧任务当场死掉** ——
    #    模型照常在算，一帧也发不出去。表现是「进房了、音频也收到了、
    #    就是没有画面」，跟模型崩了长得一模一样。
    #
    #    2026-09-19 我拿这个脚本跑 48 k / 24 k 各三次全军覆没，一度以为
    #    「线上那条 48 kHz 的路是坏的」—— 其实 CloseCrab 老老实实发了
    #    `sample_rate=48000`，是这个脚本没发。**探针自己造出来的故障，
    #    差点被当成生产故障报上去。**
    body = dict(provider="liveavatar", livekit_url=a.livekit_url,
                room_name=room_name, room_sid=f"RM_{int(secs * 1000)}",
                avatar_identity=avatar_identity, avatar_name="E2E",
                agent_identity=agent_identity, sample_rate=SAMPLE_RATE)
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{a.gateway}/avatar/sessions",
                             json=body, headers=headers) as r:
            if r.status != 200:
                print(f"❌ 建会话失败 {r.status}：{(await r.text())[:300]}")
                return 1
            sess = await r.json()
        psid = sess["provider_session_id"]
        # ⚠️ terminate 认的是**这个 token**，不是调用者身份（契约如此）。
        #    不带它 422，会话就一直挂着 —— 下次跑直接 429「所有槽位都忙」。
        tt = sess.get("terminate_token", "")
        print(f"会话 {psid} 已派出，等 worker 进房…")
        # ⚠️ 从这一行往下**任何**出路都必须还槽位。上一版把 terminate 放在
        #    正常路径末尾，结果脚本中途抛异常，槽位被一个已经没人的会话
        #    占死，下一次跑直接 429「所有槽位都忙」—— 看起来像容量不够。
        try:
            return await _drive(http, a, headers, psid, pcm, secs, out_dir)
        finally:
            await _terminate(http, a, headers, psid, tt)


async def _drive(http, a, headers, psid, pcm, secs, out_dir) -> int:
    lk_key = os.environ["LIVEKIT_API_KEY"]
    lk_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = a.room
    avatar_identity, agent_identity = "e2e-avatar", "e2e-agent"

    t_created = time.monotonic()

    # 2. 扮演 agent 进房
    tok = (api.AccessToken(lk_key, lk_secret)
           .with_identity(agent_identity).with_name("E2E Agent")
           .with_grants(api.VideoGrants(room_join=True, room=room_name,
                                        can_publish=True, can_subscribe=True))
           .to_jwt())
    agent_room = rtc.Room()
    await agent_room.connect(a.livekit_url, tok)

    # ⚠️ **假 agent 必须真发一条音轨，哪怕全是静音。**
    #
    #   worker 那边 `DataStreamAudioReceiver.start()` 调的是
    #   `wait_for_participant(identity=agent_identity)`，而那个函数只认
    #   **ACTIVE** 状态的参与者（`p.state == PARTICIPANT_STATE_ACTIVE`，
    #   或者等 `participant_active` 事件）。一个轨都不发布的参与者在对端
    #   SDK 眼里停在 JOINED，永远不会 active —— 于是 `runner.start()` 卡死在
    #   第一步，音视频轨一个都发不出来。
    #
    #   生产上碰不到这一条，因为真 bot（`livekit_out`）本来就往房间里发音轨
    #   （摘掉数字人时手机听的就是它）。**这个脚本原来比生产少做了一件事，
    #   于是造出一个生产不存在的死锁。** 症状是「数字人进房了但一帧没有」，
    #   跟模型没跑起来一模一样 —— 查了三轮才查到。
    #   而且**光 publish 不够，要真有帧在流**：只发布不喂数据的话媒体协商
    #   可能一直不落地，参与者照样不 active。所以起一条后台任务持续推静音，
    #   就跟真 bot 不说话时那条麦克风轨一样。
    silent = rtc.AudioSource(SAMPLE_RATE, 1)
    await agent_room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track("e2e-agent-mic", silent),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

    async def _pump_silence() -> None:
        n = SAMPLE_RATE // 50                      # 20 ms
        buf = np.zeros(n, dtype=np.int16).tobytes()
        while True:
            await silent.capture_frame(rtc.AudioFrame(
                data=buf, sample_rate=SAMPLE_RATE, num_channels=1,
                samples_per_channel=n))

    silence_task = asyncio.create_task(_pump_silence())

    # 3. 扮演观众进房，订阅数字人的视频轨
    vtok = (api.AccessToken(lk_key, lk_secret)
            .with_identity("e2e-viewer").with_name("E2E Viewer")
            .with_grants(api.VideoGrants(room_join=True, room=room_name,
                                         can_publish=False, can_subscribe=True))
            .to_jwt())
    viewer = rtc.Room()
    col = _Collector(avatar_identity, out_dir, every=a.save_every)
    col.attach(viewer)
    await viewer.connect(a.livekit_url, vtok)

    # 等数字人进来。**进不来就别往下走** —— 后面推音频只会喂给空气。
    deadline = time.monotonic() + a.join_timeout
    while avatar_identity not in agent_room.remote_participants:
        if time.monotonic() > deadline:
            print(f"❌ 等了 {a.join_timeout}s 数字人没进房。"
                  f"看 worker 日志：是没派到，还是进房失败")
            return 1
        await asyncio.sleep(0.3)
    t_joined = time.monotonic()
    print(f"数字人已进房（{t_joined - t_created:.1f} s）")

    # ⚠️ **进房 ≠ 能收音频**，但「等视频轨」这个判据是个死锁，别再写回来。
    #
    # 真实约束：`AvatarRunner.start()` 先 `audio_recv.start()`（挂上
    # `lk.audio_stream` 回调）、**再**发布音视频轨。进房那一刻两件事都没做，
    # 这中间要加载形象、reset 模型。往里推音频的话 LiveKit 只在 worker 日志
    # 留一句 `ignoring byte stream ... no callback attached` —— 不报错、
    # 不重试，这一段**直接丢掉**。
    #
    # 所以这里原来等「数字人发布了视频轨」再推。那在 `_lazy_publish=False`
    # 的年代是对的，现在**必然超时**：
    #
    #     发视频轨 ← 要第一帧 ← 要音频 ← 我在等视频轨
    #     └──────────────── 转圈 ────────────────┘
    #
    # `AvatarRunner.__init__` 的默认值 `_lazy_publish=True`（官方注释：
    # publish tracks **until the first frame pushed**）。2026-09-19 实测：
    # 探针每次都停在「60s 内没发布视频轨」，而 worker 日志里连
    # 「第一帧音频到了」都没有 —— 因为一个字节都没推出去。
    # 同一个坑 CloseCrab 侧刚踩过一遍，判据换成「人进房了」才解开。
    #
    # 换成**静音前导**：接收端没挂上之前丢掉的是静音，损失为零；
    # 真音频从 `t_audio0` 起算，首帧延迟照样准。比猜一个 sleep 秒数稳，
    # 因为它不依赖「多久能就位」这个会随机器和形象变的量。
    LEADIN_S = a.leadin
    # ⚠️ **默认不垫静音了。** Chris 2026-09-19：「那个前导静音咱也不要了吧？
    #    没必要，徒增烦恼。上来就正常音频。」
    #
    #    它当初是为了绕开「接收端还没挂上、推进去的音频被静默丢掉」。
    #    代价是那几秒也会生成自己的帧（一张不说话的脸），把「哪些帧属于
    #    我的音频」这件事搅浑 —— 而那正是今天要查清的东西。
    #
    #    不垫的代价：开头零点几秒的话可能丢。但**音视频会一起丢**
    #    （丢掉的音频压根没进管线，也就没有对应的帧），所以不影响同步判断。
    print("开始推音频" + (f"（前面垫 {LEADIN_S:.1f} s 静音）" if LEADIN_S else "（不垫静音，上来就是真音频）"))

    # 4. 推音频。按实时速率推 —— **一次性灌进去量不出真实延迟**。
    out = DataStreamAudioOutput(agent_room, destination_identity=avatar_identity,
                                sample_rate=SAMPLE_RATE)
    chunk = SAMPLE_RATE // 10                     # 100 ms 一包

    # 静音前导。**不能用 sleep 代替** —— 要的就是「真的在推」，
    # 这样接收端一挂上就立刻开始收，不用再等下一个时间点。
    quiet = np.zeros(chunk, dtype=np.int16)
    for _ in range(int(LEADIN_S * 10)):
        await out.capture_frame(rtc.AudioFrame(
            data=quiet.tobytes(), sample_rate=SAMPLE_RATE,
            num_channels=1, samples_per_channel=chunk))
        await asyncio.sleep(0.1)

    t_audio0 = time.monotonic()
    for i in range(0, len(pcm), chunk):
        part = pcm[i:i + chunk]
        await out.capture_frame(rtc.AudioFrame(
            data=part.tobytes(), sample_rate=SAMPLE_RATE,
            num_channels=1, samples_per_channel=len(part)))
        await asyncio.sleep(len(part) / SAMPLE_RATE)
    out.flush()
    print(f"音频推完（{time.monotonic() - t_audio0:.1f} s），再收 {a.drain}s 尾巴…")
    await asyncio.sleep(a.drain)

    # 5. 结账
    silence_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await silence_task
    await col.stop()
    await agent_room.disconnect()
    await viewer.disconnect()
    return _report(a, col, secs, t_audio0)


def _report(a, col: _Collector, secs: float, t_audio0: float) -> int:
    print("\n── 结果 ──")
    if not col.count:
        print("❌ **一帧都没收到。**")
        print("   多半是 worker 退回静帧了（那样连视频轨都不会发）或者模型崩了。")
        print("   去 worker 日志里找启动时那条 warning —— 它写了是哪一条不满足。")
        return 1

    first = col.first_at - t_audio0
    span = (col.last_at - col.first_at) or 1e-9
    fps = (col.count - 1) / span
    print(f"  首帧延迟   {first:.2f} s   （音频开推 → 第一帧到手）")
    print(f"  收到帧数   {col.count}")
    print(f"  分辨率     {col.size[0]}×{col.size[1]}")
    print(f"  实测帧率   {fps:.1f} fps")
    print(f"  覆盖时长   {col.count / max(fps, 1e-9):.1f} s（音频 {secs:.1f} s）")
    if a.out:
        print(f"  抽帧       {a.out}/frame-*.jpg（每 {a.save_every} 帧一张）")

    ok = True
    if first > a.max_first_frame:
        print(f"  ⚠️ 首帧 {first:.2f}s 超过阈值 {a.max_first_frame}s")
        ok = False
    if fps < a.min_fps:
        print(f"  ⚠️ 帧率 {fps:.1f} 低于阈值 {a.min_fps} —— 生成跟不上，"
              f"降分辨率或加卡")
        ok = False
    print("\n✅ 端到端通" if ok else "\n⚠️ 通了但没达标")
    return 0 if ok else 2


async def _terminate(http, a, headers, psid: str, terminate_token: str) -> None:
    """还槽位。**失败要说出来** —— 静默失败下次就是 429，而 429 看起来像
    「容量不够」，跟「上次没还干净」是完全不同的诊断方向。
    """
    try:
        async with http.post(
            f"{a.gateway}/avatar/sessions/terminate",
            json={"provider": "liveavatar", "provider_session_id": psid,
                  "terminate_token": terminate_token},
            headers=headers,
        ) as r:
            if r.status != 200:
                print(f"⚠️ 会话没还干净（{r.status}：{(await r.text())[:120]}）"
                      f"，下次跑可能 429")
    except Exception as e:                    # noqa: BLE001
        print(f"⚠️ 还槽位时出错：{e}")


def main() -> int:
    p = argparse.ArgumentParser(description="CloseCrab Avatar 端到端自证")
    p.add_argument("--gateway", default=os.environ.get("LA_GATEWAY_URL",
                                                       "http://127.0.0.1:8080"))
    p.add_argument("--livekit-url", required=True)
    p.add_argument("--audio", required=True,
                   help="单声道 16 bit wav，16k/24k/48k 之一（按文件里的采样率推）")
    p.add_argument("--leadin", type=float, default=2.0,
                   help="真音频之前垫多少秒静音，等接收端挂上（默认 2）")
    p.add_argument("--room", default=f"e2e-{os.getpid()}")
    p.add_argument("--out", help="抽帧存哪（不给就不存）")
    p.add_argument("--save-every", type=int, default=25,
                   help="每几帧存一张。想拼成片就给 1")
    p.add_argument("--join-timeout", type=float, default=60.0)
    p.add_argument("--drain", type=float, default=5.0, help="推完音频再收几秒")
    p.add_argument("--max-first-frame", type=float, default=3.0)
    p.add_argument("--min-fps", type=float, default=20.0)
    p.add_argument("--env-file", help="systemd EnvironmentFile，默认 /etc/closecrab-avatar/env")
    a = p.parse_args()
    envf = a.env_file or "/etc/closecrab-avatar/env"
    if os.path.exists(envf):
        load_env_file(envf)
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
