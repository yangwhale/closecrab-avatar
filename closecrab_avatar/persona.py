"""形象库 —— 每个房间当前用哪张脸。

## 它解决的问题

参考图原来是 worker 启动时的一个命令行参数，写死在进程里。想换张脸
就得重启那一组五卡，两三分钟；而且**外面没有任何地方看得到当前用的是哪张**
—— 想知道只能去 ssh 到 GPU 机器上看命令行。

这一层把「当前形象」变成一个可读可写的东西：

    PUT  /avatar/persona/{room}          传一张新的（原始字节）
    GET  /avatar/persona/{room}          问现在用的是哪张（元数据 + 版本）
    GET  /avatar/persona/{room}/image    把那张图拿回来

worker 在**重开生成循环**时按 room 取图，所以换脸的代价从「重启进程」
降到「重开循环」。

## 为什么存文件不存进 SQLite

图片是几十 KB 到几 MB 的二进制，塞进状态库会让每次备份和 diff 都变重，
而它本来就不需要事务。文件系统 + 一个 json 元数据刚好。

## ⚠️ 版本号不是装饰

worker 得知道「这张图跟我正在用的是不是同一张」。比时间戳可靠：
同一秒内换两次、或者机器时钟回拨，时间戳都会骗人。这里用**内容哈希**，
同样的图重复上传不会触发无谓的循环重启。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import asdict, dataclass

log = logging.getLogger("closecrab.avatar.persona")

# 认得的图片魔数。**按内容判，不按扩展名** —— 客户端传上来的文件名
# 完全不可信，而一个伪装成 jpg 的 HTML 会让模型在加载时抛一个
# 跟图片八竿子打不着的错。
_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"RIFF", "image/webp", ".webp"),          # 还要再看 8..12 是不是 WEBP
)

HISTORY_KEEP = 10
"""换掉的旧形象图留几张。**不能不留** —— 用户手里未必有副本
（相册那张可能已经裁过，或者干脆是别人发来的）。也不能无限留，
一张几 MB，迟早把盘塞满。"""

MAX_BYTES = 12 * 1024 * 1024
"""单张上限。参考图是拿来当一帧画面用的，没有理由超过这个量级；
不设上限的话一个手滑就能把控制面的磁盘塞满。"""


class PersonaError(ValueError):
    """传上来的东西不能用。**消息直接回给客户端**，所以要写人话。"""


@dataclass(frozen=True)
class Persona:
    room: str
    version: str          # 内容哈希前 16 位
    content_type: str
    bytes_len: int
    width: int
    height: int
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def sniff(data: bytes) -> tuple[str, str]:
    """认出图片类型。认不出就报错，**不猜**。"""
    if len(data) > MAX_BYTES:
        raise PersonaError(f"图片太大（{len(data) // 1024} KB），上限 {MAX_BYTES // 1024 // 1024} MB")
    for magic, ctype, ext in _MAGIC:
        if not data.startswith(magic):
            continue
        if magic == b"RIFF" and data[8:12] != b"WEBP":
            continue
        return ctype, ext
    raise PersonaError("认不出这是什么图片（支持 JPEG / PNG / WebP）。"
                       "注意是按文件内容判的，改扩展名没用")


def measure(data: bytes) -> tuple[int, int]:
    """量宽高。**拿不到就是 0×0，不要因此拒绝上传** ——

    控制面上可能没装 Pillow（它本来就只需要 fastapi 那几个），
    宽高只是给人看的信息，不该变成上传的硬依赖。
    """
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return int(im.width), int(im.height)
    except Exception:                           # noqa: BLE001
        return 0, 0


class PersonaStore:
    """一个房间一张脸，落在磁盘上。"""

    def __init__(self, root: str):
        self._root = pathlib.Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ── 路径 ──────────────────────────────────────────────────────

    def _safe(self, room: str) -> str:
        """房间名直接当文件名是危险的 —— `../` 能写到任何地方。

        白名单收窄到「字母数字和三个符号」，其余一律拒。**不做静默替换**：
        替换之后两个不同的房间可能落到同一个文件上，那比报错难查得多。
        """
        if not room or len(room) > 64:
            raise PersonaError("房间名为空或过长")
        ok = all(c.isalnum() or c in "-_." for c in room)
        if not ok or room.startswith("."):
            raise PersonaError(f"房间名 {room!r} 含不允许的字符（只收字母数字和 - _ .）")
        return room

    def _meta_path(self, room: str) -> pathlib.Path:
        return self._root / f"{self._safe(room)}.json"

    def _img_path(self, room: str, ext: str) -> pathlib.Path:
        return self._root / f"{self._safe(room)}{ext}"

    # ── 读写 ──────────────────────────────────────────────────────

    def put(self, room: str, data: bytes, *, note: str = "") -> Persona:
        ctype, ext = sniff(data)
        w, h = measure(data)
        version = hashlib.sha256(data).hexdigest()[:16]
        p = Persona(room=self._safe(room), version=version, content_type=ctype,
                    bytes_len=len(data), width=w, height=h, note=note)

        # ⭐ **换之前先把旧的那张归档。**
        #
        # 原来这里是直接 unlink 掉旧格式那份，理由是「目录里留两张说不清谁是
        # 当前的」。理由没错，但代价没想清楚：2026-09-18 Chris 从手机传了一张
        # 5712×4284 的照片，十二分钟后我用生成图覆盖，**他那张原图就没了**。
        #
        # 形象图是**用户手里可能没有副本**的东西（相册里那张已经被裁过、
        # 或者干脆是别人发来的）。这一层不该做不可逆的删除。
        self._archive_current(room)

        # 先写临时文件再原子改名 —— 半截文件会让 worker 在加载时报一个
        # 看起来像「模型坏了」的错。
        img = self._img_path(room, ext)
        self._atomic_write(img, data)
        # 当前那张只能有一份，其余格式的清掉（上面已经归档过了）。
        for _, _, other in _MAGIC:
            if other != ext:
                self._img_path(room, other).unlink(missing_ok=True)
        self._atomic_write(self._meta_path(room),
                           json.dumps({**p.to_json(), "ext": ext},
                                      ensure_ascii=False).encode())
        log.info("房间 %s 换了形象图：%s %dx%d %d 字节 版本 %s",
                 room, ctype, w, h, len(data), version)
        return p

    def get(self, room: str) -> Persona | None:
        try:
            raw = self._meta_path(room).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        d = json.loads(raw)
        d.pop("ext", None)
        return Persona(**d)

    def read_image(self, room: str) -> tuple[bytes, str] | None:
        try:
            d = json.loads(self._meta_path(room).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        img = self._img_path(room, d.get("ext", ".jpg"))
        try:
            return img.read_bytes(), d["content_type"]
        except FileNotFoundError:
            # 元数据在、图没了。**说出来** —— 静默当作「没设置过」的话，
            # worker 会退回默认脸，而你以为自己换过了。
            log.warning("房间 %s 的元数据在但图片文件不见了：%s", room, img)
            return None

    def delete(self, room: str) -> bool:
        found = self._meta_path(room).exists()
        self._meta_path(room).unlink(missing_ok=True)
        for _, _, ext in _MAGIC:
            self._img_path(room, ext).unlink(missing_ok=True)
        return found

    # ── 历史 ──────────────────────────────────────────────────────

    def _hist_dir(self, room: str) -> pathlib.Path:
        return self._root / "history" / self._safe(room)

    def _archive_current(self, room: str) -> None:
        """把当前那张挪进 history。**失败不能挡住上传** ——
        归档是保险，不是主线；因为存不下历史而拒绝换脸是本末倒置。
        """
        try:
            got = self.read_image(room)
            meta = self.get(room)
            if got is None or meta is None:
                return
            data, _ = got
            d = self._hist_dir(room)
            d.mkdir(parents=True, exist_ok=True)
            ext = json.loads(self._meta_path(room).read_text(encoding="utf-8")).get("ext", ".jpg")
            self._atomic_write(d / f"{meta.version}{ext}", data)
            self._atomic_write(d / f"{meta.version}.json",
                               json.dumps(meta.to_json(), ensure_ascii=False).encode())
            self._trim_history(room)
        except Exception:                            # noqa: BLE001
            log.warning("房间 %s 的旧形象图没归档成功（不影响这次上传）", room, exc_info=True)

    def _trim_history(self, room: str, keep: int = HISTORY_KEEP) -> None:
        """只留最近 keep 张。图是几 MB 一张，不设上限迟早把盘塞满。"""
        d = self._hist_dir(room)
        imgs = sorted((p for p in d.glob("*") if p.suffix != ".json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for old in imgs[keep:]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

    def history(self, room: str) -> list[Persona]:
        """按新到旧列出换掉过的那些。"""
        d = self._hist_dir(room)
        if not d.is_dir():
            return []
        out = []
        for meta in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                out.append(Persona(**json.loads(meta.read_text(encoding="utf-8"))))
            except Exception:                        # noqa: BLE001
                continue
        return out

    def read_history_image(self, room: str, version: str) -> tuple[bytes, str] | None:
        d = self._hist_dir(room)
        for _, ctype, ext in _MAGIC:
            f = d / f"{version}{ext}"
            if f.exists():
                return f.read_bytes(), ctype
        return None

    def rooms(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.json"))

    @staticmethod
    def _atomic_write(path: pathlib.Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise
