# GPU 机器环境

`./scripts/install.sh --role worker` 已经把下面的事全做了。
**这份是给「装不起来要查为什么」和「想知道为什么是这几个版本」的人看的。**

## 三个坑

三条的共同点：**pip 返回 0、`import` 也成功、真跑才炸。**

| ⛔ 别做 | 症状 | 正确做法 |
|---|---|---|
| venv 带 `--system-site-packages` | `ImportError: libcusparseLt.so.0` / `hf` 命令不存在但包能 import | 建**干净** venv |
| 装 FlashAttention **3** | 五个 rank 一起崩：`flash-attention/hopper/...: no kernel image is available` | 装 **FA2 2.8.3** |
| 照 `requirements.txt` 装完就跑 | `from transformers import PreTrainedModel` → 循环导入 RuntimeError | 装完**卸掉 deepspeed** |

### ① `--system-site-packages` 为什么是陷阱

pip 的「已满足」只看**包名和版本区间**，不看它是不是**配套的那一份**：

- `pip install huggingface_hub` → 系统里有，什么都不装 → **入口脚本 `hf` 没进 venv/bin**
- `pip install torch==2.8.0 --index-url .../cu128` → torch 本体进来了，但配套的
  `nvidia-*-cu12` 被系统的 cu129 版「满足」掉 → **cu128 那套压根没装**

⇒ **要某个包的特定版本、而系统里也有它时，就别共享 site-packages。**

### ② FlashAttention：FA3 崩、零 FA 也崩

| 配置 | 结果 |
|---|---|
| FA3 | ❌ wheel 是 **Hopper（sm_90）** 编的，B200 是 **Blackwell（sm_100）** |
| 一个都不装 | ❌ `AssertionError: FLASH_ATTN_2_AVAILABLE` |
| **FA2 2.8.3** | ✅ **唯一能跑的** |

为什么零 FA 也不行 —— 主干自注意力确实走 cuDNN
（`wan_2_2/modules/attention.py` 的 `attention()` 里 `cudnn_require()` 优先），
但 **cross-attention 在 `model.py:175` 直接调 `flash_attention()`，绕过了那个入口**：

```
model.py:175  x = flash_attention(q, k, v, k_lens=context_lens)
                → attention.py:143  assert FLASH_ATTN_2_AVAILABLE   ← 死在这
```

> 想做到零 FA：把那几处直调（`model.py:145/175`、`causal_model_s2v.py:153/176`、
> `wan_base/modules/model.py`）换成 `attention(`，让它们也走 cuDNN 分支。
> **一行的事，但那是改上游行为，要单独验数值和性能**，还没做。

### ③ deepspeed 会把 transformers 弄坏

```
transformers/modeling_utils.py:158   import deepspeed
  → deepspeed/runtime/hybrid_engine.py:26  取 transformers.models.opt.modeling_opt
    → modeling_opt.py:37  from ...modeling_utils import PreTrainedModel
      → 而 modeling_utils 正卡在自己第 158 行没初始化完 → 循环导入
```

**transformers 只在检测到 deepspeed 存在时才走那一行** —— 装了它反而坏。
推理用不到它（仓库里只在训练路径的 `if strategy == 'deepspeed'` 里，惰性 import）。

## 版本表

| 组件 | 版本 | 不这么配会怎样 |
|---|---|---|
| torch | **2.8.0 + cu128** | 2.9.x 上 transformers 4.51.3 循环导入 |
| flash-attn | **2.8.3** | 见上 |
| deepspeed | **卸掉** | 见上 |
| transformers | 4.51.3（requirements 上限） | — |

## 关键参数（别自己 grep，这张表是核对过的）

| 参数 | 值 | 出处 / 坑 |
|---|---|---|
| `sample_fps` | **25** | `wan_2_2/configs/shared_config.py`。⚠️ `wan_base/configs/` 里是 **16**，**s2v-14B 不走那棵树** |
| `infer_frames` | **48** | 官方两个 `.sh` 都传 48（函数签名默认是 80，别被骗） |
| `num_frames_per_block` | 3 | 一个 block = **12 帧 = 0.48 s** |
| 音频嵌入帧率 | 30 Hz | wav2vec 出 50 Hz，`audio_encoder.py:86` 重采样。**别拿 PCM 采样率换算** |

自检：一个 clip 吃的音频时长应该等于出的视频时长 ——
48×(30/25) = 57.6 帧 ÷ 30 Hz = 1.92 s；48 帧 ÷ 25 fps = 1.92 s。
**两个数走完全不同的公式，对不上就是哪儿拿错了常数。**

## 流式出帧

上游有三个 pipeline，**只有一个会 `yield`，而且没人 import 它**：

| pipeline | `yield` | 谁在用 |
|---|---|---|
| `causal_s2v_pipeline`（单卡） | 0 | 官方入口 |
| `causal_s2v_pipeline_tpp`（5 卡） | 0 | 官方入口 |
| `causal_s2v_pipeline_tpp_blockwise` | **2** | **没有任何地方** |

好消息：blockwise 的**构造函数和 `generate()` 签名跟 `causal_s2v_pipeline_tpp`
逐字一致**，纯 drop-in，换个 import 就能用。

它还需要调用方装一个钩子：

```python
wan_s2v.get_audio_callback = lambda: <下一块 PCM，16 kHz，0.48 s>
```

第 0、1 轮用预置音频预热，**从第 2 轮起每个 block 向这个回调要一次实时音频**
（`_streaming_encode_next_audio_block_or_random`）。
**这就是「流式音频」缺的唯一一环**，上游没有任何入口装它。

⚠️ 流式模式**不会自己停**（infinite inference）——`--num_clip` 限不住，
要在消费侧自己截断。
