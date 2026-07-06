# M1:LLM 轻量层 —— 实现指导

> 目标:实现 `mini_researcher/llm.py`,提供 `chat()` 和 `chat_json()` 两个函数,作为后续所有里程碑(M2-M9)的 LLM 调用统一入口。
>
> 参考源码:`gpt-researcher/gpt_researcher/utils/llm.py`、`gpt_researcher/llm_provider/generic/base.py`、`gpt_researcher/gpt_researcher/actions/agent_creator.py:71-131`。
>
> 预计工时:半天。

---

## 一、前置确认(动手前花 5 分钟检查)

| 检查项 | 命令 | 期望结果 |
|---|---|---|
| M0 配置层就位 | `python -c "from mini_researcher import Config; print(Config.from_env())"` | 打印 `Config(...)`,key 脱敏 |
| openai SDK 装好 | `python -c "import openai; print(openai.__version__)"` | ≥ 1.30 |
| json-repair 装好 | `python -c "import json_repair; print(json_repair.__version__)"` | ≥ 0.25 |
| `.env` 已填 key | `cat .env \| grep OPENAI_API_KEY` | 非 `sk-xxxx` 占位 |
| 能调通兼容接口 | `curl $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY"` | 返回 model 列表 |

任意一项失败先解决再继续,否则写到一半发现配额/网络问题难定位。

---

## 二、设计要点(先想清楚为什么,再动手)

### 2.1 为什么不用 langchain

原项目 `gpt_researcher/utils/llm.py` 全程基于 `langchain_openai.ChatOpenAI` + `astream`,简版**有意绕开**:

- langchain 抽象层会吞掉底层 SDK 的细节(比如 `usage_metadata`、错误类型),调试困难
- 多一层依赖、多一层版本耦合,简版目标是「无 LangChain」
- 我们要的功能(发消息、流式、JSON 解析)`openai` SDK 直接给得了

**替代方案**:直接用 `openai.AsyncOpenAI` 异步客户端,base_url 指向 DeepSeek / Qwen / 任意兼容端点。

### 2.2 客户端用 lazy 单例,不要每次调用都 new

`AsyncOpenAI` 内部维护 httpx 连接池,每次 `chat()` 都 new 一个客户端 = 每次都重建连接池 = 慢且浪费端口。

但**也不要在模块顶层直接实例化**——import 时就连接,测试时想替换客户端很难。

**推荐做法**:模块级 `_client: AsyncOpenAI | None = None`,首次调用时从 `Config.from_env()` 读 key/base_url 构造。`chat()` 额外接受可选 `client` 参数,测试时注入 mock。

### 2.3 不要用 `response_format={"type": "json_object"}`

PLAN 坑 1:OpenAI 兼容接口兼容度参差,DeepSeek、部分本地 vLLM 不支持这个参数,直接 422 报错。

**替代方案**:`chat_json` 完全靠 prompt 约定("请返回合法 JSON")+ 三层解析兜底。这样在任何兼容接口上都能跑。

### 2.4 流式不重试,非流式重试

原项目 `utils/llm.py:106` 有个细节:

```python
max_attempts = 1 if (stream and websocket is not None) else 10
```

含义:流式输出一旦开始,已经吐了几个 token 给用户,这时候失败重试会导致**重复输出**(用户先看到"你好我是",然后报错,然后又看到"你好我是...")。所以流式直接抛,不重试。

非流式可以重试,因为用户还没看到任何输出。简版保留这个语义:**非流式 3 次指数退避,流式 1 次到底**。

---

## 三、分步实现

### Step 1:导入 + 客户端单例

**做什么**:在 `mini_researcher/llm.py` 顶部加导入,写 `_get_client()` 函数。

**关键骨架**:

```python
"""LLM 轻量层:封装 OpenAI 兼容接口的 chat / chat_json。

对应原项目 gpt_researcher/utils/llm.py + llm_provider/generic/base.py,
但去掉 langchain,直接用 openai SDK。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Callable

import json_repair
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from .config import Config

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """lazy 单例:首次调用时从 Config 读 key/base_url 构造客户端。

    放在函数里而不是模块顶层,是为了让 import 此模块不触发网络配置,
    且测试时可以用 monkeypatch 替换 _client。
    """
    global _client
    if _client is None:
        cfg = Config.from_env()
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置,请检查 .env")
        _client = AsyncOpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
        )
    return _client
```

**坑**:
- 不要 `from openai import OpenAI`,要用 `AsyncOpenAI`(全链路 async 是 PLAN 通用注意事项第 1 条)。
- `openai` SDK 的异常类路径:`openai.APIError`(基类)、`openai.APITimeoutError`、`openai.RateLimitError`、`openai.APIConnectionError`。导入前确认你的 openai 版本(1.30+ 这些都在 `openai` 包根)。

---

### Step 2:重试包装器

**做什么**:写一个内部用的 `_call_with_retry(fn, stream=False, max_attempts=3)`,封装指数退避。

**关键骨架**:

```python
async def _call_with_retry(
    fn: Callable[[], "Awaitable"],
    *,
    stream: bool = False,
    max_attempts: int = 3,
):
    """对一次 LLM 调用做指数退避重试。

    stream=True 时强制 max_attempts=1,因为流式已经吐过 token,
    重试会让用户看到重复输出(对齐原项目 utils/llm.py:106 的语义)。
    """
    if stream:
        max_attempts = 1
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except (APITimeoutError, RateLimitError, APIError) as e:
            last_exc = e
            logger.warning(f"LLM 调用失败 (第 {attempt}/{max_attempts} 次): {e}")
            if attempt < max_attempts:
                # 1s, 2s, 4s... 上限 8s,避免退避太久卡死流程
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
            else:
                raise
    # 理论上不会走到这,for 循环里要么 return 要么 raise
    raise RuntimeError("unreachable") from last_exc
```

**坑**:
- 别捕获 `Exception` 全吞,只捕获网络/ API 类异常。`ValueError`、`KeyError` 这种是代码 bug,该让它炸。
- `RateLimitError` 退避可以更长一点(配额限制通常 60s 才解),但简版统一处理够用。

---

### Step 3:实现 `chat()` 非流式分支

**做什么**:写 `chat()` 主签名 + 非流式分支。

**关键骨架**:

```python
async def chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.4,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
    *,
    client: AsyncOpenAI | None = None,
) -> str:
    """向 LLM 发一次聊天补全,返回完整字符串。

    stream=True 时,每收到一个 token 片段就调 on_chunk(text),
    最后仍返回完整字符串(调用方既能实时打印又能拿到全文)。

    client 参数用于测试注入;生产路径走 _get_client() 单例。
    """
    if model is None:
        raise ValueError("model is required")
    cli = client or _get_client()

    if not stream:
        # 非流式:一次性拿完整响应,可重试
        async def _call():
            resp = await cli.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            return resp.choices[0].message.content or ""

        return await _call_with_retry(_call, stream=False)

    # 流式分支见 Step 4
    ...
```

**坑**:
- `resp.choices[0].message.content` 理论上可能为 `None`(某些提供商 content 过滤时返回空),用 `or ""` 兜底。
- 不要传 `max_tokens`,让模型用默认上限。原项目传 `max_tokens=4000` 是配合 langchain 的限制,简版没必要。

---

### Step 4:实现 `chat()` 流式分支

**做什么**:补全 `chat()` 的 `stream=True` 分支。

**关键骨架**:

```python
    # 接 Step 3 的 else 分支
    response = ""

    async def _stream_call():
        nonlocal response
        response = ""  # 重置,避免重试时残留(虽然流式不重试,防御性写)
        stream_obj = await cli.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream_obj:
            # OpenAI 兼容接口第一个 chunk 常只带 role,content 为 None
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue
            response += delta
            if on_chunk:
                on_chunk(delta)

    await _call_with_retry(_stream_call, stream=True)
    return response
```

**坑(PLAN 坑 1 的核心)**:
- `chunk.choices[0].delta.content` **可能是 None**,必须 `if delta is None: continue`。新手常犯的错是直接 `response += delta`,然后 TypeError。
- 有些提供商的 chunk 里 `choices` 数组可能空(结束信号 chunk),最好也防御一下:`if not chunk.choices: continue`。
- `on_chunk` 是同步回调,不要 `await` 它。M8 接 WebSocket 时会在 `on_chunk` 内部 `await ws.send`,那时候要把 `on_chunk` 改成 async 或者用 `asyncio.create_task`。**M1 先保持同步**,M8 再回头改。

---

### Step 5:实现 `chat_json()` 三层解析

这是 M1 最容易出 bug 的地方,务必看懂「为什么要三层」。

**做什么**:写 `chat_json()` 和两个辅助函数 `_strip_json_fences()` / `_extract_json_block()`。

**为什么需要三层**(对应 PLAN 坑 2):

LLM 返回 JSON 时常见的脏数据形态:

| 形态 | 例子 | 哪层处理 |
|---|---|---|
| 干净 JSON | `{"server":"X","agent_role_prompt":"Y"}` | 第一层 `json.loads` |
| 带 ```json``` 围栏 | ```` ```json\n{"server":...}\n``` ```` | 第二层剥围栏 |
| 尾逗号 / 单引号 | `{"server":"X",}` 或 `{'server':'X'}` | 第二层 `json_repair` |
| JSON 前后有解释文字 | `好的,这是结果:{"server":"X"} 希望有帮助` | 第三层正则提取 |

**关键骨架**:

```python
async def chat_json(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = 0.4,
    client: AsyncOpenAI | None = None,
) -> dict:
    """调 LLM 拿 JSON dict。失败抛 ValueError,调用方自己决定兜底。

    不要依赖 response_format=json_object(PLAN 坑 1),
    靠 prompt 约定 + 三层解析兜底(PLAN 坑 2)。
    """
    text = await chat(
        messages, model, temperature=temperature,
        stream=False, client=client,
    )
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    """三层容错解析,对齐原项目 agent_creator.py:71 的设计。"""
    # 第一层:理想情况,直接解析
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 第二层:剥 ```json``` 围栏 + json_repair(容错尾逗号、单引号、缺引号)
    stripped = _strip_json_fences(text)
    try:
        result = json_repair.loads(stripped)
        if isinstance(result, dict):
            return result
    except Exception as e:
        logger.warning(f"json_repair 解析失败: {e}")

    # 第三层:正则提取最外层 {...} + json_repair 兜底
    block = _extract_json_block(text)
    if block:
        try:
            result = json_repair.loads(block)
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"正则提取后仍解析失败: {e}")

    raise ValueError(
        f"无法从 LLM 响应中解析 JSON,前 200 字符: {text[:200]!r}"
    )


def _strip_json_fences(text: str) -> str:
    """剥 ```json ... ``` 或 ``` ... ``` 围栏,没有围栏则原样返回。"""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_json_block(text: str) -> str | None:
    """正则提取最外层 {...}。

    注意:原项目 agent_creator.py:128 用 r"{.*?}" 非贪婪,
    对嵌套 JSON(如 {"a":{"b":1}})会匹配到 {"a":{"b":1} 截断。
    简版改贪婪 r"\{.*\}" 匹配最外层,对嵌套更稳。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None
```

**坑**:
- `json_repair.loads` 对非字符串输入会炸,对空字符串也会,要 try/except 包住。
- `json_repair.loads("好的,这是结果:{...}")` 会**尝试**从文本里提取,但不一定成功,所以仍需要第三层正则。
- `isinstance(result, dict)` 检查很关键:LLM 偶尔返回 `[...]` 列表,`json.loads` 不报错但调用方拿 `result["key"]` 会 TypeError。
- 全失败时**抛 ValueError**,不要返回 `{}` 或 `None`。原项目 `handle_json_error` 返回默认 agent 是业务层兜底,通用层不该替业务做决策。

**为什么不在这里加"解析失败 → 重试调 LLM"**:
M1 保持简单。如果实际使用发现成功率不够(比如某些模型经常返回带解释文字的 JSON),M5 之后再回来加:把"你上一次返回的不是合法 JSON,请只返回 JSON"塞回 messages 再调一次。这属于 prompt 工程范畴,放业务层更合适。

---

### Step 6:导出 + 冒烟测试

**做什么 1**:在 `mini_researcher/__init__.py` 追加导出。

```python
from .config import Config
from .llm import chat, chat_json

__all__ = ["Config", "chat", "chat_json"]
__version__ = "0.1.0"
```

**做什么 2**:新建 `tests/smoke_llm.py`,写 4 个冒烟测试。

测试需要真实 API key(消耗少量配额),用 `pytest.mark.skipif` 在无 key 环境自动跳过。

**关键骨架**:

```python
"""M1 LLM 层冒烟测试。需要真实 OPENAI_API_KEY,无 key 时自动跳过。"""
import os
import asyncio
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="需要 OPENAI_API_KEY 才能跑 LLM 冒烟测试",
)

from mini_researcher import Config, chat, chat_json

@pytest.fixture
def cfg():
    return Config.from_env()

@pytest.fixture
def model(cfg):
    return cfg.fast_llm


def test_chat_nonstream(cfg, model):
    """非流式 chat 应返回非空字符串。"""
    result = asyncio.run(chat(
        [{"role": "user", "content": "用一句话介绍量子计算。"}],
        model=model,
    ))
    assert isinstance(result, str)
    assert len(result) > 10


def test_chat_stream(model):
    """流式 chat:on_chunk 被多次调用,拼接后等于返回值。"""
    chunks = []
    result = asyncio.run(chat(
        [{"role": "user", "content": "数到 5,用空格分隔。"}],
        model=model,
        stream=True,
        on_chunk=chunks.append,
    ))
    assert len(chunks) > 1, "应该收到多个 chunk"
    assert "".join(chunks) == result, "on_chunk 拼接应等于返回值"


def test_chat_json(model):
    """chat_json 应返回 dict。"""
    result = asyncio.run(chat_json(
        [
            {"role": "system", "content": "你是一个 JSON 生成器,只返回合法 JSON。"},
            {"role": "user", "content": '返回 {"status":"ok","count":3}。'},
        ],
        model=model,
    ))
    assert isinstance(result, dict)
    assert result.get("status") == "ok"


def test_chat_json_with_fences(model):
    """chat_json 应能处理带 ```json``` 围栏的返回(三层解析第二层)。"""
    result = asyncio.run(chat_json(
        [
            {"role": "system", "content": "你是 JSON 生成器。"},
            {"role": "user", "content": '请用 ```json``` 围栏包裹返回 {"status":"ok"}。'},
        ],
        model=model,
    ))
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
```

**坑**:
- 测试用 `fast_llm`(便宜模型)省钱,不要用 `smart_llm`。
- `test_chat_stream` 的 prompt 用"数到 5"这种短任务,确保流式 chunk 数 > 1(长输出才能拆 chunk)。
- 跑测试前先 `pip install pytest` 如果没装。

---

## 四、验收脚本

按顺序执行,全过即 M1 完成:

```bash
# 1. 模块能 import,不报错
python -c "from mini_researcher import chat, chat_json; print('import ok')"

# 2. 非流式 chat 返回中文
python -c "
import asyncio
from mini_researcher import chat, Config
cfg = Config.from_env()
print(asyncio.run(chat([{'role':'user','content':'你好'}], model=cfg.fast_llm)))
"

# 3. 流式 chat 逐 token 打印
python -c "
import asyncio
from mini_researcher import chat, Config
cfg = Config.from_env()
asyncio.run(chat(
    [{'role':'user','content':'写一首关于秋天的四行诗'}],
    model=cfg.smart_llm, stream=True,
    on_chunk=lambda t: print(t, end='', flush=True),
))
print()
"

# 4. chat_json 拿到 dict
python -c "
import asyncio
from mini_researcher import chat_json, Config
cfg = Config.from_env()
result = asyncio.run(chat_json(
    [{'role':'user','content':'返回一个 JSON,字段 name 填你的名字,字段 ok 填 true'}],
    model=cfg.fast_llm,
))
print(type(result), result)
assert isinstance(result, dict)
"

# 5. 跑冒烟测试
pytest tests/smoke_llm.py -v
```

**验收标准**(对齐 PLAN):
- `chat_json` 稳定拿到 dict
- 流式模式逐 token 打印(`on_chunk` 被多次调用)
- `pytest tests/smoke_llm.py` 全绿

---

## 五、常见问题排查

| 症状 | 可能原因 | 排查方式 |
|---|---|---|
| `AuthenticationError` | `.env` 里 key 没填或填错 | `echo $OPENAI_API_KEY` 看是否生效 |
| `APIConnectionError` | base_url 错 / 网络不通 | `curl $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY"` |
| `NotFoundError` | model 名拼错 | 用 `curl` 列出 `/models` 看支持哪些 |
| 流式只收到 1 个 chunk | prompt 太短,模型一次吐完 | 换长一点的 prompt 验证 |
| `chat_json` 经常抛 ValueError | 模型不听话,返回带大量解释文字 | 检查 system prompt 是否强调"只返回 JSON";考虑 M5 后加"重试调 LLM" |
| `TypeError: can't concat NoneType` | 流式没判 `delta is None` | 回看 Step 4 的坑 |
| `json_repair` 把列表当 dict 返回 | 没加 `isinstance(result, dict)` 检查 | 回看 Step 5 的坑 |

---

## 六、完成后下一步

M1 完成后,后续里程碑如何使用这一层:

- **M4 Prompts**:不需要直接调 `chat`,只生成 prompt 字符串
- **M5 研究循环**:`chat_json` 用来生成子查询(prompt 让 LLM 返回 `{"queries":[...]}`)
- **M7 报告生成**:`chat(stream=True, on_chunk=print)` 流式写报告
- **M8 WebSocket**:`on_chunk` 改成 `lambda t: asyncio.create_task(ws.send_json({"type":"report","output":t}))`

**M1 的 `on_chunk` 同步设计会在 M8 被打破**,届时回来重构。M1 先保持同步,简单优先。

---

## 七、交付清单(自检)

- [ ] `mini_researcher/llm.py`(约 120 行,含 `chat` / `chat_json` / `_get_client` / `_call_with_retry` / `_parse_json` / `_strip_json_fences` / `_extract_json_block`)
- [ ] `mini_researcher/__init__.py` 导出 `chat`、`chat_json`
- [ ] `tests/smoke_llm.py`(4 个测试,无 key 自动跳过)
- [ ] 5 条验收脚本全过
- [ ] `pytest tests/smoke_llm.py -v` 全绿

全部勾选即 M1 完成,可以进 M2(Tavily 检索器)。
