"""LLM 轻量层：封装 OpenAI 兼容接口的 chat / chat_json。

对应原项目 gpt_researcher/utils/llm.py + llm_provider/generic/base.py，
但去掉 langchain，直接用 openai SDK。
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

# ---- 简单的 token 成本追踪（累加每次调用的 usage，跨调用全局累计） ----
_total_prompt_tokens = 0
_total_completion_tokens = 0


def get_token_usage() -> dict[str, int]:
    """返回自进程启动以来累计的 token 用量。"""
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
    }


def _record_usage(usage) -> None:
    global _total_prompt_tokens, _total_completion_tokens
    if usage is None:
        return
    _total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
    _total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0


def _get_client() -> AsyncOpenAI:
    """lazy 单例：首次调用时从 Config 读 key/base_url 构造客户端。

    放在函数里而不是模块顶层，是为了让 import 此模块不触发网络配置，
    且测试时可以用 monkeypatch 替换 _client。
    """
    global _client
    if _client is None:
        cfg = Config.from_env()
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置，请检查 .env")
        _client = AsyncOpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
        )
    return _client


async def _call_with_retry(
    fn: Callable[[], "asyncio.Future"],
    *,
    stream: bool = False,
    max_attempts: int = 3,
):
    """对一次 LLM 调用做指数退避重试。

    stream=True 时强制 max_attempts=1，因为流式已经吐过 token 给用户，
    重试会导致重复输出（对齐原项目 utils/llm.py:106 的语义）。
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
                # 1s, 2s, 4s... 上限 8s，避免退避太久卡死流程
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
            else:
                raise
    raise RuntimeError("unreachable") from last_exc


async def chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.4,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
    *,
    client: AsyncOpenAI | None = None,
) -> str:
    """向 LLM 发一次聊天补全，返回完整字符串。

    stream=True 时，每收到一个 token 片段就调 on_chunk(text)，
    最后仍返回完整字符串（调用方既能实时打印又能拿到全文）。

    client 参数用于测试注入；生产路径走 _get_client() 单例。
    """
    if model is None:
        raise ValueError("model is required")
    cli = client or _get_client()

    if not stream:
        async def _call():
            resp = await cli.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            _record_usage(getattr(resp, "usage", None))
            return resp.choices[0].message.content or ""

        return await _call_with_retry(_call, stream=False)

    response = ""

    async def _stream_call():
        nonlocal response
        response = ""  # 重置，避免重试时残留（虽然流式不重试，防御性写）
        stream_obj = await cli.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream_obj:
            if not chunk.choices:
                # 部分兼容接口在结束时发一个只带 usage、没有 choices 的 chunk
                _record_usage(getattr(chunk, "usage", None))
                continue
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue
            response += delta
            if on_chunk:
                on_chunk(delta)

    await _call_with_retry(_stream_call, stream=True)
    return response


async def chat_json(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = 0.4,
    client: AsyncOpenAI | None = None,
) -> dict:
    """调 LLM 拿 JSON dict。失败抛 ValueError，调用方自己决定兜底。

    不要依赖 response_format=json_object（PLAN 坑 1），
    靠 prompt 约定 + 三层解析兜底（PLAN 坑 2）。
    """
    text = await chat(
        messages, model, temperature=temperature,
        stream=False, client=client,
    )
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    """三层容错解析，对齐原项目 agent_creator.py:71 的设计。"""
    # 第一层：理想情况，直接解析
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 第二层：剥 ```json``` 围栏 + json_repair（容错尾逗号、单引号、缺引号）
    stripped = _strip_json_fences(text)
    try:
        result = json_repair.loads(stripped)
        if isinstance(result, dict):
            return result
    except Exception as e:
        logger.warning(f"json_repair 解析失败: {e}")

    # 第三层：正则提取最外层 {...} + json_repair 兜底
    block = _extract_json_block(text)
    if block:
        try:
            result = json_repair.loads(block)
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"正则提取后仍解析失败: {e}")

    raise ValueError(
        f"无法从 LLM 响应中解析 JSON，前 200 字符: {text[:200]!r}"
    )


def _strip_json_fences(text: str) -> str:
    """剥 ```json ... ``` 或 ``` ... ``` 围栏，没有围栏则原样返回。"""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_json_block(text: str) -> str | None:
    """正则提取最外层 {...}。

    注意：原项目 agent_creator.py:128 用 r"{.*?}" 非贪婪，
    对嵌套 JSON（如 {"a":{"b":1}}）会匹配到 {"a":{"b":1} 截断。
    简版改贪婪 r"\\{.*\\}" 匹配最外层，对嵌套更稳。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None
