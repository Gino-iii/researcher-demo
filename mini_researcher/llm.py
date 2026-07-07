"""LLM 轻量层：封装 OpenAI 兼容接口的 chat / chat_json。

对应原项目 gpt_researcher/utils/llm.py + llm_provider/generic/base.py，
但去掉 langchain，直接用 openai SDK。
"""

import os
from typing import Callable

from openai import AsyncOpenAI

from mini_researcher.config import Config

_client: AsyncOpenAI | None = None

# ---- 简单的 token 成本追踪（累加每次调用的 usage，跨调用全局累计） ----
_total_prompt_tokens = 0 # 累计的 prompt 用量
_total_completion_tokens = 0 # 累计的 completion 用量

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
            raise RuntimeError("OPENAI_API_KEY 未配置,请检查 .env")
        _client = AsyncOpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
        )
    return _client


async def _call_with_retry(
    func: Callable,
    *,
    stream: bool = False,
    max_retries: int = 3,
):

    if stream:
        max_attempts = 1
    last


# async def chat(
#     messages: list[dict[str, str]],
#     model: str,
#     temperature: float = 0.5,
#     stream: bool = False,
#     on_chunk: Callable[[str], None] = None,
# ) -> str:

#     if model is None:
#         raise ValueError("model is required")
