"""
本模块提供一系列工具函数，通过统一接口与各种 LLM 提供方进行交互。
"""

import os
from typing import Callable

from openai import AsyncOpenAI

from mini_researcher.config import Config

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
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
