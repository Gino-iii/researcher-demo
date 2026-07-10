"""M1 冒烟：逐一验证 chat / 流式 / chat_json 三条路径。

运行：python tests/smoke_m1_llm.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_researcher.config import Config
from mini_researcher import llm


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = Config.from_env()
    model = cfg.fast_llm
    print(f"模型: {model}")
    print(f"base_url: {cfg.openai_base_url}\n")

    # 1) 非流式 chat
    print("=== 1/3 非流式 chat ===")
    text = await llm.chat(
        [{"role": "user", "content": "只回两个字:就绪"}],
        model=model,
        temperature=0,
    )
    print(f"-> {text!r}\n")

    # 2) 流式 chat
    print("=== 2/3 流式 chat ===")
    chunks: list[str] = []
    text2 = await llm.chat(
        [{"role": "user", "content": "从 1 数到 5,只输出数字和空格"}],
        model=model,
        temperature=0,
        stream=True,
        on_chunk=lambda s: chunks.append(s),
    )
    print(f"片段数: {len(chunks)}")
    print(f"片段: {chunks!r}")
    print(f"全文: {text2!r}\n")

    # 3) chat_json
    print("=== 3/3 chat_json ===")
    obj = await llm.chat_json(
        [
            {
                "role": "user",
                "content": '返回 JSON:{"ok": true, "n": 7},不要任何多余文字',
            }
        ],
        model=model,
        temperature=0,
    )
    print(f"-> {obj}  类型={type(obj).__name__}\n")

    # 4) token 用量
    print(f"=== token 用量 ===\n{llm.get_token_usage()}")
    print("\n🎉 M1 三条路径冒烟通过")


if __name__ == "__main__":
    asyncio.run(main())
