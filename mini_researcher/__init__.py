"""Mini Researcher —— 学习性复刻 gpt-researcher 的简版实现。

主链路：query → 生成子查询 → 并发搜索/抓取 → 上下文压缩 → 流式写报告。
详见项目根目录 PLAN.html。
"""

from .config import Config
from .llm import chat, chat_json, get_token_usage

__all__ = ["Config", "chat", "chat_json", "get_token_usage"]
__version__ = "0.1.0"
