"""配置层。

对应原项目 ``gpt_researcher/config/``，但大幅简化：
- 原项目支持 ``provider:model`` 语法（"openai:gpt-4o-mini"）在运行时选提供商；
  简版只对接一个 OpenAI 兼容接口，所以模型名就是裸字符串，provider 由 base_url 决定。
- 原项目用 ``BaseConfig`` TypedDict + 反射做 env 覆盖（见 config.py:60 _set_attributes）；
  简版直接用 dataclass + ``from_env()``，代码更透明。

设计要点：
1. FAST/SMART 双模型分工保留 —— 这是原项目控制成本的核心。
2. chat 与 embedding 的 base_url/key 分开配置（见 PLAN 坑 10）——很多 OpenAI 兼容
   提供商（如 DeepSeek）没有 embedding 端点，需要指向另一个服务，或干脆留空降级。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

try:
    from dotenv import load_dotenv
    load_dotenv()  # 导入即加载 .env，缺失也不报错
except ImportError:  # 允许没装 python-dotenv 时仍能靠真实环境变量运行
    pass


def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val in (None, ""):
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val in (None, ""):
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val in (None, ""):
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# 与原项目默认值对齐的 USER_AGENT（default.py:17），伪装成正常浏览器以降低被反爬概率。
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
)


@dataclass
class Config:
    """运行期全部配置。字段默认值对齐原项目 default.py，可被同名环境变量覆盖。"""

    # ---- LLM：chat（OpenAI 兼容） ----
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    fast_llm: str = "gpt-4o-mini"   # 便宜快模型：子查询生成、单页摘要
    smart_llm: str = "gpt-4.1"      # 强模型：写最终报告
    temperature: float = 0.4

    # ---- Embedding：M6 上下文压缩用，与 chat 分开配置 ----
    # 留空 => M6 降级为"截断丐版"，不做相似度过滤。
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    similarity_threshold: float = 0.42  # 余弦相似度阈值，对齐原项目

    # ---- 检索器 ----
    tavily_api_key: str = ""
    max_search_results_per_query: int = 5

    # ---- 研究行为 ----
    max_iterations: int = 3      # 生成的子查询数量（原项目 MAX_ITERATIONS）
    total_words: int = 1200      # 报告目标字数下限
    browse_chunk_max_length: int = 8192  # 单页正文截断上限
    max_scraper_workers: int = 15        # 抓取并发上限
    language: str = "chinese"
    user_agent: str = _DEFAULT_USER_AGENT
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量（含 .env）构造配置。缺失项回落到 dataclass 默认值。"""
        return cls(
            openai_api_key=_env_str("OPENAI_API_KEY", ""),
            openai_base_url=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            fast_llm=_env_str("FAST_LLM", "gpt-4o-mini"),
            smart_llm=_env_str("SMART_LLM", "gpt-4.1"),
            temperature=_env_float("TEMPERATURE", 0.4),
            embedding_base_url=_env_str("EMBEDDING_BASE_URL", ""),
            embedding_api_key=_env_str("EMBEDDING_API_KEY", ""),
            embedding_model=_env_str("EMBEDDING_MODEL", ""),
            similarity_threshold=_env_float("SIMILARITY_THRESHOLD", 0.42),
            tavily_api_key=_env_str("TAVILY_API_KEY", ""),
            max_search_results_per_query=_env_int("MAX_SEARCH_RESULTS_PER_QUERY", 5),
            max_iterations=_env_int("MAX_ITERATIONS", 3),
            total_words=_env_int("TOTAL_WORDS", 1200),
            browse_chunk_max_length=_env_int("BROWSE_CHUNK_MAX_LENGTH", 8192),
            max_scraper_workers=_env_int("MAX_SCRAPER_WORKERS", 15),
            language=_env_str("LANGUAGE", "chinese"),
            user_agent=_env_str("USER_AGENT", _DEFAULT_USER_AGENT),
            verbose=_env_bool("VERBOSE", True),
        )

    def __post_init__(self) -> None:
        # embedding 未单独配置时，回落到 chat 的 base_url/key（同一提供商恰好也支持 embedding 的情况）。
        # 仍需显式配置 embedding_model 才会真正启用相似度压缩，否则 M6 走丐版。
        if not self.embedding_base_url:
            self.embedding_base_url = self.openai_base_url
        if not self.embedding_api_key:
            self.embedding_api_key = self.openai_api_key

    @property
    def embedding_enabled(self) -> bool:
        """是否具备做 embedding 相似度压缩的条件（M6 用来决定走正式版还是丐版）。"""
        return bool(self.embedding_model and self.embedding_api_key)

    def validate(self) -> list[str]:
        """返回缺失的必需配置项列表（空列表表示配置完整）。不抛异常，交给调用方决定。"""
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing

    def __repr__(self) -> str:
        # 打印时对密钥脱敏，避免日志泄露。
        def mask(s: str) -> str:
            if not s:
                return "<未设置>"
            return s[:4] + "…" + s[-2:] if len(s) > 6 else "***"

        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name.endswith(("_api_key",)):
                val = mask(val)
            parts.append(f"{f.name}={val!r}")
        return "Config(\n  " + ",\n  ".join(parts) + "\n)"
