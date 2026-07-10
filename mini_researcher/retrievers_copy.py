"""检索器层。

对应原项目 gpt_researcher/retrievers/tavily/tavily_search.py，但做两处关键改造：

1. 全程 async —— 原项目用同步 requests（tavily_search.py:89），在 event loop 里会阻塞，
   M5 的 asyncio.gather 并发就名存实亡。这里用 httpx.AsyncClient，和 M3 抓取器技术栈统一，
   顺便少一个 tavily-python 依赖。
2. 加磁盘缓存（PLAN 坑 3）—— Tavily 免费档 1000 次/月，一次研究 ≈ 5-6 次调用，
   开发期反复跑同一 query 会烧光配额。query+参数哈希 → JSON 文件，命中直接读盘。

返回结构与原项目对齐：list[{"href": url, "body": content}]，
这样 M3/M5 拿到的字段名不用改。
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from pathlib import Path

import httpx

from mini_researcher.config import Config

logger = logging.getLogger(__name__)


class BaseRetriever(abc.ABC):
    """检索器基类。

    哪怕简版只有 Tavily 一个实现，这层抽象仍值得保留 —— 它正是原项目能塞进
    18 个搜索引擎的原因（DuckDuckGo/Bing/Searx…）。M9 想加检索器，只要再写一个
    子类实现 search()，上层 researcher 无感。
    """

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_raw_content: bool = False,
    ) -> list[dict]:
        """执行一次检索。

        返回 list[{"href": str, "body": str}]，与原项目对齐。
        include_raw_content=True 时每条额外带 "raw_content"（M3 抓取被反爬/超时
        干掉的 URL 可退回用它顶上，报告不至于缺料）。

        约定：检索失败返回空列表，不抛异常 —— 让研究循环能带着部分结果继续
        （对齐原项目 tavily_search.py:121 的容错语义）。
        """
        raise NotImplementedError


class _DiskCache:
    """query 哈希 → JSON 文件的极简磁盘缓存。

    key 由 query + 影响结果的参数共同哈希，避免"同 query 不同 max_results"串味。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{digest}.json"

    def get(self, key: str) -> list[dict] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"读取缓存失败,忽略: {e}")
            return None

    def set(self, key: str, value: list[dict]) -> None:
        try:
            self._path(key).write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"写入缓存失败,忽略: {e}")


class TavilyRetriever(BaseRetriever):
    """Tavily REST 异步检索器。"""

    BASE_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache_dir: str | Path | None = ".cache/tavily",
    ):
        self.config = config or Config.from_env()
        if not self.config.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY 未配置,请检查 .env")
        # cache_dir=None 可关缓存（如生产环境要实时结果）
        self._cache = _DiskCache(cache_dir) if cache_dir else None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_raw_content: bool = False,
    ) -> list[dict]:
        cache_key = f"{query}|{max_results}|raw={include_raw_content}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(f"[Tavily] 命中缓存: {query!r}")
                return cached

        try:
            raw = await self._request(query, max_results, include_raw_content)
        except httpx.HTTPError as e:
            # 网络/HTTP 错误：记日志返回空,不阻断整轮研究（坑 9 的先声）
            logger.warning(f"[Tavily] 检索失败,返回空: {query!r} -> {e}")
            return []

        sources = raw.get("results", []) or []
        results: list[dict] = []
        for obj in sources:
            item = {"href": obj.get("url", ""), "body": obj.get("content", "")}
            if include_raw_content:
                item["raw_content"] = obj.get("raw_content") or ""
            results.append(item)

        if self._cache is not None and results:
            self._cache.set(cache_key, results)
        logger.info(f"[Tavily] {query!r} -> {len(results)} 条")
        return results

    async def _request(
        self, query: str, max_results: int, include_raw_content: bool
    ) -> dict:
        """向 Tavily REST 发一次异步请求。

        对齐原项目 _search()（tavily_search.py:56）的字段，去掉简版用不到的
        include_images / days 等。api_key 放 body（与原项目一致；放 header 也行）。
        """
        payload = {
            "api_key": self.config.tavily_api_key,
            "query": query,
            "search_depth": "basic",   # basic 一次算 1 credit,advanced 算 2
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": include_raw_content,
            "include_images": False,
        }
        # 10s 硬超时：搜索本该秒回,卡住就该失败,别拖垮 M5 并发
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.BASE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
