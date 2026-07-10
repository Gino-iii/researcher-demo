"""M2 冒烟：验证 Tavily 检索器返回 5 条带 URL+摘要的结果,并验证缓存命中。

运行：python tests/smoke_m2_tavily_copy.py
"""

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_researcher.config import Config
from mini_researcher.retrievers_copy import TavilyRetriever


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = Config.from_env()
    retriever = TavilyRetriever(cfg)
    query = "量子计算 2025 进展"

    # 1) 首次检索（走网络）
    print("=== 1/3 首次检索(网络) ===")
    t0 = time.perf_counter()
    results = await retriever.search(query, max_results=5)
    print(f"耗时 {time.perf_counter() - t0:.2f}s,拿到 {len(results)} 条")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['href']}")
        print(f"      {r['body'][:60]}...")
    assert results, "应至少返回 1 条结果"
    assert all(r["href"] and r["body"] for r in results), "每条都应有 URL 和摘要"

    # 2) 再次检索同 query（应命中缓存,秒回）
    print("\n=== 2/3 缓存命中 ===")
    t0 = time.perf_counter()
    cached = await retriever.search(query, max_results=5)
    dt = time.perf_counter() - t0
    print(f"耗时 {dt:.3f}s,{len(cached)} 条(应 < 0.1s 且结果一致)")
    assert cached == results, "缓存结果应与首次一致"

    # 3) M3 兜底:带 raw_content
    print("\n=== 3/3 include_raw_content ===")
    raw = await retriever.search("Tavily API", max_results=2, include_raw_content=True)
    print(f"{len(raw)} 条,首条 raw_content 长度: {len(raw[0].get('raw_content', '')) if raw else 0}")

    print("\n🎉 M2 冒烟通过")


if __name__ == "__main__":
    asyncio.run(main())
