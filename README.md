# Mini Researcher

学习性复刻 [gpt-researcher](../gpt-researcher) 的简版实现。目标是通过亲手实现主链路来读懂原项目，不追求功能完备。

**技术选型**：OpenAI 兼容接口（DeepSeek/Qwen/…）· Tavily 搜索 · 手写轻量 LLM 层（无 LangChain）· 先 CLI 后 Web。

完整实施计划见 [PLAN.html](PLAN.html)。

## 进度

- [x] **M0** 脚手架 + 配置层
- [ ] M1 LLM 轻量层
- [ ] M2 Tavily 检索器
- [ ] M3 抓取器
- [ ] M4 Prompts
- [ ] M5 研究循环（核心）
- [ ] M6 上下文压缩
- [ ] M7 报告生成 + CLI（首个可用版）
- [ ] M8 FastAPI + WebSocket + 前端

## 快速开始

```bash
# 1. 建虚拟环境并安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. 配置密钥
cp .env.example .env    # 然后填入 OPENAI_API_KEY / OPENAI_BASE_URL / TAVILY_API_KEY

# 3. 冒烟测试当前里程碑
python tests/test_m0_config.py
```

## 目录结构

```
mini_researcher/
├── config.py      # M0 配置层（已完成）
├── llm.py         # M1 chat / embedding 轻量封装
├── retrievers.py  # M2 搜索
├── scraper.py     # M3 抓取
├── prompts.py     # M4 提示词
├── context.py     # M6 上下文压缩
├── researcher.py  # M5 研究循环
├── writer.py      # M7 报告生成
└── agent.py       # 编排层
cli.py             # 命令行入口
server/            # M8 Web 服务
tests/             # 每个里程碑一个冒烟脚本
```
