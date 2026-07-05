"""M0 冒烟测试：配置层能正确从环境变量加载、脱敏打印、校验缺失项。

运行：python tests/test_m0_config.py
不依赖真实密钥，用临时环境变量验证行为。
"""

import os
import sys

# 让脚本能直接 python tests/xxx.py 运行（把项目根加入 import 路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_researcher.config import Config


def test_defaults():
    """不设任何环境变量时，回落到 dataclass 默认值。"""
    cfg = Config()
    assert cfg.fast_llm == "gpt-4o-mini"
    assert cfg.smart_llm == "gpt-4.1"
    assert cfg.max_iterations == 3
    assert cfg.similarity_threshold == 0.42
    print("✓ 默认值正确")


def test_from_env_override():
    """环境变量能覆盖默认值，且类型转换正确。"""
    os.environ["FAST_LLM"] = "deepseek-chat"
    os.environ["MAX_ITERATIONS"] = "2"
    os.environ["TEMPERATURE"] = "0.7"
    os.environ["VERBOSE"] = "false"
    os.environ["OPENAI_API_KEY"] = "sk-test1234567890"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

    cfg = Config.from_env()
    assert cfg.fast_llm == "deepseek-chat"
    assert cfg.max_iterations == 2 and isinstance(cfg.max_iterations, int)
    assert cfg.temperature == 0.7 and isinstance(cfg.temperature, float)
    assert cfg.verbose is False
    print("✓ 环境变量覆盖 + 类型转换正确")


def test_embedding_fallback():
    """未单独配 embedding 时，base_url/key 回落到 chat；但未配 model 则视为未启用。"""
    os.environ.pop("EMBEDDING_BASE_URL", None)
    os.environ.pop("EMBEDDING_API_KEY", None)
    os.environ.pop("EMBEDDING_MODEL", None)

    cfg = Config.from_env()
    assert cfg.embedding_base_url == cfg.openai_base_url
    assert cfg.embedding_api_key == cfg.openai_api_key
    assert cfg.embedding_enabled is False  # 没有 model => 走丐版
    print("✓ embedding 回落逻辑正确（未配 model => 降级丐版）")

    os.environ["EMBEDDING_MODEL"] = "text-embedding-v3"
    cfg2 = Config.from_env()
    assert cfg2.embedding_enabled is True
    print("✓ 配了 embedding_model 后正式启用")


def test_validate():
    """缺关键密钥时 validate 报出，齐全时返回空。"""
    for k in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        os.environ.pop(k, None)
    missing = Config.from_env().validate()
    assert "OPENAI_API_KEY" in missing and "TAVILY_API_KEY" in missing
    print("✓ validate 正确报出缺失项")

    os.environ["OPENAI_API_KEY"] = "sk-x"
    os.environ["TAVILY_API_KEY"] = "tvly-x"
    assert Config.from_env().validate() == []
    print("✓ 配置齐全时 validate 通过")


def test_repr_masks_secrets():
    """repr 必须对密钥脱敏，避免日志泄露。"""
    os.environ["OPENAI_API_KEY"] = "sk-supersecret-abcdef"
    text = repr(Config.from_env())
    assert "sk-supersecret-abcdef" not in text
    assert "openai_api_key" in text
    print("✓ repr 已对密钥脱敏")


if __name__ == "__main__":
    test_defaults()
    test_from_env_override()
    test_embedding_fallback()
    test_validate()
    test_repr_masks_secrets()
    print("\n🎉 M0 全部冒烟测试通过")

    # 顺便演示实际打印效果（脱敏后）
    print("\n--- Config.from_env() 打印示例 ---")
    print(Config.from_env())
