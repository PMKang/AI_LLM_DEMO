"""
LLM 提供商配置模块

使用方法：
在你的代码中修改 LLM_CHOICE 的值来切换不同的 LLM：
  1 = DeepSeek
  2 = OpenAI (GPT)
  3 = 硅基流动 (SiliconFlow)
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================
# 🔧 在这里修改数字来切换 LLM
# ============================================
#   1 = DeepSeek（默认）
#   2 = OpenAI (GPT)
#   3 = 硅基流动 (SiliconFlow)
#   4 = Custom Cloud (Claude/Other)
# ============================================
LLM_CHOICE = 1


# LLM 配置字典
LLM_CONFIGS = {
    1: {
        "name": "DeepSeek",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": "deepseek-chat"
    },
    2: {
        "name": "OpenAI",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": "gpt-5"  # 或 gpt-4o, gpt-3.5-turbo
    },
    3: {
        "name": "硅基流动",
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "model": "deepseek-ai/DeepSeek-V3"  # 硅基流动的模型名
    },
    4: {
        "name": "Custom Cloud (Claude/Other)",
        "api_key": os.getenv("CUSTOM_API_KEY"),
        "base_url": os.getenv("CUSTOM_BASE_URL", "https://api.example.com/v1"),
        "model": os.getenv("CUSTOM_MODEL_NAME", "claude-3-5-sonnet-20240620") # 默认使用 Claude 3.5 Sonnet，可通过 .env 修改
    }
}


def get_llm_config():
    """获取当前选择的 LLM 配置"""
    config = LLM_CONFIGS.get(LLM_CHOICE)
    if not config:
        raise ValueError(f"无效的 LLM_CHOICE: {LLM_CHOICE}，请设置为 1/2/3")
    
    if not config["api_key"]:
        raise ValueError(f"请在 .env 文件中配置 {config['name']} 的 API Key")
    
    print(f"📡 当前使用: {config['name']} ({config['model']})")
    return config


# 导出配置
config = get_llm_config()
API_KEY = config["api_key"]
BASE_URL = config["base_url"]
MODEL_NAME = config["model"]

from openai import OpenAI
def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_model_name():
    return MODEL_NAME
