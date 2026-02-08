# AI_LLM_DEMO (LLM Prompt Caching Demo)

这是一个用于演示和验证 LLM (Large Language Model) Prompt Caching (提示词缓存) 机制的 Demo 项目。通过对比实验，直观展示缓存技术对成本降低（90%）和首字延迟（TTFT）优化的效果。

## 📂 项目结构

- `deepseek_cache.py`: 主演示脚本，包含“有缓存”和“无缓存”的对比实验逻辑。
- `llm_config.py`: LLM 配置文件，支持切换 DeepSeek, OpenAI, 硅基流动等提供商。
- `.env.example`: 配置文件模板，**请复制并重命名为 `.env`**。
- `requirements.txt`: 依赖库列表。

## 🚀 快速开始

### 1. 克隆/下载代码

将 `LLM_PROMPT_CACHE_demo` 文件夹下载到本地。

### 2. 安装依赖

确保已安装 Python 环境，然后运行：

```bash
pip install langchain langchain-openai python-dotenv
```

### 3. 配置环境 (关键步骤 🔑)

项目中提供了一个 `.env.example` 模板文件。你需要将其复制一份并重命名为 `.env`，然后填入你的 API Key。

**Mac/Linux:**

```bash
cp .env.example .env
nano .env  # 或者用你喜欢的编辑器打开 .env
```

**Windows:**

1. 复制 `.env.example` 文件。
2. 将副本重命名为 `.env`。
3. 用记事本打开 `.env`，填入你的 `DEEPSEEK_API_KEY`。

**`.env` 文件内容示例：**

```ini
# 必填：DeepSeek 配置
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 选填：其他模型配置...
```

> ⚠️ **注意**：`.env` 文件包含敏感密钥，**切勿**上传到 GitHub 或公开分享！

### 4. 运行演示

```bash
python deepseek_cache.py
```

## 📊 预期结果

程序将运行两组实验：

1. **利用缓存**：使用固定前缀，你将看到 `prompt_cache_hit_tokens` 显示通过，且计算成本大幅降低 90%。
2. **无法利用缓存**：使用动态时间戳前缀，强制每次重新计算，展示无缓存的基准数据。

最终会输出一份详细的对比表格。

## 🛠️ 切换模型 (实验性)

本 Demo **主要针对 DeepSeek API 进行开发和测试**。
虽然代码中预留了 OpenAI 和 SiliconFlow 的配置入口，但由于 LangChain 对不同厂商缓存机制的支持程度不同（部分厂商可能需要特定的 Header 或参数），**目前仅保证 DeepSeek 能跑通并看到缓存效果**。

如果你想自行测试其他模型，可以尝试修改 `llm_config.py`，但可能需要根据官方文档调整 `langchain` 的调用方式。

```python
# 1 = DeepSeek (✅ 已验证)
# 2 = OpenAI (⚠️ 未验证/实验性)
# 3 = SiliconFlow (⚠️ 未验证/实验性)
LLM_CHOICE = 1
```

---

**Enjoy Caching! 🚀**
