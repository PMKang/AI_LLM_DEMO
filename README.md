# AI_LLM_DEMO 🤖

## 🤝 关注作者

更多硬核技术文章与 AI 实战思路，欢迎关注公众号：**[产品阿康日记]**

![公众号二维码](qrcode.jpg)

这里汇集了公众号及日常技术分享中涉及的各种 AI & LLM 相关的实战 Demo 代码。

旨在通过最精简的代码，展示大模型开发中的核心概念与工程实践，方便大家“拿来即用”或进行二次开发。

## 📂 目录 (Table of Contents)

### 1. [LLM Prompt Caching (提示词缓存)](./LLM_PROMPT_CACHE_DEMO)

> 对应文章：《DeepSeek 提示词缓存实战：如何通过 Context Caching 节省 90% 成本？》

- **核心演示**: 对比开启缓存前后的 Token 消耗与响应速度。
- **技术要点**:
  - DeepSeek API 的 `prompt_cache_hit_tokens` 实测
  - 静态/动态前缀对缓存命中的影响
  - 成本计算 (1.0元 vs 0.1元)
- **快速开始**: [点击此处查看详细文档](./LLM_PROMPT_CACHE_DEMO/README.md)

### 2. [Dify Essay Master (AI 作文大师)](./dify_Essay_master)

> 对应文章：《手把手教你搭建 Agent：小白也能搭出“会自我进化”的 AI 写作大师》

- **核心演示**: 利用 Dify 工作流构建一个具备“反思-修正”闭环的 AI 写作智能体。
- **技术要点**:
  - **迭代闭环**: 使用反馈环路（Feedback Loop）实现内容的自我进化。
  - **多步编排**: 结合预览、评分与修正节点，模拟真人写作过程。
  - **低代码实现**: 展示如何通过 Dify 可视化界面快速落地复杂的 Agent 逻辑。
- **快速开始**: [直接导入 YAML 文件](./dify_Essay_master/essay_master_v1.yml) 即可运行。

---

## 🚀 如何使用

每个子文件夹都是一个独立的项目，包含各自的 `requirements.txt` 和说明文档。请进入子目录查看具体的运行指南。
