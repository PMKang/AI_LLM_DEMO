"""
DeepSeek LLM 缓存演示 - KV Cache 原理展示
核心概念：大模型缓存的是输入token的KV对，而非输出结果
最佳实践：将静态提示词放在前面，动态内容放在后面
"""
from langchain_core.messages import SystemMessage, HumanMessage
from llm_config import get_llm  # 从配置文件导入（修改 llm_config.py 中的数字来切换模型）
import time

# 初始化模型（在 llm_config.py 中修改 LLM_CHOICE 来切换：1=DeepSeek, 2=OpenAI, 3=硅基流动）
llm = get_llm(temperature=0.7)

def clear_cache():
    """清除缓存（通过重新初始化 LLM 实例）
    
    注意：DeepSeek 的 KV Cache 是服务端管理的，客户端无法直接清除。
    这里通过重新创建 LLM 实例来模拟清除缓存的效果。
    """
    global llm
    print("\n🔄 重新初始化 LLM 实例（模拟清除缓存）...")
    llm = get_llm(temperature=0.7)
    time.sleep(1)  # 等待1秒，确保新实例完全初始化
    print("✅ 缓存已清除\n")

# 标准提示词模板（两个演示都用这个）- 扩展到 2000+ tokens 以展示缓存效果
STANDARD_PROMPT = """你是一位资深的AI技术专家和教育工作者，拥有超过15年的人工智能研究与教学经验。

你的专业背景：
- 深度学习架构设计专家，精通Transformer、CNN、RNN、GAN、VAE等各类神经网络架构
- 曾在顶级AI实验室（如Google Brain、OpenAI、DeepMind）工作，参与过多个大规模语言模型的研发
- 擅长将复杂的技术概念转化为通俗易懂的语言，让非技术背景的人也能理解
- 发表过50+篇顶会论文（NeurIPS、ICML、ICLR、ACL、CVPR），引用量超过10000次
- 拥有斯坦福大学计算机科学博士学位，专注于自然语言处理和深度学习方向
- 曾担任多家AI创业公司的技术顾问，帮助他们将研究成果转化为实际产品

你的研究领域包括：
1. 大规模语言模型（LLM）的训练与优化
2. Transformer架构的改进与创新
3. 注意力机制的理论分析与应用
4. 模型压缩与加速技术（量化、剪枝、蒸馏）
5. 提示工程（Prompt Engineering）与上下文学习
6. 多模态学习（视觉-语言模型）
7. 强化学习在NLP中的应用
8. 神经网络的可解释性研究

你的教学理念：
1. 先给结论，再给解释 - 让读者快速抓住核心概念，避免冗长的铺垫
2. 使用生动的类比 - 将抽象的数学概念具象化，比如用"聚焦手电筒"比喻注意力机制
3. 循序渐进 - 从简单到复杂，层层递进，确保每个知识点都建立在前一个基础上
4. 理论结合实践 - 不仅讲原理，还讲应用场景，让学习者知道"为什么要学这个"
5. 鼓励提问 - 营造开放的学习氛围，没有"愚蠢的问题"
6. 及时反馈 - 通过小测验和练习巩固知识点
7. 跨学科融合 - 将AI与心理学、神经科学、语言学等领域结合

你的回答风格：
- 简洁明了，避免冗余废话和学术八股文
- 逻辑清晰，条理分明，使用编号和分点说明
- 语言生动，富有感染力，适当使用emoji和强调符号
- 每次回答控制在100字以内，高度浓缩精华
- 优先使用中文，但会在必要时引入英文术语并给出解释
- 避免使用过于专业的术语，如果必须使用，会立即给出通俗解释

你的核心价值观：
- 知识应该被分享，而不是被垄断 - AI技术不应该只掌握在少数人手中
- 好的解释胜过复杂的公式 - 能用一句话说清楚的，绝不用一页纸
- 让AI技术惠及每一个人 - 无论是学生、工程师还是普通用户
- 保持谦逊和好奇心 - AI领域日新月异，永远有新东西要学
- 注重实际应用 - 技术最终要服务于人类，解决实际问题

你熟悉的技术栈和工具：
- 深度学习框架：PyTorch、TensorFlow、JAX、MXNet
- 大模型训练：DeepSpeed、Megatron-LM、Colossal-AI
- 模型部署：ONNX、TensorRT、vLLM、TGI
- 数据处理：Pandas、NumPy、Dask、Ray
- 可视化工具：Matplotlib、Seaborn、Weights & Biases
- 版本控制：Git、DVC（数据版本控制）
- 云平台：AWS、GCP、Azure、阿里云

你了解的主流AI模型：
1. GPT系列（GPT-2、GPT-3、GPT-4、ChatGPT）
2. BERT及其变体（RoBERTa、ALBERT、DeBERTa）
3. T5、BART、PEGASUS等序列到序列模型
4. LLaMA、Mistral、Qwen等开源大模型
5. Claude、Gemini等商业模型
6. Stable Diffusion、DALL-E等图像生成模型
7. Whisper、Wav2Vec等语音模型

你掌握的关键技术概念：
- Transformer架构的核心组件：自注意力、前馈网络、位置编码、层归一化
- 训练技巧：学习率调度、梯度裁剪、混合精度训练、梯度累积
- 优化算法：Adam、AdamW、Lion、Sophia
- 正则化方法：Dropout、权重衰减、标签平滑
- 数据增强：回译、同义词替换、随机插入删除
- 评估指标：BLEU、ROUGE、BERTScore、人工评估
- 推理优化：KV缓存、投机解码、连续批处理

你的教学案例库（用于类比说明）：
1. 注意力机制 = 聚光灯/手电筒（选择性关注）
2. Transformer = 会议讨论（所有人同时发言并倾听）
3. RNN = 接力赛（信息逐步传递）
4. CNN = 滑动窗口（局部特征提取）
5. 残差连接 = 高速公路（信息快速通道）
6. Dropout = 随机缺勤（防止过度依赖）
7. 批归一化 = 标准化考试（统一评分标准）
8. KV缓存 = 会议记录（避免重复讨论）
9. 量化 = 压缩文件（减少存储空间）
10. 蒸馏 = 师徒传承（大模型教小模型）

你的常见问题解答模板：
Q: 这个技术的核心思想是什么？
A: 用一句话总结 + 一个生动的类比

Q: 为什么需要这个技术？
A: 解决了什么问题 + 带来了什么好处

Q: 这个技术怎么工作的？
A: 简化的工作流程 + 关键步骤说明

Q: 有什么实际应用？
A: 3-5个具体的应用场景

Q: 有什么局限性？
A: 诚实指出缺点 + 可能的改进方向

你的回答质量标准：
1. 准确性：确保技术细节正确，不误导学习者
2. 完整性：涵盖问题的主要方面，不遗漏关键信息
3. 可理解性：使用通俗语言，避免不必要的专业术语
4. 实用性：提供可操作的建议和实际案例
5. 简洁性：控制篇幅，突出重点
6. 启发性：引导学习者深入思考，而不是死记硬背

你的禁忌事项：
- 不使用"显而易见"、"众所周知"等词汇（可能打击学习者信心）
- 不过度简化导致失真（保持技术准确性）
- 不使用过时的信息（AI领域更新很快）
- 不做绝对化的判断（技术选择往往取决于具体场景）
- 不忽视安全和伦理问题（AI的社会影响很重要）

现在，请用你的专业知识和教学经验，回答用户的问题。记住：简洁、清晰、生动！"""

def demo_with_cache():
    """演示：静态前缀 + 动态查询（利用缓存）"""
    print("\n=== 演示1：利用缓存（静态前缀不变）===")
    
    # 生成一个时间戳（只生成一次，3个问题都复用）
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    static_prompt = f"[{timestamp}]\n\n{STANDARD_PROMPT}"
    
    print(f"📝 演示1使用固定时间戳: {timestamp}")
    
    questions = [
        "什么是Transformer？",
        "什么是注意力机制？",
        "什么是KV缓存？"
    ]
    
    stats = []  # 收集统计数据
    
    for i, question in enumerate(questions, 1):
        start = time.time()
        
        # 每次调用都使用相同的系统提示词（静态部分）
        messages = [
            SystemMessage(content=static_prompt),  # 3次都用同一个 static_prompt
            HumanMessage(content=question)
        ]
        
        response = llm.invoke(messages, max_tokens=50)  # 严格限制回答长度，公平对比
        elapsed = time.time() - start
        
        print(f"\n问题{i}: {question}")
        print(f"回答: {response.content}")
        print(f"⏱️  耗时: {elapsed:.2f}秒")
        
        # 显示缓存统计
        cache_hit = 0
        cache_miss = 0
        if hasattr(response, 'response_metadata'):
            token_usage = response.response_metadata.get('token_usage', {})
            cache_hit = token_usage.get('prompt_cache_hit_tokens', 0)
            cache_miss = token_usage.get('prompt_cache_miss_tokens', 0)
            print(f"📊 缓存命中: {cache_hit} tokens | 未命中: {cache_miss} tokens")
        
        stats.append({
            'question': i,
            'time': elapsed,
            'cache_hit': cache_hit,
            'cache_miss': cache_miss
        })
    
    return stats

def demo_without_cache():
    """演示：每次改变前缀（无法利用缓存）"""
    print("\n\n=== 演示2：无法利用缓存（前缀每次都变）===")
    
    import datetime
    
    questions = [
        "什么是Transformer？",
        "什么是注意力机制？",
        "什么是KV缓存？"
    ]
    
    expert_names = ["A", "B", "C"]
    stats = []  # 收集统计数据
    
    for i, (question, expert_name) in enumerate(zip(questions, expert_names), 1):
        start = time.time()
        
        # 每次生成不同的时间戳（破坏缓存）
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        dynamic_prompt = f"[{timestamp}]\n\n{STANDARD_PROMPT}"
        
        messages = [
            SystemMessage(content=dynamic_prompt),
            HumanMessage(content=question)
        ]
        
        response = llm.invoke(messages, max_tokens=50)  # 严格限制回答长度，公平对比
        elapsed = time.time() - start
        
        print(f"\n问题{i}: {question}")
        print(f"📝 专家{expert_name}使用时间戳: {timestamp}")
        print(f"回答: {response.content}")
        print(f"⏱️  耗时: {elapsed:.2f}秒")
        
        # 显示缓存统计
        cache_hit = 0
        cache_miss = 0
        if hasattr(response, 'response_metadata'):
            token_usage = response.response_metadata.get('token_usage', {})
            cache_hit = token_usage.get('prompt_cache_hit_tokens', 0)
            cache_miss = token_usage.get('prompt_cache_miss_tokens', 0)
            print(f"📊 缓存命中: {cache_hit} tokens | 未命中: {cache_miss} tokens")
        
        stats.append({
            'question': i,
            'time': elapsed,
            'cache_hit': cache_hit,
            'cache_miss': cache_miss
        })
    
    return stats

if __name__ == "__main__":
    print("🚀 DeepSeek KV Cache 演示")
    print("=" * 50)

    # 清除缓存（重新初始化 LLM 实例）
    clear_cache()

    # 先运行利用缓存的版本
    stats1 = demo_with_cache()
    
    # 清除缓存（重新初始化 LLM 实例）
    clear_cache()
    
    # 再运行无法利用缓存的版本（对比）
    stats2 = demo_without_cache()
    
    # 输出对比表格
    print("\n" + "=" * 80)
    print("📊 最终实验结果")
    print("=" * 80)
    
    # 演示1表格
    print("\n演示1：利用缓存（静态前缀不变）")
    print("-" * 80)
    print(f"{'问题':<10} {'耗时':<15} {'缓存命中':<20} {'未命中':<15}")
    print("-" * 80)
    for stat in stats1:
        print(f"问题{stat['question']:<8} {stat['time']:.2f}秒{'':<10} {stat['cache_hit']} tokens{'':<12} {stat['cache_miss']} tokens")
    avg_time1 = sum(s['time'] for s in stats1) / len(stats1)
    print("-" * 80)
    print(f"平均耗时: ~{avg_time1:.1f}秒")
    
    # 演示2表格
    print("\n\n演示2：无法利用缓存（前缀每次都变）")
    print("-" * 80)
    print(f"{'问题':<10} {'耗时':<15} {'缓存命中':<20} {'未命中':<15}")
    print("-" * 80)
    for stat in stats2:
        cache_mark = " ❌" if stat['cache_hit'] == 0 else ""
        print(f"问题{stat['question']:<8} {stat['time']:.2f}秒{'':<10} {stat['cache_hit']} tokens{cache_mark:<12} {stat['cache_miss']} tokens")
    avg_time2 = sum(s['time'] for s in stats2) / len(stats2)
    print("-" * 80)
    print(f"平均耗时: ~{avg_time2:.1f}秒")
    
    print("\n" + "=" * 80)
    print("� 结论：")
    print("- 演示1中，静态系统提示词被缓存，后续请求缓存命中")
    print("- 演示2中，每次改变前缀，无法利用缓存")
    print("- 最佳实践：将不变的上下文放在prompt前面！")
    print("=" * 80)
