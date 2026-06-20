# LLM 自学路线

**主轴顺序：Transformer → Tokenizer → Pretraining → Inference → Scaling → Data → PEFT/SFT → Alignment → RAG → Agent → Evaluation**

以三门课为主轴：

- **CS336：主线，动手实现 LLM from scratch**（Stanford 2026 "Language Modeling from Scratch"）→ [cs336.stanford.edu](https://cs336.stanford.edu/)
- **CS324：理论框架**，补 modeling / training / data / scaling / adaptation / parallelism → [stanford-cs324.github.io](https://stanford-cs324.github.io/winter2022/lectures/)
- **CS25：视野扩展**，只选和 LLM 主线强相关的讲座，不全看 → [web.stanford.edu/class/cs25](https://web.stanford.edu/class/cs25/)

辅助资源：

- **Karpathy: Neural Networks: Zero to Hero** → [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)（Week 1–4 必配）
- **nanoGPT**（Week 4–5 代码精读）→ [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

---

## 总体原则：每周固定四步法

```
1. 看课：先建立概念地图
2. 读论文：只读和本周主题强相关的部分
3. 写代码：必须产出 notebook / repo commit
4. 写总结：用自己的话解释，形成 Obsidian 笔记
```

**只看课是最大的陷阱。代码 + 总结才是真正的学习。**

---

## 三个项目：有联动，不孤立

```
tiny-gpt-from-scratch          ← 基础模型，Week 2–5 建立
  ↓ 复用 train.py
data-quality-for-language-modeling  ← 数据实验，Week 7 建立
  ↓ clean data 反哺训练
paper-rag-assistant            ← 应用层，Week 11 建立
  ↓ 用于 Week 12 benchmark 的答案来源
```

三个 repo 形成一条链：训练 → 数据 → 应用。

---

## 时间分配（每周 8–10 小时）

| 模块   | 时间     |
| ------ | -------- |
| 看课   | 2 小时   |
| 读论文 | 2 小时   |
| 写代码 | 4 小时   |
| 写总结 | 1–2 小时 |

Week 5（代码重整）和 Week 9（LoRA 实验）预留 12–15 小时，不要压缩。

---

# Week 0：环境准备

## 目标

把开发环境、阅读环境、笔记结构搭好。不算正式学习周。

## 代码任务

```bash
# 建立 repo 结构
mkdir llm-learning && cd llm-learning
mkdir tiny-gpt-from-scratch
mkdir data-quality-for-language-modeling
mkdir paper-rag-assistant
mkdir notes

# 推荐用 uv 管理环境（CS336 官方推荐）
pip install uv
uv venv --python 3.11
source .venv/bin/activate

# 基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm einops tiktoken
pip install transformers datasets accelerate peft trl
pip install sentence-transformers faiss-cpu chromadb
pip install jupyter ipykernel
```

## Obsidian 结构

```
LLM/
  00_Roadmap.md
  01_Transformer.md
  02_Tokenization.md
  03_Pretraining.md
  04_Scaling_Laws.md
  05_Data.md
  06_PEFT_SFT.md
  07_Alignment.md
  08_Inference.md
  09_RAG.md
  10_Agent.md
  11_Evaluation.md
  Papers/
  Code_Notes/
  Projects/
```

---

# Week 1：LLM 全局框架 + Transformer 直觉

## 目标

建立完整的 LLM pipeline 心智模型：

```
data → tokenizer → Transformer → pretraining → SFT → alignment → inference → evaluation
```

## Step 1：看课

1. **CS25 V6：Overview of Transformers**
   - Transformer 为什么重要
   - NLP → LLM 的演变史
   - attention 的直觉
   - 当前挑战

2. **CS324：Introduction**

3. **CS324：Capabilities**

4. **Karpathy：The spelled-out intro to neural networks and backpropagation**（makemore Part 1）
   - 不要跳过，这是理解梯度流动的最佳入门

## Step 2：读论文

精读：

1. **Attention Is All You Need**
   - Abstract + Introduction
   - Section 3 Model Architecture
   - Figure 1
   - Table 2（BLEU scores，感受规模）

略读：

1. **GPT-2: Language Models are Unsupervised Multitask Learners**
   - Abstract + Introduction
   - Dataset 部分
   - Results 部分

## Step 3：代码任务

写 `attention_demo.ipynb`：

```python
# 三个实验
# 1. 不加 mask 的 self-attention
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

# 2. 加 causal mask 的 self-attention
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))

# 3. 改变 sequence length，观察 attention matrix 大小
```

## Step 4：总结

写：

```
01_Transformer.md
Papers/Attention_Is_All_You_Need.md
```

必须回答：

```
1. Transformer 解决了 RNN 的什么问题？
2. attention 的直觉是什么？
3. causal self-attention 和普通 self-attention 的区别？
4. 为什么 attention 适合大规模并行训练？
5. 我还有哪些不懂？
```

## 验收标准

> 我能不能手画 Transformer pipeline（从 token → embedding → attention → logits）？

---

# Week 2：PyTorch / einops / Transformer Block

## 目标

能手写一个最小 Transformer Block，理解每个 tensor 的 shape。

## Step 1：看课

1. **CS336 Lecture 1：Overview, tokenization**
2. **CS336 Lecture 2：PyTorch, einops, resource accounting**
3. **CS336 Lecture 3：Architectures, hyperparameters**
4. **Karpathy：makemore Part 3**（Backprop ninja，理解梯度计算）

## Step 2：读论文

精读：

1. **Attention Is All You Need**（第二遍，精读架构细节）
   - Section 3.1 Encoder and Decoder Stacks
   - Section 3.2 Attention（多头，公式）
   - Section 3.4 Embeddings and Softmax
   - Section 3.5 Positional Encoding

辅助：

1. **The Illustrated Transformer**（Jay Alammar）
   - 只看架构图和 attention 部分

## Step 3：代码任务

在 `tiny-gpt-from-scratch/model.py` 实现：

```python
class CausalSelfAttention(nn.Module):
    # Q, K, V 投影
    # scaled dot-product
    # causal mask
    # multi-head concat

class FeedForward(nn.Module):
    # Linear → GELU → Linear

class TransformerBlock(nn.Module):
    # Pre-LN: LayerNorm → Attention → residual
    # Pre-LN: LayerNorm → FFN → residual

class GPT(nn.Module):
    # token embedding + positional embedding
    # N × TransformerBlock
    # final LayerNorm → linear head
```

验证：

```python
x = torch.randint(0, vocab_size, (batch_size, seq_len))
logits = model(x)
assert logits.shape == (batch_size, seq_len, vocab_size)
```

## Step 4：总结

写：

```
Code_Notes/Transformer_Block_From_Scratch.md
```

回答：

```
1. input token ids 如何一步步变成 logits？（shape 全程追踪）
2. Q/K/V 的 shape 分别是什么？
3. Pre-LN 和 Post-LN 有什么区别？
4. residual connection 为什么重要？
5. einops 解决了什么问题？
```

## 验收标准

> 我能不能解释 Q/K/V 的 shape，并说清楚 multi-head attention 的拼接逻辑？

---

# Week 3：Tokenizer / BPE

## 目标

理解 tokenizer 不是小细节，而是 LLM 的输入层。中文和代码的 tokenization 问题尤其重要。

## Step 1：看课

1. **CS336 Lecture 1：tokenization 部分**（重看）
2. **CS324：Data**（重点：数据来源、pretraining data 构成）
3. **Karpathy：Let's build the GPT tokenizer**（YouTube，约 2 小时，强烈推荐）

## Step 2：读论文

精读：

1. **Neural Machine Translation of Rare Words with Subword Units**（BPE 原论文）
   - Abstract + Introduction
   - BPE merge 规则部分

略读：

1. **SentencePiece**
   - Abstract + Introduction
   - 和 BPE / WordPiece 的区别

## Step 3：代码任务

实现简化版 BPE：

```python
# tokenizer.py
def train_bpe(corpus: str, vocab_size: int) -> dict:
    # 统计字符对频率
    # 迭代 merge 最高频对
    # 返回 merge rules

def encode(text: str) -> list[int]: ...
def decode(token_ids: list[int]) -> str: ...
```

实验：用三段文本对比：

```
1. 英文新闻段落
2. 中文段落（体会为什么中文 token 效率低）
3. Python 代码片段
```

记录：原始字符数 / token 数 / 压缩比 / 奇怪切分案例

## Step 4：总结

写 `02_Tokenization.md`，回答：

```
1. 为什么不能直接用 word-level tokenizer？
2. BPE 的 merge 规则如何学习？
3. 中文 token 效率低的根本原因？
4. 代码数据为什么对 tokenizer 特别敏感？
5. tokenizer 如何影响 context window 利用率？
```

## 验收标准

> 我能不能自己实现一个简化 BPE，并解释中文 token 效率问题？

---

# Week 4：训练 tiny GPT

## 目标

跑通第一个真正的 language model training loop。**本周重点是 nanoGPT 代码精读，不建议完全从零造轮子。**

## Step 1：看课

1. **CS336 Lecture 3：Architectures, hyperparameters**
2. **CS324：Modeling**
3. **CS324：Training**
4. **Karpathy：Let's build GPT from scratch**（YouTube，最重要的一讲）

## Step 2：读论文

精读：

1. **GPT-2**
   - Model 部分
   - Training dataset 部分
   - Zero-shot results

略读：

1. **GPT-3**
   - Abstract + Introduction
   - Model and Architectures
   - Few-shot / one-shot / zero-shot 定义

## Step 3：代码任务

**先精读 nanoGPT**（[github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)），在注释里写清楚每个部分做什么。

然后在自己的 `train.py` 里完成：

```python
# train.py
def load_data(path): ...
def get_batch(split): ...
def train():
    for step in range(max_steps):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_interval == 0:
            val_loss = estimate_loss('val')
            print(f"step {step}: val loss {val_loss:.4f}")
            generate_sample()
```

最小模型配置：

```python
n_layer = 2
n_head = 2
n_embd = 128
block_size = 128
batch_size = 32
```

数据集优先用 TinyStories，或 Shakespeare / WikiText-2。

产出：training loss curve + validation loss curve + 每 500 步生成一次文本。

## Step 4：总结

写 `03_Pretraining.md`，回答：

```
1. next-token prediction 在预测什么？
2. cross entropy loss 和 perplexity 的关系？
3. 为什么 validation loss 比 training loss 更重要？
4. temperature 如何影响生成？
5. 你的 tiny GPT 现在最明显的缺陷是什么？
```

## 验收标准

> 我能不能训练一个 tiny GPT 并生成文本，并解释 loss curve 说明了什么？

---

# Week 5：代码整理 + CS336 Assignment 1 标准化

## 目标

把 Week 2–4 的代码整理到接近生产标准。**这周是工程周，不引入新概念。**

## Step 1：看课

1. **CS336 Assignment 1 handout**（精读要求）
2. **CS336 Lecture 1–3 回看**（补漏洞）
3. **CS336 Lecture 4：Attention alternatives and MoE**（只看前半部分）

## Step 2：读论文

精读：

1. **AdamW: Decoupled Weight Decay Regularization**
   - Abstract
   - AdamW 和 Adam 的核心区别

略读：

1. **Switch Transformer**（只需理解 MoE 是什么）

## Step 3：代码任务

整理 repo 结构：

```
tiny-gpt-from-scratch/
  README.md          ← 必须完整
  config.py          ← 所有超参数集中管理
  tokenizer.py
  model.py
  train.py
  generate.py
  requirements.txt
  notebooks/
    loss_curve.ipynb
    attention_demo.ipynb
```

补齐以下内容：

```python
# AdamW + 学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

# Warmup + cosine decay
def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # cosine decay
    ...

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Checkpoint
torch.save({'model': model.state_dict(), 'step': step}, 'ckpt.pt')
```

README 必须包括：

```
1. Problem（这个项目解决什么问题）
2. Architecture（模型结构）
3. Dataset（数据集说明）
4. Training setup（超参数）
5. Loss curve（截图）
6. Sample generations（生成样例）
7. Limitations（当前局限）
8. What I learned
```

## Step 4：总结

写：

```
Projects/tiny-gpt-from-scratch.md
```

这是第一个可以放 GitHub 的项目报告。

## 验收标准

> 我的 tiny-gpt-from-scratch repo 是否有完整 README，能让别人直接跑起来？

---

# Week 6：Inference / Decoding / KV Cache

> **⚠️ 调整说明：Inference 从原 Week 8 提前到 Week 6。**
> 理由：理解 generation（temperature、top-k、KV cache）是理解 SFT 实验结果的前提，不应该放在 alignment 之后。

## 目标

理解 LLM 推理为什么贵，以及 decoding 策略如何影响输出质量。

## Step 1：看课

1. **CS336 Lecture 10：Inference**
2. **CS324：Selective architectures**（了解 SSM 背景）
3. **CS25：On the Tradeoffs of State Space Models and Transformers**

## Step 2：读论文

精读：

1. **FlashAttention**
   - Abstract + Introduction
   - IO-aware attention 的核心想法（tiling，不是完整数学证明）

2. **vLLM / PagedAttention**
   - Abstract + Introduction
   - KV cache 碎片化问题

略读：

1. **Mamba**
   - Abstract + Introduction
   - 为什么它是 Transformer alternative

## Step 3：代码任务

在 `tiny-gpt-from-scratch/generate.py` 实现四种解码策略：

```python
def greedy_decode(model, prompt, max_new_tokens): ...
def temperature_sample(model, prompt, temperature): ...
def top_k_sample(model, prompt, k): ...
def top_p_sample(model, prompt, p): ...  # nucleus sampling
```

实验：同一个 prompt `"Once upon a time"`，对比：

```
temperature = 0.3 / 0.8 / 1.2
top_k = 10 / 50
top_p = 0.9 / 0.95
```

进阶：实现简化 KV cache：

```python
# 没有 cache：每次重算整个 prefix
# 有 cache：只算新 token 的 K/V，past_kv 累积
```

## Step 4：总结

写 `08_Inference.md`，回答：

```
1. greedy decoding 为什么容易出现重复和无聊？
2. temperature 控制什么？为什么 > 1 会乱，< 0.3 会重复？
3. top-k 和 top-p 的区别和各自适用场景？
4. KV cache 为什么能加速？它节省了什么计算？
5. inference bottleneck 和 training bottleneck 有何本质不同？
```

## 验收标准

> 我能不能解释 KV cache 的原理，并演示不同 decoding 策略对生成结果的影响？

---

# Week 7：Scaling Laws

## 目标

理解为什么"大模型 + 大数据 + 大算力"有效，以及 Chinchilla 为什么重新定义了训练范式。

## Step 1：看课

三个来源，各有侧重，不是重复内容：

1. **CS336 Lecture 9：Scaling laws**（侧重 Kaplan empirical scaling，模型大小 vs. loss 的幂律关系）
2. **CS336 Lecture 11：Scaling laws**（侧重 Chinchilla compute-optimal，给定算力如何分配模型和数据）
3. **CS324：Scaling laws**（理论视角 + 实际训练决策的影响）

## Step 2：读论文

精读：

1. **Scaling Laws for Neural Language Models**（Kaplan et al.）
   - Abstract + Introduction
   - scaling law 公式
   - compute / data / model size 三角关系图

2. **Training Compute-Optimal Large Language Models**（Chinchilla）
   - Abstract + Introduction
   - Figure 1（Chinchilla vs GPT-3 vs Gopher 的对比）
   - 结论：给定 compute，模型大小和 token 数如何平衡

## Step 3：代码任务

Mini scaling experiment。训练 3 个模型，在同一数据集上对比：

```python
configs = {
    'small':  {'n_layer': 2, 'n_head': 2, 'n_embd': 128},
    'medium': {'n_layer': 4, 'n_head': 4, 'n_embd': 256},
    'large':  {'n_layer': 6, 'n_head': 6, 'n_embd': 384},
}
```

记录并画图：

```
x-axis: log(parameter count)
y-axis: validation loss
```

如果算力不够，只跑 small / medium。

## Step 4：总结

写 `04_Scaling_Laws.md`，回答：

```
1. scaling law 试图预测什么？
2. 参数量、数据量、算力三者是什么关系？
3. Chinchilla 为什么说很多模型"训练 token 不够"？
4. LLaMA 系列为什么能用小模型打败大模型？
5. 对个人学习者，scaling law 告诉了我们什么？
```

## 验收标准

> 我能不能解释 Chinchilla 的核心结论，并说明它对 LLaMA 的影响？

---

# Week 8：Data-Centric LLM

## 目标

把 Data Science 背景用起来。理解数据质量是 LLM 效果的天花板。

## Step 1：看课

1. **CS336 Lecture 13：Data, sources, datasets**
2. **CS336 Lecture 14：Data filtering, deduplication, mixing, synthetic data**
3. **CS324：Data**

## Step 2：读论文

精读：

1. **LLaMA**（Meta，2023）
   - Data 部分（数据来源 + 过滤策略）
   - Training 部分

略读：

1. **The Pile**（了解 pretraining data 构成）
2. **FineWeb / RefinedWeb 技术报告**（现代 CC 数据清洗范式）
3. **LLaMA 2**（Pretraining Data 部分）

## Step 3：代码任务

新建项目 `data-quality-for-language-modeling/`，构造两个数据集：

```
clean_dataset:   高质量 story / wiki / 课程文本
dirty_dataset:   重复文本 + 垃圾符号 + HTML 残留 + 随机拼接
```

用同一个 tiny GPT 配置训练三次：

```
A: clean dataset
B: dirty dataset
C: deduplicated dirty dataset（从 B 去重后）
```

比较：valid loss / 生成样例 / 重复率 / 乱码率

这个项目复用 `tiny-gpt-from-scratch/train.py`，数据集替换即可。

## Step 4：总结

写：

```
05_Data.md
Projects/data-quality-for-language-modeling.md
```

回答：

```
1. 数据质量如何直接影响 validation loss？
2. 重复数据会造成什么具体问题（举 GPT-2 memorization 例子）？
3. deduplication 为什么是 pretraining 的标配？
4. synthetic data 的风险是什么（model collapse）？
5. Data Science 背景在 LLM 工程里有哪些实际用武之地？
```

## 验收标准

> 我能不能用实验数据证明数据质量影响 loss，并说明 deduplication 的必要性？

---

# Week 9：PEFT / LoRA / QLoRA

> **⚠️ 新增独立章节。**
> 原路线把 LoRA 压缩在 SFT 代码任务里一笔带过。对 AI engineering 目标而言，这是必须独立掌握的核心技能。

## 目标

理解 LoRA 的数学原理，能独立跑完 LoRA fine-tuning 全流程，理解 QLoRA 的量化机制。

## Step 1：看课

1. **CS336 Lecture 15：Mid/post-training, SFT/RLHF**（只看 adaptation 相关部分）
2. **CS324：Adaptation**（完整看）
3. Hugging Face PEFT 文档：[huggingface.co/docs/peft](https://huggingface.co/docs/peft)

## Step 2：读论文

精读：

1. **LoRA: Low-Rank Adaptation of Large Language Models**
   - Abstract + Introduction
   - Section 4：LoRA 方法（$W = W_0 + \Delta W = W_0 + BA$）
   - Section 7：实验结果

精读：

2. **QLoRA: Efficient Finetuning of Quantized LLMs**
   - Abstract + Introduction
   - 4-bit NormalFloat 量化的直觉
   - 和 LoRA 的组合方式

## Step 3：代码任务

新建子目录 `tiny-gpt-from-scratch/peft-experiments/`：

**实验 1：手动实现 LoRA 层**

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scale
```

**实验 2：用 `peft` 库对小模型做 LoRA SFT**

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
model.print_trainable_parameters()
# 对比：full FT 参数量 vs. LoRA 参数量
```

数据：Alpaca-style 小样本（1k–5k 条）  
平台：Mac M3 Pro 可跑 0.5B；大模型用 Colab / Kaggle

对比：

```
base model output（同一 prompt）
LoRA SFT output
full FT output（如果算力允许）
```

## Step 4：总结

写 `06_PEFT_SFT.md`，回答：

```
1. LoRA 的低秩分解是什么意思？为什么有效？
2. rank 和 alpha 如何影响训练效果？
3. 为什么 LoRA 比 full fine-tuning 省显存？（参数量对比）
4. QLoRA 在 LoRA 基础上做了什么额外工作？
5. 哪些层应该加 LoRA？（q_proj, v_proj 为什么是首选）
```

## 验收标准

> 我能不能手写 LoRA 层，解释低秩分解原理，并跑完一个完整的 LoRA SFT 实验？

---

# Week 10：Instruction Tuning / SFT 全流程

## 目标

理解 base model 为什么不是 chat model，掌握 chat template 和 instruction dataset 的工程细节。

## Step 1：看课

1. **CS336 Lecture 15：Mid/post-training, SFT/RLHF**（完整看）
2. **CS324：Adaptation**（回看，结合代码理解）

## Step 2：读论文

精读：

1. **InstructGPT: Training Language Models to Follow Instructions with Human Feedback**
   - Abstract + Introduction
   - SFT 部分（数据格式、训练细节）
   - RLHF 部分先读概念，不需要深挖 PPO

略读：

1. **Self-Instruct**
2. **Alpaca**（了解 self-instruct 的实际应用）

## Step 3：代码任务

在 Week 9 的 LoRA SFT 基础上，扩展到完整 SFT pipeline：

```
sft-experiments/
  prepare_data.py    ← 格式化 instruction dataset
  train_sft.py       ← LoRA SFT 训练
  inference.py       ← 对比 base / sft 输出
  README.md
```

chat template 示例：

```python
# Qwen2.5 chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain overfitting in one sentence."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

必须做的对比实验：

```
同一个 prompt，base model vs. SFT model，记录：
- 指令遵循程度
- 格式规范性
- 幻觉率（主观判断）
```

## Step 4：总结

补充 `06_PEFT_SFT.md`，回答：

```
1. base model 和 instruct model 的本质差异是什么？
2. instruction dataset 长什么样？（prompt / response 对）
3. chat template 是什么？为什么不同模型格式不同？
4. SFT 为什么不能完全解决 alignment？
5. SFT 数据质量 vs. 数量，哪个更重要？
```

## 验收标准

> 我能不能解释 base model 和 instruct model 的差异，并展示 SFT 前后的输出对比？

---

# Week 11：RLHF / DPO / Reasoning RL

## 目标

理解现代 LLM 对齐的主线，以及为什么 DPO 正在取代传统 RLHF。

## Step 1：看课

1. **CS336 Lecture 15：Mid/post-training, SFT/RLHF**
2. **CS336 Lecture 16：Post-training - RLVR**
3. **CS336 Assignment 5：Alignment and Reasoning RL**（读 handout，了解任务设计）

## Step 2：读论文

精读：

1. **InstructGPT**
   - RLHF 三阶段（SFT → RM → PPO）
   - reward model 的训练方式
   - PPO 在语言模型里的变体

2. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**
   - Abstract + Introduction
   - DPO 相比 RLHF 简化在哪里（消掉了显式 reward model）

略读：

1. **Constitutional AI**（Anthropic）
2. **GRPO / DeepSeek-R1 技术报告**（了解 reasoning RL 现状）

## Step 3：代码任务

**选择 A（推荐）：DPO toy experiment**

构造 preference dataset：

```json
{
  "prompt": "Explain overfitting.",
  "chosen": "Overfitting occurs when a model learns noise in training data...",
  "rejected": "Overfitting means the model is too big."
}
```

用 `trl` 跑 DPO：

```python
from trl import DPOTrainer, DPOConfig
trainer = DPOTrainer(model=model, args=config, train_dataset=dataset)
trainer.train()
```

**选择 B：只做 preference data 分析**

如果算力不够，至少完成：

```
- preference dataset 格式解析
- chosen / rejected 长度分布对比
- reward hacking 案例举例
- 手画 RLHF vs. DPO 流程图
```

## Step 4：总结

写 `07_Alignment.md`，回答：

```
1. RLHF 三阶段分别做什么？
2. reward model 学的是什么？
3. PPO 在语言模型里解决什么问题？
4. DPO 为什么更简单？它的假设是什么？
5. RLHF / DPO 可能引入什么新问题（reward hacking、分布偏移）？
```

## 验收标准

> 我能不能画出 SFT / RLHF / DPO 完整流程图，并解释三者的关系？

---

# Week 12：RAG / 上下文检索

## 目标

做一个实用项目，理解"参数记忆"和"上下文检索"的本质区别。

## Step 1：看课

1. **CS25：Distinct Modes of Generalization from Parameters and Context**
2. **CS324：Adaptation**（retrieval 相关部分）
3. **CS336 Lecture 12：Evaluation**（先看 task evaluation 部分）

## Step 2：读论文

精读：

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**（RAG 原论文）
   - Abstract + Introduction
   - Method：dense retrieval + generator 结构

2. **ReAct: Synergizing Reasoning and Acting in Language Models**
   - Abstract + Introduction
   - reasoning + acting 的输出格式

略读：

1. **Self-RAG**
2. **HyDE（Hypothetical Document Embeddings）**

## Step 3：代码任务

新建项目 `paper-rag-assistant/`：

```
paper-rag-assistant/
  ingest.py       ← 读 PDF，chunk，embedding，存入向量库
  retrieval.py    ← 查询，返回 top-k chunks
  generate.py     ← 拼 prompt，调 API，输出答案 + 来源
  evaluate.py     ← 评估 retrieval precision 和 answer quality
  README.md
```

技术选型：

```python
# embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# vector store
import chromadb

# generation
import anthropic  # 或 openai
```

数据源：你已经读过的 LLM 论文 PDF

实验：对比三种 chunk size 的效果：

```
300 tokens / 600 tokens / 1000 tokens
```

记录：retrieval precision / answer quality / 引用是否准确

## Step 4：总结

写：

```
09_RAG.md
Projects/paper-rag-assistant.md
```

回答：

```
1. RAG 解决什么问题？为什么不直接 fine-tune？
2. chunk size 为什么重要？太大和太小各有什么问题？
3. dense retrieval 和 keyword search 的本质区别？
4. 为什么有了 RAG 还会 hallucination？
5. RAG 和 fine-tuning 各自适合什么场景？
```

## 验收标准

> 我能不能做一个能输出答案 + 引用来源的 RAG，并解释 chunk size 对结果的影响？

---

# Week 13：Agent / Tool Use

> **⚠️ 新增章节。**
> 原路线 RAG 之后没有 Agent，但 Agent 是 2025–2026 AI engineering 岗位的高频考点。

## 目标

在 RAG 基础上加一层 Agent 能力：模型自己决定何时检索、何时调用工具。

## Step 1：看课 / 读文档

1. **CS25：任何 agent / tool use 相关讲座**
2. Anthropic Tool Use 文档：[docs.anthropic.com/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
3. LangGraph 文档（如果你计划用框架）：[langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)

## Step 2：读论文

精读：

1. **ReAct**（第二遍，这次重点看代码实现格式）

略读：

1. **Toolformer**
2. **HuggingGPT / HuggingFace Agents**（了解 multi-tool 场景）

## Step 3：代码任务

在 `paper-rag-assistant/` 里加一层 Agent：

```python
# 手写简单 ReAct loop（不依赖框架）
def react_loop(user_query, max_steps=5):
    messages = [{"role": "user", "content": user_query}]
    for step in range(max_steps):
        response = llm(messages, tools=[search_tool, calculator_tool])
        if response.stop_reason == "tool_use":
            tool_result = execute_tool(response.tool_call)
            messages.append({"role": "tool", "content": tool_result})
        else:
            return response.text  # 最终答案
```

实现两个工具：

```
search_tool    ← 调用你的 RAG retrieval
calculator_tool ← 简单的数学计算
```

场景测试：

```
"FlashAttention 的作者是谁，他现在在哪里工作？"
→ Agent 应该先 search_tool，再用 retrieval 结合回答
```

## Step 4：总结

写 `10_Agent.md`，回答：

```
1. ReAct 的 Reason + Act 循环是什么？
2. function calling 和 prompt engineering 有什么区别？
3. tool use 和 RAG 如何配合？
4. agent loop 什么时候会死循环？如何防止？
5. 单 agent vs. multi-agent，各自适合什么场景？
```

## 验收标准

> 我能不能手写一个 ReAct 循环，让模型自己决定是否调用 search 工具？

---

# Week 14：Evaluation + 部署 + 项目整理

## 目标

从"我学过"变成"我有成果可以展示"。**加入 LLM-as-judge 和本地部署实践。**

## Step 1：看课

1. **CS336 Lecture 12：Evaluation**
2. **CS324：Capabilities**
3. **CS324：Harms I / Harms II**（bias、safety 部分）

## Step 2：读论文 / benchmark

精读：

1. **HELM**（evaluation framework）
   - accuracy / calibration / robustness / fairness / efficiency

2. **MMLU**
   - benchmark 构造原理
   - 为什么它不等于真正的智能

略读：

1. **GSM8K**
2. **HumanEval**
3. **MT-Bench / AlpacaEval**（了解 LLM-as-judge 范式）

## Step 3：代码任务

**Part 1：自定义 benchmark**

测试对象：

```
1. 你的 tiny GPT
2. 一个 base small model（Qwen2.5-0.5B base）
3. 一个 instruct small model（Qwen2.5-0.5B instruct）
4. Claude / GPT API
```

测试集（20 个问题）：

```
5 个 CS 概念
5 个数学推理
5 个代码生成
5 个 RAG 问答（用你的 paper-rag-assistant）
```

**Part 2：LLM-as-judge（新增）**

```python
def llm_judge(question, answer, reference=None):
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Rate the answer on: accuracy, completeness, hallucination (1-5 each).
    Return JSON.
    """
    return claude_api(prompt)
```

**Part 3：本地部署实践（新增）**

用 Ollama 在本地跑一个小模型：

```bash
# 安装 Ollama
curl https://ollama.ai/install.sh | sh

# 拉取并运行
ollama run qwen2.5:0.5b
ollama run llama3.2:1b

# Python 调用
import ollama
response = ollama.chat(model='qwen2.5:0.5b', messages=[...])
```

理解：vLLM / Ollama / HuggingFace TGI 的区别，以及什么场景用哪个。

## Step 4：最终整理

整理三个 README（每个都要完整）：

```
tiny-gpt-from-scratch/README.md
data-quality-for-language-modeling/README.md
paper-rag-assistant/README.md
```

每个 README 必须包括：

```
1. Problem（解决什么问题）
2. Method（方法说明）
3. Implementation（关键实现细节）
4. Experiments（实验设置）
5. Results（结果 + 图表）
6. Failure cases（哪里没做好）
7. What I learned
8. Future work
```

## 验收标准

> 我有没有 3 个可展示的 GitHub 项目，能用 LLM-as-judge 评估模型输出，并在本地跑通一个模型？

---

# 论文精读优先级

## 必须精读（按顺序）

```
1. Attention Is All You Need
2. GPT-2
3. GPT-3
4. Scaling Laws for Neural Language Models（Kaplan）
5. Chinchilla
6. FlashAttention
7. LLaMA
8. LoRA
9. QLoRA
10. InstructGPT
11. DPO
12. RAG（Lewis et al.）
13. ReAct
```

## 可以略读

```
1. BERT
2. T5
3. RoBERTa
4. PaLM
5. LLaMA 2
6. Mamba
7. Constitutional AI
8. Self-Instruct
9. Alpaca
10. HELM
11. Switch Transformer
12. Self-RAG
13. Toolformer
```

---

# 每周验收标准（完整版）

| 周   | 验收问题                                                     |
| ---- | ------------------------------------------------------------ |
| W1   | 能不能手画 Transformer pipeline（token → embedding → attention → logits）？ |
| W2   | 能不能解释 Q/K/V 的 shape，说清楚 multi-head attention 的拼接逻辑？ |
| W3   | 能不能自己实现简化 BPE，解释中文 token 效率问题？           |
| W4   | 能不能训练 tiny GPT 并生成文本，解释 loss curve？           |
| W5   | tiny-gpt-from-scratch repo 是否有完整 README，别人能直接跑？ |
| W6   | 能不能解释 KV cache 原理，演示不同 decoding 策略的效果？    |
| W7   | 能不能解释 Chinchilla 核心结论，说明它对 LLaMA 的影响？     |
| W8   | 能不能用实验数据证明数据质量影响 loss？                     |
| W9   | 能不能手写 LoRA 层，解释低秩分解原理，跑完 LoRA SFT 实验？  |
| W10  | 能不能展示 SFT 前后的输出对比，解释 base vs instruct 的差异？ |
| W11  | 能不能画出 SFT / RLHF / DPO 完整流程图，解释三者关系？      |
| W12  | 能不能做一个有引用的 RAG，解释 chunk size 对结果的影响？     |
| W13  | 能不能手写 ReAct 循环，让模型自己决定是否调用 search？       |
| W14  | 有没有 3 个完整 GitHub 项目，会用 LLM-as-judge 评估？        |

---

# 学习路线全局图

```
Week 0   环境 + Obsidian
  ↓
Week 1   Transformer 直觉 + attention demo
  ↓
Week 2   Transformer Block 实现
  ↓
Week 3   BPE Tokenizer
  ↓
Week 4   tiny GPT 训练（nanoGPT 精读）
  ↓
Week 5   代码整理 + CS336 标准化           → 项目 1 完成：tiny-gpt-from-scratch
  ↓
Week 6   Inference / Decoding / KV Cache
  ↓
Week 7   Scaling Laws（Kaplan + Chinchilla）
  ↓
Week 8   Data-Centric LLM                 → 项目 2 完成：data-quality
  ↓
Week 9   PEFT / LoRA / QLoRA（独立章节）
  ↓
Week 10  SFT 全流程 + chat template
  ↓
Week 11  RLHF / DPO / Reasoning RL
  ↓
Week 12  RAG + 向量检索                   → 项目 3 完成：paper-rag-assistant
  ↓
Week 13  Agent / Tool Use / ReAct
  ↓
Week 14  Evaluation + 本地部署 + 项目整理  → 3 个 GitHub 项目
```

---

# 面向 AI Engineering 的额外提示

以下是优先级建议：

**最高优先（面试高频）**
- LoRA / QLoRA 全链路（Week 9）
- 本地推理部署：Ollama / vLLM（Week 14）
- RAG pipeline 工程细节（Week 12）
- Prompt engineering：few-shot、chain-of-thought、structured output

**中等优先（理解原理）**
- Scaling Laws（Week 7）——理解就够，不需要自己跑大规模实验
- RLHF / DPO（Week 11）——理解 pipeline，能解释概念

**了解即可（研究方向，不是工程方向）**
- Data filtering 精细操作（Week 8 做了实验就够）
- 从零实现 BPE（Week 3 做了就够，工作中用 tiktoken）

**建议额外关注（路线未覆盖但实际工作常见）**
- Structured output / JSON mode（Pydantic + instructor 库）
- LangChain / LangGraph（工程框架，Week 13 后可选学）
- 模型量化：GPTQ / AWQ / bitsandbytes
- 成本优化：prompt caching、batch inference
