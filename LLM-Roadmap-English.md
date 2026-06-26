# LLM Self-Study Roadmap

**Main sequence: Transformer → Tokenizer → Pretraining → Inference → Scaling → Data → PEFT/SFT → Alignment → RAG → Agent → Evaluation**

Use three courses as the main axis:

- **CS336: the main line, hands-on implementation of an LLM from scratch** (Stanford 2026 "Language Modeling from Scratch") → [cs336.stanford.edu](https://cs336.stanford.edu/)
- **CS324: theoretical framework**, supplementing modeling / training / data / scaling / adaptation / parallelism → [stanford-cs324.github.io](https://stanford-cs324.github.io/winter2022/lectures/)
- **CS25: broader perspective**, only select lectures strongly related to the LLM main line; do not watch everything → [web.stanford.edu/class/cs25](https://web.stanford.edu/class/cs25/)

Supplementary resources:

- **Karpathy: Neural Networks: Zero to Hero** → [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html) (essential companion for Weeks 1–4)
- **nanoGPT** (close code reading in Weeks 4–5) → [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

---

## Overall Principle: Fixed Four-Step Method Each Week

```
1. Watch lectures: first build a conceptual map
2. Read papers: only read the parts strongly related to this week’s topic
3. Write code: must produce a notebook / repo commit
4. Write a summary: explain in your own words and form Obsidian notes
```

**Only watching lectures is the biggest trap. Code + summaries are the real learning.**

---

## Three Projects: Connected, Not Isolated

```
tiny-gpt-from-scratch          ← foundational model, built in Weeks 2–5
  ↓ reuse train.py
data-quality-for-language-modeling  ← data experiment, built in Week 7
  ↓ clean data feeds back into training
paper-rag-assistant            ← application layer, built in Week 11
  ↓ used as the answer source for the Week 12 benchmark
```

The three repos form one chain: training → data → application.

---

## Time Allocation (8–10 hours per week)

| Module         | Time      |
| -------------- | --------- |
| Watch lectures | 2 hours   |
| Read papers    | 2 hours   |
| Write code     | 4 hours   |
| Write summary  | 1–2 hours |

Reserve 12–15 hours for Week 5 (code cleanup) and Week 9 (LoRA experiments). Do not compress them.

---

# Week 0: Environment Setup

## Goal

Set up the development environment, reading environment, and note structure. This is not counted as an official learning week.

## Code Tasks

```bash
# Create repo structure
mkdir llm-learning && cd llm-learning
mkdir tiny-gpt-from-scratch
mkdir data-quality-for-language-modeling
mkdir paper-rag-assistant
mkdir notes

# Recommended to use uv for environment management (officially recommended by CS336)
pip install uv
uv venv --python 3.11

# Basic dependencies
uv add numpy pandas matplotlib tqdm einops tiktoken
uv add torch torchvision torchaudio
uv add transformers datasets accelerate peft trl
uv add sentence-transformers faiss-cpu chromadb
uv add jupyter ipykernel
```

## Obsidian Structure

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

# Week 1: Global LLM Framework + Transformer Intuition

## Goal

Build a complete mental model of the LLM pipeline:

```
data → tokenizer → Transformer → pretraining → SFT → alignment → inference → evaluation
```

## Step 1: Watch Lectures

1. **CS25 V6: Overview of Transformers**
   - Why Transformer matters
   - The evolution from NLP to LLMs
   - Intuition behind attention
   - Current challenges

2. **CS324: Introduction**

3. **CS324: Capabilities**

4. **Karpathy: The spelled-out intro to neural networks and backpropagation** (makemore Part 1)
   - Do not skip this. It is the best entry point for understanding gradient flow.

## Step 2: Read Papers

Read closely:

1. **Attention Is All You Need**
   - Abstract + Introduction
   - Section 3 Model Architecture
   - Figure 1
   - Table 2 (BLEU scores; get a feel for scale)

Skim:

1. **GPT-2: Language Models are Unsupervised Multitask Learners**
   - Abstract + Introduction
   - Dataset section
   - Results section

## Step 3: Code Task

Write `attention_demo.ipynb`:

```python
# Three experiments
# 1. Self-attention without mask
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

# 2. Self-attention with causal mask
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))

# 3. Change sequence length and observe the size of the attention matrix
```

## Step 4: Summary

Write:

```
01_Transformer.md
Papers/Attention_Is_All_You_Need.md
```

Must answer:

```
1. What problem of RNNs did Transformer solve?
2. What is the intuition behind attention?
3. What is the difference between causal self-attention and ordinary self-attention?
4. Why is attention suitable for large-scale parallel training?
5. What do I still not understand?
```

## Acceptance Standard

> Can I draw the Transformer pipeline by hand, from token → embedding → attention → logits?

---

# Week 2: PyTorch / einops / Transformer Block

## Goal

Be able to hand-write a minimal Transformer Block and understand the shape of every tensor.

## Step 1: Watch Lectures

1. **CS336 Lecture 1: Overview, tokenization**
2. **CS336 Lecture 2: PyTorch, einops, resource accounting**
3. **CS336 Lecture 3: Architectures, hyperparameters**
4. **Karpathy: makemore Part 3** (Backprop ninja; understand gradient computation)

## Step 2: Read Papers

Read closely:

1. **Attention Is All You Need** (second pass, close reading of architectural details)
   - Section 3.1 Encoder and Decoder Stacks
   - Section 3.2 Attention (multi-head, formulas)
   - Section 3.4 Embeddings and Softmax
   - Section 3.5 Positional Encoding

Supplement:

1. **The Illustrated Transformer** (Jay Alammar)
   - Only look at the architecture diagrams and attention sections

## Step 3: Code Task

Implement in `tiny-gpt-from-scratch/model.py`:

```python
class CausalSelfAttention(nn.Module):
    # Q, K, V projections
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

Validation:

```python
x = torch.randint(0, vocab_size, (batch_size, seq_len))
logits = model(x)
assert logits.shape == (batch_size, seq_len, vocab_size)
```

## Step 4: Summary

Write:

```
Code_Notes/Transformer_Block_From_Scratch.md
```

Answer:

```
1. How do input token IDs become logits step by step? Track shapes throughout.
2. What are the shapes of Q/K/V respectively?
3. What is the difference between Pre-LN and Post-LN?
4. Why are residual connections important?
5. What problem does einops solve?
```

## Acceptance Standard

> Can I explain the shapes of Q/K/V and clearly describe the concatenation logic of multi-head attention?

---

# Week 3: Tokenizer / BPE

## Goal

Understand that the tokenizer is not a minor detail, but the input layer of an LLM. Tokenization issues in Chinese and code are especially important.

## Step 1: Watch Lectures

1. **CS336 Lecture 1: tokenization section** (rewatch)
2. **CS324: Data** (focus: data sources and the composition of pretraining data)
3. **Karpathy: Let’s build the GPT tokenizer** (YouTube, about 2 hours, strongly recommended)

## Step 2: Read Papers

Read closely:

1. **Neural Machine Translation of Rare Words with Subword Units** (original BPE paper)
   - Abstract + Introduction
   - BPE merge rule section

Skim:

1. **SentencePiece**
   - Abstract + Introduction
   - Differences from BPE / WordPiece

## Step 3: Code Task

Implement a simplified BPE:

```python
# tokenizer.py
def train_bpe(corpus: str, vocab_size: int) -> dict:
    # Count character-pair frequencies
    # Iteratively merge the most frequent pair
    # Return merge rules

def encode(text: str) -> list[int]: ...
def decode(token_ids: list[int]) -> str: ...
```

Experiment: compare three text segments:

```
1. English news paragraph
2. Chinese paragraph (understand why Chinese token efficiency is low)
3. Python code snippet
```

Record: original character count / token count / compression ratio / strange segmentation cases

## Step 4: Summary

Write `02_Tokenization.md`, and answer:

```
1. Why not directly use a word-level tokenizer?
2. How are BPE merge rules learned?
3. What is the root cause of low token efficiency in Chinese?
4. Why is code data especially sensitive to tokenizer design?
5. How does the tokenizer affect context window utilization?
```

## Acceptance Standard

> Can I implement a simplified BPE by myself and explain the Chinese token efficiency problem?

---

# Week 4: Training a Tiny GPT

## Goal

Run through the first real language model training loop. **This week’s focus is close reading of nanoGPT code. It is not recommended to build everything completely from scratch.**

## Step 1: Watch Lectures

1. **CS336 Lecture 3: Architectures, hyperparameters**
2. **CS324: Modeling**
3. **CS324: Training**
4. **Karpathy: Let’s build GPT from scratch** (YouTube, the most important lecture)

## Step 2: Read Papers

Read closely:

1. **GPT-2**
   - Model section
   - Training dataset section
   - Zero-shot results

Skim:

1. **GPT-3**
   - Abstract + Introduction
   - Model and Architectures
   - Definitions of few-shot / one-shot / zero-shot

## Step 3: Code Task

**First read nanoGPT closely** ([github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)), and write comments explaining what each part does.

Then complete the following in your own `train.py`:

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

Minimal model configuration:

```python
n_layer = 2
n_head = 2
n_embd = 128
block_size = 128
batch_size = 32
```

Prefer using TinyStories, Shakespeare, or WikiText-2 as the dataset.

Deliverables: training loss curve + validation loss curve + generated text every 500 steps.

## Step 4: Summary

Write `03_Pretraining.md`, and answer:

```
1. What is next-token prediction predicting?
2. What is the relationship between cross entropy loss and perplexity?
3. Why is validation loss more important than training loss?
4. How does temperature affect generation?
5. What is the most obvious limitation of your tiny GPT right now?
```

## Acceptance Standard

> Can I train a tiny GPT, generate text, and explain what the loss curve shows?

---

# Week 5: Code Cleanup + CS336 Assignment 1 Standardization

## Goal

Clean up the code from Weeks 2–4 to near production standard. **This is an engineering week and does not introduce new concepts.**

## Step 1: Watch Lectures

1. **CS336 Assignment 1 handout** (read requirements closely)
2. **CS336 Lectures 1–3 rewatch** (fill gaps)
3. **CS336 Lecture 4: Attention alternatives and MoE** (only watch the first half)

## Step 2: Read Papers

Read closely:

1. **AdamW: Decoupled Weight Decay Regularization**
   - Abstract
   - Core difference between AdamW and Adam

Skim:

1. **Switch Transformer** (only need to understand what MoE is)

## Step 3: Code Task

Clean up the repo structure:

```
tiny-gpt-from-scratch/
  README.md          ← must be complete
  config.py          ← centralize all hyperparameters
  tokenizer.py
  model.py
  train.py
  generate.py
  requirements.txt
  notebooks/
    loss_curve.ipynb
    attention_demo.ipynb
```

Complete the following:

```python
# AdamW + learning rate schedule
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

README must include:

```
1. Problem (what problem this project solves)
2. Architecture (model structure)
3. Dataset (dataset description)
4. Training setup (hyperparameters)
5. Loss curve (screenshot)
6. Sample generations (generation examples)
7. Limitations (current limitations)
8. What I learned
```

## Step 4: Summary

Write:

```
Projects/tiny-gpt-from-scratch.md
```

This is the first project report that can be put on GitHub.

## Acceptance Standard

> Does my tiny-gpt-from-scratch repo have a complete README that lets others run it directly?

---

# Week 6: Inference / Decoding / KV Cache

> **⚠️ Adjustment note: Inference is moved up from the original Week 8 to Week 6.**
> Reason: Understanding generation (temperature, top-k, KV cache) is a prerequisite for understanding SFT experiment results. It should not be placed after alignment.

## Goal

Understand why LLM inference is expensive and how decoding strategies affect output quality.

## Step 1: Watch Lectures

1. **CS336 Lecture 10: Inference**
2. **CS324: Selective architectures** (understand SSM background)
3. **CS25: On the Tradeoffs of State Space Models and Transformers**

## Step 2: Read Papers

Read closely:

1. **FlashAttention**
   - Abstract + Introduction
   - Core idea of IO-aware attention (tiling, not the full mathematical proof)

2. **vLLM / PagedAttention**
   - Abstract + Introduction
   - KV cache fragmentation problem

Skim:

1. **Mamba**
   - Abstract + Introduction
   - Why it is a Transformer alternative

## Step 3: Code Task

Implement four decoding strategies in `tiny-gpt-from-scratch/generate.py`:

```python
def greedy_decode(model, prompt, max_new_tokens): ...
def temperature_sample(model, prompt, temperature): ...
def top_k_sample(model, prompt, k): ...
def top_p_sample(model, prompt, p): ...  # nucleus sampling
```

Experiment: use the same prompt `"Once upon a time"` and compare:

```
temperature = 0.3 / 0.8 / 1.2
top_k = 10 / 50
top_p = 0.9 / 0.95
```

Advanced: implement a simplified KV cache:

```python
# Without cache: recompute the entire prefix every time
# With cache: only compute K/V for the new token; past_kv accumulates
```

## Step 4: Summary

Write `08_Inference.md`, and answer:

```
1. Why does greedy decoding easily become repetitive and boring?
2. What does temperature control? Why does > 1 become chaotic and < 0.3 become repetitive?
3. What is the difference between top-k and top-p, and what scenarios are they suitable for?
4. Why can KV cache speed things up? What computation does it save?
5. What is the fundamental difference between the inference bottleneck and the training bottleneck?
```

## Acceptance Standard

> Can I explain the principle of KV cache and demonstrate how different decoding strategies affect generation results?

---

# Week 7: Scaling Laws

## Goal

Understand why "large models + large data + large compute" works, and why Chinchilla redefined the training paradigm.

## Step 1: Watch Lectures

Three sources, each with a different emphasis. They are not repetitive:

1. **CS336 Lecture 9: Scaling laws** (focus on Kaplan empirical scaling, the power-law relationship between model size and loss)
2. **CS336 Lecture 11: Scaling laws** (focus on Chinchilla compute-optimal training, how to allocate model size and data given compute)
3. **CS324: Scaling laws** (theoretical perspective + impact on practical training decisions)

## Step 2: Read Papers

Read closely:

1. **Scaling Laws for Neural Language Models** (Kaplan et al.)
   - Abstract + Introduction
   - scaling law formula
   - compute / data / model size triangle diagram

2. **Training Compute-Optimal Large Language Models** (Chinchilla)
   - Abstract + Introduction
   - Figure 1 (comparison of Chinchilla vs GPT-3 vs Gopher)
   - Conclusion: how to balance model size and token count under fixed compute

## Step 3: Code Task

Mini scaling experiment. Train 3 models and compare them on the same dataset:

```python
configs = {
    'small':  {'n_layer': 2, 'n_head': 2, 'n_embd': 128},
    'medium': {'n_layer': 4, 'n_head': 4, 'n_embd': 256},
    'large':  {'n_layer': 6, 'n_head': 6, 'n_embd': 384},
}
```

Record and plot:

```
x-axis: log(parameter count)
y-axis: validation loss
```

If compute is insufficient, only run small / medium.

## Step 4: Summary

Write `04_Scaling_Laws.md`, and answer:

```
1. What do scaling laws try to predict?
2. What is the relationship among parameter count, data volume, and compute?
3. Why did Chinchilla say many models were "undertrained on tokens"?
4. Why could the LLaMA series use smaller models to beat larger models?
5. What do scaling laws tell individual learners?
```

## Acceptance Standard

> Can I explain Chinchilla’s core conclusion and describe its impact on LLaMA?

---

# Week 8: Data-Centric LLM

## Goal

Use your Data Science background. Understand that data quality is the ceiling of LLM performance.

## Step 1: Watch Lectures

1. **CS336 Lecture 13: Data, sources, datasets**
2. **CS336 Lecture 14: Data filtering, deduplication, mixing, synthetic data**
3. **CS324: Data**

## Step 2: Read Papers

Read closely:

1. **LLaMA** (Meta, 2023)
   - Data section (data sources + filtering strategies)
   - Training section

Skim:

1. **The Pile** (understand the composition of pretraining data)
2. **FineWeb / RefinedWeb technical reports** (modern Common Crawl data cleaning paradigm)
3. **LLaMA 2** (Pretraining Data section)

## Step 3: Code Task

Create a new project `data-quality-for-language-modeling/`, and construct two datasets:

```
clean_dataset:   high-quality story / wiki / course text
dirty_dataset:   repeated text + garbage symbols + HTML remnants + random concatenation
```

Train three times with the same tiny GPT configuration:

```
A: clean dataset
B: dirty dataset
C: deduplicated dirty dataset (deduplicate B first)
```

Compare: valid loss / generated samples / repetition rate / garbled-text rate

This project reuses `tiny-gpt-from-scratch/train.py`; just replace the dataset.

## Step 4: Summary

Write:

```
05_Data.md
Projects/data-quality-for-language-modeling.md
```

Answer:

```
1. How does data quality directly affect validation loss?
2. What specific problems does duplicated data cause? Give the GPT-2 memorization example.
3. Why is deduplication standard in pretraining?
4. What is the risk of synthetic data (model collapse)?
5. What practical uses does a Data Science background have in LLM engineering?
```

## Acceptance Standard

> Can I use experimental data to prove that data quality affects loss and explain the necessity of deduplication?

---

# Week 9: PEFT / LoRA / QLoRA

> **⚠️ Newly added independent chapter.**
> The original route compressed LoRA into one sentence inside the SFT code task. For an AI engineering goal, this is a core skill that must be mastered independently.

## Goal

Understand the mathematical principle of LoRA, independently run the full LoRA fine-tuning process, and understand QLoRA’s quantization mechanism.

## Step 1: Watch Lectures

1. **CS336 Lecture 15: Mid/post-training, SFT/RLHF** (only watch the adaptation-related part)
2. **CS324: Adaptation** (watch in full)
3. Hugging Face PEFT documentation: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)

## Step 2: Read Papers

Read closely:

1. **LoRA: Low-Rank Adaptation of Large Language Models**
   - Abstract + Introduction
   - Section 4: LoRA method ($W = W_0 + \Delta W = W_0 + BA$)
   - Section 7: experimental results

Read closely:

2. **QLoRA: Efficient Finetuning of Quantized LLMs**
   - Abstract + Introduction
   - Intuition behind 4-bit NormalFloat quantization
   - How it combines with LoRA

## Step 3: Code Task

Create a new subdirectory `tiny-gpt-from-scratch/peft-experiments/`:

**Experiment 1: manually implement a LoRA layer**

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

**Experiment 2: use the `peft` library to perform LoRA SFT on a small model**

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
model.print_trainable_parameters()
# Compare: full FT parameter count vs. LoRA parameter count
```

Data: small Alpaca-style sample (1k–5k examples)  
Platform: Mac M3 Pro can run 0.5B; use Colab / Kaggle for larger models

Compare:

```
base model output (same prompt)
LoRA SFT output
full FT output (if compute allows)
```

## Step 4: Summary

Write `06_PEFT_SFT.md`, and answer:

```
1. What does low-rank decomposition in LoRA mean? Why does it work?
2. How do rank and alpha affect training results?
3. Why does LoRA save VRAM compared with full fine-tuning? (parameter count comparison)
4. What extra work does QLoRA do on top of LoRA?
5. Which layers should LoRA be applied to? (Why are q_proj and v_proj the first choices?)
```

## Acceptance Standard

> Can I hand-write a LoRA layer, explain the principle of low-rank decomposition, and complete a full LoRA SFT experiment?

---

# Week 10: Instruction Tuning / Full SFT Pipeline

## Goal

Understand why a base model is not a chat model, and master the engineering details of chat templates and instruction datasets.

## Step 1: Watch Lectures

1. **CS336 Lecture 15: Mid/post-training, SFT/RLHF** (watch in full)
2. **CS324: Adaptation** (rewatch and connect it with the code)

## Step 2: Read Papers

Read closely:

1. **InstructGPT: Training Language Models to Follow Instructions with Human Feedback**
   - Abstract + Introduction
   - SFT section (data format, training details)
   - Read the RLHF section conceptually first; no need to go deep into PPO yet

Skim:

1. **Self-Instruct**
2. **Alpaca** (understand the practical use of self-instruct)

## Step 3: Code Task

Based on the LoRA SFT from Week 9, expand it into a complete SFT pipeline:

```
sft-experiments/
  prepare_data.py    ← format instruction dataset
  train_sft.py       ← LoRA SFT training
  inference.py       ← compare base / sft outputs
  README.md
```

Chat template example:

```python
# Qwen2.5 chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain overfitting in one sentence."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

Required comparison experiment:

```
Same prompt, base model vs. SFT model. Record:
- instruction-following ability
- format regularity
- hallucination rate (subjective judgment)
```

## Step 4: Summary

Expand `06_PEFT_SFT.md`, and answer:

```
1. What is the essential difference between a base model and an instruct model?
2. What does an instruction dataset look like? (prompt / response pair)
3. What is a chat template? Why do different models use different formats?
4. Why can’t SFT fully solve alignment?
5. Which matters more for SFT data: quality or quantity?
```

## Acceptance Standard

> Can I explain the difference between a base model and an instruct model, and show output comparisons before and after SFT?

---

# Week 11: RLHF / DPO / Reasoning RL

## Goal

Understand the main line of modern LLM alignment, and why DPO is replacing traditional RLHF.

## Step 1: Watch Lectures

1. **CS336 Lecture 15: Mid/post-training, SFT/RLHF**
2. **CS336 Lecture 16: Post-training - RLVR**
3. **CS336 Assignment 5: Alignment and Reasoning RL** (read the handout and understand the task design)

## Step 2: Read Papers

Read closely:

1. **InstructGPT**
   - The three stages of RLHF (SFT → RM → PPO)
   - How the reward model is trained
   - PPO variants in language models

2. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**
   - Abstract + Introduction
   - How DPO simplifies RLHF (removes the explicit reward model)

Skim:

1. **Constitutional AI** (Anthropic)
2. **GRPO / DeepSeek-R1 technical report** (understand the current state of reasoning RL)

## Step 3: Code Task

**Option A (recommended): DPO toy experiment**

Construct a preference dataset:

```json
{
  "prompt": "Explain overfitting.",
  "chosen": "Overfitting occurs when a model learns noise in training data...",
  "rejected": "Overfitting means the model is too big."
}
```

Run DPO with `trl`:

```python
from trl import DPOTrainer, DPOConfig
trainer = DPOTrainer(model=model, args=config, train_dataset=dataset)
trainer.train()
```

**Option B: only analyze preference data**

If compute is insufficient, at least complete:

```
- parse preference dataset format
- compare chosen / rejected length distributions
- give examples of reward hacking
- draw the RLHF vs. DPO flowchart by hand
```

## Step 4: Summary

Write `07_Alignment.md`, and answer:

```
1. What do the three stages of RLHF do respectively?
2. What does the reward model learn?
3. What problem does PPO solve in language models?
4. Why is DPO simpler? What assumption does it make?
5. What new problems might RLHF / DPO introduce? (reward hacking, distribution shift)
```

## Acceptance Standard

> Can I draw the full SFT / RLHF / DPO flowchart and explain their relationship?

---

# Week 12: RAG / Context Retrieval

## Goal

Build a practical project and understand the essential difference between "parametric memory" and "context retrieval."

## Step 1: Watch Lectures

1. **CS25: Distinct Modes of Generalization from Parameters and Context**
2. **CS324: Adaptation** (retrieval-related section)
3. **CS336 Lecture 12: Evaluation** (first watch the task evaluation section)

## Step 2: Read Papers

Read closely:

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (original RAG paper)
   - Abstract + Introduction
   - Method: dense retrieval + generator structure

2. **ReAct: Synergizing Reasoning and Acting in Language Models**
   - Abstract + Introduction
   - reasoning + acting output format

Skim:

1. **Self-RAG**
2. **HyDE (Hypothetical Document Embeddings)**

## Step 3: Code Task

Create a new project `paper-rag-assistant/`:

```
paper-rag-assistant/
  ingest.py       ← read PDF, chunk, embed, store in vector database
  retrieval.py    ← query and return top-k chunks
  generate.py     ← assemble prompt, call API, output answer + sources
  evaluate.py     ← evaluate retrieval precision and answer quality
  README.md
```

Technical choices:

```python
# embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# vector store
import chromadb

# generation
import anthropic  # or openai
```

Data source: LLM paper PDFs you have already read

Experiment: compare the effects of three chunk sizes:

```
300 tokens / 600 tokens / 1000 tokens
```

Record: retrieval precision / answer quality / whether citations are accurate

## Step 4: Summary

Write:

```
09_RAG.md
Projects/paper-rag-assistant.md
```

Answer:

```
1. What problem does RAG solve? Why not directly fine-tune?
2. Why does chunk size matter? What are the problems when it is too large or too small?
3. What is the essential difference between dense retrieval and keyword search?
4. Why can hallucination still happen with RAG?
5. What scenarios are RAG and fine-tuning each suitable for?
```

## Acceptance Standard

> Can I build a RAG system that outputs answers + cited sources, and explain how chunk size affects results?

---

# Week 13: Agent / Tool Use

> **⚠️ Newly added chapter.**
> The original route had no Agent section after RAG, but Agent is a frequent topic for 2025–2026 AI engineering roles.

## Goal

Add an Agent layer on top of RAG: the model decides when to retrieve and when to call tools.

## Step 1: Watch Lectures / Read Documentation

1. **CS25: any lecture related to agent / tool use**
2. Anthropic Tool Use documentation: [docs.anthropic.com/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
3. LangGraph documentation (if you plan to use a framework): [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)

## Step 2: Read Papers

Read closely:

1. **ReAct** (second pass, this time focus on the code implementation format)

Skim:

1. **Toolformer**
2. **HuggingGPT / HuggingFace Agents** (understand multi-tool scenarios)

## Step 3: Code Task

Add an Agent layer inside `paper-rag-assistant/`:

```python
# Hand-write a simple ReAct loop (do not rely on a framework)
def react_loop(user_query, max_steps=5):
    messages = [{"role": "user", "content": user_query}]
    for step in range(max_steps):
        response = llm(messages, tools=[search_tool, calculator_tool])
        if response.stop_reason == "tool_use":
            tool_result = execute_tool(response.tool_call)
            messages.append({"role": "tool", "content": tool_result})
        else:
            return response.text  # final answer
```

Implement two tools:

```
search_tool    ← call your RAG retrieval
calculator_tool ← simple mathematical calculation
```

Scenario test:

```
"Who is the author of FlashAttention, and where does he work now?"
→ The Agent should first use search_tool, then answer using retrieval.
```

## Step 4: Summary

Write `10_Agent.md`, and answer:

```
1. What is ReAct’s Reason + Act loop?
2. What is the difference between function calling and prompt engineering?
3. How do tool use and RAG work together?
4. When will an agent loop get stuck in an infinite loop? How can it be prevented?
5. Single-agent vs. multi-agent: what scenarios are they each suitable for?
```

## Acceptance Standard

> Can I hand-write a ReAct loop that lets the model decide whether to call a search tool?

---

# Week 14: Evaluation + Deployment + Project Cleanup

## Goal

Move from "I have studied it" to "I have results I can show." **Add LLM-as-judge and local deployment practice.**

## Step 1: Watch Lectures

1. **CS336 Lecture 12: Evaluation**
2. **CS324: Capabilities**
3. **CS324: Harms I / Harms II** (bias and safety sections)

## Step 2: Read Papers / Benchmarks

Read closely:

1. **HELM** (evaluation framework)
   - accuracy / calibration / robustness / fairness / efficiency

2. **MMLU**
   - benchmark construction principle
   - why it is not equivalent to true intelligence

Skim:

1. **GSM8K**
2. **HumanEval**
3. **MT-Bench / AlpacaEval** (understand the LLM-as-judge paradigm)

## Step 3: Code Task

**Part 1: custom benchmark**

Test targets:

```
1. your tiny GPT
2. a base small model (Qwen2.5-0.5B base)
3. an instruct small model (Qwen2.5-0.5B instruct)
4. Claude / GPT API
```

Test set (20 questions):

```
5 CS concepts
5 mathematical reasoning
5 code generation
5 RAG Q&A (using your paper-rag-assistant)
```

**Part 2: LLM-as-judge (new)**

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

**Part 3: local deployment practice (new)**

Run a small model locally with Ollama:

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull and run
ollama run qwen2.5:0.5b
ollama run llama3.2:1b

# Python call
import ollama
response = ollama.chat(model='qwen2.5:0.5b', messages=[...])
```

Understand the differences among vLLM / Ollama / HuggingFace TGI, and which scenario each is suitable for.

## Step 4: Final Cleanup

Clean up three READMEs (each must be complete):

```
tiny-gpt-from-scratch/README.md
data-quality-for-language-modeling/README.md
paper-rag-assistant/README.md
```

Each README must include:

```
1. Problem (what problem it solves)
2. Method (method description)
3. Implementation (key implementation details)
4. Experiments (experiment setup)
5. Results (results + charts)
6. Failure cases (what did not work well)
7. What I learned
8. Future work
```

## Acceptance Standard

> Do I have 3 presentable GitHub projects, can I evaluate model outputs with LLM-as-judge, and can I run a model locally?

---

# Paper Close-Reading Priority

## Must Read Closely (in order)

```
1. Attention Is All You Need
2. GPT-2
3. GPT-3
4. Scaling Laws for Neural Language Models (Kaplan)
5. Chinchilla
6. FlashAttention
7. LLaMA
8. LoRA
9. QLoRA
10. InstructGPT
11. DPO
12. RAG (Lewis et al.)
13. ReAct
```

## Can Skim

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

# Weekly Acceptance Standards (Full Version)

| Week | Acceptance question |
| ---- | ------------------------------------------------------------ |
| W1   | Can I draw the Transformer pipeline by hand (token → embedding → attention → logits)? |
| W2   | Can I explain the shapes of Q/K/V and clearly describe the concatenation logic of multi-head attention? |
| W3   | Can I implement a simplified BPE and explain the Chinese token efficiency problem? |
| W4   | Can I train a tiny GPT, generate text, and explain the loss curve? |
| W5   | Does the tiny-gpt-from-scratch repo have a complete README that others can run directly? |
| W6   | Can I explain the principle of KV cache and demonstrate the effects of different decoding strategies? |
| W7   | Can I explain Chinchilla’s core conclusion and describe its impact on LLaMA? |
| W8   | Can I use experimental data to prove that data quality affects loss? |
| W9   | Can I hand-write a LoRA layer, explain the principle of low-rank decomposition, and complete a LoRA SFT experiment? |
| W10  | Can I show output comparisons before and after SFT, and explain the difference between base and instruct models? |
| W11  | Can I draw the full SFT / RLHF / DPO flowchart and explain their relationship? |
| W12  | Can I build a RAG system with citations and explain how chunk size affects results? |
| W13  | Can I hand-write a ReAct loop that lets the model decide whether to call search? |
| W14  | Do I have 3 complete GitHub projects and know how to evaluate with LLM-as-judge? |

---

# Global Map of the Learning Route

```
Week 0   Environment + Obsidian
  ↓
Week 1   Transformer intuition + attention demo
  ↓
Week 2   Transformer Block implementation
  ↓
Week 3   BPE Tokenizer
  ↓
Week 4   tiny GPT training (close reading of nanoGPT)
  ↓
Week 5   Code cleanup + CS336 standardization           → Project 1 completed: tiny-gpt-from-scratch
  ↓
Week 6   Inference / Decoding / KV Cache
  ↓
Week 7   Scaling Laws (Kaplan + Chinchilla)
  ↓
Week 8   Data-Centric LLM                 → Project 2 completed: data-quality
  ↓
Week 9   PEFT / LoRA / QLoRA (independent chapter)
  ↓
Week 10  Full SFT pipeline + chat template
  ↓
Week 11  RLHF / DPO / Reasoning RL
  ↓
Week 12  RAG + vector retrieval                   → Project 3 completed: paper-rag-assistant
  ↓
Week 13  Agent / Tool Use / ReAct
  ↓
Week 14  Evaluation + local deployment + project cleanup  → 3 GitHub projects
```

---

# Additional Tips for AI Engineering

The following are priority recommendations:

**Highest priority (frequent in interviews)**
- Full LoRA / QLoRA workflow (Week 9)
- Local inference deployment: Ollama / vLLM (Week 14)
- RAG pipeline engineering details (Week 12)
- Prompt engineering: few-shot, chain-of-thought, structured output

**Medium priority (understanding principles)**
- Scaling Laws (Week 7) — understanding is enough; no need to run large-scale experiments yourself
- RLHF / DPO (Week 11) — understand the pipeline and be able to explain the concepts

**Understand only (research direction, not engineering direction)**
- Fine-grained data filtering operations (the Week 8 experiment is enough)
- Implementing BPE from scratch (Week 3 is enough; in real work, use tiktoken)

**Recommended extra focus (not covered in the route but common in real work)**
- Structured output / JSON mode (Pydantic + instructor library)
- LangChain / LangGraph (engineering frameworks, optional after Week 13)
- Model quantization: GPTQ / AWQ / bitsandbytes
- Cost optimization: prompt caching, batch inference
