# 📖 读书笔记：*Build a Large Language Model (From Scratch)* — 第2章 Working with Text Data

> **书籍**：Sebastian Raschka《Build a Large Language Model (From Scratch)》  
> **仓库**：[rasbt/LLMs-from-scratch/tree/main/ch02/01_main-chapter-code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/01_main-chapter-code)  
> **核心主题**：文本数据预处理——从原始文本到模型可用的张量输入

---

## 🗺️ 章节总览

本章解决一个核心问题：**如何将原始文本转化为 LLM 可以处理的数值输入？**

整体流程如下：
```

原始文本 → 分词(Tokenization) → Token IDs → Token Embedding → + Positional Embedding → LLM 输入
```


---

## 1️⃣ 文本分词（Tokenization）

### 1.1 基于正则的简单分词

第一步是将文本拆分为 token。书中使用正则表达式进行初步分词：
```
python
import re

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```
**要点**：
- 标点符号被独立为单独的 token
- 空白符作为分隔但自身不保留
- `--` 被整体视为一个 token

### 1.2 构建词汇表（Vocabulary）

将所有唯一 token 排序并映射为整数 ID：
```
python
all_words = sorted(set(preprocessed))
vocab = {token: integer for integer, token in enumerate(all_words)}
```
---

## 2️⃣ SimpleTokenizerV1 —— 最简版分词器
```
python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```
**局限性**：遇到不在词汇表中的词（OOV）会直接报错。

---

## 3️⃣ SimpleTokenizerV2 —— 加入特殊 Token

为了处理未知词和文本分隔，引入两个特殊 token：

| 特殊 Token | 用途 |
|---|---|
| `<\|unk\|>` | 代替所有词汇表外的未知词 |
| `<\|endoftext\|>` | 标记文档/文本的结束边界 |
```
python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```
**💡 关键认识**：SimpleTokenizer 仅用于教学，实际 LLM 使用更先进的子词分词算法。

---

## 4️⃣ Byte Pair Encoding（BPE）—— 实际使用的分词算法

GPT-2/3/4 使用的是 **BPE（字节对编码）** 分词器。

### 核心思想

- 将罕见词拆分为更小的**子词单元**（subword units）
- 常见词保持完整，稀有词被拆为子字符串
- 彻底解决 OOV（词汇表外）问题

### 使用 tiktoken（OpenAI 的 BPE 实现）
```
python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114]

strings = tokenizer.decode(integers)
print(strings)
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
```
**GPT-2 BPE 词汇表大小**：50,257 个 token

---

## 5️⃣ 数据采样与滑动窗口（Sliding Window）

### 核心概念：输入-目标对

LLM 的训练任务是 **next token prediction**（预测下一个 token）：
```

输入:   [A, B, C, D]
目标:   [B, C, D, E]
```
每个位置的目标都是对应输入位置的**下一个** token。

### GPTDatasetV1 —— 自定义 PyTorch Dataset
```
python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```
### create_dataloader_v1 —— 数据加载器工厂函数
```
python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
```
**关键参数解读**：

| 参数 | 含义 |
|---|---|
| `max_length` | 每个样本的上下文窗口长度（即 context size） |
| `stride` | 滑动步长；`stride < max_length` 时窗口有重叠 |
| `drop_last` | 丢弃最后不满一个 batch 的数据 |

**💡 stride vs max_length**：
- `stride = max_length`：窗口无重叠，最大化数据利用效率
- `stride = 1`：最大重叠，数据量最大但冗余高

---

## 6️⃣ Token Embedding 与 Positional Embedding

### 6.1 Token Embedding

将每个 token ID 映射为一个连续向量：
```
python
import torch

vocab_size = 50257
output_dim = 256  # embedding 维度

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(torch.tensor([token_ids]))
print(token_embeddings.shape)  # (1, seq_len, 256)
```
**本质**：`nn.Embedding` 就是一个**可训练的查找表**（lookup table），维度为 `(vocab_size, embedding_dim)`。

### 6.2 Positional Embedding

Transformer 没有 RNN 的顺序结构，需要**显式注入位置信息**：
```
python
context_length = 1024  # GPT-2 最大上下文长度
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(
    torch.arange(context_length)  # [0, 1, 2, ..., 1023]
)
```
### 6.3 最终输入 = Token Embedding + Positional Embedding
```
python
input_embeddings = token_embeddings + pos_embeddings
```
**GPT-2 参数参考**：

| 模型 | 参数量 | Embedding 维度 | 上下文长度 |
|---|---|---|---|
| GPT-2 Small | 124M | 768 | 1024 |
| GPT-2 Medium | 355M | 1024 | 1024 |
| GPT-2 Large | 774M | 1280 | 1024 |
| GPT-3 | 175B | 12288 | 2048 |

---

## 🧠 核心概念总结
```

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  原始文本     │───▶│   分词器      │───▶│  Token IDs   │
│ "Hello world" │    │ (BPE/tiktoken)│    │ [15496, 995] │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐            ▼
                    │ Positional   │    ┌──────────────┐
                    │ Embedding    │───▶│              │
                    │ pos(0,1,...) │  + │  最终输入     │
                    └──────────────┘    │  Embedding   │
                                       │ (batch, seq, │
                    ┌──────────────┐    │  embed_dim)  │
                    │ Token        │───▶│              │
                    │ Embedding    │    └──────────────┘
                    │ lookup table │
                    └──────────────┘
```
---

## 📌 关键收获

1. **分词不是切词那么简单** —— BPE 子词分词是 LLM 的工业标准，它在词汇表大小和表达能力之间取得了平衡。

2. **Embedding 是可训练的** —— 不是固定的 Word2Vec，而是随模型一起学习的参数，这样 embedding 可以针对下游任务最优化。

3. **位置编码必不可少** —— Transformer 是并行处理 token 的，没有内建的顺序概念，Positional Embedding 弥补了这一点。

4. **滑动窗口数据采样** —— 通过 `stride` 控制训练样本的重叠度，`max_length` 定义上下文窗口大小。

5. **Next Token Prediction** —— LLM 预训练的基本范式：给定 `[t₁, t₂, ..., tₙ]`，预测 `[t₂, t₃, ..., tₙ₊₁]`。

---


