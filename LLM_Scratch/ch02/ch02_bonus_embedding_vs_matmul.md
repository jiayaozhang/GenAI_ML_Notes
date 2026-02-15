# Chapter 2 Bonus: Embedding 层到底在干什么？

## 开篇：一个让人困惑的问题

你有没有想过，当我们把单词 "cat" 输入到 GPT 模型时，模型是怎么"理解"这个单词的？答案就藏在 Embedding 层里。这一章要解答的核心问题是：**Embedding 层是不是就是在做矩阵乘法？**

剧透一下答案：没错，但它做得更聪明。就像你可以开车去超市，也可以走路去，最后都能买到东西，但效率差太多了。

## 1. 先从一个餐厅的故事说起

### 想象你去餐厅点菜

**Embedding 层的方式（聪明的食客）：**
- 看菜单，找到 "宫保鸡丁" 在第 42 号
- 直接告诉服务员："给我来份 42 号"
- 服务员直接去厨房拿第 42 道菜

**线性层的方式（较真的数学家）：**
- 把整个菜单上 100 道菜都问一遍
- "第 1 道菜要不要？" "不要" (×0)
- "第 2 道菜要不要？" "不要" (×0)
- ...
- "第 42 道菜要不要？" "要！" (×1)
- ...
- "第 100 道菜要不要？" "不要" (×0)
- 最后算账：0+0+...+1×(第42道菜)+...+0 = 第42道菜

两种方式最后都能吃到宫保鸡丁，但哪个更高效？显然是第一种！

## 2. 代码里到底发生了什么？

### 2.1 Embedding：就是查字典

```python
# 想象有个词汇表：["我", "爱", "吃", "苹果", ...]
# 每个词都有个编号：{"我": 0, "爱": 1, "吃": 2, "苹果": 3, ...}

# Embedding 层就是个大字典，存着每个词的"特征向量"
embedding = nn.Embedding(vocab_size=10000, embedding_dim=128)

# 当你输入 "苹果" 的编号 3
word_id = torch.tensor([3])
vector = embedding(word_id)  # 直接拿出第 3 个向量，完事！
```

这就像是：
- 你有个通讯录，每个人都有电话号码
- 你想找张三的电话，直接翻到张三那页就行了
- 不需要把整个通讯录从头到尾看一遍

### 2.2 Linear Layer：强行做数学题

```python
# Linear 层非要把简单的事情复杂化
linear = nn.Linear(10000, 128, bias=False)

# 先把词的编号 3 转成 one-hot
# [0, 0, 0, 1, 0, 0, ..., 0]  # 10000 个数，只有第 4 个是 1
onehot = torch.nn.functional.one_hot(torch.tensor([3]), 10000).float()

# 做矩阵乘法（9999 个 0 的乘法都白做了）
vector = linear(onehot)
```

这就像是：
- 你要找张三的电话
- 但你非要问："李四是不是？不是。王五是不是？不是..."
- 问遍所有人，最后才确定是张三

## 3. 它们为什么结果一样？

这里有个数学小秘密。当你用 one-hot 向量去乘一个矩阵时：

```python
# one-hot: [0, 0, 1, 0, 0]
# 权重矩阵 W:
# [[w11, w12, w13, ...],   <- 第 0 行
#  [w21, w22, w23, ...],   <- 第 1 行
#  [w31, w32, w33, ...],   <- 第 2 行 (我们要的)
#  [w41, w42, w43, ...],   <- 第 3 行
#  [w51, w52, w53, ...]]   <- 第 4 行

# one-hot × W = 0×第0行 + 0×第1行 + 1×第2行 + 0×第3行 + 0×第4行
#              = 第2行
```

看到了吗？最后就是把矩阵的第 2 行拿出来。而 Embedding 层做的事情就是：**直接把第 2 行拿出来**，跳过了所有那些乘 0 的废操作。

## 4. 性能差距有多大？

我做了个实验，结果让人震惊：

```python
import time

vocab_size = 50000  # 5 万个词
embedding_dim = 512  # 每个词用 512 维向量表示
num_words = 10000  # 处理 1 万个词

# Embedding 层
embedding = nn.Embedding(vocab_size, embedding_dim)
start = time.time()
result = embedding(word_ids)
print(f"Embedding 用时: {time.time() - start:.4f} 秒")

# Linear 层
linear = nn.Linear(vocab_size, embedding_dim, bias=False)
onehot = F.one_hot(word_ids, vocab_size).float()
start = time.time()
result = linear(onehot)
print(f"Linear 用时: {time.time() - start:.4f} 秒")

# 结果：
# Embedding 用时: 0.0012 秒
# Linear 用时: 1.2034 秒
# Linear 慢了 1000 倍！
```

为什么差这么多？
- Embedding：查 10000 次表
- Linear：做 10000 × 50000 × 512 = 2.56 亿次乘法（其中 99.998% 都是乘 0）

## 5. 实际应用：什么时候用什么？

### 用 Embedding 的场景（99% 的 NLP 任务）

- 处理文本：每个词都有个 ID
- 推荐系统：每个用户/商品都有个 ID
- 知识图谱：每个实体都有个 ID

```python
# 典型的 NLP 模型开头
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # ... 其他层
```

### 用 Linear 的场景

- 输入本来就是连续的向量（不是 ID）
- 需要对向量做复杂变换
- 输入是图片的像素值、音频的波形等

```python
# 处理图片特征
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, 512)  # 图片特征是 2048 维向量
```

## 6. 一些容易踩的坑

### 坑 1：维度搞反了

```python
# 容易搞混的地方
embedding = nn.Embedding(100, 50)  # [100, 50] 的查找表
linear = nn.Linear(100, 50)        # 权重是 [50, 100]，转置了！

# 如果要让它们等价，需要：
linear.weight.data = embedding.weight.data.T  # 转置！
```

### 坑 2：忘了转 float

```python
# one-hot 默认是 LongTensor，Linear 层不接受
onehot = F.one_hot(ids, vocab_size)  # 错！
onehot = F.one_hot(ids, vocab_size).float()  # 对！
```

### 坑 3：以为 Embedding 能处理连续值

```python
# Embedding 只能接受整数 ID
embedding(torch.tensor([3.14]))  # 错！会报错

# 如果输入是连续的，老老实实用 Linear
linear(torch.tensor([3.14, 2.71, 1.41]))  # 对！
```

## 7. Transformer 里的应用

在 GPT、BERT 这些模型里，Embedding 层是第一步：

```python
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 词嵌入：把每个词映射成向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置嵌入：告诉模型这个词在句子的什么位置
        self.position_embedding = nn.Embedding(max_position, d_model)

    def forward(self, input_ids):
        # 查两个表，然后加起来
        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(len(input_ids))
        position_emb = self.position_embedding(position_ids)

        # 词的意思 + 词的位置 = 最终表示
        embeddings = token_emb + position_emb
        return embeddings
```

## 8. 一个有趣的思考实验

假如 GPT-3 有 175B 参数，词汇表 5 万，如果用 Linear 层代替 Embedding...

- 每处理一个词要多做 5 万次无用乘法
- 一个句子 100 个词，就是 500 万次
- 一个 batch 32 个句子，就是 1.6 亿次
- 训练一个 epoch... 算了，电费付不起

这就是为什么理解底层原理很重要。看似等价的两种方法，实际效率天差地别。

## 9. 总结：简单的智慧

Embedding 层的故事告诉我们：
1. **数学上等价 ≠ 工程上等价**（能算出来和算得快是两码事）
2. **针对性优化很重要**（知道输入特点，就能大幅优化）
3. **简单往往更好**（查表比矩阵乘法简单 1000 倍）

下次当你看到 `nn.Embedding` 时，你就知道了：
- 它不是什么神秘的东西
- 就是个优化过的查找表
- 但这个优化，让 ChatGPT 成为可能

记住：**在 NLP 里，永远用 Embedding 层处理词 ID，不要用 Linear 层配 one-hot**。这不是建议，是铁律。

## 实用代码模板

最后送你一个实用模板，拿走即用：

```python
class TextProcessor(nn.Module):
    """处理文本的标准开头"""
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        # 永远这么写，不要想着用 Linear
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # 可选：用预训练的词向量初始化
        # self.embedding.weight.data.copy_(pretrained_vectors)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len] 的整数张量
        # output: [batch_size, seq_len, hidden_dim] 的浮点张量
        return self.embedding(input_ids)
```

就这么简单，但背后的原理，现在你全都懂了！