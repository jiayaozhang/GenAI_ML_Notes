# Chapter 3: Attention Mechanisms - 让模型学会"专注"

## 开篇：一个课堂的比喻

想象你在教室里，老师说："明天的考试重点是第三章和第五章。"

这时你的大脑会做什么？
- 自动给"明天"、"考试"、"第三章"、"第五章"打上重点标记
- 忽略"的"、"是"、"和"这些不重要的词
- 把"考试"和"第三章、第五章"联系起来

这就是 **Attention（注意力）机制**！让模型像人脑一样，知道该关注什么，忽略什么。

## 1. Self-Attention：让词语互相"打分"

### 核心思想：每个词都要看看其他词

想象一个句子："小明喜欢他的猫"

当模型处理"他"这个词时：
- 需要知道"他"指的是谁？
- 通过注意力机制，"他"会更关注"小明"
- "他"和"小明"的关联分数会很高

### 用代码说话

```python
import torch
import torch.nn as nn

# 假设我们有个句子，6个词，每个词用8维向量表示
sentence = torch.randn(6, 8)  # [句子长度, 嵌入维度]

# Self-Attention 的三个主角
d_in = 8   # 输入维度
d_out = 8  # 输出维度

# 创建三个变换矩阵（这就是要学习的参数）
W_query = nn.Linear(d_in, d_out, bias=False)  # 生成查询向量
W_key = nn.Linear(d_in, d_out, bias=False)    # 生成键向量
W_value = nn.Linear(d_in, d_out, bias=False)  # 生成值向量
```

### Q、K、V 是什么？用恋爱来理解

把 Attention 想象成相亲大会：

- **Query（查询）**：你想找什么样的人？
- **Key（键）**：每个人的特征是什么？
- **Value（值）**：这个人的实际信息

```python
# 每个词都变成 Q、K、V 三种身份
queries = W_query(sentence)  # "我想找什么"
keys = W_key(sentence)        # "我是什么样"
values = W_value(sentence)    # "我的实际内容"

# 计算匹配度（点积 = 相似度）
attention_scores = queries @ keys.T  # @ 是矩阵乘法
# 结果是 6×6 的矩阵，表示每个词对每个词的关注度

print(attention_scores.shape)  # [6, 6]
```

## 2. Attention 的计算步骤

### Step 1: 计算注意力分数

```python
# 假设处理句子 "The cat sat"
# 每个词都要给其他词打分

def compute_attention_simple(queries, keys, values):
    # 1. 计算所有词对之间的相似度
    attention_scores = queries @ keys.T

    # 看看 "cat" 对其他词的注意力分数
    # attention_scores[1] = [0.5, 2.0, 0.3]
    #                       [The, cat, sat]
    # "cat" 最关注自己（2.0分）

    return attention_scores
```

### Step 2: 归一化（Softmax）

```python
def compute_attention_with_softmax(queries, keys, values):
    # 计算原始分数
    attention_scores = queries @ keys.T

    # 问题：分数可能很大或很小，不好比较
    # 解决：用 softmax 转换成概率（和为1）
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # 现在每行的和都是 1
    # [0.2, 0.7, 0.1] <- "cat" 70%关注自己，20%关注"The"，10%关注"sat"

    return attention_weights
```

### Step 3: 缩放（为什么要除以 √d_k？）

```python
def scaled_dot_product_attention(queries, keys, values):
    d_k = keys.shape[-1]  # 键向量的维度

    # 为什么要缩放？
    # 当维度很大时，点积结果会很大
    # softmax 对大数值会产生极端的概率分布（接近0或1）
    # 除以 √d_k 让分布更平滑

    attention_scores = queries @ keys.T
    attention_scores = attention_scores / (d_k ** 0.5)  # 缩放！

    attention_weights = torch.softmax(attention_scores, dim=-1)

    # 用注意力权重加权求和
    context_vectors = attention_weights @ values

    return context_vectors, attention_weights
```

## 3. Causal Attention：不能"偷看"未来

### 问题：生成文本时的作弊

想象你在做完形填空：
```
"我今天去了____，买了苹果"
```

如果模型能看到"苹果"，就知道空格应该填"超市"或"水果店"。
但这是作弊！生成文本时，模型不应该看到后面的内容。

### 解决方案：因果掩码（Causal Mask）

```python
def create_causal_mask(size):
    """创建一个下三角掩码"""
    # 只能看到当前位置及之前的内容
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    # triu = 上三角，diagonal=1 表示主对角线上方

    # mask 看起来像这样：
    # [[0, 1, 1, 1],
    #  [0, 0, 1, 1],
    #  [0, 0, 0, 1],
    #  [0, 0, 0, 0]]

    # 0 = 可以看，1 = 不能看
    return mask.bool()

def causal_attention(queries, keys, values):
    seq_len = queries.shape[0]
    d_k = keys.shape[-1]

    # 计算注意力分数
    scores = queries @ keys.T / (d_k ** 0.5)

    # 应用因果掩码
    mask = create_causal_mask(seq_len)
    scores = scores.masked_fill(mask, float('-inf'))
    # -inf 经过 softmax 会变成 0，完美屏蔽！

    # Softmax
    weights = torch.softmax(scores, dim=-1)

    # 看看第一个词能关注谁
    print("第1个词的注意力:", weights[0])
    # [1.0, 0.0, 0.0, 0.0] <- 只能看自己！

    print("第3个词的注意力:", weights[2])
    # [0.3, 0.5, 0.2, 0.0] <- 能看前3个，不能看第4个

    # 计算输出
    output = weights @ values
    return output, weights
```

## 4. Multi-Head Attention：多个角度看问题

### 为什么需要多头？

单头注意力就像用一个角度看问题。多头注意力就像：
- 一个头关注语法关系
- 一个头关注语义相似性
- 一个头关注位置关系
- ...

### 实现多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length):
        super().__init__()

        # 确保能平均分配给各个头
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 为所有头创建 Q、K、V 投影
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

        # 最后的输出投影
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout（防止过拟合）
        self.dropout = nn.Dropout(0.1)

        # 因果掩码
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                      diagonal=1)
        )

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape

        # 1. 生成 Q、K、V
        queries = self.W_query(x)  # [batch, seq, d_out]
        keys = self.W_key(x)
        values = self.W_value(x)

        # 2. 重塑成多个头
        # [batch, seq, d_out] -> [batch, seq, num_heads, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. 转置以便并行计算各个头
        # [batch, num_heads, seq, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. 计算注意力
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # 5. 应用因果掩码
        mask_expanded = self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask_expanded.bool(), float('-inf'))

        # 6. Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. 加权求和
        context = attn_weights @ values  # [batch, heads, seq, head_dim]

        # 8. 合并所有头
        context = context.transpose(1, 2)  # [batch, seq, heads, head_dim]
        context = context.reshape(batch_size, seq_len, self.d_out)

        # 9. 最终投影
        output = self.out_proj(context)

        return output
```

### 使用示例

```python
# 创建多头注意力层
mha = MultiHeadAttention(
    d_in=256,           # 输入维度
    d_out=256,          # 输出维度
    num_heads=8,        # 8个注意力头
    context_length=100  # 最大序列长度
)

# 输入：批次大小=2，序列长度=10，特征维度=256
x = torch.randn(2, 10, 256)

# 前向传播
output = mha(x)
print(output.shape)  # [2, 10, 256]

# 每个头看到的维度：256 / 8 = 32维
```

## 5. 实际效果：Attention 到底在关注什么？

### 可视化注意力权重

```python
def visualize_attention(sentence, attention_weights):
    """可视化注意力矩阵"""
    import matplotlib.pyplot as plt

    words = sentence.split()

    plt.figure(figsize=(8, 8))
    plt.imshow(attention_weights, cmap='Blues')
    plt.colorbar()

    # 设置坐标轴
    plt.xticks(range(len(words)), words, rotation=45)
    plt.yticks(range(len(words)), words)

    plt.xlabel('被关注的词')
    plt.ylabel('当前词')
    plt.title('注意力权重可视化')

    # 添加数值
    for i in range(len(words)):
        for j in range(len(words)):
            plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                    ha='center', va='center')

    plt.show()

# 示例
sentence = "The cat sat on the mat"
# 假设这是计算出的注意力权重
attention = torch.tensor([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  # "The" 主要看自己
    [0.2, 0.7, 0.1, 0.0, 0.0, 0.0],  # "cat" 主要看自己
    [0.1, 0.3, 0.5, 0.1, 0.0, 0.0],  # "sat" 看"cat"和自己
    [0.0, 0.1, 0.2, 0.6, 0.1, 0.0],  # "on" 主要看自己
    [0.1, 0.2, 0.1, 0.1, 0.4, 0.1],  # "the" 分散注意力
    [0.0, 0.3, 0.2, 0.1, 0.1, 0.3],  # "mat" 看"cat"和自己
])

visualize_attention(sentence, attention)
```

## 6. 位置编码：告诉模型词的顺序

### 为什么需要位置信息？

Attention 机制本身不知道词的顺序！
```
"猫追狗" vs "狗追猫"
```
如果没有位置信息，这两个句子对模型来说是一样的！

### 解决方案：位置嵌入

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 方法1：学习位置嵌入（GPT风格）
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]

        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)

        # 获取位置嵌入
        pos_emb = self.pos_embedding(positions)

        # 加到输入上
        return x + pos_emb

# 使用
pos_encoder = PositionalEncoding(d_model=256)
x_with_pos = pos_encoder(x)
```

## 7. 训练技巧和注意事项

### Dropout：防止过拟合

```python
# 在注意力权重上应用 dropout
attn_weights = torch.softmax(scores, dim=-1)
attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
# 训练时随机将 10% 的注意力权重设为 0
```

### 梯度裁剪：防止梯度爆炸

```python
# 训练循环中
optimizer.zero_grad()
loss.backward()

# 裁剪梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### 学习率预热

```python
def get_lr(step, d_model, warmup_steps):
    """Transformer 论文中的学习率调度"""
    lr = d_model ** (-0.5)
    lr = lr * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return lr
```

## 8. 常见问题解答

### Q1: 为什么叫"自注意力"？

因为序列在关注**自己**的其他部分。不像 CNN 看固定窗口，也不像 RNN 按顺序处理，而是让每个位置都能看到所有位置。

### Q2: 计算复杂度是多少？

- 时间复杂度：O(n² × d)，n 是序列长度，d 是维度
- 这就是为什么 GPT 对长文本会变慢

### Q3: 多少个头比较好？

- GPT-2：12 个头
- GPT-3：96 个头（大模型）
- 一般原则：模型越大，头越多

## 9. 完整示例：构建一个迷你 Transformer 块

```python
class TransformerBlock(nn.Module):
    """一个完整的 Transformer 块"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 多头注意力
        self.attention = MultiHeadAttention(
            d_model, d_model, num_heads, context_length=100
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 自注意力（带残差连接）
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)

        # 2. 前馈网络（带残差连接）
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x

# 使用
transformer_block = TransformerBlock(
    d_model=256,
    num_heads=8,
    d_ff=1024
)

# 输入
x = torch.randn(2, 10, 256)  # [batch, seq, features]

# 输出
output = transformer_block(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # 形状不变！
```

## 总结：Attention 的魔力

Attention 机制就像给模型装上了"眼睛"：
1. **看得全**：每个词都能看到所有词
2. **有重点**：通过权重决定关注什么
3. **够灵活**：多头机制从多角度理解
4. **防作弊**：因果掩码确保生成的合理性

记住核心公式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

这个简单的公式，撑起了整个 ChatGPT 的脊梁！

下一步：把多个 Transformer 块堆叠起来，就能构建真正的 GPT 了！🚀