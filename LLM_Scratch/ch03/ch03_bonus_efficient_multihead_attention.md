# Chapter 3 Bonus：高效多头注意力实现对比

> 📖 本笔记基于 [LLMs-from-scratch / ch03 / 02_bonus_efficient-multihead-attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention)

## 开篇：同一件事，九种做法

上一节我们学了多头注意力（MHA）的原理。但在实际工程中，**同样的数学公式可以有很多种代码写法**，它们在速度上天差地别。

打个比方：从家到公司，你可以走路、骑车、开车、坐地铁……目的地一样，但效率完全不同。

本笔记对比了 **9 种** 实现因果多头注意力的方式，从最朴素的写法到 PyTorch 最新的 FlashAttention / FlexAttention。

---

## 实验设置

```python
batch_size = 8
context_len = 1024
embed_dim = 768       # GPT-2 小尺寸的参数
num_heads = 12        # 12 个注意力头 → 每头 64 维
```

所有实现接收相同的随机 embeddings 张量 `(8, 1024, 768)`，输出也是 `(8, 1024, 768)`。

---

## 实现 ①：Wrapper 包装类（最直观，最慢）

### 思路

把第 3 章的 **单头** `CausalAttention` 类复制 `num_heads` 份，放进 `nn.ModuleList`，各自独立计算，最后把结果拼起来。

```python
class Ch03_MHA_Wrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]        # 创建 num_heads 个独立的注意力头
        )
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        # 每个头单独算，最后 cat 拼接
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)
```

### 🐢 为什么慢？

- 每个头是一个 **独立的 Python 循环迭代**，GPU 无法并行
- 12 个头 = 调用 12 次 `forward`，每次都有 kernel launch 开销
- 就像请 12 个厨师**轮流**做菜，而不是同时做

---

## 实现 ②：Ch03 原版 MHA（统一矩阵 + reshape 拆头）

### 思路

用一个大的 `W_query`、`W_key`、`W_value` 矩阵一次性投影，然后通过 **reshape + transpose** 把结果拆成多个头。

```python
# 关键步骤：
keys = self.W_key(x)                     # (b, num_tokens, d_out)
keys = keys.view(b, num_tokens, num_heads, head_dim)  # 拆头
keys = keys.transpose(1, 2)              # (b, num_heads, num_tokens, head_dim)
```

### 🔑 核心改进

| 对比项 | Wrapper 写法 | Ch03 写法 |
|--------|-------------|-----------|
| 权重矩阵 | 每头独立 `nn.Linear` | 一个大矩阵，一次乘法 |
| GPU 并行 | ❌ 循环串行 | ✅ 一次性计算所有头 |
| 拆头方式 | 物理分离（多个类实例） | 逻辑分离（view/reshape） |

> **类比**：不再让 12 个厨师轮流用灶台，而是给他们一个 12 口灶的大灶台，同时开火。

---

## 实现 ③：合并 QKV 权重矩阵（进一步减少矩阵乘法次数）

### 思路

把 Q、K、V 三个投影矩阵**合并成一个** `nn.Linear(d_in, 3 * d_out)`。

```python
self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)  # 一次算完 Q, K, V

def forward(self, x):
    qkv = self.qkv(x)  # (b, n, 3 * d_out)  -- 一次矩阵乘法搞定三个投影

    # 拆分并重排
    qkv = qkv.view(batch_size, num_tokens, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, b, heads, n, head_dim)
    queries, keys, values = qkv.unbind(0)  # 分出 Q, K, V
```

### 🚀 为什么更快？

- **3 次矩阵乘法 → 1 次**：合并权重后只需一次 `x @ W_qkv`
- 减少了 GPU kernel 启动次数
- 这是 **大多数生产级 Transformer 库**（如 HuggingFace）的标准做法

---

## 实现 ④：Einsum（爱因斯坦求和）

### 思路

用 `torch.einsum` 以更数学化的方式写矩阵运算：

```python
Q = torch.einsum("bnd,do->bno", x, self.W_query)   # 等价于 x @ W_query
scores = torch.einsum("bhnd,bhmd->bhnm", Q, K)      # 等价于 Q @ K.T
context = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)  # 加权求和
```

### 🎯 优缺点

| 优点 | 缺点 |
|------|------|
| 代码即公式，数学含义清晰 | 对不熟悉 einsum 的人来说可读性差 |
| 灵活处理任意维度 | 编译器优化可能不如显式写法好 |

> **类比**：就像用数学符号写解题过程 vs 用白话文写——前者简洁但需要数学基础。

---

## 实现 ⑤：PyTorch `scaled_dot_product_attention` + FlashAttention ⭐

### 思路

用 PyTorch 内置的 `F.scaled_dot_product_attention()`，它会自动调用 **FlashAttention** 内核。

```python
context_vec = nn.functional.scaled_dot_product_attention(
    queries, keys, values,
    attn_mask=None,
    dropout_p=use_dropout,
    is_causal=True       # ← 告诉它做因果注意力
)
```

### ⚡ FlashAttention 为什么这么快？

传统注意力的瓶颈不是计算，而是**内存搬运**：

```
传统做法               FlashAttention
──────────             ──────────────
1. 算 QK^T (n×n)      整体用 tiling（分块）策略
2. 存到 GPU 显存       在 SRAM（快速缓存）里
3. 读回来做 softmax    边算边做 softmax
4. 存 softmax 结果     从不生成完整的 n×n 矩阵
5. 读回来 × V
```

| 对比项 | 传统注意力 | FlashAttention |
|--------|-----------|----------------|
| 显存 | O(n²)——存完整注意力矩阵 | O(n)——分块处理 |
| 速度 | 受限于 HBM 带宽 | 利用更快的 SRAM |
| 数值精度 | 标准 | online softmax，一样精确 |

> **类比**：传统方式像把所有食材摆在大桌上再做菜，FlashAttention 像小灶台上来一点做一点，反而更快。

---

## 实现 ⑥：SDPA 但**不用** FlashAttention

### 思路

传入显式的 `attn_mask`（而不是 `is_causal=True`），这会阻止 FlashAttention 的激活：

```python
attn_mask = self.mask[:num_tokens, :num_tokens]  # 显式因果掩码

context_vec = nn.functional.scaled_dot_product_attention(
    queries, keys, values,
    attn_mask=attn_mask,    # ← 显式掩码
    dropout_p=use_dropout,
    is_causal=False          # ← 不使用内置因果优化
)
```

### 🤔 什么时候需要这样做？

- 需要**非因果**的自定义掩码（如稀疏注意力、局部注意力）
- 调试时想对比 FlashAttention 的等价性
- 速度会比 FlashAttention 版本慢

---

## 实现 ⑦：`nn.MultiheadAttention`（PyTorch 官方类）

### 思路

直接用 PyTorch 提供的 `nn.MultiheadAttention` 类：

```python
self.multihead_attn = nn.MultiheadAttention(
    embed_dim=d_out,
    num_heads=num_heads,
    dropout=dropout,
    bias=qkv_bias,
    batch_first=True,   # 输入格式 (batch, seq, dim)
)

attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
```

### 📝 注意事项

- 默认 `need_weights=True`，会返回注意力权重——方便可视化，但**会拖慢速度**
- 参数 `(x, x, x)` 表示自注意力（Q=K=V 来源相同）

---

## 实现 ⑧：`nn.MultiheadAttention` + `need_weights=False`

### 关键改动

```python
MHAPyTorchClass(
    ...,
    need_weights=False   # ← 不需要返回注意力权重
)
```

设置 `need_weights=False` 后，PyTorch 内部会自动使用 `scaled_dot_product_attention`，从而获得 FlashAttention 加速。

> **一个参数带来质的飞跃**：只是关掉了注意力权重的输出，速度就大幅提升。

---

## 实现 ⑨：FlexAttention（PyTorch 2.5+ 新功能）

### 思路

FlexAttention 允许你用**简单的 Python 函数定义注意力掩码模式**，然后自动编译成高效 CUDA 内核：

```python
# 用一个简单函数定义因果掩码
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx    # query 位置 >= key 位置才允许

# 编译成高效的块状掩码
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=n, KV_LEN=n)

# 使用 FlexAttention
context_vec = flex_attention(queries, keys, values, block_mask=block_mask)
```

### 🎯 FlexAttention 的优势

| 特性 | 说明 |
|------|------|
| 灵活性 | 任意掩码模式只需写一个 Python 函数 |
| 性能 | 自动编译为优化 CUDA 内核，接近 FlashAttention |
| 场景覆盖 | 因果、滑动窗口、稀疏、前缀 LM 等各种模式 |
| 限制 | 暂不支持 dropout；需 CUDA 且 PyTorch ≥ 2.5 |

> **类比**：FlashAttention 像一把锋利的固定用途刀，FlexAttention 像瑞士军刀——多功能且依然锋利。

---

## 速度对比总览

### 🖥️ M3 MacBook Air CPU（粗略 `%timeit`）

在 CPU 上主要体现代码写法的开销差异（没有 FlashAttention 加速）。

### 🎮 NVIDIA A100 GPU（带 warmup 的精确测量）

notebook 用 CUDA Event 做了三组基准测试：

| 测试场景 | 说明 |
|----------|------|
| **仅前向** | 只跑一次 `forward()`，1000 次取平均 |
| **前向 + 反向** | 加上 `loss.backward()`，更贴近训练场景 |
| **前向 + 反向 + `torch.compile`** | 用 `torch.compile()` 编译后再测 |

### 速度排名（从快到慢，GPU 上大致趋势）

```
🥇 FlexAttention / SDPA + FlashAttention (is_causal=True)
🥈 nn.MultiheadAttention (need_weights=False)
🥉 合并 QKV 权重 / Ch03 MHA
4️⃣  Einsum / SDPA 无 FlashAttention / nn.MHA (默认)
🐢 Wrapper 写法（循环拼接）
```

> **关键结论**：
> - **FlashAttention 是最大的加速点**，比手写注意力快好几倍
> - **合并 QKV** 比分开三个矩阵快约 10-20%
> - **`torch.compile`** 能进一步压榨性能，对所有实现都有提升
> - Wrapper 写法在任何平台上都是最慢的

---

## 9 种实现的全景对比

| # | 实现方式 | QKV 权重 | 注意力计算 | FlashAttention | 适用场景 |
|---|---------|----------|-----------|----------------|----------|
| 1 | Wrapper 包装 | 每头独立 | 手写 + 循环 | ❌ | 教学演示 |
| 2 | Ch03 MHA | 分开三个 `Linear` | 手写 | ❌ | 教学/原型 |
| 3 | 合并 QKV | 一个 `Linear(d, 3d)` | 手写 | ❌ | 生产代码 |
| 4 | Einsum | `nn.Parameter` + einsum | einsum | ❌ | 研究代码 |
| 5 | PyTorch SDPA | 合并 QKV | `F.scaled_dot_product_attention` | ✅ `is_causal` | **推荐** |
| 6 | SDPA 无 Flash | 合并 QKV | `F.scaled_dot_product_attention` | ❌ 显式 mask | 自定义掩码 |
| 7 | `nn.MHA` 默认 | 内部管理 | 内部实现 | ❌ `need_weights` | 需要权重时 |
| 8 | `nn.MHA` 优化 | 内部管理 | 内部 → SDPA | ✅ `need_weights=False` | **推荐** |
| 9 | FlexAttention | 合并 QKV | `flex_attention` | ✅ 自动编译 | 自定义掩码 |

---

## 实战建议

### 日常训练 → 用实现 ⑤

```python
# 最简洁、最快的写法
context = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

### 需要 PyTorch 原生集成 → 用实现 ⑧

```python
# 已有代码用了 nn.MultiheadAttention？加一个参数就能加速
nn.MultiheadAttention(..., need_weights=False)
```

### 需要自定义掩码模式 → 用实现 ⑨

```python
# 比如滑动窗口注意力
def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= window_size
```

### 想提升已有模型的速度 → 加上 `torch.compile`

```python
model = torch.compile(model)  # 一行代码，免费加速
```

---

## 总结

```
       写法越简单                  写法越底层
       ──────────────────────────────────────>
       nn.MHA    →   SDPA    →   手写 matmul

       速度越快                    速度越可控
       <──────────────────────────────────────
       FlashAttn → 合并QKV →   Wrapper循环
```

**核心要点**：
1. **不要循环拼接头**——用 reshape 做逻辑拆分
2. **合并 QKV 权重**——减少矩阵乘法次数
3. **用 `F.scaled_dot_product_attention`**——自动获得 FlashAttention
4. **`torch.compile` 是免费午餐**——几乎不改代码就能加速

数学不变，工程在变。选对实现方式，同样的模型可以训得更快、推理更省！🚀
