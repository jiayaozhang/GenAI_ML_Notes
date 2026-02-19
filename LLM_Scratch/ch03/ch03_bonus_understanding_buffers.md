# Chapter 3 Bonus：理解 PyTorch Buffers

> 📖 本笔记基于 [LLMs-from-scratch / ch03 / 03_understanding-buffers](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/03_understanding-buffers)

## 开篇：一个让人抓狂的 Bug

你写了一个注意力模块，在 CPU 上跑得好好的。然后你把模型搬到 GPU 上：

```python
model = model.to("cuda")
batch = batch.to("cuda")
output = model(batch)   # 💥 RuntimeError: 张量不在同一个设备上！
```

明明已经 `.to("cuda")` 了，怎么还报错？

问题出在那个**因果掩码 `mask`** 上——它被落在 CPU 了，像搬家时忘在旧房子里的东西。

---

## 核心问题：Parameters vs 普通张量

PyTorch 的 `.to(device)` 只会自动搬运两种东西：

| 类型 | 自动搬运？ | 自动求梯度？ | 例子 |
|------|-----------|-------------|------|
| **`nn.Parameter`** | ✅ | ✅ | `W_query.weight`、`W_key.weight` |
| **Buffer**（`register_buffer`） | ✅ | ❌ | 因果掩码 `mask` |
| **普通 `torch.Tensor`** | ❌ | ❌ | `self.mask = torch.triu(...)` |

> **类比**：`model.to("cuda")` 就像叫搬家公司，他们只搬**登记在册**的物品（参数和 buffer）。随手放在角落的东西（普通张量）不在清单上，自然会被遗漏。

---

## 反面教材：不用 Buffer 的写法

```python
class CausalAttentionWithoutBuffers(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)   # ← 参数，会自动搬
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)   # ← 参数，会自动搬
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)   # ← 参数，会自动搬
        self.dropout = nn.Dropout(dropout)

        # ⚠️ 这只是一个普通张量，不会跟着 model.to("cuda") 搬走！
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
```

### 在 CPU 上正常运行 ✅

```python
batch = torch.stack((inputs, inputs), dim=0)
ca = CausalAttentionWithoutBuffers(d_in=3, d_out=2, context_length=6, dropout=0.0)

with torch.no_grad():
    output = ca(batch)    # 完全没问题，一切都在 CPU 上
print(output)             # 正常输出
```

### 搬到 GPU 后崩溃 ❌

```python
ca = ca.to("cuda")        # 权重搬走了 ✅
batch = batch.to("cuda")  # 数据搬走了 ✅
# ca.mask 还在 CPU 上！ ❌

with torch.no_grad():
    output = ca(batch)     # 💥 RuntimeError!
```

检查设备位置就能发现问题：

```python
print(ca.W_query.weight.device)  # → cuda:0  ✅ 参数自动搬了
print(ca.mask.device)            # → cpu     ❌ 掩码被遗忘了！
```

### 手动修复（麻烦）

```python
ca.mask = ca.mask.to("cuda")   # 每次都要手动搬，容易忘
```

---

## 正确做法：`register_buffer`

只需改一行代码：

```python
class CausalAttentionWithBuffer(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # ✅ 用 register_buffer 注册！
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
```

### 现在一切自动

```python
ca = CausalAttentionWithBuffer(d_in=3, d_out=2, context_length=6, dropout=0.0)
ca.to("cuda")   # mask 也自动跟着到 GPU 了！

print(ca.W_query.weight.device)  # → cuda:0  ✅
print(ca.mask.device)            # → cuda:0  ✅  再也不用手动搬了
```

---

## Buffer 的另一个好处：被 `state_dict` 记住

### 什么是 `state_dict`？

`state_dict` 是模型的"快照"字典，保存和加载模型时用的：

```python
# 保存
torch.save(model.state_dict(), "model.pth")
# 加载
model.load_state_dict(torch.load("model.pth"))
```

### 不用 Buffer → mask 不在快照里

```python
ca_without_buffer.state_dict()
# OrderedDict({
#     'W_query.weight': tensor(...),
#     'W_key.weight':   tensor(...),
#     'W_value.weight': tensor(...),
#     # ← mask 不在这里！
# })
```

### 用了 Buffer → mask 在快照里

```python
ca_with_buffer.state_dict()
# OrderedDict({
#     'W_query.weight': tensor(...),
#     'W_key.weight':   tensor(...),
#     'W_value.weight': tensor(...),
#     'mask':           tensor(...),   # ← mask 被保存了！✅
# })
```

### 为什么这很重要？

假设 `mask` 在训练过程中被修改了：

```python
# 模拟训练中修改了 mask
ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.

# 保存 → 加载
torch.save(ca_with_buffer.state_dict(), "model.pth")
new_model = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
new_model.load_state_dict(torch.load("model.pth"))

print(new_model.mask)   # → 包含修改后的2.0  ✅ 修改被保留了
```

如果没用 buffer：

```python
# 不用 buffer 的版本
torch.save(ca_without_buffer.state_dict(), "model.pth")
new_model = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)
new_model.load_state_dict(torch.load("model.pth"))

print(new_model.mask)   # → 全是原始的1.0  ❌ 修改丢失了！
```

---

## 一图总结

```
模块中的张量
│
├── nn.Parameter（权重/偏置）
│     ✅ 自动跟随 .to(device)
│     ✅ 包含在 state_dict
│     ✅ 参与梯度计算（会被优化器更新）
│
├── register_buffer（Buffer）
│     ✅ 自动跟随 .to(device)
│     ✅ 包含在 state_dict
│     ❌ 不参与梯度计算（训练时不更新）
│
└── 普通 self.xxx = torch.Tensor
      ❌ 不跟随 .to(device) ← 这是 Bug 的根源！
      ❌ 不包含在 state_dict
      ❌ 不参与梯度计算
```

---

## 实战规则

> **什么时候用 `register_buffer`？**
>
> 当你的模块里有一个张量满足以下条件时：
> 1. **不需要梯度**（不是可学习的参数）
> 2. **需要跟模型在同一个设备上**
> 3. **需要被保存/加载**
>
> → 就用 `self.register_buffer("name", tensor)`

### 常见场景

| 场景 | 示例 |
|------|------|
| **因果掩码** | 注意力中的上三角 mask |
| **位置编码** | 固定的正弦/余弦位置编码（非学习型） |
| **归一化统计量** | BatchNorm 的 running_mean / running_var |
| **固定常数** | 预计算的频率表、缩放因子等 |

### 一行代码，三个好处

```python
# ❌ 别这样写
self.mask = torch.triu(torch.ones(n, n), diagonal=1)

# ✅ 这样写
self.register_buffer("mask", torch.triu(torch.ones(n, n), diagonal=1))
```

就这一行改动，你就永远不会再遇到"张量不在同一设备"的 bug 了！🎯
