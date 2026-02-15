# Chapter 2 Bonus: DataLoader - 训练 LLM 时如何"喂"数据？

## 引子：为什么需要 DataLoader？

想象你要教一个小孩认字，你有一本 1000 页的书：

**没有 DataLoader 的世界：**
- 要么一次性把整本书砸给小孩（内存爆炸💥）
- 要么一个字一个字地教（慢得要命🐌）
- 要么永远从第一页开始教（小孩只记住了开头）

**有了 DataLoader：**
- 每次给几页（批量处理）
- 随机翻页教（避免死记硬背）
- 用滑动窗口看上下文（理解连贯性）

这就是 DataLoader 的精髓：**把大数据切成小块，智能地喂给模型**。

## 1. 核心概念：滑动窗口

### 什么是滑动窗口？

想象你在读一本书，但你的视野只能看到 4 个字：

```
原文：我爱吃苹果和香蕉
窗口大小：4个字

第1次看：[我爱吃苹]
第2次看：[爱吃苹果]  # 窗口向右滑动1个字
第3次看：[吃苹果和]
第4次看：[苹果和香]
第5次看：[果和香蕉]
```

这就是滑动窗口！每次看固定长度，然后向前滑动。

### 在代码里长什么样？

为了更直观，我们用数字代替文字：

```python
# 假设我们的"文本"是 0 到 10 的数字
text = "0 1 2 3 4 5 6 7 8 9 10"

# 窗口大小 = 4，步长 = 1
max_length = 4
stride = 1

# 滑动窗口切分
for i in range(0, len(numbers) - max_length, stride):
    window = numbers[i:i + max_length]
    print(window)

# 输出：
# [0, 1, 2, 3]
# [1, 2, 3, 4]
# [2, 3, 4, 5]
# [3, 4, 5, 6]
# ...
```

## 2. 关键参数：步长（stride）的魔力

步长决定了窗口每次移动多少：

### stride = 1（密集采样，重叠最多）
```
数据：0 1 2 3 4 5 6 7 8
窗口1：[0 1 2 3]
窗口2：  [1 2 3 4]  # 重叠3个数字
窗口3：    [2 3 4 5]
```
- 优点：数据利用充分，模型看到更多上下文组合
- 缺点：训练样本多，训练慢

### stride = 4（无重叠）
```
数据：0 1 2 3 4 5 6 7 8 9 10 11
窗口1：[0 1 2 3]
窗口2：        [4 5 6 7]  # 完全不重叠
窗口3：                [8 9 10 11]
```
- 优点：训练快，没有重复
- 缺点：可能错过一些上下文组合

### stride = 2（中等重叠）
```
数据：0 1 2 3 4 5 6 7 8
窗口1：[0 1 2 3]
窗口2：    [2 3 4 5]  # 重叠2个数字
窗口3：        [4 5 6 7]
```
平衡了训练速度和数据利用率。

## 3. 输入与目标：预测下一个

LLM 的核心任务是预测下一个词（或标记）。所以：

```python
输入：  [0, 1, 2, 3]
目标：  [1, 2, 3, 4]  # 就是输入向右移一位！

# 模型要学习的是：
# 看到0，预测1
# 看到0,1，预测2
# 看到0,1,2，预测3
# 看到0,1,2,3，预测4
```

这就像是填空题：
- 输入："我爱吃___"
- 目标："苹果"

## 4. 完整的 DataLoader 实现

### Step 1: 创建数据集类

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 把文本转成数字（这里简化处理）
        token_ids = [int(i) for i in txt.strip().split()]

        # 滑动窗口切分
        for i in range(0, len(token_ids) - max_length, stride):
            # 输入窗口
            input_chunk = token_ids[i:i + max_length]
            # 目标窗口（向右移一位）
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### Step 2: 创建 DataLoader

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True):

    # 创建数据集
    dataset = GPTDatasetV1(txt, None, max_length, stride)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 每批多少个样本
        shuffle=shuffle,         # 是否打乱顺序
        drop_last=True           # 丢弃最后不完整的批次
    )

    return dataloader
```

## 5. 批处理（Batching）：一次处理多个样本

### 没有批处理
```python
# 一个一个处理（慢）
for sample in dataset:
    model.train(sample)  # 每次只处理1个
```

### 有批处理
```python
# batch_size = 4，一次处理4个
batch = [
    [0, 1, 2, 3],  # 样本1
    [4, 5, 6, 7],  # 样本2
    [8, 9, 10, 11], # 样本3
    [12, 13, 14, 15] # 样本4
]
model.train(batch)  # 一次处理4个，快！
```

为什么批处理更快？
- GPU 擅长并行计算
- 减少数据传输次数
- 更好的矩阵运算优化

## 6. 打乱（Shuffle）：避免模型"死记硬背"

### 不打乱的问题

```python
# 每次都是这个顺序
epoch 1: 样本1 → 样本2 → 样本3 → 样本4
epoch 2: 样本1 → 样本2 → 样本3 → 样本4
epoch 3: 样本1 → 样本2 → 样本3 → 样本4

# 模型可能会记住："样本2总是在样本1后面"
```

### 打乱后

```python
# 每次顺序都不同
epoch 1: 样本3 → 样本1 → 样本4 → 样本2
epoch 2: 样本2 → 样本4 → 样本1 → 样本3
epoch 3: 样本4 → 样本3 → 样本2 → 样本1

# 模型被迫学习真正的模式，而不是顺序
```

## 7. 实战示例：从创建到使用

### 创建测试数据

```python
# 生成简单的数字序列
with open("number-data.txt", "w") as f:
    for number in range(1001):
        f.write(f"{number} ")

# 读取数据
with open("number-data.txt", "r") as f:
    raw_text = f.read()
```

### 创建 DataLoader

```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=2,    # 每批2个样本
    max_length=4,    # 窗口大小4
    stride=1,        # 步长1（最大重叠）
    shuffle=True     # 打乱顺序
)
```

### 使用 DataLoader

```python
# 方法1：手动获取批次
data_iter = iter(dataloader)
first_batch_inputs, first_batch_targets = next(data_iter)
print("第一批输入:", first_batch_inputs)
print("第一批目标:", first_batch_targets)

# 方法2：循环遍历（训练时常用）
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"批次 {batch_idx}:")
    print(f"  输入形状: {inputs.shape}")  # [batch_size, max_length]
    print(f"  目标形状: {targets.shape}")

    # 这里通常会：
    # 1. 将数据送入模型
    # 2. 计算损失
    # 3. 反向传播
    # 4. 更新参数

    if batch_idx >= 2:  # 只看前3批
        break
```

## 8. 常见参数组合及其效果

### 配置1：密集学习（适合小数据集）
```python
dataloader = create_dataloader_v1(
    text,
    batch_size=8,
    max_length=256,
    stride=1,        # 最大重叠
    shuffle=True
)
# 特点：数据利用充分，训练慢
```

### 配置2：快速训练（适合大数据集）
```python
dataloader = create_dataloader_v1(
    text,
    batch_size=64,
    max_length=256,
    stride=128,      # 50%重叠
    shuffle=True
)
# 特点：训练快，仍有一定重叠
```

### 配置3：无重叠训练（最快）
```python
dataloader = create_dataloader_v1(
    text,
    batch_size=128,
    max_length=256,
    stride=256,      # 无重叠
    shuffle=True
)
# 特点：最快，但可能错过一些模式
```

## 9. 实用技巧和注意事项

### 技巧1：选择合适的 batch_size

```python
# 小 batch_size (如 8-16)
# ✅ 内存占用少
# ✅ 更新频繁，可能找到更好的最优解
# ❌ 训练慢，GPU利用率低

# 大 batch_size (如 64-256)
# ✅ 训练快，GPU利用率高
# ✅ 梯度估计更稳定
# ❌ 内存占用大，可能陷入局部最优
```

### 技巧2：drop_last 的使用

```python
# 假设有 100 个样本，batch_size=8
# 100 ÷ 8 = 12 批 + 4 个剩余样本

drop_last=True   # 丢弃最后4个，只用96个样本（12批）
drop_last=False  # 最后一批只有4个样本（13批）

# 训练时通常设为 True，保证批次大小一致
# 验证时设为 False，不浪费数据
```

### 技巧3：num_workers 并行加载

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4  # 使用4个进程并行加载数据
)
# 加速数据加载，但占用更多CPU和内存
```

## 10. 调试技巧：可视化你的数据

```python
def visualize_dataloader(dataloader, num_batches=3):
    """查看 DataLoader 产生的数据"""

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        print(f"\n=== 批次 {batch_idx + 1} ===")
        print(f"输入: {inputs.tolist()}")
        print(f"目标: {targets.tolist()}")

        # 显示输入-目标对应关系
        for i in range(len(inputs)):
            print(f"  样本{i}: {inputs[i].tolist()} → {targets[i].tolist()}")

# 使用
small_loader = create_dataloader_v1(
    "0 1 2 3 4 5 6 7 8 9",
    batch_size=2,
    max_length=3,
    stride=1,
    shuffle=False  # 关闭打乱，便于观察
)
visualize_dataloader(small_loader)
```

## 总结：DataLoader 的哲学

DataLoader 就像是一个聪明的"投喂器"：

1. **切分**：把大数据切成小块（滑动窗口）
2. **打包**：把小块组成批次（batching）
3. **打乱**：随机化顺序（shuffle）
4. **配送**：高效地送给模型（并行加载）

记住这个公式：
```
好的 DataLoader = 合适的窗口 + 合理的步长 + 恰当的批次 + 必要的打乱
```

最后的建议：
- 刚开始时，用小的 batch_size 和 max_length 调试
- 确认数据正确后，再调大参数加速训练
- 永远记得检查第一个批次的数据，确保符合预期

现在，你已经掌握了喂养 LLM 的艺术！🎉