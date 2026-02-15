# Chapter 2: BPE from Scratch - 手把手教你做分词器

## 开篇：一个汉字游戏的启发

还记得小时候玩的"组词游戏"吗？

```
日 + 月 = 明
木 + 木 = 林
木 + 木 + 木 = 森
```

BPE（Byte Pair Encoding）就是玩类似的游戏，只不过是用字母：

```
t + h = th
th + e = the
i + n = in
```

通过不断合并常见的字母组合，我们能用更少的"零件"表示更多的词。这就是 BPE 的精髓！

## 1. BPE 的核心思想：像拼乐高一样拼词

### 传统方法的困境

**按词切分（Word-level）：**
```python
"I love artificial intelligence"
→ ["I", "love", "artificial", "intelligence"]

问题：遇到新词怎么办？
"I love ChatGPT"
→ ["I", "love", "???"]  # ChatGPT 不在词表里！
```

**按字母切分（Character-level）：**
```python
"cat" → ["c", "a", "t"]

问题：太碎了！
"artificial" → ["a","r","t","i","f","i","c","i","a","l"]  # 10个token！
```

### BPE 的解决方案：智能组合

```python
"artificial" → ["art", "ific", "ial"]  # 只要3个token！
```

BPE 找到了平衡点：
- 常见的词保持完整（如 "the"）
- 罕见的词被拆分（如 "antidisestablishmentarianism"）
- 新词也能处理（通过组合已知部分）

## 2. BPE 算法：一步步看懂

### Step 1: 从字符开始

```python
text = "the cat in the hat"

# 初始状态：每个字符都是一个token
tokens = ['t','h','e',' ','c','a','t',' ','i','n',' ','t','h','e',' ','h','a','t']
```

### Step 2: 统计相邻字符对的频率

```python
# 数一数哪两个字符经常一起出现
pairs_count = {
    ('t', 'h'): 2,  # "th" 出现了2次
    ('h', 'e'): 2,  # "he" 出现了2次
    ('e', ' '): 2,  # "e " 出现了2次
    (' ', 'c'): 1,  # " c" 出现了1次
    ('c', 'a'): 1,  # "ca" 出现了1次
    ('a', 't'): 2,  # "at" 出现了2次
    # ... 等等
}
```

### Step 3: 合并最频繁的对

```python
# 找出最频繁的对（假设是 'th'）
most_frequent = ('t', 'h')  # 出现2次

# 合并它们
# 原来：['t','h','e',' ','c','a','t',' ','i','n',' ','t','h','e',' ','h','a','t']
# 现在：['th','e',' ','c','a','t',' ','i','n',' ','th','e',' ','h','a','t']

# 给这个新组合一个ID
new_token_id = 256  # 假设0-255已经被单个字符占用了
vocab[256] = 'th'
```

### Step 4: 重复直到满意

```python
# 继续合并下一个最频繁的对
# 假设是 ('th', 'e')
# 原来：['th','e',' ','c','a','t',' ','i','n',' ','th','e',' ','h','a','t']
# 现在：['the',' ','c','a','t',' ','i','n',' ','the',' ','h','a','t']

vocab[257] = 'the'
```

重复这个过程，直到：
- 达到目标词汇表大小（如 GPT-2 的 50,257）
- 或者没有值得合并的对了

## 3. 动手实现：从零开始写 BPE

### 简化版实现

```python
class SimpleBPE:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}  # ID到token的映射
        self.merges = {}  # 记录哪些对被合并了

    def train(self, text, verbose=True):
        """训练BPE分词器"""

        # Step 1: 初始化 - 每个字符是一个token
        tokens = list(text.encode('utf-8'))  # 转成字节

        # 初始词汇表：256个字节值
        for i in range(256):
            self.vocab[i] = bytes([i])

        next_token_id = 256

        # Step 2: 迭代合并
        while next_token_id < self.vocab_size:
            # 统计所有相邻对
            pairs = self.get_pairs(tokens)
            if not pairs:
                break

            # 找最频繁的对
            most_frequent = max(pairs, key=pairs.get)

            if verbose:
                print(f"合并: {most_frequent} (出现 {pairs[most_frequent]} 次)")

            # 合并这个对
            tokens = self.merge_pair(tokens, most_frequent, next_token_id)

            # 记录合并
            self.merges[most_frequent] = next_token_id
            self.vocab[next_token_id] = (
                self.vocab[most_frequent[0]] +
                self.vocab[most_frequent[1]]
            )

            next_token_id += 1

    def get_pairs(self, tokens):
        """统计所有相邻token对的频率"""
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def merge_pair(self, tokens, pair, new_id):
        """合并指定的token对"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            # 如果找到要合并的对
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_id)
                i += 2  # 跳过两个token
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
```

### 使用示例

```python
# 创建并训练分词器
bpe = SimpleBPE(vocab_size=300)
bpe.train("the cat in the hat sat on the mat")

# 看看学到了什么
print("学到的合并规则:")
for pair, new_id in list(bpe.merges.items())[:5]:
    token1 = bpe.vocab[pair[0]].decode('utf-8', errors='replace')
    token2 = bpe.vocab[pair[1]].decode('utf-8', errors='replace')
    merged = bpe.vocab[new_id].decode('utf-8', errors='replace')
    print(f"  '{token1}' + '{token2}' = '{merged}'")
```

## 4. 编码和解码：使用训练好的 BPE

### 编码（文本 → Token IDs）

```python
def encode(self, text):
    """将文本编码为token IDs"""
    # 先转成字节
    tokens = list(text.encode('utf-8'))

    # 应用所有学到的合并规则
    while True:
        pairs = self.get_pairs(tokens)
        if not pairs:
            break

        # 找到可以合并的对（按学习顺序）
        mergeable_pair = None
        for pair in pairs:
            if pair in self.merges:
                mergeable_pair = pair
                break

        if not mergeable_pair:
            break

        # 合并
        tokens = self.merge_pair(
            tokens,
            mergeable_pair,
            self.merges[mergeable_pair]
        )

    return tokens

# 使用
text = "the cat"
token_ids = bpe.encode(text)
print(f"'{text}' → {token_ids}")
```

### 解码（Token IDs → 文本）

```python
def decode(self, token_ids):
    """将token IDs解码回文本"""
    bytes_list = []
    for token_id in token_ids:
        bytes_list.append(self.vocab[token_id])

    # 拼接所有字节
    all_bytes = b''.join(bytes_list)

    # 转回文本
    return all_bytes.decode('utf-8', errors='replace')

# 使用
decoded = bpe.decode(token_ids)
print(f"{token_ids} → '{decoded}'")
```

## 5. 实际应用：处理真实文本

### 处理空格的小技巧

GPT 的 BPE 用特殊符号 Ġ 表示空格开头：

```python
# 预处理：空格变成特殊符号
text = "hello world"
text_processed = text.replace(' ', 'Ġ')  # "helloĠworld"

# 训练BPE...

# 后处理：特殊符号变回空格
decoded = decoded.replace('Ġ', ' ')
```

这样做的好处：
- 区分词首和词中的字母
- "cat" 和 " cat" 会有不同的表示
- 保留了词边界信息

### 处理未知字符

```python
def safe_decode(self, token_ids):
    """安全解码，处理可能的错误"""
    try:
        return self.decode(token_ids)
    except:
        # 如果解码失败，用 � 替换问题字符
        return self.decode_with_replacement(token_ids)
```

## 6. 性能优化：让 BPE 更快

### 问题：朴素实现太慢

```python
# 每次都要遍历整个序列找对
for i in range(len(tokens) - 1):  # O(n)
    pair = (tokens[i], tokens[i + 1])

# 重复很多次
while vocab_size < target:  # O(vocab_size)
    # ...

# 总复杂度：O(n × vocab_size) 😱
```

### 优化技巧

1. **使用堆（Heap）跟踪最频繁的对**
```python
import heapq

# 用堆维护频率最高的对
heap = [(-count, pair) for pair, count in pairs.items()]
heapq.heapify(heap)

# 直接取最频繁的
neg_count, most_frequent = heapq.heappop(heap)
```

2. **缓存中间结果**
```python
# 记住已经处理过的文本片段
cache = {}
def encode_with_cache(text):
    if text in cache:
        return cache[text]
    result = encode(text)
    cache[text] = result
    return result
```

3. **并行处理**
```python
from multiprocessing import Pool

# 把大文本分块并行处理
def parallel_encode(texts):
    with Pool() as pool:
        return pool.map(encode, texts)
```

## 7. 与 Tiktoken 对比

### Tiktoken（OpenAI 的生产级实现）

```python
import tiktoken

# 加载 GPT-2 的编码器
enc = tiktoken.get_encoding("gpt2")

# 编码
tokens = enc.encode("hello world")
print(f"Tiktoken: {tokens}")  # [31373, 995]

# 解码
text = enc.decode(tokens)
print(f"解码: {text}")  # "hello world"
```

### 我们的实现 vs Tiktoken

| 特性 | 我们的实现 | Tiktoken |
|-----|---------|----------|
| 速度 | 慢（Python） | 快（Rust） |
| 词汇表大小 | 可自定义 | 固定（50k+） |
| 特殊token | 基础支持 | 完整支持 |
| 适用场景 | 学习理解 | 生产环境 |

## 8. 常见问题和解决方案

### Q1: 为什么词汇表大小很重要？

```python
# 小词汇表（如 1000）
"artificial" → ["art", "if", "ic", "ial"]  # 4个token

# 大词汇表（如 50000）
"artificial" → ["artificial"]  # 1个token！

# 权衡：
# - 大词汇表：序列短，但模型参数多
# - 小词汇表：序列长，但模型参数少
```

### Q2: 如何选择合并次数？

```python
def analyze_compression(text, vocab_sizes=[500, 1000, 5000]):
    """分析不同词汇表大小的压缩效果"""

    original_length = len(text.encode('utf-8'))

    for vocab_size in vocab_sizes:
        bpe = SimpleBPE(vocab_size)
        bpe.train(text)
        tokens = bpe.encode(text)

        print(f"词汇表 {vocab_size}:")
        print(f"  原始字节: {original_length}")
        print(f"  Token数: {len(tokens)}")
        print(f"  压缩率: {len(tokens)/original_length:.2%}")
```

### Q3: BPE 的局限性

1. **语言依赖**：英语效果好，中文可能需要调整
2. **贪婪算法**：不一定是全局最优
3. **训练数据依赖**：词汇表质量取决于训练文本

## 9. 扩展：支持特殊 Token

```python
class BPEWithSpecialTokens(SimpleBPE):
    def __init__(self, vocab_size=300):
        super().__init__(vocab_size)
        self.special_tokens = {
            '<|endoftext|>': 300,
            '<|pad|>': 301,
            '<|unk|>': 302
        }

    def encode(self, text, add_special=True):
        """编码，可选添加特殊token"""

        # 处理特殊token
        if add_special:
            text = f"<|endoftext|>{text}<|endoftext|>"

        # ... 正常编码逻辑

        return tokens
```

## 10. 总结：BPE 的智慧

BPE 就像是个聪明的"拼字游戏"：

1. **从小到大**：先认识字母，再学会拼词
2. **频率驱动**：常见的组合优先学习
3. **灵活适应**：能处理任何文本，包括新词
4. **可控大小**：词汇表大小可以调节

记住核心流程：
```
统计对 → 找最频繁 → 合并 → 重复
```

最后的建议：
- **学习用**：自己实现一遍，理解原理
- **生产用**：用 Tiktoken，别重新发明轮子
- **记住**：BPE 不是魔法，只是聪明的统计

现在你知道了 ChatGPT 是怎么"读懂"你的话了！下次输入文字时，想想它是如何被切成一个个 token 的吧。🎯