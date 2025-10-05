# Lecture1_Notes

## “书生・浦语” 目前开源了三个量级的大模型

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/20c1a246-0244-465f-8e10-44006df92d99)

上海AI实验室联合多家机构推出了中量级参数的InternLM-20B大模型，性能先进且应用便捷，以不足三分之一的参数量，达到了当前被视为开源模型标杆的Llama2-70B的能力水平

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/93e5d0a9-e72b-4b02-9956-71ae4a17d201)

## 书生·浦语大模型全链路工具体系

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/bde11eea-341a-45f3-b88f-4486690c57f9)

* 在多元融合方面，书生·万卷1.0包含文本、图文、视频等多模态数据，范围覆盖科技、文学、媒体、教育、法律等多个领域，在训练提升模型知识含量、逻辑推理和泛化能力方面具有显著效果。

* 在精细处理方面，书生·万卷1.0经历了语言甄别、正文抽取、格式标准化、基于规则及模型的数据过滤与清洗、多尺度去重、数据质量评估等精细化数据处理环节，因而能更好地适配后续的模型训练需求。

* 在价值对齐方面，研究人员在书生·万卷1.0的构建过程中，着眼于内容与中文主流价值观的对齐，通过算法与人工评估结合的方式，提升了语料的纯净度。

* 在易用高效方面，研究人员在书生·万卷1.0采用统一格式，并提供详细的字段说明和工具指导，使其兼顾了易用性和效率，可快速应用于语言、多模态等大模型训练。

* Sample

* ![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/ccbc655e-9cc3-44b2-b7eb-d0cdfaf25d9f)

* ![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/cdb76517-717f-46d2-97bc-ed2739d9f66e)

## 大模型微调

大模型的下游应用，增量训练和有监督微调是主流的两种方式。

增量训练：一般用于让基座模型学习知识，如某一个垂直领域的知识，一般将文章、书籍、代码作为训练数据；

有监督微调：用于让模型理解、遵循各种指令，或注入少量领域知识，一般将对话数据、问答数据作为训练数据。

上海人工智能实验室开发了低成本大模型训练工具箱 XTuner，旨在让大模型训练不再有门槛。

XTuner 首次尝试将 HuggingFace 与 OpenMMLab 进行结合，兼顾易用性和可配置性。支持使用 MMEngine Runner 和 HuggingFace Trainer 两种训练引擎.

```bash
pip install xtuner

# 使用 MMEngine Runner 训练
xtuner train internlm_7b_qlora_oasst1_e3

# 使用 HugingFace Trainer 训练
xtuner train internlm_7b_qlora_oasst1_e3_hf
```

## 模型评测

opencompass开源评测平台架构如下：

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/31a3585d-e1ac-4cd8-834e-7f5c3680968e)

具有丰富的模型支持

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/d2700f9e-c8cd-4dbf-a8f2-eb02ff06f33f)

* 分布式的高效评测；

* 便捷的数据集接口；

* 敏捷并发的迭代能力。

* 用户广泛分布于国内外的知名企业与科研机构。

## 模型部署

LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发，涵盖了 LLM 任务的全套轻量化、部署和服务解决方案，提供大模型在GPU部署的全流程解决方法。

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/96a07ebb-f709-49e4-944c-7fdb34055aa4)

这个强大的工具箱提供以下核心功能：

高效推理引擎 TurboMind：基于 FasterTransformer，实现了高效推理引擎 TurboMind，支持 InternLM、LLaMA、vicuna等模型在 NVIDIA GPU 上的推理。

交互推理方式：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。

多 GPU 部署和量化：我们提供了全面的模型部署和量化支持，已在不同规模上完成验证。

persistent batch 推理：进一步优化模型执行效率。

具有如下高效的推理引擎、完备工具链。

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/cb914e8b-04a9-4848-b318-37a3819ee53a)

## 模型应用-智能体

大语言模型在信息可靠性、数值计算、工具使用交互具有局限性。智能体将完美的补上这个短板。

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/3faa785d-862b-453f-888d-74de26bcf4b1)

Lagent是轻量级框架，用于构建基于LLM（Logical Layered Modeling）的代理。设计目的是为了简化和提高基于这种模型的代理的开发效率。LLM模型用于模拟和管理复杂的系统，而Lagent就是这种模型的实现。
轻量级智能体框架lagent具有以下优势：
高效的推理引擎：Lagent 支持 lmdeploy turbomind，高效推理引擎，让你的代理运行得更快、更流畅。

* 多代理支持：Lagent 支持 ReAct、AutoGPT 还是 ReWOO。

* 极简易扩展：需大约 20 行代码，可以打造自己的代理。Lagent 还支持 Python 解释器、API 调用和谷歌搜索等工具，让你的代理更加全能。

* 支持多种 LLM：从 GPT-3.5/4 到 LLaMA 2 和 InternLM，Lagent 支持各种大型语言模型，让你的选择更加多样。

* 丰富的工具集合，提供了大量视觉、多模态相关领域的前沿算法功能；涉及图像理解、音频文字转换、图像生成等功能，让你的 LLM 能力大大增强。

* 支持主流智能体系统，如 LangChain，Transformers Agent，lagent等

  ```python
  from agentlego import load_tool
  tool = load_tool('ImageCaption')
  
  # 在 LangChain 中
  from langchain import initialize_agent
  agent = initialize_agent(
      agent="structured-chat-zero-shot-react-description",
      tools=[tool.to_langchain()],
      ...
  )
  
  # 在 Transformers Agent 中
  from transformers import HfAgent
  agent = HfAgent(
      'https://api-inference.huggingface.co/models/bigcode/starcoder',
      additional_tools=[tool.to_transformers_agent()],
  )
  
  # 在 Lagent 中
  from lagent import ReAct, ActionExecutor
  agent = ReAct(
      action_executor=ActionExecutor(actions=[tool.to_lagent()]),
      ...
  )
  ```

  * 多模态工具调用接口，可以轻松支持各类输入输出格式的工具函数。通过统一的多模态输入输出接口，在不同的 Agent 系统中自动进行数据格式转换。

![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/f9138c74-bb46-434a-a99c-a425b29f29ea)


  * 一键式远程工具部署，轻松使用和调试大模型智能体。

    ![image](https://github.com/jiayaozhang/InternLM_Notes/assets/38579506/faa50c84-c387-4b27-a8a1-46643ef49752)

   * 用一行命令启动工具服务器，并指定在该工具服务器上运行的工具列表：
```bash
python server.py ImageCaption TextToImage VQA OCR
```
* Agent 系统运行的平台，只需要配置最基本的环境，配合 RemoteTool，即可通过网络通讯的方式，调用远程机器上部署的工具。
```python
from agentlego.tools.remote import RemoteTool 
 tools = RemoteTool.from_server('Tool server address')
```
