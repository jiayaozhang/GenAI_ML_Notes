# LMDeploy 大模型量化部署实践 学习笔记

* 视频教程链接：
https://www.bilibili.com/video/BV1iW4y1A77P

* 文档链接：
https://github.com/InternLM/tutorial/blob/main/lmdeploy/lmdeploy.md

* 大模型上下文扩展技术原理

![大模型参数高效微调及扩展1.png](images/大模型参数高效微调及扩展1.png)




## 大模型部署的背景
* 模型部署：将训练好的模型在特定的软硬件环境中启动，使模型能够接收输入并返回预测结果。为了满足性能和效率的需求，常常要对模型进行优化，例如模型压缩和硬件加速。

* 大模型的特点：内存开销巨大，参数量庞大，7B模型仅权重就需要14G内存。采用自回归生成token，需要缓存Attention的k/v。动态shape，请求数不固定，token逐个生成，且数量不定。相对视觉模型，LLM结构简单，Transformers结构大部分是decoder-only。

大模型部署挑战：

* 设备：如何应对巨大的存储问题？低存储设备，消费级显卡，手机等如何部署？
* 推理：如何加速token的生成，如何解决动态shape，让推理可以不间断，如何有效管理和利用内存。
* 服务：如何提升系统吞吐量，降低响应时间。

* 部署方案：

## LMDeploy简介
LMDeploy是LLM在英伟达设备上部署的全流程解决方案。包括轻量化，推理和服务。提供了高效的推理引擎和完备易用的工具链。

* 核心功能：

* 量化

* 推理引擎TurboMind

* 推理服务api server

## 项目实战\

* 环境准备
```python
/root/share/install_conda_env_internlm_base.sh lmdeploy
conda activate lmdeploy
```

* 安装LMDeploy

```bash
pip install 'lmdeploy[all]==v0.1.0'
```
* 离线转换

```bash
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
```

* 模型部署

方案1.命令行
```bash 
lmdeploy chat turbomind ./workspace
```


方案2. API服务

![FastAPI.png](images/FastAPI.png)


```bash
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1

lmdeploy serve api_client http://localhost:23333
```

方案3.gradio

![weather.png](images/weather.png)


```bash
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True
  
  lmdeploy serve gradio ./workspace
```

方案4. TurboMind 推理 + Python 代码集成

```bash
from lmdeploy import turbomind as tm

# load model
model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-20b')
generator = tm_model.create_instance()

# process query
query = "你好啊兄嘚"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)

# inference
for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]

response = tm_model.tokenizer.decode(res.tolist())
print(response)
```

![300wordsStory.png](images/300wordsStory.png)


## 模型量化

那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用 KV Cache 量化和 4bit Weight Only 量化（W4A16）。KV Cache 量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。4bit Weight 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

### KV Cache 量化

```bash
# 第一步：计算 minmax
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "ptb" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_output
```
第一步：计算 minmax。主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。

<!-- ![kv.png](images/kv.png) -->

```cpp
model.layers.0,  samples: 128, max gpu memory: 8.18 GB
model.layers.1,  samples: 128, max gpu memory: 10.18 GB
model.layers.2,  samples: 128, max gpu memory: 10.18 GB
model.layers.3,  samples: 128, max gpu memory: 10.18 GB
model.layers.4,  samples: 128, max gpu memory: 10.18 GB
model.layers.5,  samples: 128, max gpu memory: 10.18 GB
model.layers.6,  samples: 128, max gpu memory: 10.18 GB
model.layers.7,  samples: 128, max gpu memory: 10.18 GB
model.layers.8,  samples: 128, max gpu memory: 10.18 GB
model.layers.9,  samples: 128, max gpu memory: 10.18 GB
model.layers.10, samples: 128, max gpu memory: 10.18 GB
model.layers.11, samples: 128, max gpu memory: 10.18 GB
model.layers.12, samples: 128, max gpu memory: 10.18 GB
model.layers.13, samples: 128, max gpu memory: 10.18 GB
model.layers.14, samples: 128, max gpu memory: 10.18 GB
model.layers.15, samples: 128, max gpu memory: 10.18 GB
model.layers.16, samples: 128, max gpu memory: 10.18 GB
model.layers.17, samples: 128, max gpu memory: 10.18 GB
model.layers.18, samples: 128, max gpu memory: 10.18 GB
model.layers.19, samples: 128, max gpu memory: 10.18 GB
model.layers.20, samples: 128, max gpu memory: 10.18 GB
model.layers.21, samples: 128, max gpu memory: 10.18 GB
model.layers.22, samples: 128, max gpu memory: 10.18 GB
model.layers.23, samples: 128, max gpu memory: 10.18 GB
model.layers.24, samples: 128, max gpu memory: 10.18 GB
model.layers.25, samples: 128, max gpu memory: 10.18 GB
model.layers.26, samples: 128, max gpu memory: 10.18 GB
model.layers.27, samples: 128, max gpu memory: 10.18 GB
model.layers.28, samples: 128, max gpu memory: 10.18 GB
model.layers.29, samples: 128, max gpu memory: 10.18 GB
model.layers.30, samples: 128, max gpu memory: 10.18 GB
model.layers.31, samples: 128, max gpu memory: 10.18 GB

```

* 对于 Attention 的 K 和 V：取每个 Head 各自维度在所有Token的最大、最小和绝对值最大值。对每一层来说，上面三组值都是 (num_heads, head_dim) 的矩阵。这里的统计结果将用于本小节的 KV Cache。

* 对于模型每层的输入：取对应维度的最大、最小、均值、绝对值最大和绝对值均值。每一层每个位置的输入都有对应的统计值，它们大多是 (hidden_dim, ) 的一维向量，当然在 FFN 层由于结构是先变宽后恢复，因此恢复的位置维度并不相同。这里的统计结果用于下个小节的模型参数量化，主要用在缩放环节（回顾PPT内容）。


```bash
x lmdeploy lite calibrate \  --model  /root/share/temp/model_repos/internlm-chat-7b/ \  --calib_dataset "ptb" \  --calib_samples 128 \  --calib_seqlen 2048 \  --work_dir ./quant_output
```


在这个命令行中，会选择 128 条输入样本，每条样本长度为 2048，数据集选择 ptb，输入模型后就会得到上面的各种统计值。值得说明的是，如果显存不足，可以适当调小 samples 的数量或 sample 的长度。

第二步：通过 minmax 获取量化参数。主要就是利用下面这个公式，获取每一层的 K V 中心值（zp）和缩放值（scale）。

```bash
# 第二步：获取量化参数
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp

# 修改 weights/config.ini 文件

```

有这两个值就可以进行量化和解量化操作了。具体来说，就是对历史的 K 和 V 存储 quant 后的值，使用时在 dequant。

```bash
# 通过 minmax 获取量化参数
lmdeploy lite kv_qparams \
  --work_dir ./quant_output  \
  --turbomind_dir workspace/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1

```

<!-- ![kv_quarms.png](images/kv_quarms.png) -->
```cpp
Layer 0 MP 0 qparam:    0.056243896484375       -1.078125       0.00868988037109375     -0.037109375
Layer 1 MP 0 qparam:    0.06939697265625        -0.375          0.0149688720703125      -0.12255859375
Layer 2 MP 0 qparam:    0.0789794921875         0.1015625       0.033538818359375       -0.7255859375
Layer 3 MP 0 qparam:    0.098388671875         -0.21484375      0.034149169921875       -1.0361328125
Layer 4 MP 0 qparam:    0.11614990234375        0.29296875      0.031097412109375        0.2919921875
Layer 5 MP 0 qparam:    0.12646484375           0.203125        0.0303955078125         -0.328125
Layer 6 MP 0 qparam:    0.128662109375         -0.015625        0.0256805419921875       0.0693359375
Layer 7 MP 0 qparam:    0.139892578125         -1.140625        0.038238525390625       -0.36328125
Layer 8 MP 0 qparam:    0.140380859375         -0.3203125       0.029388427734375       -0.3583984375
Layer 9 MP 0 qparam:    0.1376953125           -0.4375          0.03778076171875         0.59765625
Layer 10 MP 0 qparam:   0.126220703125         -1.34375         0.026580810546875        0.07421875
Layer 11 MP 0 qparam:   0.130859375            -0.015625        0.0322265625            -0.033203125
Layer 12 MP 0 qparam:   0.1312255859375        -0.015625        0.030792236328125        0.1845703125
Layer 13 MP 0 qparam:   0.132568359375          0.3046875       0.0361328125            -0.2890625
Layer 14 MP 0 qparam:   0.12646484375           0.46875         0.030975341796875        0.109375
Layer 15 MP 0 qparam:   0.131103515625          0.296875        0.02978515625            0.015625
Layer 16 MP 0 qparam:   0.1324462890625        -0.859375        0.029632568359375       -0.1220703125
Layer 17 MP 0 qparam:   0.12646484375           0.3515625       0.041534423828125       -0.291015625
Layer 18 MP 0 qparam:   0.130126953125         -0.0625          0.04766845703125         0.326171875
Layer 19 MP 0 qparam:   0.126220703125          0.30859375      0.051300048828125        0.28125
Layer 20 MP 0 qparam:   0.135498046875         -0.09375         0.04998779296875        -0.240234375
Layer 21 MP 0 qparam:   0.13427734375          -0.2265625       0.046600341796875        0.0625
Layer 22 MP 0 qparam:   0.140380859375         -0.3359375       0.053009033203125        0.18359375
Layer 23 MP 0 qparam:   0.1337890625           -0.4453125       0.062286376953125        0.73046875
Layer 24 MP 0 qparam:   0.1240234375           -0.09765625      0.048248291015625       -0.14453125
Layer 25 MP 0 qparam:   0.139404296875          0.65625         0.06695556640625         1.056640625
Layer 26 MP 0 qparam:   0.1304931640625        -0.89453125      0.08197021484375         0.15625
Layer 27 MP 0 qparam:   0.134521484375         -0.25            0.07171630859375        -0.2890625
Layer 28 MP 0 qparam:   0.13916015625           0.2578125       0.078125                 0.109375
Layer 29 MP 0 qparam:   0.143798828125         -0.03125         0.08172607421875        -0.62109375
Layer 30 MP 0 qparam:   0.1318359375           -0.78125         0.08551025390625        -0.33203125
Layer 31 MP 0 qparam:   0.12890625              0.3203125       0.152099609375          -0.265625
```
第三步：修改配置。也就是修改 weights/config.ini 文件，这个我们在《2.6.2 模型配置实践》中已经提到过了（KV int8 开关），只需要把 quant_policy 改为 4 即可

* lmdeploy==0.2.0后，c4数据集会报错，这里改成了ptb数据集后成功

### W4A16量化

```bash
# 第二步：量化权重模型
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output 

# 转换成 TurboMind 格式
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128

```

通过比较未使用量化&使用量化 

![origin.png](images/origin.png)

![optimize.png](images/optimize.png)

显存明显下降

* 使用量化`output log results`

```cpp
remove workspace in directory ./workspace
create workspace in directory ./workspace
copy triton model templates from "/root/.local/lib/python3.10/site-packages/lmdeploy/serve/turbomind/triton_models" to "./workspace/triton_models"
copy service_docker_up.sh from "/root/.local/lib/python3.10/site-packages/lmdeploy/serve/turbomind/service_docker_up.sh" to "./workspace"
model_name             internlm-chat-7b
model_format           awq
inferred_model_format  hf-awq
model_path             ./quant_output
tokenizer_path         ./quant_output/tokenizer.model
output_format          w4
WARNING: Can not find tokenizer.json. It may take long time to initialize the tokenizer.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
*** splitting layers.0.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.0.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.0.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.0.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.0.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.0.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.0.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.0.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.0.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.0.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.1.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.1.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.1.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.1.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.1.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.1.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.1.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.1.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.1.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.1.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.2.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.2.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.2.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.2.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.2.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.2.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.2.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.2.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.2.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.2.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.3.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.3.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.3.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.3.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.3.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.3.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.3.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.3.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.3.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.3.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.4.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.4.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.4.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.4.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.4.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.4.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.4.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.4.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.4.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.4.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.5.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.5.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.5.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.5.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.5.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.5.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.5.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.5.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.5.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.5.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.6.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.6.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.6.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.6.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.6.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.6.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.6.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.6.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.6.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.6.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.7.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.7.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.7.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.7.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.7.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.7.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.7.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.7.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.7.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.7.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.8.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.8.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.8.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.8.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.8.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.8.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.8.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.8.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.8.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.8.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.9.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                         
*** splitting layers.9.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                           
*** splitting layers.9.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                    
*** splitting layers.9.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                                
*** splitting layers.9.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                    
### copying layers.9.attention.wo.bias, shape=torch.Size([4096])                                                                                                 
*** splitting layers.9.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.9.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                          
*** splitting layers.9.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                                
*** splitting layers.9.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                             
*** splitting layers.10.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.10.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.10.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.10.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.10.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.10.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.10.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.10.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.10.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.10.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.11.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.11.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.11.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.11.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.11.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.11.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.11.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.11.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.11.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.11.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.12.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.12.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.12.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.12.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.12.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.12.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.12.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.12.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.12.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.12.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.13.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.13.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.13.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.13.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.13.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.13.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.13.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.13.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.13.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.13.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.14.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.14.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.14.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.14.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.14.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.14.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.14.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.14.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.14.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.14.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.15.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.15.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.15.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.15.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.15.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.15.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.15.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.15.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.15.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.15.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.16.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.16.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.16.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.16.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.16.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.16.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.16.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.16.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.16.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.16.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.17.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.17.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.17.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.17.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.17.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.17.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.17.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.17.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.17.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.17.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.18.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.18.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.18.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.18.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.18.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.18.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.18.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.18.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.18.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.18.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.19.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.19.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.19.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.19.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.19.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.19.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.19.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.19.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.19.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.19.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.20.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.20.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.20.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.20.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.20.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.20.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.20.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.20.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.20.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.20.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.21.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.21.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.21.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.21.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.21.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.21.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.21.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.21.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.21.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.21.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.22.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.22.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.22.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.22.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.22.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.22.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.22.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.22.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.22.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.22.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.23.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.23.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.23.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.23.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.23.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.23.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.23.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.23.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.23.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.23.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.24.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.24.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.24.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.24.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.24.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.24.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.24.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.24.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.24.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.24.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.25.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.25.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.25.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.25.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.25.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.25.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.25.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.25.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.25.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.25.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.26.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.26.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.26.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.26.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.26.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.26.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.26.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.26.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.26.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.26.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.27.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.27.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.27.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.27.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.27.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.27.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.27.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.27.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.27.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.27.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.28.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.28.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.28.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.28.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.28.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.28.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.28.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.28.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.28.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.28.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.29.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.29.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.29.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.29.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.29.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.29.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.29.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.29.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.29.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.29.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.30.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.30.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.30.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.30.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.30.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.30.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.30.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.30.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.30.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.30.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1                                                            
*** splitting layers.31.attention.w_qkv.qweight, shape=torch.Size([128, 384, 128, 1]), split_dim=-1, tp=1                                                        
*** splitting layers.31.attention.w_qkv.scales_zeros, shape=torch.Size([32, 12288]), split_dim=-1, tp=1                                                          
*** splitting layers.31.attention.wo.qweight, shape=torch.Size([4096, 512]), split_dim=0, tp=1                                                                   
*** splitting layers.31.attention.wo.scales_zeros, shape=torch.Size([32, 4096]), split_dim=0, tp=1                                                               
*** splitting layers.31.attention.w_qkv.bias, shape=torch.Size([1, 12288]), split_dim=-1, tp=1                                                                   
### copying layers.31.attention.wo.bias, shape=torch.Size([4096])                                                                                                
*** splitting layers.31.feed_forward.w13.qweight, shape=torch.Size([128, 688, 128, 1]), split_dim=-1, tp=1                                                       
*** splitting layers.31.feed_forward.w13.scales_zeros, shape=torch.Size([32, 22016]), split_dim=-1, tp=1                                                         
*** splitting layers.31.feed_forward.w2.qweight, shape=torch.Size([11008, 512]), split_dim=0, tp=1                                                               
*** splitting layers.31.feed_forward.w2.scales_zeros, shape=torch.Size([86, 4096]), split_dim=0, tp=1 

```