# 糖三藏

面向 2 型糖尿病的可溯源循证问答系统。  
副标题：三藏辨经，答案有据。

系统先检索指南与说明书，再按权威、适用、一致审视证据并给出可信度，由选择性门控决定采信检索生成或保留直接判断，最终回答可定位到 PDF 原文。

正式评测（测试集 685 题，温度 0）：直接生成 609/685（88.9%），普通稠密检索 618/685（90.2%），本系统 619/685（90.4%）。有害检索由约 19/609（3.12%）降至约 4/609（0.66%）。

## 仓库里有什么

本仓库提供源代码、评测划分和建库用文献，不附带模型权重和向量库。

| 路径 | 作用 |
|---|---|
| `src/rag/` | 专科判断、三藏辨经、选择性门控 |
| `src/eval/` | 评测脚本 |
| `web/` | 演示端与原文定位 |
| `data/eval/` | 评测划分与锁定阈值 |
| `data/raw/manual/` | 指南与说明书，供重建知识库 |
| `build_kb.py` | 构建向量库 |

## 运行前准备

1. 安装 Python 3.10 及以上，然后安装依赖：

```bash
pip install -r requirements.txt
pip install modelscope
```

2. 复制环境文件并填写自己的大模型密钥（不要把真实密钥推送到仓库）：

```bash
copy .env.example .env
```

`.env` 中设置：

```text
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

3. 下载开源向量模型到 `models/`（国内建议用 ModelScope）：

- BGE-M3：https://www.modelscope.cn/models/BAAI/bge-m3  
  备用：https://huggingface.co/BAAI/bge-m3
- bge-reranker-v2-m3（证据适用性打分时使用）：https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3  
  备用：https://huggingface.co/BAAI/bge-reranker-v2-m3

```bash
modelscope download --model BAAI/bge-m3 --local_dir models/bge-m3
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir models/bge-reranker-v2-m3
```

4. 用仓库中的文献重建向量库（首次需要一定时间）：

```bash
python build_kb.py
```

5. 启动演示：

```bash
python web/app.py
```

浏览器打开 http://127.0.0.1:7860 。

## 开源组件

检索编码使用 BAAI/bge-m3，向量库存放使用 Chroma，重排按需使用 bge-reranker-v2-m3。演示服务使用 FastAPI。文本生成通过 OpenAI 兼容接口调用大模型，密钥由运行者自行配置。测评题筛选自 CMB-Exam、CMExam、MedQA-ZH 等公开中文医学数据。自研部分是三藏辨经、选择性门控与原文溯源。
