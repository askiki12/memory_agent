# 长期记忆对话 Agent (Long-Term Memory Dialog Agent)

NLP 课程项目：从多会话对话中自动提取、存储、检索并管理记忆，在 LoCoMo 评测集上评估长期记忆问答能力。

## 目录结构

```
memory_agent/
├── agent/                        # Agent 主控模块
│   └── controller.py             # MyMemoryAgent 主流程编排：写入 → 检索 → 生成
├── memory/                       # 记忆子系统（核心实现）
│   ├── store.py                  # FAISS 向量存储与索引（MemoryStore）
│   ├── writer.py                 # LLM 驱动的记忆提取（MemoryWriter）
│   ├── retriever.py              # 语义检索 + 新近度加权（MemoryRetriever）
│   └── updater.py                # 去重合并 + 容量裁剪（MemoryUpdater）
├── eval/                         # 评测入口
│   └── run_eval.py               # 一键评测流水线（自动准备数据、跑基线、生成+评判）
├── eval_kit/                     # TA 提供的评测工具包
│   ├── agent_template.py         # Agent 接口定义 + FullContextAgent 基线
│   ├── no_memory_agent.py        # No-Memory 基线（仅用 query，不看历史）
│   ├── vanilla_rag_agent.py      # Vanilla RAG 基线（原始对话切片 → 向量检索）
│   ├── llm_client.py             # OpenAI 兼容 LLM 客户端
│   ├── metrics.py                # SQuAD 风格 F1 / Exact Match
│   ├── prepare_eval_set.py       # 下载 LoCoMo → 分层抽样 → eval_set.json
│   ├── run_generation.py         # 加载 Agent → ingest() → answer() → 输出预测
│   ├── run_judge.py              # LLM-as-Judge 评测（CORRECT/PARTIAL/WRONG）
│   ├── eval_set.json             # 已准备好的评测集（需运行 prepare_eval_set.py 生成）
│   └── requirements.txt          # eval_kit 依赖
├── models/                       # 本地下载的模型权重（.gitignore 排除，需自行下载）
│   ├── Qwen2.5-3B-Instruct-AWQ/  # 主 LLM（~2.6 GB，AWQ 量化）
│   └── bge-small-zh-v1.5/        # Embedding 模型（~184 MB）
├── experiments/results/          # 实验输出目录
│   ├── predictions/              # 各 Agent 的预测 JSON
│   └── evals/                    # Judge 评测结果 JSON
├── download_models.py            # 模型下载脚本（HF / ModelScope 双通道）
├── pyproject.toml                # uv 项目配置与依赖声明
├── .env.example                  # 环境变量模板
├── .gitignore                    # 排除 .env、models/、__pycache__ 等
├── report.md                     # 期中进展报告
└── README.md                     # 本文件
```

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                  Agent Controller                │
│       ingest(conversation) → answer(question)   │
└──────┬──────────┬───────────┬───────────────────┘
       │          │           │
       ▼          ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Writer  │ │Retriever │ │ Updater  │
│ LLM 提取 │ │ 语义检索  │ │ 去重/裁剪 │
│ 原子事实  │ │+新近度加权│ │          │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     ▼            ▼            │
┌─────────────────────────────────────┐
│           Memory Store              │
│    FAISS IndexFlatIP + Metadata     │
└─────────────────────────────────────┘
```

**核心流程：**

1. **写入 (Write)**：`MemoryWriter` 对每个 session 调用 LLM，抽取原子事实（个人信息、事件、偏好、关系、计划、知识 6 类），输出结构化 JSON。
2. **去重合并 (Update)**：`MemoryUpdater` 用 embedding 相似度检测新记忆与已有记忆的冲突，保留较新版本；当记忆数超限时裁掉最旧的。
3. **存储 (Store)**：`MemoryStore` 基于 FAISS IndexFlatIP（内积=余弦相似度），L2 归一化向量 + 内存元数据字典。
4. **检索 (Retrieve)**：`MemoryRetriever` 对 query 编码后做语义搜索，按 session_id 施加新近度 boost，返回 top-k 记忆。
5. **生成 (Answer)**：`MyMemoryAgent.answer()` 将检索到的记忆拼入 prompt，调用 LLM 生成简短答案。

## 环境准备

### 1. 系统要求

- **Python** ≥ 3.10
- **GPU**：NVIDIA RTX 3070/4060 8GB+（本地部署 LLM 用；若无 GPU，可用云端 API 替代）
- **CUDA** 12.x（vLLM 依赖）
- **Docker**（推荐用 Docker 运行 vLLM，避免环境污染）

### 2. 创建虚拟环境

```bash
cd memory_agent

# 方式一：使用 uv（推荐）
uv sync

# 方式二：使用 pip
python -m venv .venv
source .venv/bin/activate
pip install openai numpy sentence-transformers faiss-cpu
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入实际配置
```

`.env` 说明：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | LLM 服务地址（vLLM / 云端 API） |
| `LLM_API_KEY` | `EMPTY` | API Key（vLLM 本地不需要） |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct-AWQ` | 模型名称 |
| `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding 模型（HuggingFace ID） |
| `EMBED_MODEL_PATH` | (空) | Embedding 模型本地路径（优先级高于 EMBED_MODEL） |

### 4. 下载模型

```bash
# 自动下载 LLM + Embedding 模型到 models/ 目录
python download_models.py
```

> 国内网络自动尝试 `hf-mirror.com` 镜像，失败则回退到 ModelScope。

### 5. 启动 LLM 服务

**方式一：Docker vLLM（推荐）**

```bash
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/models \
    vllm/vllm-openai:latest \
    --model /models/Qwen2.5-3B-Instruct-AWQ \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75
```

**方式二：本地 vLLM**

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75
```

启动后，服务在 `http://localhost:8000/v1` 提供 OpenAI 兼容 API。

### 6. 验证环境

```bash
# 确认 vLLM 服务可用
curl http://localhost:8000/v1/models

# 快速验证流水线（2 个对话）
python eval_kit/run_generation.py \
    --eval_set eval_kit/eval_set.json \
    --agent agent_template:FullContextAgent \
    --output /tmp/test_pred.json \
    --limit_conversations 2
```

## 使用指南

### 三步评测流程

#### 第 1 步：准备评测集

```bash
cd memory_agent
source .venv/bin/activate
python eval_kit/prepare_eval_set.py \
    --output eval_kit/eval_set.json \
    --per_category 40 \
    --seed 42

# 快速调试：--per_category 5
```

- 脚本自动从 GitHub 克隆 LoCoMo 数据集（~2.7 MB），缓存到 `.locomo_cache/`
- 默认排除对抗性类别（cat 5），共 4 类 × 40 题 = 160 题
- 类别：`single_hop` / `temporal` / `multi_hop` / `open_domain`

#### 第 2 步：运行 Agent 生成预测

```bash
# 运行自定义记忆系统
python eval_kit/run_generation.py \
    --eval_set eval_kit/eval_set.json \
    --agent agent.controller:MyMemoryAgent \
    --output experiments/results/predictions/predictions_mine.json

# 运行基线（FullContext / NoMemory / VanillaRAG）
python eval_kit/run_generation.py \
    --eval_set eval_kit/eval_set.json \
    --agent agent_template:FullContextAgent \
    --output experiments/results/predictions/predictions_fullctx.json

# 调试模式：只跑前 2 个对话，断点续跑
python eval_kit/run_generation.py \
    --eval_set eval_kit/eval_set.json \
    --agent agent.controller:MyMemoryAgent \
    --output experiments/results/predictions/predictions_mine.json \
    --limit_conversations 2 \
    --resume
```

`--agent` 参数格式：`模块路径:类名`，脚本会动态导入并以零参数实例化。

#### 第 3 步：LLM-as-Judge 评测

Judge 推荐使用云端 API（避免与本地生成模型共用显存，且不同模型族可避免 self-evaluation bias）：

```bash
# 配置 Judge API（例如 DeepSeek V4 Flash，极便宜）
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_API_KEY="sk-xxxxxxxx"
export LLM_MODEL="deepseek-v4-flash"

# 运行评测
python eval_kit/run_judge.py \
    --predictions experiments/results/predictions/predictions_mine.json \
    --output experiments/results/evals/results_mine.json \
    --num_workers 4
```

输出示例：

```
===== 评测结果 =====
Judge 模型：deepseek-v4-flash
类别             题数     得分      F1      EM   正确   部分   错误
multi_hop        40    0.387    0.241    0.075    12     7    21
open_domain      40    0.525    0.318    0.125    17     9    14
single_hop       40    0.712    0.456    0.250    25     7     8
temporal         40    0.400    0.198    0.050    14     4    22
总体            160    0.506    0.303    0.125
```

### 一键评测脚本

```bash
# 自动跑全流程：准备数据 → 跑 3 个基线 → 评判 → 重新跑你的 Agent → 评判
python eval/run_eval.py
```

该脚本会智能跳过已存在的评测集和基线结果，每次必然重新生成你的 Agent 的预测和评判。

## Agent 接口

自定义 Agent 必须实现以下接口（参考 `eval_kit/agent_template.py`）：

```python
class MyMemoryAgent:
    def __init__(self):
        """初始化：每个对话新建一个实例，状态不跨对话共享。"""
        ...

    def ingest(self, conversation: dict) -> None:
        """读入完整的多会话对话，构建记忆。仅调用一次，在所有 answer() 之前。"""
        ...

    def answer(self, question: str) -> str:
        """基于已有记忆回答问题，返回简短字符串（短语或单句）。"""
        ...
```

`conversation` 结构：

```json
{
  "speaker_a": "Caroline",
  "speaker_b": "Melanie",
  "sessions": [
    {
      "session_id": 1,
      "date_time": "1:56 pm on 8 May 2023",
      "turns": [
        {"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}
      ]
    }
  ]
}
```

## 记忆系统设计

### MemoryWriter — 记忆提取

- 对每个 session 调用 LLM，抽取 6 类原子事实：`personal_info` / `event` / `preference` / `relationship` / `plan` / `knowledge`
- 每条记忆为自包含的陈述句，脱离上下文也可理解
- 输出为 JSON 数组，每项含 `fact` 和 `category`

### MemoryStore — 向量存储

- **FAISS IndexFlatIP**：内积索引，配合 L2 归一化实现余弦相似度
- **元数据字典**：FAISS ID → `{text, category, session_id, date_time, mem_id}`
- 支持增删查，删除为软删除（清元数据）

### MemoryUpdater — 去重与裁剪

- **去重**：新记忆与已有记忆的 embedding 余弦相似度 > `similarity_threshold`（默认 0.90）时，保留日期较新者
- **裁剪**：记忆数超过 `max_memories`（默认 500）时，按日期删最旧的

### MemoryRetriever — 检索

- 对 query 编码后做语义搜索，返回 top-k（默认 10）
- 按 `session_id` 施加新近度加权：`score += (session_id / max_session_id) × recency_weight`
- 输出格式化为带日期标注的上下文文本

## 实验基线

| 基线 | Agent 类 | 说明 |
|------|----------|------|
| No-Memory | `no_memory_agent:NoMemoryAgent` | 仅用当前 query，不看对话历史 |
| Full-Context | `agent_template:FullContextAgent` | 全部对话历史截断后塞入 prompt |
| Vanilla RAG | `vanilla_rag_agent:VanillaRAGAgent` | 原始对话切片 → 向量检索 → prompt |
| **自定义系统** | `agent.controller:MyMemoryAgent` | 提取记忆 → 去重存储 → 检索 → 生成 |

## 设计约束

- 核心记忆逻辑自实现，不使用 LangChain 的 `ConversationBufferMemory` / `VectorStoreMemory`
- 明确区分原始对话日志与派生的记忆单元——不直接将原始对话扔进向量库
- 所有记忆操作可追踪：`save_log()` 保存每次问答的检索记忆、完整 prompt 和模型输出
- API Key 放在 `.env`（已 gitignore），不写入源码
- Judge 使用与生成不同的模型族（推荐 DeepSeek），避免 self-evaluation bias

## 常见问题

**Q: vLLM 启动 OOM？**
降低 `--max-model-len 4096`，或 `--gpu-memory-utilization 0.65`。8G 显存必须用 AWQ 量化版。

**Q: `run_generation.py` 找不到模块？**
脚本需要从 `memory_agent/` 目录运行，或设置 `PYTHONPATH=.`。

**Q: 如何调试？**
加 `--limit_conversations 1` 只跑一个对话，确认 ingest → answer 全链路通后再扩到全集。

**Q: 如何切换云端 API？**
修改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_API_KEY` 指向云端服务（DeepSeek / DashScope / SiliconFlow 等），无需本地 GPU。
