# Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.57+-green.svg)](https://github.com/langchain-ai/langgraph)

基于 LangGraph 和 LangChain 构建的面向生产环境的多智能体自主研究系统。四个专业智能体协同工作，对任何主题进行深入研究，生成带有引用支持和可信度评分的详细报告。

**支持：** 本地模型（Ollama、llama.cpp）和云 API（Google Gemini、OpenAI）

---

## 目录

- [演示](#演示)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [安装](#安装)
- [使用方法](#使用方法)
- [配置](#配置)
- [项目结构](#项目结构)
- [核心组件](#核心组件)
- [API 参考](#api-参考)
- [贡献](#贡献)
- [许可证](#许可证)

---

## 演示

https://github.com/user-attachments/assets/df8404c6-7423-4a49-864a-bd4d59885c1b

*观看完整演示视频，了解 Deep Research Agent 的实际运行效果，展示多智能体工作流程、实时进度更新和综合报告生成。*

---

## 功能特性

### 核心能力

| 功能 | 描述 |
|------|------|
| **多智能体架构** | 由 LangGraph 的 StateGraph 编排的四个专业自主智能体 |
| **自主研究** | 搜索智能体动态决定查询、来源和提取深度 |
| **可信度评分** | 基于域名权威性的自动来源评估（0-100 分） |
| **质量验证** | 节级验证，支持重试逻辑和指数退避 |
| **多格式导出** | 支持 Markdown、HTML 和纯文本格式的报告 |
| **LLM 使用追踪** | 实时监控 API 调用、token 和成本 |
| **研究缓存** | 基于文件的研究缓存，7天 TTL，MD5 主题哈希 |
| **Web 界面** | 交互式 Chainlit UI，实时进度显示 |

### 生产级特性

| 功能 | 描述 |
|------|------|
| **熔断器** | 外部服务的自动故障检测和恢复 |
| **连接池** | 通过 httpx 实现 HTTP/2 持久连接 |
| **检查点** | 工作流状态持久化，支持崩溃恢复 |
| **类型化异常** | 领域特定的错误处理，便于调试 |
| **依赖注入** | 可注入 LLM 的可测试智能体架构 |
| **搜索提供商抽象** | 可扩展的搜索后端（DuckDuckGo，易于添加其他） |

---

## 系统架构

### 高级流程

![Deep Research Agent Flow Diagram](assets/flow.png)

### 智能体职责

#### ResearchPlanner（研究规划器）
- 分析研究主题并生成 3-5 个 SMART 目标
- 创建涵盖不同方面的针对性搜索查询
- 设计包含最多 8 个章节的报告大纲
- 使用结构化 JSON 输出确保可靠性

#### ResearchSearcher（研究搜索器 - 自主智能体）
- 基于 LangChain 的自主智能体，使用 `create_agent()`
- 动态决定要执行的查询
- 使用 `web_search` 和 `extract_webpage_content` 工具
- 所有来源均进行可信度评分和过滤（默认阈值：40）
- 针对服务故障的熔断器保护

#### ResearchSynthesizer（研究综合器）
- 以可信度感知方式分析聚合结果
- 优先考虑高可信度来源（评分 ≥70）
- 使用可信度层次结构解决矛盾
- 渐进式截断处理 token 限制

#### ReportWriter（报告撰写器）
- 以学术语调生成结构化章节
- 添加正确的引用（APA、MLA、Chicago、IEEE）
- 验证章节质量，失败时重试
- 编译包含参考文献的最终 markdown

---

## 安装

### 前置要求

- Python 3.11+
- pip 或 uv 包管理器
- 以下任一选项：
  - [Ollama](https://ollama.com/)（本地模型）
  - [llama.cpp](https://github.com/ggerganov/llama.cpp)（本地模型，最高性能）
  - [Google Gemini API](https://makersuite.google.com/app/apikey)（云端）
  - [OpenAI API](https://platform.openai.com/api-keys)（云端）

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/tarun7r/deep-research-agent.git
cd deep-research-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

### 使用 Ollama（推荐用于本地）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen2.5:7b

# 配置 .env
MODEL_PROVIDER=ollama
MODEL_NAME=qwen2.5:7b
SUMMARIZATION_MODEL=qwen2.5:7b
```

### 使用 llama.cpp（最高性能）

```bash
# 下载 GGUF 模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models

# 启动支持工具调用的服务器
./llama-server -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 -ngl 35 --ctx-size 4096 --jinja

# 配置 .env
MODEL_PROVIDER=llamacpp
MODEL_NAME=qwen2.5-7b-instruct-q4_k_m
LLAMACPP_BASE_URL=http://localhost:8080
```

### 使用云 API

```bash
# Gemini
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.5-flash

# OpenAI
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

---

## 使用方法

### 命令行

```bash
# 交互模式
python main.py

# 直接指定主题
python main.py "量子计算对密码学的影响"
```

### Web 界面

```bash
chainlit run app.py --host 127.0.0.1 --port 8000
```

功能：
- 实时进度与阶段指示器
- 质量指标和 LLM 使用统计
- 多格式下载（MD、HTML、TXT）
- 研究历史追踪

### 编程 API

```python
import asyncio
from src.graph import run_research

async def research():
    # 基本用法
    state = await run_research(
        topic="您的研究主题",
        verbose=True,
        use_cache=True
    )

    # 访问结果
    print(state["final_report"])
    print(f"来源数量: {len(state['search_results'])}")
    print(f"发现数量: {len(state['key_findings'])}")
    print(f"Token数: {state['total_input_tokens'] + state['total_output_tokens']:,}")

asyncio.run(research())
```

### 持久化（崩溃恢复）

```python
from src.graph import run_research_with_persistence, resume_research

# 使用 SQLite 持久化运行
state = await run_research_with_persistence(
    topic="您的研究主题",
    thread_id="my-research-001"
)

# 恢复中断的工作流程
state = await resume_research(thread_id="my-research-001")
```

---

## 配置

### 环境变量

```bash
# =============================================================================
# 模型提供商（必需）
# =============================================================================
MODEL_PROVIDER=gemini              # 选项：ollama、llamacpp、gemini、openai

# =============================================================================
# 提供商特定设置
# =============================================================================

# Ollama
MODEL_NAME=qwen2.5:7b
SUMMARIZATION_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434

# llama.cpp
MODEL_NAME=qwen2.5-7b-instruct-q4_k_m
LLAMACPP_BASE_URL=http://localhost:8080

# Gemini
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.5-flash
SUMMARIZATION_MODEL=gemini-2.5-flash

# OpenAI
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com  # 可选
MODEL_NAME=gpt-4o-mini
SUMMARIZATION_MODEL=gpt-4o-mini

# =============================================================================
# 搜索设置（可选）
# =============================================================================
MAX_SEARCH_QUERIES=3               # 搜索查询数量
MAX_SEARCH_RESULTS_PER_QUERY=3     # 每个查询的结果数
MIN_CREDIBILITY_SCORE=40           # 过滤阈值（0-100）

# =============================================================================
# 报告设置（可选）
# =============================================================================
MAX_REPORT_SECTIONS=8              # 报告最大章节数
CITATION_STYLE=apa                 # 选项：apa、mla、chicago、ieee
```

### 模型提供商对比

| 提供商 | 成本 | 隐私 | 速度 | 设置 |
|--------|------|------|------|------|
| **Ollama** | 免费 | 本地 | 快 | 简单 |
| **llama.cpp** | 免费 | 本地 | 最快 | 手动 |
| **Gemini** | 免费额度 | 云端 | 快 | API 密钥 |
| **OpenAI** | 按量付费 | 云端 | 快 | API 密钥 |

---

## 项目结构

```
deep-research-agent/
├── src/
│   ├── __init__.py           # 包初始化
│   ├── config.py             # 配置管理（Pydantic）
│   ├── state.py              # 状态模型（ResearchState 等）
│   ├── agents.py             # 带依赖注入的智能体实现
│   ├── graph.py              # LangGraph 工作流 + 检查点
│   ├── callbacks.py          # 进度回调系统
│   ├── llm_tracker.py        # Token 和成本追踪
│   ├── exceptions.py         # 类型化领域异常
│   │
│   ├── prompts/              # 提取的提示模板
│   │   ├── __init__.py
│   │   ├── planner.py        # 规划提示
│   │   ├── searcher.py       # 搜索提示
│   │   ├── synthesizer.py    # 综合提示
│   │   └── writer.py         # 撰写提示
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tools.py          # LangChain @tool 函数
│       ├── web_utils.py      # httpx 客户端、熔断器、搜索提供商
│       ├── cache.py          # 研究缓存（7天 TTL）
│       ├── credibility.py    # 来源可信度评分
│       ├── citations.py      # 引用格式化
│       ├── exports.py        # 多格式导出
│       └── history.py        # 研究历史
│
├── outputs/                  # 生成的报告
├── .cache/
│   ├── research/             # 缓存结果
│   ├── checkpoints/          # 工作流检查点（SQLite）
│   └── research_history.json
│
├── assets/                   # 文档资源
├── main.py                   # CLI 入口点
├── app.py                    # Chainlit Web 界面
├── requirements.txt          # 依赖
├── pyproject.toml            # 项目元数据
├── LICENSE                   # MIT 许可证
└── README.md
```

---

## 核心组件

### 异常层次结构

```python
DeepResearchError
├── ConfigurationError
├── PlanningError
├── SearchError
│   └── RateLimitError
├── ContentExtractionError
├── SynthesisError
├── ReportGenerationError
├── CircuitOpenError
└── LLMError
```

### 可信度评分

来源基于以下因素评分（0-100）：

| 因素 | 分数 |
|------|------|
| 受信任的域名（.edu、.gov、学术） | +30 |
| 启用 HTTPS | +5 |
| 学术/研究路径 | +10 |
| 可疑的 TLD（.xyz、.tk） | -20 |
| 无 HTTPS | -10 |

默认过滤阈值：40（可通过 `MIN_CREDIBILITY_SCORE` 配置）

### 熔断器状态

```
CLOSED ──► (5 次失败) ──► OPEN ──► (30s 超时) ──► HALF_OPEN ──► (成功) ──► CLOSED
                              │                            │
                              └──────── (失败) ◄────────┘
```

---

## API 参考

### 核心函数

```python
# 主要研究函数
async def run_research(
    topic: str,
    verbose: bool = True,
    use_cache: bool = True,
    use_checkpoints: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]

# 使用 SQLite 持久化
async def run_research_with_persistence(
    topic: str,
    verbose: bool = True,
    use_cache: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]

# 恢复中断的工作流程
async def resume_research(
    thread_id: str,
    additional_input: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]

# 检查工作流程状态
async def get_workflow_state(thread_id: str) -> Optional[Dict[str, Any]]

# 列出已保存的线程
def list_research_threads() -> List[str]
```

### 响应结构

```python
{
    "research_topic": str,
    "plan": ResearchPlan,
    "search_results": List[SearchResult],
    "credibility_scores": List[Dict],
    "key_findings": List[str],
    "report_sections": List[ReportSection],
    "final_report": str,
    "current_stage": str,
    "error": Optional[str],
    "iterations": int,
    "llm_calls": int,
    "total_input_tokens": int,
    "total_output_tokens": int,
    "llm_call_details": List[Dict]
}
```

---

## 输出格式

报告遵循以下结构：

```markdown
# [研究主题]

**深度研究报告**

## 执行摘要
[概述，包含来源数量和章节数量]

## 研究目标
1. [目标 1]
2. [目标 2]
...

---

## [章节 1 标题]
[内容，包含内联引用 [1]、[2]]

## [章节 2 标题]
[内容，包含内联引用 [3]、[4]]

---

## 参考文献
1. [格式化引用 - APA/MLA/Chicago/IEEE]
2. [格式化引用]
...

---

**注意：** 优先考虑了 X 个高可信度来源。
```

---

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 添加新的搜索提供商

```python
# src/utils/web_utils.py
class GoogleSearchProvider(SearchProvider):
    @property
    def name(self) -> str:
        return "google"

    async def search(self, query: str, max_results: int) -> List[SearchResult]:
        # 实现
        pass

# 在 WebSearchTool 中注册
tool = WebSearchTool(providers=[
    DuckDuckGoProvider(),
    GoogleSearchProvider()  # 备用
])
```

### 自定义提示

编辑 `src/prompts/` 中的文件：
- `planner.py` - 研究规划策略
- `searcher.py` - 搜索智能体指令
- `synthesizer.py` - 综合方法论
- `writer.py` - 报告撰写风格

---

## 贡献

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'Add amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 打开 Pull Request

---

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 致谢

构建于：
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排
- [LangChain](https://github.com/langchain-ai/langchain) - LLM 框架
- [Chainlit](https://github.com/Chainlit/chainlit) - Web 界面
- [httpx](https://www.python-httpx.org/) - 异步 HTTP 客户端
- [DuckDuckGo](https://duckduckgo.com/) - 网页搜索

支持：
- [Ollama](https://ollama.com/) & [llama.cpp](https://github.com/ggerganov/llama.cpp) - 本地模型
- [Google Gemini](https://ai.google.dev/) & [OpenAI](https://openai.com/) - 云 API

---
