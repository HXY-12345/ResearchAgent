"""
深度研究代理系统 - 状态管理模块
============================================

功能：定义工作流中所有数据模型和状态结构

核心概念：
    - ResearchState：LangGraph 工作流的核心状态对象
    - 状态在智能体间自动合并（LangGraph 自动处理）
    - 每个智能体返回字典，自动更新到 ResearchState

数据流向：
    Planner ──▶ Searcher ──▶ Synthesizer ──▶ Writer
       │           │              │              │
       ▼           ▼              ▼              ▼
    plan    search_results   key_findings   final_report
"""

from typing import Annotated, List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


# ═══════════════════════════════════════════════════════════════════════════════
# 基础数据模型
# ═══════════════════════════════════════════════════════════════════════════════

class SearchQuery(BaseModel):
    """
    搜索查询模型

    用途：存储单个搜索查询及其元数据
    """
    query: str = Field(description="搜索查询文本")
    purpose: str = Field(description="执行此查询的目的/意图")
    completed: bool = Field(default=False, description="查询是否已完成")


class SearchResult(BaseModel):
    """
    搜索结果模型

    用途：存储单条搜索结果，包含原始摘要和抓取的完整内容
    """
    query: str = Field(description="原始搜索查询")
    title: str = Field(description="结果标题")
    url: str = Field(description="结果 URL")
    snippet: str = Field(description="结果摘要（来自搜索引擎）")
    content: Optional[str] = Field(default=None, description="完整抓取的网页内容")


class ReportSection(BaseModel):
    """
    报告章节模型

    用途：存储报告的单一章节内容
    """
    title: str = Field(description="章节标题")
    content: str = Field(description="章节内容（Markdown 格式）")
    sources: List[str] = Field(default_factory=list, description="使用的来源 URL 列表")


class ResearchPlan(BaseModel):
    """
    研究计划模型

    用途：存储 ResearchPlanner 智能体生成的研究计划
    包含：研究目标、搜索查询列表、报告大纲
    """
    topic: str = Field(description="研究主题")
    objectives: List[str] = Field(description="研究目标列表（通常 3-5 个）")
    search_queries: List[SearchQuery] = Field(description="待执行的搜索查询列表")
    report_outline: List[str] = Field(description="报告章节大纲（最多 8 节）")


# ═══════════════════════════════════════════════════════════════════════════════
# 核心状态模型
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchState(BaseModel):
    """
    研究工作流状态（核心数据容器）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    LangGraph 状态合并机制：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    每个智能体节点返回一个字典，LangGraph 自动将其合并到状态中：

        # 智能体返回
        def plan_node(state: ResearchState) -> Dict[str, Any]:
            return {"plan": new_plan, "current_stage": "searching"}

        # LangGraph 自动执行（伪代码）
        state = {**state, **{"plan": new_plan, "current_stage": "searching"}}

    这意味着智能体只需要返回它要更新的字段，其他字段保持不变。

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    字段分类：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    # ═══════════════════════════════════════════════════════════════
    # 用户输入
    # ═══════════════════════════════════════════════════════════════
    research_topic: str = Field(description="研究主题（用户输入）")

    # ═══════════════════════════════════════════════════════════════
    # 规划阶段输出
    # ═══════════════════════════════════════════════════════════════
    plan: Optional[ResearchPlan] = Field(
        default=None,
        description="研究计划（由 ResearchPlanner 生成）"
    )

    # ═══════════════════════════════════════════════════════════════
    # 搜索阶段输出
    # ═══════════════════════════════════════════════════════════════
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="所有搜索结果（由 ResearchSearcher 收集）"
    )

    # ═══════════════════════════════════════════════════════════════
    # 综合阶段输出
    # ═══════════════════════════════════════════════════════════════
    key_findings: List[str] = Field(
        default_factory=list,
        description="关键发现列表（由 ResearchSynthesizer 提取）"
    )

    # ═══════════════════════════════════════════════════════════════
    # 报告生成阶段输出
    # ═══════════════════════════════════════════════════════════════
    report_sections: List[ReportSection] = Field(
        default_factory=list,
        description="报告章节列表（由 ReportWriter 生成）"
    )

    final_report: Optional[str] = Field(
        default=None,
        description="完整最终报告（Markdown 格式）"
    )

    # ═══════════════════════════════════════════════════════════════
    # 工作流控制
    # ═══════════════════════════════════════════════════════════════
    current_stage: Literal[
        "planning",      # 规划阶段
        "searching",     # 搜索阶段
        "synthesizing",  # 综合阶段
        "reporting",     # 报告生成阶段
        "complete"       # 完成状态
    ] = Field(default="planning", description="当前工作流阶段")

    error: Optional[str] = Field(
        default=None,
        description="错误信息（如果有，工作流将终止）"
    )

    # ═══════════════════════════════════════════════════════════════
    # 元数据
    # ═══════════════════════════════════════════════════════════════
    iterations: int = Field(default=0, description="工作流迭代次数")

    # ═══════════════════════════════════════════════════════════════
    # 质量与指标
    # ═══════════════════════════════════════════════════════════════
    quality_score: Optional[Dict] = Field(
        default=None,
        description="报告质量评分指标"
    )

    credibility_scores: List[Dict] = Field(
        default_factory=list,
        description="来源可信度评分列表（与 search_results 一一对应）"
    )

    # ═══════════════════════════════════════════════════════════════
    # LLM 调用追踪
    # ═══════════════════════════════════════════════════════════════
    llm_calls: int = Field(default=0, description="LLM API 调用总次数")
    total_input_tokens: int = Field(default=0, description="输入 token 总数")
    total_output_tokens: int = Field(default=0, description="输出 token 总数")

    llm_call_details: List[Dict] = Field(
        default_factory=list,
        description="每次 LLM 调用的详细信息（agent、operation、tokens 等）"
    )

    # ═══════════════════════════════════════════════════════════════
    # Pydantic 配置
    # ═══════════════════════════════════════════════════════════════
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型的字段

