"""
深度研究代理系统 - LangGraph 工作流编排模块
============================================================

功能：定义并管理研究工作流的状态图

核心组件：
    - StateGraph: 工作流状态图
    - Checkpointing: 检查点持久化（支持崩溃恢复）
    - 路由函数: 条件边决策逻辑
    - 执行函数: 运行研究工作流的入口

工作流结构：
    START → plan → search → synthesize → write_report → END
              ↓         ↓           ↓
            END      END         END
              (任何节点出错都可以终止)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
状态合并机制（LangGraph 核心特性）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每个节点返回一个字典，LangGraph 自动将其合并到全局状态中：

    节点返回: {"plan": new_plan, "current_stage": "searching"}
    LangGraph: state = {**old_state, **return_value}
    结果: state.plan 被更新，state.current_stage 被更新

这样节点只需要关心"我要更新什么"，不需要关心其他字段。
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 标准库导入
# ═══════════════════════════════════════════════════════════════════════════════
import os
import uuid        # 用于生成唯一的 thread_id
import sqlite3     # 用于查询 SQLite 检查点数据库
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager  # 用于上下文管理器

# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph 导入
# ═══════════════════════════════════════════════════════════════════════════════
from langgraph.graph import StateGraph, START, END     # 状态图核心组件
from langgraph.checkpoint.memory import MemorySaver    # 内存检查点
from langgraph.checkpoint.sqlite import SqliteSaver    # SQLite 检查点

# ═══════════════════════════════════════════════════════════════════════════════
# 项目内部导入
# ═══════════════════════════════════════════════════════════════════════════════
from src.state import ResearchState                           # 状态模型
from src.agents import ResearchPlanner, ResearchSearcher, ResearchSynthesizer, ReportWriter  # 四个智能体
from src.utils.cache import ResearchCache                    # 研究缓存
from src.config import config                                 # 全局配置
from src.exceptions import DeepResearchError                # 自定义异常

import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 检查点管理（Checkpointing）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 检查点功能：将工作流状态持久化到存储，支持：
# 1. 崩溃恢复：程序中断后可以从上次检查点继续
# 2. 状态查询：查看工作流当前进度
# 3. 长运行任务：避免长时间运行后失败需要重头开始
# ═══════════════════════════════════════════════════════════════════════════════

def get_checkpoint_path() -> Path:
    """
    获取 SQLite 检查点数据库的文件路径

    Returns:
        Path: .cache/checkpoints/research_checkpoints.db
    """
    cache_dir = Path(".cache/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "research_checkpoints.db"


def create_memory_checkpointer() -> MemorySaver:
    """
    创建内存检查点（不持久化）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    特点：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 状态保存在内存中
    - 程序重启后丢失
    - 适合短期测试和开发

    Returns:
        MemorySaver: LangGraph 内存检查点实例
    """
    return MemorySaver()


@contextmanager
def create_sqlite_checkpointer():
    """
    创建 SQLite 检查点（上下文管理器）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    特点：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 状态持久化到磁盘
    - 程序重启后可以恢复
    - 适合生产环境

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    使用方式：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)
        result = await graph.ainvoke(...)

    Yields:
        SqliteSaver: LangGraph SQLite 检查点实例
    """
    checkpoint_path = get_checkpoint_path()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        logger.info(f"SQLite checkpointer initialized: {checkpoint_path}")
        yield checkpointer


# ═══════════════════════════════════════════════════════════════════════════════
# 图构建（Graph Construction）
# ═══════════════════════════════════════════════════════════════════════════════

def create_research_graph(checkpointer=None):
    """
    创建研究工作流状态图

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    工作流结构：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    ┌─────────────────────────────────────┐
                    │         START                       │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │          plan 节点                   │
                    │  (ResearchPlanner.plan)             │
                    │  - 生成研究计划                      │
                    │  - 输出: plan                      │
                    └──────────────┬──────────────────────┘
                                   │
                        ┌──────────┴───────────┐
                        │ (路由决策)            │
                        ▼                       │
                  ┌─────────────┐           │
                  │ 有 error?   │           │
                  └─────┬───────┘           │
                   Yes│ No                  │
                      ▼                     │
                    END                    ▼
                                          ┌───────────────────┐
                                          │    search 节点      │
                                          │ (ResearchSearcher.  │
                                          │  search)           │
                                          │ - 执行网络搜索      │
                                          │ - 输出: search_results│
                                          └──────────┬─────────┘
                                                     │
                                    ┌────────────────┴─────────┐
                                    │ (路由决策)              │
                                    ▼                         │
                              ┌─────────────┐                     │
                              │ 有 error?   │                     │
                              └─────┬───────┘                     │
                                 │ No                        │
                                 ▼                           │
                               END                          ▼
                                                    ┌───────────────────┐
                                                    │  synthesize 节点   │
                                                    │ (ResearchSynthesizer│
                                                    │  .synthesize)      │
                                                    │ - 分析结果         │
                                                    │ - 输出: key_findings│
                                                    └──────────┬─────────┘
                                                               │
                                                   ┌──────────────┴───────┐
                                                   │ (路由决策)          │
                                                   ▼                      │
                                             ┌─────────────┐            │
                                             │ 有 error?   │            │
                                             └─────┬───────┘            │
                                                │ No                  │
                                                ▼                     │
                                               END                    ▼
                                                                   ┌───────────────────┐
                                                                   │ write_report 节点  │
                                                                   │ (ReportWriter.     │
                                                                   │  write_report)     │
                                                                   │ - 生成报告         │
                                                                   │ - 输出: final_report│
                                                                   └──────────┬─────────┘
                                                                              │
                                                                              ▼
                                                                           END

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Args:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        checkpointer: 可选的检查点实例（MemorySaver 或 SqliteSaver）
                       用于状态持久化，支持崩溃恢复

    Returns:
        CompiledGraph: 编译后的 LangGraph 工作流（可执行）
    """

    # ═══════════════════════════════════════════════════════════════
    # 步骤 1: 实例化智能体
    # ═══════════════════════════════════════════════════════════════
    planner = ResearchPlanner()                          # 规划智能体
    searcher = ResearchSearcher()                         # 搜索智能体
    synthesizer = ResearchSynthesizer()                   # 综合智能体
    writer = ReportWriter(citation_style=config.citation_style)  # 写作智能体

    # ═══════════════════════════════════════════════════════════════
    # 步骤 2: 创建 StateGraph（指定状态类型）
    # ═══════════════════════════════════════════════════════════════
    # ResearchState 是状态的数据结构，LangGraph 会自动处理状态合并
    workflow = StateGraph(ResearchState)

    # ═══════════════════════════════════════════════════════════════
    # 步骤 3: 添加节点
    # ═══════════════════════════════════════════════════════════════
    # 节点名称 = 智能体方法（异步函数）
    workflow.add_node("plan", planner.plan)
    workflow.add_node("search", searcher.search)
    workflow.add_node("synthesize", synthesizer.synthesize)
    workflow.add_node("write_report", writer.write_report)

    # ═══════════════════════════════════════════════════════════════
    # 步骤 4: 添加固定边（START → plan）
    # ═══════════════════════════════════════════════════════════════
    workflow.add_edge(START, "plan")

    # ═══════════════════════════════════════════════════════════════
    # 步骤 5: 定义路由函数
    # ═══════════════════════════════════════════════════━━━━━━━━━━━━━━━
    # 路由函数检查状态并决定下一个节点：
    # - 返回节点名称（如 "search"）
    # - 返回 END 终止工作流
    # ═══════════════════════════════════════════════════════════════

    def should_continue_after_plan(state: ResearchState) -> str:
        """
        plan 节点后的路由决策

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        检查条件：
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. state.error 存在 → 终止（规划失败）
        2. state.plan 为空 → 终止（无有效计划）
        3. state.plan.search_queries 为空 → 终止（无搜索查询）
        4. 其他 → 继续 → search

        Args:
            state: 当前研究状态

        Returns:
            str: "search" 或 END
        """
        if state.error:
            logger.error(f"Planning failed: {state.error}")
            return END

        if not state.plan or not state.plan.search_queries:
            logger.error("No search queries generated in plan")
            return END

        logger.info(f"Plan validated: {len(state.plan.search_queries)} queries")
        return "search"

    def should_continue_after_search(state: ResearchState) -> str:
        """
        search 节点后的路由决策

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        检查条件：
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. state.error 存在 → 终止（搜索失败）
        2. search_results 为空 → 终止（无结果）
        3. search_results 少于 2 个 → 终止（结果不足）
        4. 其他 → 继续 → synthesize

        Args:
            state: 当前研究状态

        Returns:
            str: "synthesize" 或 END
        """
        if state.error:
            logger.error(f"Search failed: {state.error}")
            return END

        if not state.search_results:
            logger.warning("No search results found")
            return END

        if len(state.search_results) < 2:
            logger.warning(f"Insufficient search results: {len(state.search_results)}")
            return END

        logger.info(f"Search validated: {len(state.search_results)} results")
        return "synthesize"

    def should_continue_after_synthesize(state: ResearchState) -> str:
        """
        synthesize 节点后的路由决策

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        检查条件：
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. state.error 存在 → 终止（综合失败）
        2. key_findings 为空 → 终止（无发现）
        3. 其他 → 继续 → write_report

        Args:
            state: 当前研究状态

        Returns:
            str: "write_report" 或 END
        """
        if state.error:
            logger.error(f"Synthesis failed: {state.error}")
            return END

        if not state.key_findings:
            logger.warning("No key findings extracted")
            return END

        logger.info(f"Synthesis validated: {len(state.key_findings)} findings")
        return "write_report"

    def should_continue_after_report(state: ResearchState) -> str:
        """
        write_report 节点后的路由决策

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        这是最后一个节点，总是终止工作流

        检查条件：
        1. state.error 存在 → 记录错误日志
        2. final_report 不存在 → 记录错误日志
        3. 其他 → 记录成功日志

        Args:
            state: 当前研究状态

        Returns:
            str: 总是返回 END
        """
        if state.error:
            logger.error(f"Report generation failed: {state.error}")
        elif not state.final_report:
            logger.error("No report generated")
        else:
            logger.info("Report generation complete")

        return END

    # ═══════════════════════════════════════════════════════════════
    # 步骤 6: 添加条件边（连接节点与路由函数）
    # ═══════════════════════════════════════════════════════════════
    # 条件边：根据路由函数的返回值决定下一步

    workflow.add_conditional_edges(
        "plan",                              # 源节点
        should_continue_after_plan,         # 路由函数
        {"search": "search", END: END}      # 路由映射：返回值 → 目标节点
    )

    workflow.add_conditional_edges(
        "search",
        should_continue_after_search,
        {"synthesize": "synthesize", END: END}
    )

    workflow.add_conditional_edges(
        "synthesize",
        should_continue_after_synthesize,
        {"write_report": "write_report", END: END}
    )

    workflow.add_conditional_edges(
        "write_report",
        should_continue_after_report,
        {END: END}
    )

    # ═════════════════════════════════════════════════════════════════
    # 步骤 7: 编译图（生成可执行工作流）
    # ═══════════════════════════════════════════════════════════════
    # checkpointer 参数启用状态持久化
    return workflow.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# 研究执行函数（Research Execution）
# ═══════════════════════════════════════════════════════════════════════════════

async def run_research(
    topic: str,
    verbose: bool = True,
    use_cache: bool = True,
    use_checkpoints: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行研究工作流（标准入口函数）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    执行流程：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. 检查缓存（可选）
    2. 创建初始状态
    3. 配置检查点（可选）
    4. 运行工作流
    5. 保存结果到缓存（可选）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    检查点类型：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - use_checkpoints=True: MemorySaver（内存，重启后丢失）
    - use_checkpoints=False: 无检查点（最快，无法恢复）
    - 如需持久化，使用 run_research_with_persistence()

    Args:
        topic: 研究主题
        verbose: 是否启用详细日志
        use_cache: 是否使用缓存（7天 TTL）
        use_checkpoints: 是否启用内存检查点
        thread_id: 可选的线程 ID（自动生成如果未提供）

    Returns:
        Dict[str, Any]: 完整的研究状态（包含 final_report）
    """
    logger.info(f"Starting research on: {topic}")

    # ═══════════════════════════════════════════════════════════════
    # 步骤 1: 检查缓存
    # ═══════════════════════════════════════════════════════════════
    cache = ResearchCache()
    if use_cache:
        cached_result = cache.get(topic)
        if cached_result:
            logger.info("Using cached research result")
            return cached_result

    # ═══════════════════════════════════════════════════════════════
    # 步骤 2: 创建初始状态
    # ═══════════════════════════════════════════════════════════════
    initial_state = ResearchState(research_topic=topic)

    # ═══════════════════════════════════════════════════════════════
    # 步骤 3: 配置运行参数
    # ═══════════════════════════════════════════════════════════════
    run_config: Dict[str, Any] = {}

    if use_checkpoints:
        # 内存检查点（当前会话有效）
        checkpointer = create_memory_checkpointer()
        # 生成或使用指定的 thread_id
        tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
        run_config["configurable"] = {"thread_id": tid}
        logger.info(f"Using thread_id: {tid} for checkpoint tracking")
    else:
        checkpointer = None

    # ═══════════════════════════════════════════════════════════════
    # 步骤 4: 创建并运行工作流
    # ═══════════════════════════════════════════════════════════════
    graph = create_research_graph(checkpointer=checkpointer)

    try:
        # ainvoke: 异步调用工作流
        # initial_state: 初始状态
        # config: 运行配置（包含 thread_id）
        final_state = await graph.ainvoke(initial_state, config=run_config if run_config else None)
    except Exception as e:
        logger.error(f"Research workflow failed: {e}")
        # 如果有 thread_id，记录以便恢复
        if run_config.get("configurable", {}).get("thread_id"):
            logger.info(f"Thread ID was: {run_config['configurable']['thread_id']}")
        raise

    # ═══════════════════════════════════════════════════════════════
    # 步骤 5: 保存到缓存（如果没有错误）
    # ═══════════════════════════════════════════════════════════════
    if use_cache and not final_state.get("error"):
        cache.set(topic, final_state)

    # ═══════════════════════════════════════════════════════════════
    # 步骤 6: 输出结果摘要
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        logger.info("Workflow completed")
        if final_state.get("final_report"):
            logger.info(f"Report generated: {len(final_state['final_report'])} characters")

    return final_state


async def run_research_with_persistence(
    topic: str,
    verbose: bool = True,
    use_cache: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行研究工作流（带 SQLite 持久化）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    与 run_research 的区别：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    | 特性 | run_research | run_research_with_persistence |
    |------|--------------|----------------------------|
    | 检查点类型 | MemorySaver | SqliteSaver |
    | 持久化 | 否 | 是（磁盘） |
    | 崩溃恢复 | 当前会话 | 跨重启 |
    | 适用场景 | 测试/开发 | 生产环境 |

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    恢复工作流：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    使用 resume_research(thread_id) 可以从中断点继续

    Args:
        topic: 研究主题
        verbose: 是否启用详细日志
        use_cache: 是否使用缓存
        thread_id: 可选的线程 ID

    Returns:
        Dict[str, Any]: 完整的研究状态
    """
    logger.info(f"Starting research on: {topic}")

    # 检查缓存
    cache = ResearchCache()
    if use_cache:
        cached_result = cache.get(topic)
        if cached_result:
            logger.info("Using cached research result")
            return cached_result

    # 创建初始状态
    initial_state = ResearchState(research_topic=topic)

    # 配置持久化检查点
    tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
    run_config = {"configable": {"thread_id": tid}}
    logger.info(f"Using thread_id: {tid} for persistent checkpoint tracking")

    # 使用上下文管理器创建 SQLite 检查点
    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)

        try:
            final_state = await graph.ainvoke(initial_state, config=run_config)
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            logger.info(f"Workflow state saved to disk. Resume with thread_id: {tid}")
            raise  # 状态已保存，可以用 thread_id 恢复

    # 保存到缓存
    if use_cache and not final_state.get("error"):
        cache.set(topic, final_state)

    # 输出摘要
    if verbose:
        logger.info("Workflow completed")
        if final_state.get("final_report"):
            logger.info(f"Report generated: {len(final_state['final_report'])} characters")

    return final_state


async def resume_research(
    thread_id: str,
    additional_input: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    恢复中断的研究工作流

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    使用场景：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 程序崩溃后恢复
    - 手动暂停后继续
    - 从特定检查点重新开始

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    恢复流程：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. 从 SQLite 加载检查点状态
    2. 验证检查点存在
    3. 从断点继续执行工作流

    Args:
        thread_id: 之前运行时使用的线程 ID
        additional_input: 可选的额外输入数据

    Returns:
        Dict[str, Any]: 完整的研究状态

    Raises:
        DeepResearchError: 如果未找到对应的检查点
    """
    logger.info(f"Resuming research with thread_id: {thread_id}")

    run_config = {"configurable": {"thread_id": thread_id}}

    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)

        # 获取检查点状态
        state = await graph.aget_state(run_config)
        if not state or not state.values:
            raise DeepResearchError(f"No checkpoint found for thread_id: {thread_id}")

        logger.info(f"Found checkpoint at stage: {state.values.get('current_stage', 'unknown')}")

        # 从断点继续（可以提供额外输入）
        input_state = additional_input if additional_input else None
        final_state = await graph.ainvoke(input_state, config=run_config)

    return final_state


async def get_workflow_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    获取工作流的当前状态（不执行）

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    用途：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 检查工作流进度
    - 调试检查点状态
    - UI 显示当前阶段

    Args:
        thread_id: 线程 ID

    Returns:
        Optional[Dict[str, Any]]: 当前状态字典，如果未找到则返回 None
    """
    run_config = {"configurable": {"thread_id": thread_id}}

    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)

        state = await graph.aget_state(run_config)
        if state and state.values:
            return dict(state.values)

    return None


def list_research_threads() -> list:
    """
    列出所有可用的研究线程 ID

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    用途：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 显示所有可恢复的研究任务
    - UI 显示历史记录
    - 清理旧的检查点

    Returns:
        list: 线程 ID 列表（按时间倒序）
    """
    checkpoint_path = get_checkpoint_path()
    if not checkpoint_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()
        # 查询所有不同的 thread_id，按时间倒序
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_ts DESC")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception as e:
        logger.warning(f"Failed to list threads: {e}")
        return []
