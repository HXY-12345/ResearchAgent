"""
深度研究代理系统 - 智能体节点模块
============================================

功能：实现四个核心智能体的业务逻辑

智能体列表：
    1. ResearchPlanner    - 规划研究策略
    2. ResearchSearcher   - 执行网络搜索
    3. ResearchSynthesizer - 综合分析信息
    4. ReportWriter       - 生成研究报告

设计模式：
    - 依赖注入：每个智能体可接受外部注入的 LLM 实例
    - 工厂模式：get_llm() 统一创建不同提供商的 LLM
    - 重试机制：每个智能体内置指数退避重试
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 标准库导入
# ═══════════════════════════════════════════════════════════════════════════════
import asyncio       # 异步编程支持
from typing import List, Optional, Dict, Any, Protocol
import logging       # 日志记录
import time          # 计时和延迟
import json          # JSON 解析
import re            # 正则表达式

# ═══════════════════════════════════════════════════════════════════════════════
# LangChain 相关导入
# ═══════════════════════════════════════════════════════════════════════════════
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini 模型
from langchain_ollama import ChatOllama                    # Ollama 本地模型
from langchain_openai import ChatOpenAI                      # OpenAI 兼容模型
from langchain_core.prompts import ChatPromptTemplate       # 提示词模板
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel    # LLM 基类
from langchain.agents import create_agent                   # 创建自主智能体

# ═══════════════════════════════════════════════════════════════════════════════
# 项目内部导入
# ═══════════════════════════════════════════════════════════════════════════════
from src.state import ResearchState, ResearchPlan, SearchQuery, ReportSection, SearchResult
from src.utils.tools import get_research_tools               # 获取 LangChain 工具
from src.config import config                                # 全局配置
from src.utils.credibility import CredibilityScorer         # 可信度评分
from src.utils.citations import CitationFormatter           # 引用格式化
from src.llm_tracker import estimate_tokens                 # Token 估算
from src.exceptions import PlanningError, SearchError, SynthesisError, ReportGenerationError

# 提示词模板（从 prompts 模块导入）
from src.prompts import (
    PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE,
    SEARCHER_SYSTEM_PROMPT, SEARCHER_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE,
    WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE
)

# 进度回调函数（用于 Web UI 显示）
from src.callbacks import (
    emit_planning_start, emit_planning_complete,
    emit_search_start, emit_search_results,
    emit_extraction_start, emit_extraction_complete,
    emit_synthesis_start, emit_synthesis_progress, emit_synthesis_complete,
    emit_writing_start, emit_writing_section, emit_writing_complete,
    emit_error
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def get_llm(
    temperature: float = 0.7,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None
) -> BaseChatModel:
    """
    LLM 工厂函数：根据配置创建 LLM 实例

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    支持的模型提供商：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    | provider    | 模型类型         | 配置来源              |
    |-------------|-----------------|----------------------|
    | ollama      | 本地模型         | OLLAMA_BASE_URL     |
    | openai      | OpenAI/API兼容   | OPENAI_API_KEY      |
    | llamacpp    | llama.cpp服务器  | LLAMACPP_BASE_URL   |
    | gemini      | Google Gemini    | GEMINI_API_KEY      |

    Args:
        temperature: 生成温度（0.0=确定性，1.0=创造性）
        model_override: 覆盖配置中的模型名称
        provider_override: 覆盖配置中的提供商

    Returns:
        BaseChatModel: LangChain 标准接口的 LLM 实例
    """
    # 获取模型名称和提供商（支持覆盖）
    model_name = model_override or config.model_name
    provider = provider_override or config.model_provider

    # ═══════════════════════════════════════════════════════════════
    # 根据提供商创建对应的 LLM 实例
    # ═══════════════════════════════════════════════════════════════
    if provider == "ollama":
        # Ollama 本地模型（如 qwen2.5:7b）
        logger.info(f"Using Ollama model: {model_name}")
        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
            temperature=temperature,
            num_ctx=8192,  # 上下文窗口大小
        )

    elif provider == "openai":
        # OpenAI 或兼容 API（如 OpenRouter）
        logger.info(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.openai_base_url}/v1",
            api_key=config.openai_api_key,
            temperature=temperature
        )

    elif provider == "llamacpp":
        # llama.cpp 服务器（OpenAI 兼容接口）
        logger.info(f"Using llama.cpp server model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.llamacpp_base_url}/v1",
            api_key="not-needed",  # llama.cpp 不需要 API key
            temperature=temperature
        )

    else:  # gemini（默认）
        # Google Gemini
        logger.info(f"Using Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.google_api_key,
            temperature=temperature
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 智能体 1: ResearchPlanner（规划智能体）
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchPlanner:
    """
    规划智能体：负责制定研究策略

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    职责：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. 分析研究主题
    2. 生成 3-5 个研究目标 (objectives)
    3. 生成针对性的搜索查询 (search_queries)
    4. 设计报告大纲 (report_outline，最多 8 节)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    输入/输出：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    输入: state.research_topic (研究主题)
    输出: state.plan (ResearchPlan 对象)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    实现特点：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - 使用 JsonOutputParser 要求 LLM 输出结构化 JSON
    - 内置重试机制（最多 3 次，指数退避）
    - 验证输出结构的完整性
    """

    def __init__(self, llm: Optional[BaseChatModel] = None, max_retries: int = 3):
        """
        初始化规划智能体

        Args:
            llm: 可选的 LLM 实例（依赖注入，便于测试）
            max_retries: 最大重试次数
        """
        self.llm = llm or get_llm(temperature=0.7)  # 较高温度以获得创意
        self.max_retries = max_retries

    async def plan(self, state: ResearchState) -> Dict[str, Any]:
        """
        执行规划任务

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        工作流程：
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. 构建提示词（注入配置参数）
        2. 调用 LLM 生成结构化计划
        3. 验证输出完整性
        4. 构建 ResearchPlan 对象
        5. 返回状态更新（LangGraph 自动合并）

        Args:
            state: 当前研究状态（包含 research_topic）

        Returns:
            Dict[str, Any]: 状态更新字典，LangGraph 会自动合并到全局状态
        """
        logger.info(f"Planning research for: {state.research_topic}")

        # 触发进度回调（通知 Web UI）
        await emit_planning_start(state.research_topic)

        # ═══════════════════════════════════════════════════════════════
        # 步骤 1: 构建提示词
        # ═══════════════════════════════════════════════════════════════
        # 从 prompts/planner.py 加载模板，并注入配置参数
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            max_queries=config.max_search_queries,      # 如: 3
            max_sections=config.max_report_sections     # 如: 8
        )

        # 创建 ChatPromptTemplate（system + user 消息）
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", PLANNER_USER_TEMPLATE)
        ])

        # ═══════════════════════════════════════════════════════════════
        # 步骤 2: 重试循环（处理 LLM 输出不稳定）
        # ═══════════════════════════════════════════════════════════════
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # 构建 LCEL 链: 提示词 -> LLM -> JSON 解析器
                chain = prompt | self.llm | JsonOutputParser()

                # 估算输入 token 数（用于成本追踪）
                input_text = f"{state.research_topic} {config.max_search_queries} {config.max_report_sections}"
                input_tokens = estimate_tokens(input_text)

                # ═════════════════════════════════════════════════════════
                # 调用 LLM
                # ═════════════════════════════════════════════════════════
                result = await chain.ainvoke({
                    "topic": state.research_topic,
                    "max_queries": config.max_search_queries,
                    "max_sections": config.max_report_sections
                })

                # 记录调用详情
                duration = time.time() - start_time
                output_tokens = estimate_tokens(str(result))

                call_detail = {
                    'agent': 'ResearchPlanner',
                    'operation': 'plan',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }

                # ═════════════════════════════════════════════════════════
                # 步骤 3: 验证输出结构
                # ═════════════════════════════════════════════════════════
                # 检查必需字段是否存在
                if not all(key in result for key in ["topic", "objectives", "search_queries", "report_outline"]):
                    raise PlanningError("Invalid plan structure returned")

                # 检查搜索查询是否为空
                if not result["search_queries"]:
                    raise PlanningError("No search queries generated")

                # ═════════════════════════════════════════════════════════
                # 步骤 4: 构建 ResearchPlan 对象
                # ═════════════════════════════════════════════════════════
                plan = ResearchPlan(
                    topic=result["topic"],
                    objectives=result["objectives"][:5],  # 最多 5 个目标
                    search_queries=[
                        SearchQuery(query=sq["query"], purpose=sq["purpose"])
                        for sq in result["search_queries"][:config.max_search_queries]
                    ],
                    report_outline=result["report_outline"][:config.max_report_sections]
                )

                logger.info(f"Created plan with {len(plan.search_queries)} queries (max: {config.max_search_queries})")
                logger.info(f"Report outline has {len(plan.report_outline)} sections (max: {config.max_report_sections})")

                await emit_planning_complete(len(plan.search_queries), len(plan.report_outline))

                # ═════════════════════════════════════════════════════════
                # 步骤 5: 返回状态更新
                # ═════════════════════════════════════════════════════════
                # 注意：只返回需要更新的字段，LangGraph 自动合并
                return {
                    "plan": plan,                          # 新增: 研究计划
                    "current_stage": "searching",          # 更新: 进入搜索阶段
                    "iterations": state.iterations + 1,   # 累加: 迭代计数
                    "llm_calls": state.llm_calls + 1,     # 累加: LLM 调用次数
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }

            # ═══════════════════════════════════════════════════════════════
            # 异常处理
            # ═══════════════════════════════════════════════════════════════
            except Exception as e:
                logger.warning(f"Planning attempt {attempt + 1} failed: {str(e)}")

                # 最后一次尝试失败，返回错误状态
                if attempt == self.max_retries - 1:
                    logger.error(f"Planning failed after {self.max_retries} attempts")
                    await emit_error(f"Planning failed: {str(e)}")
                    return {
                        "error": f"Planning failed: {str(e)}",  # 设置 error 字段会终止工作流
                        "iterations": state.iterations + 1
                    }
                else:
                    # 指数退避重试：2^attempt 秒（1s, 2s, 4s...）
                    await asyncio.sleep(2 ** attempt)

        # 理论上不会执行到这里（上面的逻辑已覆盖所有情况）
        return {
            "error": "Planning failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }


# =============================================================================
# Research Searcher Agent
# =============================================================================

class ResearchSearcher:
    """Autonomous agent responsible for executing research searches."""
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        credibility_scorer: Optional[CredibilityScorer] = None,
        max_retries: int = 3
    ):
        self.llm = llm or get_llm(temperature=0.3)
        self.tools = get_research_tools(agent_type="search")
        self.credibility_scorer = credibility_scorer or CredibilityScorer()
        self.max_retries = max_retries
        
    async def search(self, state: ResearchState) -> Dict[str, Any]:
        """Autonomously execute research searches using tools.
        
        Returns dict with search results that LangGraph will merge into state.
        """
        if not state.plan:
            await emit_error("No research plan available")
            return {"error": "No research plan available"}
        
        logger.info(f"Autonomous agent researching: {len(state.plan.search_queries)} planned queries")
        
        total_queries = len(state.plan.search_queries)
        for i, query in enumerate(state.plan.search_queries, 1):
            await emit_search_start(query.query, i, total_queries)
        
        max_searches = config.max_search_queries
        max_results_per_search = config.max_search_results_per_query
        expected_total_results = max_searches * max_results_per_search
        
        system_prompt = SEARCHER_SYSTEM_PROMPT.format(
            max_searches=max_searches,
            max_results_per_search=max_results_per_search,
            expected_total_results=expected_total_results
        )
        
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=system_prompt
        )
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                objectives_text = "\n".join(f"- {obj}" for obj in state.plan.objectives)
                queries_text = "\n".join(
                    f"- {q.query} (Purpose: {q.purpose})" 
                    for q in state.plan.search_queries
                )
                
                input_message = SEARCHER_USER_TEMPLATE.format(
                    topic=state.research_topic,
                    objectives=objectives_text,
                    queries=queries_text,
                    min_sources=expected_total_results
                )
                
                input_tokens = estimate_tokens(input_message)
                
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                duration = time.time() - start_time
                
                messages = result.get('messages', [])
                output_text = ""
                if messages:
                    output_text = str(messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1]))
                
                output_tokens = estimate_tokens(output_text)
                
                search_results = self._extract_results_from_messages(messages)
                
                logger.info(f"Autonomous agent collected {len(search_results)} results")
                
                total_extracted_chars = sum(
                    len(r.content) if r.content else 0 
                    for r in search_results
                )
                extracted_count = sum(1 for r in search_results if r.content)
                
                await emit_extraction_complete(extracted_count, total_extracted_chars)
                
                if not search_results:
                    await emit_error("Agent did not collect any search results")
                    raise SearchError("Agent did not collect any search results")
            
                scored_results = self.credibility_scorer.score_search_results(search_results)
                
                filtered_scored = [
                    item for item in scored_results
                    if item['credibility']['score'] >= config.min_credibility_score
                ]
                
                credibility_scores = [item['credibility'] for item in filtered_scored]
                sorted_results = [item['result'] for item in filtered_scored]
                
                logger.info(f"Filtered {len(search_results)} -> {len(sorted_results)} results (min_credibility={config.min_credibility_score})")
                
                for q in state.plan.search_queries:
                    q.completed = True
                
                call_detail = {
                    'agent': 'ResearchSearcher',
                    'operation': 'autonomous_search',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'results_count': len(sorted_results),
                    'original_results_count': len(search_results),
                    'min_credibility_score': config.min_credibility_score,
                    'attempt': attempt + 1
                }
                
                return {
                    "search_results": sorted_results,
                    "credibility_scores": credibility_scores,
                    "current_stage": "synthesizing",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Search failed after {self.max_retries} attempts")
                    await emit_error(f"Search failed: {str(e)}")
                    return {
                        "error": f"Search failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Search failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _extract_results_from_messages(self, messages: list) -> List[SearchResult]:
        """Extract search results from agent messages."""
        search_results = []
        
        for msg in messages:
            if hasattr(msg, 'name') and msg.name == 'web_search':
                try:
                    content = msg.content
                    if isinstance(content, str):
                        tool_results = json.loads(content)
                    else:
                        tool_results = content
                    
                    if isinstance(tool_results, list):
                        for item in tool_results:
                            if isinstance(item, dict):
                                search_results.append(SearchResult(
                                    query=item.get('query', ''),
                                    title=item.get('title', ''),
                                    url=item.get('url', ''),
                                    snippet=item.get('snippet', ''),
                                    content=None
                                ))
                except Exception as e:
                    logger.warning(f"Error parsing tool result: {e}")
            
            if hasattr(msg, 'name') and msg.name == 'extract_webpage_content':
                try:
                    content = msg.content
                    if search_results and content:
                        for sr in reversed(search_results):
                            if not sr.content:
                                sr.content = content
                                break
                except Exception as e:
                    logger.warning(f"Error updating content: {e}")
        
        return search_results


# =============================================================================
# Research Synthesizer Agent
# =============================================================================

class ResearchSynthesizer:
    """Autonomous agent responsible for synthesizing research findings."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.3, model_override=config.summarization_model)
        self.tools = get_research_tools(agent_type="synthesis")
        self.max_retries = max_retries
        
    async def synthesize(self, state: ResearchState) -> Dict[str, Any]:
        """Autonomously synthesize key findings using tools and reasoning.
        
        Returns dict with key findings that LangGraph will merge into state.
        """
        logger.info(f"Synthesizing findings from {len(state.search_results)} results")
        
        if not state.search_results:
            await emit_error("No search results to synthesize")
            return {"error": "No search results to synthesize"}
        
        await emit_synthesis_start(len(state.search_results))
        
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT
        )
        
        max_results = 20
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                current_max = max(5, max_results - (attempt * 5))
                
                results_to_use = state.search_results[:current_max]
                credibility_scores_to_use = state.credibility_scores[:current_max] if state.credibility_scores else []
                
                results_text = self._format_results_text(results_to_use, credibility_scores_to_use)
                
                input_message = SYNTHESIZER_USER_TEMPLATE.format(
                    topic=state.research_topic,
                    results=results_text
                )
                
                input_tokens = estimate_tokens(input_message)
                
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                duration = time.time() - start_time
                
                messages = result.get('messages', [])
                output_text = ""
                if messages:
                    last_msg = messages[-1]
                    output_text = str(last_msg.content if hasattr(last_msg, 'content') else str(last_msg))
                
                output_tokens = estimate_tokens(output_text)
                
                call_detail = {
                    'agent': 'ResearchSynthesizer',
                    'operation': 'autonomous_synthesis',
                    'model': config.summarization_model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                key_findings = self._extract_findings(output_text, state.search_results)
                
                logger.info(f"Extracted {len(key_findings)} key findings")
                
                await emit_synthesis_complete(len(key_findings))
                
                return {
                    "key_findings": key_findings,
                    "current_stage": "reporting",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Synthesis failed after {self.max_retries} attempts")
                    await emit_error(f"Synthesis failed: {str(e)}")
                    return {
                        "error": f"Synthesis failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Synthesis failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _format_results_text(self, results: list, credibility_scores: list) -> str:
        """Format search results with credibility information."""
        if len(results) != len(credibility_scores):
            return "\n\n".join([
                f"[{i+1}] {r.title}\nURL: {r.url}\nSnippet: {r.snippet}\n" +
                (f"Content: {r.content[:300]}..." if r.content else "")
                for i, r in enumerate(results)
            ])
        
        return "\n\n".join([
            f"[{i+1}] {r.title}\n"
            f"URL: {r.url}\n"
            f"Credibility: {cred.get('level', 'unknown').upper()} (Score: {cred.get('score', 'N/A')}/100) - {', '.join(cred.get('factors', []))}\n"
            f"Snippet: {r.snippet}\n" +
            (f"Content: {r.content[:300]}..." if r.content else "")
            for i, (r, cred) in enumerate(zip(results, credibility_scores))
        ])
    
    def _extract_findings(self, output_text: str, search_results: list) -> List[str]:
        """Extract key findings from synthesis output."""
        json_match = re.search(r'\[(.*?)\]', output_text, re.DOTALL)
        
        key_findings = []
        if json_match:
            try:
                findings = json.loads(json_match.group(0))
                if isinstance(findings, list):
                    key_findings = [str(f) for f in findings]
                else:
                    key_findings = [str(findings)]
            except json.JSONDecodeError:
                pass
        
        if not key_findings:
            lines = output_text.split('\n')
            for line in lines:
                line = line.strip().lstrip('-').lstrip('*').lstrip('>').strip()
                line = re.sub(r'^\d+\.\s*', '', line)
                if len(line) > 30 and not line.startswith('[') and not line.startswith(']'):
                    key_findings.append(line)
            key_findings = key_findings[:15]
        
        if not key_findings and search_results:
            logger.warning("Agent produced no findings, creating basic ones from results")
            key_findings = [
                f"{r.title}: {r.snippet[:100]}..."
                for r in search_results[:10]
                if r.snippet
            ]
        
        return key_findings


# =============================================================================
# Report Writer Agent
# =============================================================================

class ReportWriter:
    """Autonomous agent responsible for writing research reports."""
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        citation_style: str = 'apa',
        max_retries: int = 3
    ):
        self.llm = llm or get_llm(temperature=0.7)
        self.tools = get_research_tools(agent_type="writing")
        self.max_retries = max_retries
        self.citation_style = citation_style
        self.citation_formatter = citation_formatter or CitationFormatter()
        
    async def write_report(self, state: ResearchState) -> Dict[str, Any]:
        """Write the final research report with validation and retry.
        
        Returns dict with report data that LangGraph will merge into state.
        """
        logger.info("Writing final report")
        
        if not state.plan or not state.key_findings:
            await emit_error("Insufficient data for report generation")
            return {"error": "Insufficient data for report generation"}
        
        await emit_writing_start(len(state.plan.report_outline))
        
        report_llm_calls = 0
        report_input_tokens = 0
        report_output_tokens = 0
        report_call_details = []
        
        for attempt in range(self.max_retries):
            try:
                report_sections = []
                total_sections = len(state.plan.report_outline)
                
                for section_idx, section_title in enumerate(state.plan.report_outline, 1):
                    await emit_writing_section(section_title, section_idx, total_sections)
                    
                    section, section_tokens = await self._write_section(
                        state.research_topic,
                        section_title,
                        state.key_findings,
                        state.search_results
                    )
                    if section:
                        report_sections.append(section)
                        if section_tokens:
                            report_llm_calls += 1
                            report_input_tokens += section_tokens['input_tokens']
                            report_output_tokens += section_tokens['output_tokens']
                            report_call_details.append(section_tokens)
                
                if not report_sections:
                    raise ReportGenerationError("No report sections generated")
                
                temp_state = ResearchState(
                    research_topic=state.research_topic,
                    plan=state.plan,
                    report_sections=report_sections,
                    search_results=state.search_results
                )
                
                final_report = self._compile_report(temp_state)
                
                if state.search_results:
                    final_report = self.citation_formatter.update_report_citations(
                        final_report,
                        style=self.citation_style,
                        search_results=state.search_results
                    )
                
                if state.credibility_scores:
                    high_cred_sources = [
                        i+1 for i, score in enumerate(state.credibility_scores)
                        if score.get('level') == 'high'
                    ]
                    if high_cred_sources:
                        final_report += f"\n\n---\n\n**Note:** {len(high_cred_sources)} high-credibility sources were prioritized in this research."
                
                if len(final_report) < 500:
                    raise ReportGenerationError("Report too short - insufficient content")
                
                logger.info(f"Report generation complete: {len(final_report)} chars")
                
                await emit_writing_complete(len(final_report))
                
                return {
                    "report_sections": report_sections,
                    "final_report": final_report,
                    "current_stage": "complete",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + report_llm_calls,
                    "total_input_tokens": state.total_input_tokens + report_input_tokens,
                    "total_output_tokens": state.total_output_tokens + report_output_tokens,
                    "llm_call_details": state.llm_call_details + report_call_details
                }
                
            except Exception as e:
                logger.warning(f"Report attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Report generation failed after {self.max_retries} attempts")
                    await emit_error(f"Report generation failed: {str(e)}")
                    return {
                        "error": f"Report writing failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Report generation failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    async def _write_section(
        self,
        topic: str,
        section_title: str,
        findings: List[str],
        search_results: List
    ) -> tuple:
        """Write a single report section."""
        logger.info(f"Writing section: {section_title}")
        
        system_prompt = WRITER_SYSTEM_PROMPT.format(min_words=config.min_section_words)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        try:
            start_time = time.time()
            
            sources_context = ""
            if search_results:
                sources_context = "\nAvailable Sources for Citation:\n" + "\n".join(
                    f"[{i+1}] {r.title} ({r.url})"
                    for i, r in enumerate(search_results[:15])
                )
            
            input_message = WRITER_USER_TEMPLATE.format(
                topic=topic,
                section_title=section_title,
                min_words=config.min_section_words,
                findings=chr(10).join(f"- {f}" for f in findings),
                sources_context=sources_context
            )
            
            input_tokens = estimate_tokens(input_message)
            
            chain = prompt | self.llm | StrOutputParser()
            content = await chain.ainvoke({"input": input_message})
            
            if not isinstance(content, str):
                content = str(content)
            
            duration = time.time() - start_time
            output_tokens = estimate_tokens(content)
            
            call_detail = {
                'agent': 'ReportWriter',
                'operation': f'write_section_{section_title[:30]}',
                'model': config.model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'duration': round(duration, 2)
            }
            
            if not content or len(content.strip()) < 50:
                logger.warning(f"Section '{section_title}' generated insufficient content: {len(content)} chars")
                if findings:
                    logger.info(f"Creating fallback content for section '{section_title}'")
                    content = f"\n\n{chr(10).join(findings[:3])}\n\n"
                else:
                    logger.error(f"Cannot create section '{section_title}' - no content and no findings")
                    return None, None
            
            citations = re.findall(r'\[(\d+)\]', content)
            source_urls = []
            for cite_num in set(citations):
                idx = int(cite_num) - 1
                if 0 <= idx < len(search_results):
                    source_urls.append(search_results[idx].url)
            
            section = ReportSection(
                title=section_title,
                content=content,
                sources=source_urls
            )
            
            logger.info(f"Successfully wrote section '{section_title}': {len(content)} chars")
            return section, call_detail
            
        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {str(e)}")
            return None, None
    
    def _compile_report(self, state: ResearchState) -> str:
        """Compile all sections into final report."""
        search_results = getattr(state, 'search_results', []) or []
        report_sections = getattr(state, 'report_sections', []) or []
        
        unique_sources = set()
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                unique_sources.add(result.url)
        
        for section in report_sections:
            if hasattr(section, 'sources'):
                unique_sources.update(section.sources)
        
        source_count = len(unique_sources) if unique_sources else len(search_results)
        
        report_parts = [
            f"# {state.research_topic}\n",
            f"**Deep Research Report**\n",
            f"\n## Executive Summary\n",
            f"This report provides a comprehensive analysis of {state.research_topic}. ",
            f"The research was conducted across **{source_count} sources** ",
            f"and synthesized into **{len(report_sections)} key sections**.\n",
            f"\n## Research Objectives\n"
        ]
        
        if state.plan and hasattr(state.plan, 'objectives'):
            for i, obj in enumerate(state.plan.objectives, 1):
                report_parts.append(f"{i}. {obj}\n")
        
        report_parts.append("\n---\n")
        
        has_references_section = False
        for section in report_sections:
            content = section.content.strip()
            
            if "## References" in content or section.title.lower() == "references":
                has_references_section = True
            
            if content.startswith(f"## {section.title}"):
                report_parts.append(f"\n{content}\n\n")
            else:
                report_parts.append(f"\n## {section.title}\n\n")
                report_parts.append(content)
                report_parts.append("\n")
        
        if not has_references_section:
            report_parts.append("\n---\n\n## References\n\n")
        
        source_info = []
        seen_urls = set()
        
        for result in search_results:
            if hasattr(result, 'url') and result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                title = getattr(result, 'title', '')
                source_info.append((result.url, title))
        
        for section in report_sections:
            if hasattr(section, 'sources'):
                for url in section.sources:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        source_info.append((url, ''))
        
        if not has_references_section:
            if source_info:
                for i, (url, title) in enumerate(source_info[:30], 1):
                    citation = self.citation_formatter.format_apa(url, title)
                    report_parts.append(f"{i}. {citation}\n")
            else:
                report_parts.append("*No sources were available for this research.*\n")
        
        return "".join(report_parts)
