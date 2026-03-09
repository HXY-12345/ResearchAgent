"""
深度研究代理系统 - CLI 入口文件
============================================

功能：提供命令行界面，执行多智能体协同的研究工作流

使用方式：
    # 交互模式
    python main.py

    # 直接指定主题
    python main.py "量子计算对密码学的影响"

工作流程：
    1. 配置验证 (MODEL_PROVIDER, API_KEY 等)
    2. 获取研究主题
    3. 调用 run_research() 执行研究
    4. 输出统计信息
    5. 保存报告到 outputs/ 目录
"""

import asyncio
import sys
from pathlib import Path
import logging

# 项目内部模块
from src.config import config      # 配置管理（从 .env 加载）
from src.graph import run_research  # LangGraph 工作流入口

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    主函数：执行深度研究工作流

    流程：
        1. 验证配置 (API Key、模型可用性等)
        2. 获取研究主题 (命令行参数或交互输入)
        3. 调用研究工作流
        4. 展示结果并保存报告
    """

    # ═══════════════════════════════════════════════════════════════
    # 第一步：配置验证
    # ═══════════════════════════════════════════════════════════════
    # 验证环境变量配置是否正确，包括：
    # - MODEL_PROVIDER 是否支持
    # - 对应的 API Key 是否存在
    # - 本地服务 (如 Ollama) 是否可达
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("请检查 .env 文件配置")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # 第二步：获取研究主题
    # ═══════════════════════════════════════════════════════════════
    # 支持两种方式：
    # 1. 命令行参数: python main.py "研究主题"
    # 2. 交互输入: 运行后手动输入
    if len(sys.argv) > 1:
        # 从命令行参数获取主题（合并所有参数）
        topic = " ".join(sys.argv[1:])
    else:
        # 交互模式：提示用户输入
        print("\nDeep Research Agent")
        print("=" * 50)
        topic = input("\nEnter your research topic: ").strip()

    # 验证主题非空
    if not topic:
        logger.error("No research topic provided")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # 第三步：执行研究工作流
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[INFO] Starting deep research on: {topic}\n")
    print("This may take several minutes. Please wait...\n")

    try:
        # 调用核心研究函数（位于 src/graph.py）
        # 返回 final_state：包含所有研究结果的字典
        final_state = await run_research(topic, verbose=True)

        # ═══════════════════════════════════════════════════════════
        # 第四步：检查错误
        # ═══════════════════════════════════════════════════════════
        # LangGraph 返回的 state 是一个字典
        # 如果工作流中任何节点设置了 error 字段，会在这里被检测到
        if final_state.get("error"):
            logger.error(f"Research failed: {final_state.get('error')}")
            sys.exit(1)

        # ═══════════════════════════════════════════════════════════
        # 第五步：展示结果摘要
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print("RESEARCH COMPLETE")
        print("=" * 80)

        # 展示研究计划信息
        if final_state.get("plan"):
            plan = final_state["plan"]
            print(f"\nResearch Plan Summary:")
            print(f"  - Objectives: {len(plan.objectives)}")           # 研究目标数量
            print(f"  - Search Queries: {len(plan.search_queries)}")   # 执行的搜索查询数
            print(f"  - Report Sections: {len(plan.report_outline)}") # 报告章节数

        # 展示研究数据统计
        print(f"\nResearch Data Summary:")
        print(f"  - Search Results: {len(final_state.get('search_results', []))}")    # 搜索结果数量
        print(f"  - Key Findings: {len(final_state.get('key_findings', []))}")        # 关键发现数量
        print(f"  - Report Sections: {len(final_state.get('report_sections', []))}")  # 生成的章节数
        print(f"  - Iterations: {final_state.get('iterations', 0)}")                  # 工作流迭代次数

        # ═══════════════════════════════════════════════════════════
        # 第六步：保存报告
        # ═══════════════════════════════════════════════════════════
        if final_state.get("final_report"):
            # 创建输出目录 (不存在则创建)
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # 生成安全的文件名（移除特殊字符，限制长度）
            # 只保留字母、数字、空格、横线和下划线
            safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
            safe_topic = safe_topic[:50].strip()  # 限制最多 50 字符

            # 构建输出文件路径
            output_file = output_dir / f"{safe_topic}.md"
            final_report = final_state["final_report"]

            # 写入文件（UTF-8 编码）
            output_file.write_text(final_report, encoding='utf-8')

            print(f"\n[SUCCESS] Report saved to: {output_file}")
            print(f"          Report length: {len(final_report)} characters")

            # ═══════════════════════════════════════════════════════════
            # 第七步：预览报告（前 1500 字符）
            # ═══════════════════════════════════════════════════════════
            print("\n" + "=" * 80)
            print("REPORT PREVIEW")
            print("=" * 80)
            print(final_report[:1500])
            if len(final_report) > 1500:
                print(f"\n... (showing first 1500 of {len(final_report)} characters)")
            print("\n" + "=" * 80)

        else:
            logger.warning("No report was generated")

    # ═══════════════════════════════════════════════════════════════
    # 异常处理
    # ═══════════════════════════════════════════════════════════════
    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断
        print("\n\n[WARNING] Research interrupted by user")
        sys.exit(0)
    except Exception as e:
        # 其他未预期的错误
        logger.error(f"[ERROR] Unexpected error: {e}", exc_info=True)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 程序入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # asyncio.run() 创建新的事件循环并运行异步主函数
    asyncio.run(main())

