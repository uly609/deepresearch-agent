"""命令行入口。

这个模块负责把“用户怎么启动 DeepResearch”翻译成 runtime 能理解的对象：
解析参数、选择普通运行或 eval、装配 RunStore / Runtime，
最后进入一次 research run 或评估流程。

它参考 Pico 的 `cli.py` 结构，但当前项目还没有真实模型 provider、
workspace snapshot、session resume 和 REPL，所以不会硬塞那些功能。
"""

import argparse
import json
from dataclasses import asdict

from .eval_harness import default_harness
from .models import ResearchEvent
from .run_store import RunStore
from .runtime import DeepResearchRuntime

DEFAULT_QUESTION = "调研 Agent 项目为什么主流使用 Python，并比较 LangGraph、CrewAI、AutoGen 和 LangChain4j。"
DEFAULT_RUN_ROOT = ".deepresearch/runs"
DEFAULT_ENGINE = "langgraph"
EVAL_QUESTIONS = (
    "比较 Python Agent 框架 LangGraph、CrewAI、AutoGen 的核心能力。",
    "调研 DeepResearch Agent 需要哪些上下文管理和引用验证能力。",
)


def build_parser() -> argparse.ArgumentParser:
    """创建 CLI 参数解析器。

    对应 Pico 里的 parser 装配部分。这里先只保留 DeepResearch 当前真的需要的参数：
    - question：一次研究任务的问题
    - --eval：进入评估模式
    - --run-root：运行工件保存目录
    """
    parser = argparse.ArgumentParser(description="Run the Python-first DeepResearch Agent core.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--eval", action="store_true", help="Run a small evaluation harness instead of one research task.")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Directory for task_state, trace, checkpoint and reports.")
    parser.add_argument("--engine", choices=("langgraph",), default=DEFAULT_ENGINE, help="Execution engine. The project now keeps LangGraph as the only main workflow.")
    parser.add_argument("--live-tools", action="store_true", help="Use live GitHub/arXiv/Web/MCP connectors with offline fallback. One-shot runs enable this by default.")
    parser.add_argument("--offline-tools", action="store_true", help="Use only built-in offline connectors.")
    parser.add_argument("--llm", action="store_true", help="Use an OpenAI-compatible LLM provider when DEEPSEEK_API_KEY or OPENAI_API_KEY is configured.")
    parser.add_argument("--fetch-content", action="store_true", help="Fetch and clean full web-page text before RAG chunking.")
    parser.add_argument("--resume", default="", help="Resume from a previous run_id checkpoint.")
    return parser


def build_runtime(args) -> DeepResearchRuntime:
    """根据 CLI 参数装配出一个可运行的 DeepResearchRuntime。

    这个函数对应 Pico 的 `build_agent(args)`。

    Pico 里 build_agent 会装配 model client、workspace、session store。
    我们这里当前只装配 RunStore 和 DeepResearchRuntime，因为研究工具和 LLM provider
    还没有接入真实外部服务。
    """
    run_store = RunStore(args.run_root)
    use_live_tools = args.live_tools or not args.offline_tools
    return DeepResearchRuntime(run_store=run_store, engine=args.engine, use_live_tools=use_live_tools, use_llm=args.llm, fetch_content=args.fetch_content)


def run_eval(args) -> None:
    """运行评估模式。

    Eval 不是给用户生成单份报告，而是用固定题集检查 Agent 当前版本表现。
    """
    harness = default_harness(engine=args.engine, use_live_tools=args.live_tools and not args.offline_tools, use_llm=args.llm)
    results = harness.run_cases(list(EVAL_QUESTIONS))
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def run_one_shot(args) -> None:
    """运行一次普通研究任务。

    对应 Pico 的 one-shot 模式：用户给一句请求，runtime 推进一次完整 run。
    """
    runtime = build_runtime(args)

    def emit(event: ResearchEvent) -> None:
        """把运行事件实时打印到终端。"""
        print("[{}] {}".format(event.event_type, event.message))

    if args.resume:
        state = runtime.resume(args.resume, emit=emit)
    else:
        state = runtime.ask(args.question, emit=emit)
    task_state = runtime.current_task_state
    print("\n任务:", state.task_id)
    if task_state is not None:
        print("运行:", task_state.run_id)
        print("报告:", task_state.final_report_path)
    print("\n" + state.report_markdown)


def main() -> None:
    """CLI 主函数：解析参数后选择 eval 模式或普通研究模式。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.eval:
        run_eval(args)
        return

    run_one_shot(args)


if __name__ == "__main__":
    main()
