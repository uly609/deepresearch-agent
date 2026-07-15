"""一次 research run 的状态快照。

这个文件对齐 Pico 的 `task_state.py` 思路：不要只保存最后报告，而是保存
这次任务当前进行到哪一步、调用了多少次工具、为什么停下。
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

STOP_REASON_REPORT_GENERATED = "report_generated"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RUNTIME_ERROR = "runtime_error"


def new_run_id() -> str:
    """生成一次运行的 run_id，用于落盘目录和恢复任务。"""
    return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]


def new_task_id() -> str:
    """生成研究任务 ID，和 ResearchState 的 task_id 保持同一命名风格。"""
    return "research_" + uuid4().hex[:12]


@dataclass
class TaskState:
    """一次 run 的执行状态。

    它更偏运行控制，不保存完整研究内容；主要记录当前 phase、工具调用次数、
    失败原因、最终报告路径等信息。
    """

    run_id: str
    task_id: str
    user_request: str
    status: str = STATUS_RUNNING
    phase: str = "created"
    attempts: int = 0
    tool_steps: int = 0
    source_count: int = 0
    claim_count: int = 0
    weak_claim_count: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_report_path: str = ""
    checkpoint_id: str = ""
    notes: list = field(default_factory=list)

    @classmethod
    def create(cls, user_request: str, task_id: str = "", run_id: str = ""):
        """根据用户输入创建新的任务状态。"""
        return cls(
            run_id=run_id or new_run_id(),
            task_id=task_id or new_task_id(),
            user_request=user_request,
        )

    def record_attempt(self, phase: str = ""):
        """记录一次阶段尝试，通常在 plan/search 等节点开始时调用。"""
        self.attempts += 1
        if phase:
            self.phase = phase
        return self

    def record_tool(self, name: str):
        """记录一次工具调用，方便统计 Agent 调用了多少步工具。"""
        self.tool_steps += 1
        self.last_tool = str(name or "")
        return self

    def note(self, message: str):
        """给当前 run 添加一条调试/复盘备注。"""
        self.notes.append(str(message))
        return self

    def finish_success(self, report_path: str, source_count: int, claim_count: int, weak_claim_count: int):
        """标记任务成功完成，并记录最终报告和统计数据。"""
        self.status = STATUS_COMPLETED
        self.phase = "finished"
        self.stop_reason = STOP_REASON_REPORT_GENERATED
        self.final_report_path = str(report_path)
        self.source_count = int(source_count)
        self.claim_count = int(claim_count)
        self.weak_claim_count = int(weak_claim_count)
        return self

    def fail(self, message: str):
        """标记任务失败，并把失败原因写入 notes。"""
        self.status = STATUS_FAILED
        self.stop_reason = STOP_REASON_RUNTIME_ERROR
        self.note(message)
        return self

    def to_dict(self):
        """把 TaskState 转成可 JSON 序列化的 dict，用于 RunStore 落盘。"""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "phase": self.phase,
            "attempts": self.attempts,
            "tool_steps": self.tool_steps,
            "source_count": self.source_count,
            "claim_count": self.claim_count,
            "weak_claim_count": self.weak_claim_count,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_report_path": self.final_report_path,
            "checkpoint_id": self.checkpoint_id,
            "notes": list(self.notes),
        }
