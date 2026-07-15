"""Checkpoint helpers.

这个版本先保存轻量 checkpoint：当前阶段、下一步、来源数和 claim 数。
后续要做 resume 时，可以基于这份文件继续扩展。
"""

from .models import now_iso


def create_checkpoint(task_state, research_state, trigger: str) -> dict:
    """根据当前 TaskState 和 ResearchState 生成轻量 checkpoint。

    checkpoint 用于记录当前执行到哪一步、下一步应该做什么，以及当前已有
    多少来源和 claim。
    """
    checkpoint_id = "ckpt_" + task_state.run_id.split("-")[-1]
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "created_at": now_iso(),
        "trigger": trigger,
        "task_id": task_state.task_id,
        "run_id": task_state.run_id,
        "phase": task_state.phase,
        "status": task_state.status,
        "current_goal": task_state.user_request,
        "source_count": len(research_state.sources),
        "claim_count": len(research_state.claims),
        "next_step": infer_next_step(task_state),
    }
    task_state.checkpoint_id = checkpoint_id
    return checkpoint


def infer_next_step(task_state) -> str:
    """根据当前 phase 推断恢复任务时下一步应该做什么。"""
    if task_state.status == "completed":
        return "No next step; report has been generated."
    if task_state.phase == "planned":
        return "Search sources for each sub-question."
    if task_state.phase == "searched":
        return "Score sources and verify citations."
    if task_state.phase == "verified":
        return "Generate the final report."
    return "Continue the research workflow."
