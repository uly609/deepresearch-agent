"""运行工件落盘。

对齐 Pico 的 RunStore：单次 run 的状态、trace、report 分开保存。
这样恢复、复盘、评测不会混在一份聊天记录里。
"""

import json
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path


class RunStore:
    """运行工件存储器。

    每个 run 都会在 `.deepresearch/runs/{run_id}` 下保存状态、trace、
    checkpoint、向量库和报告，方便复盘和恢复。
    """

    def __init__(self, root: str = ".deepresearch/runs") -> None:
        """初始化运行目录。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, task_state) -> Path:
        """返回某次 run 的目录路径。"""
        run_id = task_state.run_id if hasattr(task_state, "run_id") else str(task_state)
        return self.root / run_id

    def task_state_path(self, task_state) -> Path:
        """返回 task_state.json 的路径。"""
        return self.run_dir(task_state) / "task_state.json"

    def trace_path(self, task_state) -> Path:
        """返回 trace.jsonl 的路径。"""
        return self.run_dir(task_state) / "trace.jsonl"

    def research_state_path(self, task_state) -> Path:
        """返回 research_state.json 的路径。"""
        return self.run_dir(task_state) / "research_state.json"

    def report_md_path(self, task_state) -> Path:
        """返回 Markdown 报告路径。"""
        return self.run_dir(task_state) / "report.md"

    def report_json_path(self, task_state) -> Path:
        """返回 JSON 报告路径。"""
        return self.run_dir(task_state) / "report.json"

    def checkpoint_path(self, task_state) -> Path:
        """返回 checkpoint.json 的路径。"""
        return self.run_dir(task_state) / "checkpoint.json"

    def vector_store_path(self, task_state) -> Path:
        """返回 vector_store.json 的路径。"""
        return self.run_dir(task_state) / "vector_store.json"

    def evidence_graph_path(self, task_state) -> Path:
        """返回 evidence_graph.json 的路径。"""
        return self.run_dir(task_state) / "evidence_graph.json"

    def start_run(self, task_state) -> Path:
        """创建 run 目录并写入初始 TaskState。"""
        run_dir = self.run_dir(task_state)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.write_task_state(task_state)
        return run_dir

    def write_task_state(self, task_state):
        """把 TaskState 写入 task_state.json。"""
        path = self.task_state_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, task_state.to_dict())
        return path

    def write_research_state(self, task_state, research_state):
        """把 ResearchState 写入 research_state.json。"""
        path = self.research_state_path(task_state)
        self._write_json_atomic(path, self._plain(research_state))
        return path

    def append_trace(self, task_state, event_type: str, payload: dict):
        """追加一条 JSONL trace 事件。"""
        path = self.trace_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {"event_type": event_type, **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return path

    def write_report(self, task_state, markdown: str, metadata: dict, structured_report: dict = None):
        """同时写入 Markdown 和带固定 schema 的 JSON 研究报告。"""
        md_path = self.report_md_path(task_state)
        json_path = self.report_json_path(task_state)
        md_path.write_text(markdown, encoding="utf-8")
        self._write_json_atomic(
            json_path,
            {
                "markdown": markdown,
                "metadata": metadata,
                "report": self._plain(structured_report or {}),
            },
        )
        return md_path

    def write_checkpoint(self, task_state, checkpoint: dict):
        """写入 checkpoint.json。"""
        path = self.checkpoint_path(task_state)
        self._write_json_atomic(path, checkpoint)
        return path

    def write_vector_store(self, task_state, vector_store):
        """保存 RAG 的本地向量库快照。"""
        return vector_store.save(self.vector_store_path(task_state))

    def write_evidence_graph(self, task_state, relations):
        """保存 GraphRAG 证据关系图。"""
        path = self.evidence_graph_path(task_state)
        self._write_json_atomic(path, self._plain(relations))
        return path

    def read_json(self, path: Path) -> dict:
        """读取 JSON 文件；不存在时返回空 dict。"""
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def read_checkpoint(self, run_id: str) -> dict:
        """按 run_id 读取 checkpoint。"""
        return self.read_json(self.root / run_id / "checkpoint.json")

    def read_task_state(self, run_id: str) -> dict:
        """按 run_id 读取 task_state。"""
        return self.read_json(self.root / run_id / "task_state.json")

    def read_report(self, run_id: str) -> str:
        """按 run_id 读取 Markdown 报告。"""
        path = self.root / run_id / "report.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _write_json_atomic(self, path: Path, payload):
        """原子写 JSON，避免中途失败导致文件只写一半。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)

    def _plain(self, value):
        """把 dataclass/list/dict 递归转换成普通 JSON 结构。"""
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._plain(item) for item in value]
        return value
