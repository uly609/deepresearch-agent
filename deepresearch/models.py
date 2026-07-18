"""DeepResearch 的核心数据结构。

这个文件只定义领域模型，不写业务流程。Agent 每一步产生的数据都会落到
这些 dataclass 里，比如计划、来源、评分、证据块、结论和最终报告状态。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4


def now_iso() -> str:
    """生成 UTC ISO 时间字符串，用于事件、状态和 checkpoint 时间戳。"""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def new_task_id() -> str:
    """生成一次研究任务的业务 ID。"""
    return "research_" + uuid4().hex[:12]


@dataclass
class ResearchPlan:
    """Planner 生成的研究计划，包含原问题和拆出来的子问题。"""

    question: str
    sub_questions: List[str]


@dataclass
class Source:
    """统一的来源结构，所有搜索工具最后都要返回 Source。"""

    title: str
    url: str
    kind: str
    snippet: str
    published_at: str
    provider: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SourceScore:
    """来源评分结果，后续引用校验和报告排序会使用它。"""

    authority: float
    freshness: float
    relevance: float
    risk: float
    final: float
    rationale: str


@dataclass
class EvidenceChunk:
    """RAG 检索使用的证据片段，由 Source 切分或拼接得到。"""

    chunk_id: str
    source_url: str
    source_title: str
    text: str
    score: float = 0.0


@dataclass
class EvidenceRelation:
    """证据图谱中的轻量关系，用于 GraphRAG 和研究过程复盘。"""

    source: str
    relation: str
    target: str
    chunk_id: str
    source_url: str
    weight: float = 1.0


@dataclass
class Claim:
    """引用校验后的结论，必须绑定来源和证据片段。"""

    text: str
    source_urls: List[str]
    confidence: float
    status: str
    evidence_excerpt: str = ""
    verification_reason: str = ""


@dataclass
class Conflict:
    """冲突检测结果，描述同一主题下不同来源是否互相矛盾。"""

    topic: str
    source_urls: List[str]
    severity: str
    explanation: str


@dataclass
class SentenceCheck:
    """报告级句子审计结果，用于判断报告句子是否有证据支撑。"""

    sentence: str
    status: str
    evidence_url: str = ""
    reason: str = ""


@dataclass
class ToolCallAudit:
    """一次 Connector 调用的审计记录。"""

    connector: str
    query: str
    status: str
    result_count: int = 0
    duration_ms: int = 0
    reason: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class ResearchEvent:
    """运行过程中的事件，用于终端输出和 trace 落盘。"""

    task_id: str
    event_type: str
    message: str
    created_at: str = field(default_factory=now_iso)


@dataclass
class ResearchState:
    """一次研究任务的完整领域状态。

    LangGraph 每个节点都会逐步填充这个对象：计划、来源、证据、评分、
    claim、冲突、报告审计和最终 Markdown 报告都在这里。
    """

    question: str
    task_id: str = field(default_factory=new_task_id)
    status: str = "queued"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    plan: Optional[ResearchPlan] = None
    sources: List[Source] = field(default_factory=list)
    evidence_chunks: List[EvidenceChunk] = field(default_factory=list)
    evidence_relations: List[EvidenceRelation] = field(default_factory=list)
    scores: Dict[str, SourceScore] = field(default_factory=dict)
    claims: List[Claim] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    report_checks: List[SentenceCheck] = field(default_factory=list)
    tool_audits: List[ToolCallAudit] = field(default_factory=list)
    events: List[ResearchEvent] = field(default_factory=list)
    report_markdown: str = ""
