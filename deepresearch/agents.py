"""DeepResearch 的 Agent 能力模块。

这个文件里不是一个单独的大 Agent，而是把 Planner、Evaluator、Verifier、
Reflector、Reporter 等能力拆成多个小 Agent。LangGraph 工作流会按节点调用
这些 Agent，共同完成研究任务。
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Set

from .llm_provider import LLMProvider, parse_json_list
from .models import Claim, Conflict, ResearchPlan, SentenceCheck, Source, SourceScore

NON_EVIDENCE_METADATA_KEYS = {
    "retrieved_query",
    "retrieved_by",
    "connectors",
    "canonical_url",
    "security_findings",
    "security_risk",
}


class PlannerAgent:
    """研究规划 Agent。

    它负责把用户的大问题拆成多个可以搜索和验证的子问题。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；没有 LLM 时使用规则版规划。"""
        self.llm = llm or LLMProvider()

    def plan(self, question: str) -> ResearchPlan:
        """生成研究计划，优先用 LLM，失败时走规则模板。"""
        llm_plan = self._llm_plan(question)
        if llm_plan:
            return ResearchPlan(question=question, sub_questions=llm_plan)
        focus_terms = self._focus_terms(question)
        framework_focus = "、".join(focus_terms) if focus_terms else "LangGraph、CrewAI、AutoGen 和 LangChain4j"
        sub_questions = [
            "明确研究范围、术语和成功标准：" + question,
            "比较当前活跃的 Agent 框架和工作流模型：" + framework_focus,
            "检查 RAG、工具调用、MCP、上下文管理、checkpoint 和安全支持：" + question,
            "收集 GitHub、论文、官方文档和深度研究示例中的证据：" + question,
            "识别缺口、风险、评估指标和下一步实现建议：" + question,
        ]
        return ResearchPlan(question=question, sub_questions=sub_questions)

    def _focus_terms(self, question: str) -> List[str]:
        """从问题里提取 Agent 相关关键词，用于规则版规划。"""
        known_terms = ["LangGraph", "CrewAI", "AutoGen", "LangChain4j", "Spring AI", "MCP", "RAG", "Eval"]
        lowered = question.lower()
        return [term for term in known_terms if term.lower() in lowered]

    def _llm_plan(self, question: str) -> List[str]:
        """调用 LLM 生成子问题列表。"""
        if not self.llm.available():
            return []
        text = self.llm.complete(
            "你是一个研究规划器。只返回 JSON：5 个简洁的子问题列表。",
            "研究问题：" + question,
        )
        items = parse_json_list(text)
        return items[:6]


class SearchQueryPlannerAgent:
    """搜索 Query 规划 Agent。

    Planner 负责拆研究任务；这个 Agent 负责把每个子问题改写成更适合搜索
    工具的 query，类似 DeepResearch 系统里的 query rewrite / query fan-out。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；无 LLM 时使用规则 query 扩展。"""
        self.llm = llm or LLMProvider()

    def expand(self, question: str, sub_question: str, limit: int = 3) -> List[str]:
        """把一个子问题扩展成少量高质量搜索 query。"""
        queries = self._llm_expand(question, sub_question, limit)
        if not queries:
            queries = self._rule_expand(question, sub_question)
        return _unique_texts(queries)[:limit]

    def _llm_expand(self, question: str, sub_question: str, limit: int) -> List[str]:
        """让 LLM 生成搜索 query，失败时返回空。"""
        if not self.llm.available():
            return []
        text = self.llm.complete(
            "你是 DeepResearch 搜索 Query 规划器。只返回 JSON 字符串列表。",
            (
                "总研究问题：{}\n当前子问题：{}\n\n"
                "请生成 {} 条适合 Web/GitHub/论文检索的短 query。"
                "要求覆盖官方文档、GitHub/论文、最近趋势；不要写解释。"
            ).format(question, sub_question, limit),
        )
        return parse_json_list(text)

    def _rule_expand(self, question: str, sub_question: str) -> List[str]:
        """无 LLM 时的规则 query fan-out。"""
        focus_terms = _focus_terms(question + " " + sub_question)
        focus = " ".join(focus_terms) if focus_terms else "agent framework RAG MCP"
        base = sub_question
        return [
            base,
            "{} official docs GitHub arXiv".format(focus),
            "{} 2026 trend comparison evaluation".format(focus),
        ]


class SourceEvaluatorAgent:
    """来源评分 Agent。

    它不负责搜索，而是在搜索完成后评价每个 Source 是否可靠。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；有 LLM 时优先用模型做来源质量判断。"""
        self.llm = llm or LLMProvider()

    def score(self, sources: List[Source], question: str, retriever=None) -> Dict[str, SourceScore]:
        """给每个来源计算 SourceScore。"""
        scores = {}
        current_year = datetime.utcnow().year
        for source in sources:
            scores[source.url] = self._score_one(source, question, current_year, retriever)
        return scores

    def _score_one(self, source: Source, question: str, current_year: int, retriever=None) -> SourceScore:
        """优先用 LLM 评分；失败时回退到确定性规则。"""
        vector_relevance = self._vector_relevance(source, question, retriever)
        rule_score = self._rule_score(source, question, current_year, vector_relevance)
        llm_score = self._llm_score(source, question, rule_score, vector_relevance)
        return llm_score or rule_score

    def _rule_score(self, source: Source, question: str, current_year: int, vector_relevance: float = 0.0) -> SourceScore:
        """无 LLM 时的来源评分兜底。"""
        authority = self._authority(source)
        freshness = self._freshness(source, current_year)
        keyword_relevance = self._relevance(source, question)
        relevance = self._combine_relevance(keyword_relevance, vector_relevance)
        risk = 0.1 if "security_findings" not in source.metadata else 0.8
        final = round(authority * 0.35 + freshness * 0.2 + relevance * 0.35 + (1 - risk) * 0.1, 2)
        rationale = "rule: authority={:.2f}, freshness={:.2f}, keyword_relevance={:.2f}, vector_relevance={:.2f}, relevance={:.2f}, risk={:.2f}".format(
            authority, freshness, keyword_relevance, vector_relevance, relevance, risk
        )
        return SourceScore(authority, freshness, relevance, risk, final, rationale)

    def _llm_score(self, source: Source, question: str, rule_score: SourceScore, vector_relevance: float = 0.0) -> Optional[SourceScore]:
        """让 LLM 按来源权威性、新鲜度、相关性和风险做结构化评分。"""
        if not self.llm.available():
            return None
        user = (
            "研究问题：\n{}\n\n来源：\n{}\n\n向量相似度：{:.4f}\n规则初评分：{}\n\n"
            "请只返回 JSON，不要 Markdown。字段：authority、freshness、relevance、risk、rationale。"
            "四个分数都在 0 到 1 之间；risk 越高表示越不可信或越可能有提示词注入/广告/过时风险。"
            "relevance 要综合你的语义判断和给定向量相似度。"
        ).format(question, _source_text(source, limit=1800), vector_relevance, rule_score.rationale)
        answer = self.llm.complete(
            "你是严谨的 DeepResearch 来源质量评估器，只根据给定来源评分。",
            user,
        )
        payload = _parse_json_object(answer)
        if not payload:
            return None
        authority = _clamp(_as_float(payload.get("authority"), rule_score.authority))
        freshness = _clamp(_as_float(payload.get("freshness"), rule_score.freshness))
        llm_relevance = _clamp(_as_float(payload.get("relevance"), rule_score.relevance))
        relevance = self._combine_relevance(llm_relevance, vector_relevance, primary_weight=0.65)
        risk = _clamp(_as_float(payload.get("risk"), rule_score.risk))
        final = round(authority * 0.35 + freshness * 0.2 + relevance * 0.35 + (1 - risk) * 0.1, 2)
        rationale = "llm: {}; llm_relevance={:.2f}, vector_relevance={:.2f}, hybrid_relevance={:.2f}".format(
            str(payload.get("rationale") or rule_score.rationale),
            llm_relevance,
            vector_relevance,
            relevance,
        )
        return SourceScore(authority, freshness, relevance, risk, final, _excerpt(rationale, 260))

    def _vector_relevance(self, source: Source, question: str, retriever=None) -> float:
        """通过 RAG 向量检索结果估算 source-level 相关性。"""
        if retriever is None or not hasattr(retriever, "source_similarity"):
            return 0.0
        return _clamp(float(retriever.source_similarity(source.url, question)))

    def _combine_relevance(self, primary_relevance: float, vector_relevance: float, primary_weight: float = 0.5) -> float:
        """融合 LLM/关键词相关性和向量相似度。"""
        primary_relevance = _clamp(primary_relevance)
        vector_relevance = _clamp(vector_relevance)
        if vector_relevance <= 0:
            return primary_relevance
        return round(primary_relevance * primary_weight + vector_relevance * (1 - primary_weight), 4)

    def _authority(self, source: Source) -> float:
        """根据来源类型估算权威性。"""
        if source.kind == "paper":
            return 0.9
        if source.kind == "official":
            return 0.95
        if source.kind == "github":
            return 0.82
        return 0.65

    def _freshness(self, source: Source, current_year: int) -> float:
        """根据发布时间估算新鲜度。"""
        try:
            year = int(source.published_at[:4])
        except ValueError:
            return 0.5
        return max(0.35, min(1.0, 1.0 - (current_year - year) * 0.12))

    def _relevance(self, source: Source, question: str) -> float:
        """根据问题关键词在来源文本中的命中情况估算相关度。"""
        haystack = (source.title + " " + source.snippet + " " + _metadata_text(source)).lower()
        keywords = [token.lower() for token in question.replace("，", " ").replace("。", " ").split() if len(token) > 2]
        hits = sum(1 for keyword in keywords if keyword in haystack)

        
        return min(1.0, 0.55 + hits * 0.08)


class CitationVerifierAgent:
    """引用验证 Agent。

    它负责把来源转换成 claim，并判断 claim 是否被 evidence 支撑。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；有 LLM 时会增加严格判断。"""
        self.llm = llm or LLMProvider()

    def verify(self, question: str, sources: List[Source], scores: Dict[str, SourceScore], retriever=None) -> List[Claim]:
        """为每个来源生成 claim，并标记 supported/weak/unsupported。"""
        claims = []
        question_terms = _keywords(question)
        for source in sources:
            score = scores[source.url]
            chunk = retriever.best_for_source(source.url, question) if retriever else None
            evidence_text = chunk.text if chunk and chunk.text else source.title + " " + source.snippet + " " + _metadata_text(source)
            llm_claim = self._llm_verify_claim(question, source, score, evidence_text)
            if llm_claim:
                claims.append(llm_claim)
                continue
            evidence_terms = _keywords(evidence_text)
            overlap = len(question_terms & evidence_terms)
            grounding = min(1.0, 0.45 + overlap * 0.08)
            if source.kind == "github":
                claim_text = source.title + " 是实现生态和框架方向的相关证据。"
            elif source.kind == "paper":
                claim_text = source.title + " 支持架构、协议、记忆或安全分析。"
            else:
                claim_text = source.title + " 为研究问题提供支撑性上下文。"
            confidence = round(score.final * 0.7 + grounding * 0.3, 2)
            if confidence >= 0.7 and overlap > 0:
                status = "supported"
            elif confidence >= 0.55:
                status = "weak"
            else:
                status = "unsupported"
            reason = "得分={:.2f}，关键词重合={}，证据贴合度={:.2f}".format(score.final, overlap, grounding)
            llm_status, llm_reason = self._llm_judge(claim_text, evidence_text)
            if llm_status:
                status = llm_status
                reason = reason + "；LLM 判断=" + llm_reason
            claims.append(
                Claim(
                    claim_text,
                    [source.url],
                    confidence,
                    status,
                    _excerpt(evidence_text),
                    reason,
                )
            )
        return claims

    def _llm_verify_claim(self, question: str, source: Source, score: SourceScore, evidence: str) -> Optional[Claim]:
        """用 LLM 直接做 claim extraction + evidence support 判断。"""
        if not self.llm.available():
            return None
        user = (
            "研究问题：\n{}\n\n来源信息：\n{}\n\n来源评分：{:.2f} ({})\n\n证据片段：\n{}\n\n"
            "请只返回 JSON，不要 Markdown。字段：claim、status、confidence、evidence_excerpt、reason。"
            "status 只能是 supported、weak、unsupported。"
            "claim 必须是证据能支撑的具体结论；如果证据不能回答研究问题，status=unsupported。"
        ).format(question, _source_text(source, limit=1200), score.final, score.rationale, evidence[:1800])
        answer = self.llm.complete(
            "你是严格的引用验证器，负责判断结论是否被证据支持，不能编造证据外的信息。",
            user,
        )
        payload = _parse_json_object(answer)
        if not payload:
            return None
        status = str(payload.get("status") or "").strip().lower()
        if status not in {"supported", "weak", "unsupported"}:
            return None
        claim_text = str(payload.get("claim") or "").strip()
        if not claim_text:
            return None
        confidence = round(_clamp(_as_float(payload.get("confidence"), score.final)), 2)
        evidence_excerpt = str(payload.get("evidence_excerpt") or "").strip() or _excerpt(evidence)
        reason = "LLM 判断：" + str(payload.get("reason") or "")
        return Claim(
            text=claim_text,
            source_urls=[source.url],
            confidence=confidence,
            status=status,
            evidence_excerpt=_excerpt(evidence_excerpt),
            verification_reason=_excerpt(reason, 260),
        )

    def _llm_judge(self, claim: str, evidence: str) -> tuple:
        """让 LLM 判断 claim 是否被 evidence 支持。"""
        if not self.llm.available():
            return "", ""
        prompt = (
            "结论：\n{}\n\n证据：\n{}\n\n"
            "只返回一行，格式必须是：supported|weak|unsupported - 简短中文理由"
        ).format(claim, evidence[:1200])
        answer = self.llm.complete(
            "你负责判断结论是否被给定证据支持。必须严格，理由用中文。",
            prompt,
        ).strip()
        lowered = answer.lower()
        if lowered.startswith("supported"):
            return "supported", _excerpt(answer, 160)
        if lowered.startswith("weak"):
            return "weak", _excerpt(answer, 160)
        if lowered.startswith("unsupported"):
            return "unsupported", _excerpt(answer, 160)
        return "", ""


class ConflictDetectorAgent:
    """冲突检测 Agent。

    它检查同一主题下是否同时存在 supported 和 unsupported 的结论。
    """

    def detect(self, claims: List[Claim]) -> List[Conflict]:
        """根据 claim 状态检测主题级冲突。"""
        conflicts = []
        by_topic = self._group_by_topic(claims)
        for topic, topic_claims in by_topic.items():
            statuses = {claim.status for claim in topic_claims}
            if "supported" in statuses and "unsupported" in statuses:
                conflicts.append(
                    Conflict(
                        topic=topic,
                        source_urls=_claim_urls(topic_claims),
                        severity="high",
                        explanation="同一主题下同时出现了支持和不支持的结论。",
                    )
                )
            elif "supported" in statuses and "weak" in statuses and len(topic_claims) >= 3:
                conflicts.append(
                    Conflict(
                        topic=topic,
                        source_urls=_claim_urls(topic_claims),
                        severity="low",
                        explanation="有些来源强支持该主题，但另一些只提供了弱支持。",
                    )
                )
        return conflicts

    def _group_by_topic(self, claims: List[Claim]) -> Dict[str, List[Claim]]:
        """把 claim 按 RAG、MCP、workflow、安全、评估等主题归类。"""
        topics = {
            "rag": ["rag", "retrieval", "grounding"],
            "mcp": ["mcp", "model context protocol"],
            "workflow": ["workflow", "langgraph", "crewai", "autogen", "framework"],
            "safety": ["safety", "prompt injection", "guardrail"],
            "evaluation": ["evaluation", "eval", "benchmark"],
        }
        grouped: Dict[str, List[Claim]] = {}
        for claim in claims:
            text = (claim.text + " " + claim.evidence_excerpt).lower()
            for topic, signals in topics.items():
                if any(signal in text for signal in signals):
                    grouped.setdefault(topic, []).append(claim)
        return grouped


class ReflectionAgent:
    """反思 Agent。

    它在 verify 后检查是否缺少关键维度，如果缺失就让 LangGraph 回到 search。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；有 LLM 时用模型判断还缺哪些研究证据。"""
        self.llm = llm or LLMProvider()

    def find_gaps(
        self,
        question: str,
        claims: List[Claim],
        sources: List[Source] = None,
        context_text: str = "",
    ) -> List[str]:
        """根据问题、claim 和来源判断还缺哪些证据。"""
        llm_gaps = self._llm_find_gaps(question, claims, sources or [], context_text)
        if llm_gaps is not None:
            return llm_gaps
        joined = " ".join(claim.text.lower() for claim in claims)
        gaps = []
        source_text = " ".join(
            [source.title + " " + source.snippet + " " + " ".join(source.metadata.values()) for source in (sources or [])]
        ).lower()
        evidence_text = joined + " " + source_text + " " + question.lower()
        for expected in ["rag", "mcp", "context", "safety", "evaluation", "checkpoint"]:
            if expected not in evidence_text:
                gaps.append("需要补充关于 " + expected + " 的证据")
        weak_claims = [claim for claim in claims if claim.status != "supported"]
        if weak_claims and len(weak_claims) > len(claims) // 2:
            gaps.append("需要更强的来源来支撑弱证据或不支持的结论")
        return gaps

    def _llm_find_gaps(
        self,
        question: str,
        claims: List[Claim],
        sources: List[Source],
        context_text: str,
    ) -> Optional[List[str]]:
        """让 LLM 根据已收集证据判断是否需要二次搜索。"""
        if not self.llm.available():
            return None
        claim_text = "\n".join(
            "- {} | {} | {}".format(claim.status, claim.text, claim.verification_reason)
            for claim in claims[:16]
        )
        source_text = "\n".join("- " + _source_text(source, limit=260) for source in sources[:12])
        user = (
            "研究问题：\n{}\n\n受预算控制的上下文：\n{}\n\n已有来源：\n{}\n\n已有结论：\n{}\n\n"
            "请只返回 JSON：{{\"gaps\": [\"需要继续搜索的具体问题\"]}}。"
            "如果证据已经足够，返回 {{\"gaps\": []}}。缺口要适合直接作为下一轮 search query。"
        ).format(question, context_text or "none", source_text, claim_text)
        answer = self.llm.complete(
            "你是 DeepResearch 反思节点，负责发现证据缺口并决定是否需要补搜。",
            user,
        )
        payload = _parse_json_object(answer)
        if payload is None:
            return None
        gaps = payload.get("gaps")
        if not isinstance(gaps, list):
            return None
        return [str(gap).strip() for gap in gaps if str(gap).strip()][:6]


class ReportAgent:
    """报告生成 Agent。

    它把计划、来源、评分、claim、冲突和缺口组织成中文 Markdown 报告。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；有 LLM 时用它生成核心判断。"""
        self.llm = llm or LLMProvider()

    def write(
        self,
        question: str,
        plan: ResearchPlan,
        sources: List[Source],
        scores: Dict[str, SourceScore],
        claims: List[Claim],
        conflicts: List[Conflict] = None,
        gaps: List[str] = None,
        context_text: str = "",
    ) -> str:
        """生成最终 Markdown 报告。"""
        llm_summary = self._llm_summary(question, sources, claims, gaps or [], context_text)
        summary = llm_summary or self._rule_summary(question, sources, scores, claims, gaps or [])
        lines = [
            "# DeepResearch Agent 报告",
            "",
            "**问题：** " + question,
            "",
            "## 研究计划",
            "",
        ]
        for index, sub_question in enumerate(plan.sub_questions, 1):
            lines.append("{}. {}".format(index, sub_question))
        lines.extend(["", "## 核心判断", ""])
        lines.append(summary)
        lines.extend(["", "## 证据", ""])
        for source in sorted(sources, key=lambda item: scores[item.url].final, reverse=True):
            score = scores[source.url]
            lines.append(
                "- [{}]({}) - 类型={}，来源={}，评分={:.2f}。{}".format(
                    source.title,
                    source.url,
                    source.kind,
                    source.provider,
                    score.final,
                    source.snippet,
                )
            )
        lines.extend(["", "## 引用校验", ""])
        for claim in claims:
            lines.append("- {} 置信度={:.2f}：{} [{}]({})".format(claim.status, claim.confidence, claim.text, "来源", claim.source_urls[0]))
            if claim.evidence_excerpt:
                lines.append("  证据片段：" + claim.evidence_excerpt)
            if claim.verification_reason:
                lines.append("  校验理由：" + claim.verification_reason)
        lines.extend(["", "## 冲突检查", ""])
        if conflicts:
            for conflict in conflicts:
                lines.append("- {} 严重程度：{}。{}".format(conflict.topic, conflict.severity, conflict.explanation))
        else:
            lines.append("- 未发现明显的来源冲突。")
        lines.extend(["", "## 剩余缺口", ""])
        if gaps:
            for gap in gaps:
                lines.append("- " + gap)
        else:
            lines.append("- 反思后未发现明显缺口。")
        lines.extend(["", "## 后续动作", ""])
        lines.extend(self._next_actions(sources, scores, claims, conflicts or [], gaps or []))
        return "\n".join(lines) + "\n"

    def _llm_summary(
        self,
        question: str,
        sources: List[Source],
        claims: List[Claim],
        gaps: List[str],
        context_text: str,
    ) -> str:
        """让 LLM 基于证据生成一段谨慎的核心判断。"""
        if not self.llm.available():
            return ""
        evidence = "\n".join(
            "- {}: {}".format(source.title, source.snippet[:240])
            for source in sorted(sources, key=lambda item: item.title)[:10]
        )
        claim_text = "\n".join("- {} ({:.2f})".format(claim.text, claim.confidence) for claim in claims[:10])
        user = (
            "问题：\n{}\n\n受预算控制的上下文：\n{}\n\n证据：\n{}\n\n结论：\n{}\n\n缺口：\n{}\n\n"
            "请写一段简洁、基于证据的核心判断，不要编造事实。"
        ).format(question, context_text or "none", evidence, claim_text, "; ".join(gaps) or "none")
        return self.llm.complete(
            "你撰写谨慎的研究报告结论，只能基于给定证据。",
            user,
        )

    def _rule_summary(
        self,
        question: str,
        sources: List[Source],
        scores: Dict[str, SourceScore],
        claims: List[Claim],
        gaps: List[str],
    ) -> str:
        """没有 LLM 时，按已验证证据生成与当前问题对应的保守摘要。"""
        supported = [claim for claim in claims if claim.status == "supported"][:3]
        if supported:
            evidence = "；".join(claim.text for claim in supported)
            return "围绕“{}”，当前有 {} 条被明确支持的结论：{}。".format(question, len(supported), evidence)

        ranked_sources = sorted(sources, key=lambda source: scores[source.url].final, reverse=True)[:3]
        source_names = "、".join(source.title for source in ranked_sources)
        gap_note = "；仍需补充：{}".format("、".join(gaps[:2])) if gaps else ""
        return "围绕“{}”，当前已收集 {} 个去重来源，并优先参考 {}。现有证据以 weak 或待验证结论为主，报告保留引用与验证状态，避免把不充分证据写成确定性结论{}。".format(
            question,
            len(sources),
            source_names or "可用来源",
            gap_note,
        )

    def _next_actions(
        self,
        sources: List[Source],
        scores: Dict[str, SourceScore],
        claims: List[Claim],
        conflicts: List[Conflict],
        gaps: List[str],
    ) -> List[str]:
        """根据本次运行结果生成后续动作，而不是固定路线图。"""
        actions = []
        for gap in gaps[:3]:
            actions.append("继续补搜：{}。".format(gap))

        weak_or_unsupported = [claim for claim in claims if claim.status != "supported"]
        if claims and len(weak_or_unsupported) / len(claims) >= 0.4:
            actions.append("补充更权威或更贴近问题的来源，优先处理 weak/unsupported claim。")

        if conflicts:
            topics = "、".join(conflict.topic for conflict in conflicts[:3])
            actions.append("对存在冲突的主题做交叉验证：{}。".format(topics))

        provider_count = len({source.provider for source in sources})
        kind_count = len({source.kind for source in sources})
        if sources and (provider_count < 3 or kind_count < 3):
            actions.append("扩展检索来源类型，补充官方文档、论文、GitHub 或 Web 搜索中的缺失维度。")

        low_score_sources = [
            source for source in sources
            if source.url in scores and scores[source.url].final < 0.6
        ]
        if low_score_sources:
            actions.append("降低低分来源在报告中的权重，必要时替换为更高分来源。")

        if not actions:
            actions.append("当前证据链基本完整，可以进入人工复核或导出报告。")
        return ["{}. {}".format(index, action) for index, action in enumerate(actions[:5], 1)]


class ReportAuditorAgent:
    """报告审计 Agent。

    它检查最终报告的关键句是否能匹配到 supported claim，防止报告阶段新增幻觉。
    """

    def __init__(self, llm: LLMProvider = None) -> None:
        """接收可选 LLM；有 LLM 时使用报告级 groundedness 审计。"""
        self.llm = llm or LLMProvider()

    def audit(self, markdown: str, claims: List[Claim]) -> List[SentenceCheck]:
        """对报告关键句进行 grounded/weak/uncited 审计。"""
        llm_checks = self._llm_audit(markdown, claims)
        if llm_checks is not None:
            return llm_checks
        checks = []
        supported_claims = [claim for claim in claims if claim.status == "supported"]
        weak_or_unsupported = [claim for claim in claims if claim.status != "supported"]
        for sentence in self._candidate_sentences(markdown):
            matched = self._match_claim(sentence, supported_claims)
            if matched:
                checks.append(
                    SentenceCheck(
                        sentence=sentence,
                        status="grounded",
                        evidence_url=matched.source_urls[0] if matched.source_urls else "",
                        reason="匹配到了 supported 结论证据。",
                    )
                )
                continue
            weak = self._match_claim(sentence, weak_or_unsupported)
            if weak:
                checks.append(
                    SentenceCheck(
                        sentence=sentence,
                        status="weak",
                        evidence_url=weak.source_urls[0] if weak.source_urls else "",
                        reason="只匹配到 weak 或 unsupported 证据。",
                    )
                )
                continue
            checks.append(
                SentenceCheck(
                    sentence=sentence,
                    status="uncited",
                    reason="没有找到匹配的结论证据。",
                )
            )
        return checks

    def append_section(self, markdown: str, checks: List[SentenceCheck]) -> str:
        """把报告证据审计结果追加到 Markdown 末尾。"""
        lines = [markdown.rstrip(), "", "## 报告证据审计", ""]
        if not checks:
            lines.append("- 没有需要额外审计的报告句子。")
            return "\n".join(lines) + "\n"
        for check in checks:
            suffix = " [{}]({})".format("来源", check.evidence_url) if check.evidence_url else ""
            lines.append("- {}：{}{} - {}".format(check.status, check.sentence, suffix, check.reason))
        return "\n".join(lines) + "\n"

    def _candidate_sentences(self, markdown: str) -> List[str]:
        """从报告核心判断部分抽取需要审计的句子。"""
        candidates = []
        capture = False
        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if line == "## 核心判断":
                capture = True
                continue
            if line.startswith("## ") and line != "## 核心判断":
                capture = False
            if capture and line and not line.startswith("#"):
                candidates.extend(_split_sentences(line))
        return [sentence for sentence in candidates if len(sentence) >= 12]

    def _match_claim(self, sentence: str, claims: List[Claim]) -> Claim:
        """用关键词重合度把报告句子匹配到最相关的 claim。"""
        sentence_terms = _keywords(sentence)
        best = None
        best_overlap = 0
        for claim in claims:
            claim_terms = _keywords(claim.text + " " + claim.evidence_excerpt)
            overlap = len(sentence_terms & claim_terms)
            if overlap > best_overlap:
                best = claim
                best_overlap = overlap
        return best if best_overlap >= 2 else None

    def _llm_audit(self, markdown: str, claims: List[Claim]) -> Optional[List[SentenceCheck]]:
        """让 LLM 对核心判断逐句做证据支撑审计。"""
        if not self.llm.available():
            return None
        sentences = self._candidate_sentences(markdown)
        if not sentences:
            return []
        claim_text = "\n".join(
            "{}. {} | {} | {} | {}".format(
                index + 1,
                claim.status,
                claim.text,
                claim.source_urls[0] if claim.source_urls else "",
                claim.evidence_excerpt,
            )
            for index, claim in enumerate(claims[:20])
        )
        sentence_text = "\n".join("{}. {}".format(index + 1, sentence) for index, sentence in enumerate(sentences[:20]))
        user = (
            "待审计句子：\n{}\n\n可用证据结论：\n{}\n\n"
            "请只返回 JSON：{{\"checks\": [{{\"sentence\": \"...\", \"status\": \"grounded|weak|uncited\", "
            "\"evidence_url\": \"...\", \"reason\": \"...\"}}]}}。"
            "只有被 supported claim 明确支撑的句子才能标 grounded。"
        ).format(sentence_text, claim_text)
        answer = self.llm.complete(
            "你是报告 groundedness 审计器，负责检查报告句子是否被证据支撑。",
            user,
        )
        payload = _parse_json_object(answer)
        if not payload:
            return None
        raw_checks = payload.get("checks")
        if not isinstance(raw_checks, list):
            return None
        checks = []
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            if status not in {"grounded", "weak", "uncited"}:
                status = "uncited"
            checks.append(
                SentenceCheck(
                    sentence=str(item.get("sentence") or "").strip(),
                    status=status,
                    evidence_url=str(item.get("evidence_url") or "").strip(),
                    reason=_excerpt(str(item.get("reason") or "LLM 报告审计"), 180),
                )
            )
        return [check for check in checks if check.sentence]


def _keywords(text: str) -> Set[str]:
    """从文本中提取关键词，用于匹配和简单 grounding 判断。"""
    normalized = text.lower()
    for char in "，。,.()[]:;!?/":
        normalized = normalized.replace(char, " ")
    stop_words = {"the", "and", "for", "with", "that", "this", "from", "into", "about"}
    return {token for token in normalized.split() if len(token) > 2 and token not in stop_words}


def _focus_terms(text: str) -> List[str]:
    """提取 Agent 项目常见技术词，用于规则 query 改写。"""
    known_terms = ["LangGraph", "CrewAI", "AutoGen", "LangChain4j", "Spring AI", "MCP", "RAG", "Eval"]
    lowered = text.lower()
    return [term for term in known_terms if term.lower() in lowered]


def _unique_texts(items: List[str]) -> List[str]:
    """保留顺序去重，并过滤过短 query。"""
    seen = set()
    results = []
    for item in items:
        cleaned = " ".join(str(item).split())
        key = cleaned.lower()
        if len(cleaned) < 4 or key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def _source_text(source: Source, limit: int = 1200) -> str:
    """把 Source 压成适合给 LLM 判断的文本。"""
    metadata = "; ".join(
        "{}={}".format(key, value)
        for key, value in source.metadata.items()
        if key not in NON_EVIDENCE_METADATA_KEYS
    )
    text = "title={}\nurl={}\nkind={}\nprovider={}\npublished_at={}\nsnippet={}\nmetadata={}".format(
        source.title,
        source.url,
        source.kind,
        source.provider,
        source.published_at,
        source.snippet,
        metadata,
    )
    return _excerpt(text, limit)


def _metadata_text(source: Source) -> str:
    """返回不含检索 provenance 的 metadata 文本，避免相关性虚高。"""
    return " ".join(
        str(value)
        for key, value in source.metadata.items()
        if key not in NON_EVIDENCE_METADATA_KEYS
    )


def _parse_json_object(text: str) -> Optional[dict]:
    """解析 LLM 返回的 JSON 对象，兼容被 Markdown 包裹的情况。"""
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(value, default: float) -> float:
    """把 LLM 字段转换为 float，失败时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """限制分数范围。"""
    return max(low, min(high, value))


def _excerpt(text: str, limit: int = 240) -> str:
    """截取一段适合放进报告的证据摘要。"""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _claim_urls(claims: List[Claim]) -> List[str]:
    """从一组 claim 中收集去重后的来源 URL。"""
    urls = []
    for claim in claims:
        for url in claim.source_urls:
            if url not in urls:
                urls.append(url)
    return urls[:6]


def _split_sentences(text: str) -> List[str]:
    """把中文/英文文本粗略切成句子。"""
    normalized = text.replace("。", ". ").replace("；", ". ").replace(";", ". ")
    parts = []
    for part in normalized.split(". "):
        cleaned = part.strip(" -")
        if cleaned:
            parts.append(cleaned)
    return parts
