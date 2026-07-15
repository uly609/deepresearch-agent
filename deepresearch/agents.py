"""DeepResearch 的 Agent 能力模块。

这个文件里不是一个单独的大 Agent，而是把 Planner、Evaluator、Verifier、
Reflector、Reporter 等能力拆成多个小 Agent。LangGraph 工作流会按节点调用
这些 Agent，共同完成研究任务。
"""

from datetime import datetime
from typing import Dict, List, Set

from .llm_provider import LLMProvider, parse_json_list
from .models import Claim, Conflict, ResearchPlan, SentenceCheck, Source, SourceScore


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


class SourceEvaluatorAgent:
    """来源评分 Agent。

    它不负责搜索，而是在搜索完成后评价每个 Source 是否可靠。
    """

    def score(self, sources: List[Source], question: str) -> Dict[str, SourceScore]:
        """给每个来源计算 SourceScore。"""
        scores = {}
        current_year = datetime.utcnow().year
        for source in sources:
            authority = self._authority(source)
            freshness = self._freshness(source, current_year)
            relevance = self._relevance(source, question)
            risk = 0.1 if "security_findings" not in source.metadata else 0.8
            final = round(authority * 0.35 + freshness * 0.2 + relevance * 0.35 + (1 - risk) * 0.1, 2)
            rationale = "authority={:.2f}, freshness={:.2f}, relevance={:.2f}, risk={:.2f}".format(
                authority, freshness, relevance, risk
            )
            scores[source.url] = SourceScore(authority, freshness, relevance, risk, final, rationale)
        return scores

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
        haystack = (source.title + " " + source.snippet + " " + " ".join(source.metadata.values())).lower()
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
            evidence_text = chunk.text if chunk and chunk.text else source.title + " " + source.snippet + " " + " ".join(source.metadata.values())
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

    def find_gaps(self, question: str, claims: List[Claim], sources: List[Source] = None) -> List[str]:
        """根据问题、claim 和来源判断还缺哪些证据。"""
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
    ) -> str:
        """生成最终 Markdown 报告。"""
        llm_summary = self._llm_summary(question, sources, claims, gaps or [])
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
        if llm_summary:
            lines.append(llm_summary)
        else:
            lines.append(
                "对于以 Agent 为核心的项目，Python 应该作为主实现语言，因为 LangGraph、CrewAI、AutoGen、RAG、评估框架、浏览器/搜索工具和 MCP 客户端的活跃生态更强。Java 可以保留给平台集成，但不应作为主叙事。"
            )
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
        lines.extend(["", "## 下一步优化", ""])
        lines.extend(
            [
                "1. 配置更稳定的搜索提供方，例如 Brave、SerpAPI 或真实 MCP 搜索网关。",
                "2. 让可选 LLM 参与报告级引用审计，而不只依赖当前的规则判断。",
                "3. 将本地哈希向量库替换为 FAISS、Chroma、Milvus 或 pgvector。",
                "4. 增加上下文压缩、证据 pinning 和完整的 GraphState 恢复。",
                "5. 扩展评估指标，覆盖引用准确率、冲突检测和来源多样性。",
            ]
        )
        return "\n".join(lines) + "\n"

    def _llm_summary(self, question: str, sources: List[Source], claims: List[Claim], gaps: List[str]) -> str:
        """让 LLM 基于证据生成一段谨慎的核心判断。"""
        if not self.llm.available():
            return ""
        evidence = "\n".join(
            "- {}: {}".format(source.title, source.snippet[:240])
            for source in sorted(sources, key=lambda item: item.title)[:10]
        )
        claim_text = "\n".join("- {} ({:.2f})".format(claim.text, claim.confidence) for claim in claims[:10])
        user = (
            "问题：\n{}\n\n证据：\n{}\n\n结论：\n{}\n\n缺口：\n{}\n\n"
            "请写一段简洁、基于证据的核心判断，不要编造事实。"
        ).format(question, evidence, claim_text, "; ".join(gaps) or "none")
        return self.llm.complete(
            "你撰写谨慎的研究报告结论，只能基于给定证据。",
            user,
        )


class ReportAuditorAgent:
    """报告审计 Agent。

    它检查最终报告的关键句是否能匹配到 supported claim，防止报告阶段新增幻觉。
    """

    def audit(self, markdown: str, claims: List[Claim]) -> List[SentenceCheck]:
        """对报告关键句进行 grounded/weak/uncited 审计。"""
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


def _keywords(text: str) -> Set[str]:
    """从文本中提取关键词，用于匹配和简单 grounding 判断。"""
    normalized = text.lower()
    for char in "，。,.()[]:;!?/":
        normalized = normalized.replace(char, " ")
    stop_words = {"the", "and", "for", "with", "that", "this", "from", "into", "about"}
    return {token for token in normalized.split() if len(token) > 2 and token not in stop_words}


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
