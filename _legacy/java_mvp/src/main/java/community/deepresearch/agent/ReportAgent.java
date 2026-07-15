package community.deepresearch.agent;

import community.deepresearch.model.Citation;
import community.deepresearch.model.ResearchPlan;
import community.deepresearch.model.ResearchReport;
import community.deepresearch.model.ResearchSource;
import community.deepresearch.model.ResearchTask;
import community.deepresearch.model.SourceScore;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public final class ReportAgent {
    public ResearchReport write(
            ResearchTask task,
            ResearchPlan plan,
            List<ResearchSource> sources,
            Map<String, SourceScore> scores,
            List<Citation> citations
    ) {
        StringBuilder markdown = new StringBuilder();
        markdown.append("# DeepResearch 报告\n\n");
        markdown.append("**研究问题：** ").append(task.question()).append("\n\n");
        markdown.append("**生成时间：** ").append(Instant.now()).append("\n\n");

        markdown.append("## 摘要\n\n");
        markdown.append("本次研究将问题拆解为框架能力、生态活跃度、协议支持、适用场景和风险边界等维度。");
        markdown.append("从当前证据看，成熟 Agent 系统的区分度不在单次问答，而在可恢复工作流、多工具接入、来源评分、引用验证和人机协同。\n\n");

        markdown.append("## 研究计划\n\n");
        for (int i = 0; i < plan.questions().size(); i++) {
            markdown.append(i + 1).append(". ").append(plan.questions().get(i)).append("\n");
        }
        markdown.append("\n");

        markdown.append("## 对比结论\n\n");
        markdown.append("| 维度 | Spring AI | LangChain4j | LangGraph4j/状态图 |\n");
        markdown.append("| --- | --- | --- | --- |\n");
        markdown.append("| 定位 | Spring 生态 AI 抽象与企业集成 | Java LLM 应用与 Agent 开发框架 | 有状态 Agent 工作流编排 |\n");
        markdown.append("| 优势 | 与 Spring Boot、Bean、配置体系契合 | 工具、记忆、RAG、模型集成更直接 | 适合长任务、循环、分支、检查点 |\n");
        markdown.append("| 风险 | Agent 编排层需要补充 | 复杂流程需要额外状态管理 | Java 生态成熟度需持续验证 |\n");
        markdown.append("| 适用 | 企业 Java 应用接入 AI | 快速构建 Java Agent/RAG | DeepResearch 这类多阶段研究流 |\n\n");

        markdown.append("## 来源可信度\n\n");
        sources.stream()
                .sorted(Comparator.comparing(source -> -scores.get(source.url()).finalScore()))
                .forEach(source -> {
                    SourceScore score = scores.get(source.url());
                    markdown.append("- [").append(source.title()).append("](").append(source.url()).append(")");
                    markdown.append("：").append(source.type()).append("，score=").append(score.finalScore());
                    markdown.append("，").append(source.snippet()).append("\n");
                });
        markdown.append("\n");

        markdown.append("## 引用校验\n\n");
        for (Citation citation : citations) {
            markdown.append("- ").append(citation.supports() ? "支持" : "存疑");
            markdown.append("，confidence=").append(citation.confidence());
            markdown.append("：").append(citation.claim()).append(" [source](").append(citation.sourceUrl()).append(")\n");
        }
        markdown.append("\n");

        markdown.append("## 风险与局限\n\n");
        markdown.append("- 当前 MVP 内置的是可替换的模拟 SourceConnector，还没有实时调用搜索引擎、GitHub API 和论文库。\n");
        markdown.append("- 真正上线前需要加入引用逐句校验、来源快照、反爬与限流、失败重试和人工确认。\n");
        markdown.append("- 对“热门程度”的判断必须使用实时 GitHub 指标，不能只依赖模型记忆。\n\n");

        markdown.append("## 下一步\n\n");
        markdown.append("优先接入真实 GitHub Search、arXiv、网页搜索和 MCP Tool Registry；随后加入任务暂停/恢复、Human-in-the-loop 和 PDF 导出。\n");
        return new ResearchReport(task.id(), markdown.toString(), Instant.now());
    }
}
