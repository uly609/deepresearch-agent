package community.deepresearch.agent;

import community.deepresearch.model.ResearchSource;
import java.time.LocalDate;
import java.util.List;

public final class BuiltInSourceConnector implements SourceConnector {
    private final String name;

    public BuiltInSourceConnector(String name) {
        this.name = name;
    }

    @Override
    public List<ResearchSource> search(String query) {
        return switch (name) {
            case "official-docs" -> List.of(
                    source("Spring AI Reference", "official", "https://docs.spring.io/spring-ai/reference/",
                            "Spring AI 的官方文档，适合确认模型抽象、工具调用、RAG 和向量库支持。", 2026, query),
                    source("LangChain4j Documentation", "official", "https://docs.langchain4j.dev/",
                            "LangChain4j 的官方文档，适合确认 Java LLM 应用、Agent、工具和记忆能力。", 2026, query),
                    source("Google Agent2Agent Protocol", "official", "https://github.com/google-a2a/A2A",
                            "A2A 协议项目，用于理解多 Agent 之间任务交换和互操作趋势。", 2026, query)
            );
            case "github" -> List.of(
                    source("spring-projects/spring-ai", "github", "https://github.com/spring-projects/spring-ai",
                            "Spring AI GitHub 仓库，可用于观察 release、star、issue、PR 和生态接入。", 2026, query),
                    source("langchain4j/langchain4j", "github", "https://github.com/langchain4j/langchain4j",
                            "LangChain4j GitHub 仓库，可用于观察 Java Agent 生态的活跃度和集成方向。", 2026, query),
                    source("langgraph4j/langgraph4j", "github", "https://github.com/langgraph4j/langgraph4j",
                            "LangGraph4j 相关仓库，可用于评估有状态 Agent 工作流在 Java 生态里的成熟度。", 2026, query)
            );
            case "papers" -> List.of(
                    source("Agentic AI Frameworks: Architectures, Protocols, and Design Challenges", "paper",
                            "https://arxiv.org/abs/2508.10146",
                            "综述 Agent 框架架构、协议、通信、记忆、安全护栏和服务化挑战。", 2025, query),
                    source("A Large-Scale Study on the Development and Issues of Multi-Agent AI Systems", "paper",
                            "https://arxiv.org/abs/2601.07136",
                            "分析多 Agent 开源系统的提交、issue、协调问题和维护趋势。", 2026, query)
            );
            default -> List.of(
                    source("OpenAI Deep Research", "industry", "https://openai.com/index/introducing-deep-research/",
                            "Deep Research 类产品说明，可用于对标自动研究、引用和长任务体验。", 2025, query),
                    source("Hugging Face Open Deep Research", "github", "https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research",
                            "开源 Deep Research 风格实现，可参考多步搜索、工具调用和报告生成方式。", 2025, query)
            );
        };
    }

    private ResearchSource source(String title, String type, String url, String snippet, int year, String query) {
        return new ResearchSource(title, type, url, snippet + " 查询上下文：" + shortQuery(query), LocalDate.of(year, 1, 1));
    }

    private String shortQuery(String query) {
        return query.length() > 80 ? query.substring(0, 80) + "..." : query;
    }
}
