package community.deepresearch.agent;

import community.deepresearch.model.Citation;
import community.deepresearch.model.ResearchEvent;
import community.deepresearch.model.ResearchPlan;
import community.deepresearch.model.ResearchReport;
import community.deepresearch.model.ResearchSource;
import community.deepresearch.model.ResearchTask;
import community.deepresearch.model.SourceScore;
import community.deepresearch.store.RunStore;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public final class ResearchWorkflow {
    private final ResearchPlanner planner;
    private final SearchAgent searchAgent;
    private final SourceEvaluationAgent evaluator;
    private final VerificationAgent verifier;
    private final ReportAgent reporter;
    private final RunStore runStore;

    public ResearchWorkflow(
            ResearchPlanner planner,
            SearchAgent searchAgent,
            SourceEvaluationAgent evaluator,
            VerificationAgent verifier,
            ReportAgent reporter,
            RunStore runStore
    ) {
        this.planner = planner;
        this.searchAgent = searchAgent;
        this.evaluator = evaluator;
        this.verifier = verifier;
        this.reporter = reporter;
        this.runStore = runStore;
    }

    public static ResearchWorkflow defaultWorkflow(RunStore runStore) {
        return new ResearchWorkflow(
                new ResearchPlanner(),
                new SearchAgent(List.of(
                        new BuiltInSourceConnector("official-docs"),
                        new BuiltInSourceConnector("github"),
                        new BuiltInSourceConnector("papers"),
                        new BuiltInSourceConnector("industry-news")
                )),
                new SourceEvaluationAgent(),
                new VerificationAgent(),
                new ReportAgent(),
                runStore
        );
    }

    public ResearchReport run(ResearchTask task, Consumer<ResearchEvent> events) {
        task.markRunning();
        emit(task, events, "run_started", "研究任务已启动");

        ResearchPlan plan = planner.plan(task.question());
        task.setPlan(plan);
        emit(task, events, "plan_created", "已拆解为 " + plan.questions().size() + " 个研究子问题");
        checkpoint(task);

        List<ResearchSource> sources = new ArrayList<>();
        for (String subQuestion : plan.questions()) {
            emit(task, events, "search_started", subQuestion);
            List<ResearchSource> found = searchAgent.search(subQuestion);
            sources.addAll(found);
            emit(task, events, "search_finished", "获得 " + found.size() + " 条候选来源");
            checkpoint(task);
        }
        sources = dedupeSources(sources);
        emit(task, events, "sources_deduped", "全局去重后保留 " + sources.size() + " 条来源");
        checkpoint(task);

        Map<String, SourceScore> scores = evaluator.score(sources);
        task.setSources(sources);
        task.setScores(scores);
        emit(task, events, "sources_scored", "已完成来源可信度评分");
        checkpoint(task);

        List<Citation> citations = verifier.verify(task.question(), sources, scores);
        task.setCitations(citations);
        long weakCitations = citations.stream().filter(citation -> citation.confidence() < 0.65).count();
        emit(task, events, "citations_verified", "引用校验完成，低置信引用 " + weakCitations + " 条");
        checkpoint(task);

        ResearchReport report = reporter.write(task, plan, sources, scores, citations);
        task.finish(report);
        runStore.writeTask(task);
        runStore.writeReport(task.id(), report.markdown());
        emit(task, events, "run_finished", "报告已生成");
        return report;
    }

    private void checkpoint(ResearchTask task) {
        task.setUpdatedAt(Instant.now());
        runStore.writeTask(task);
    }

    private List<ResearchSource> dedupeSources(List<ResearchSource> sources) {
        Map<String, ResearchSource> deduped = new LinkedHashMap<>();
        for (ResearchSource source : sources) {
            deduped.putIfAbsent(source.url(), source);
        }
        return new ArrayList<>(deduped.values());
    }

    private void emit(ResearchTask task, Consumer<ResearchEvent> events, String type, String message) {
        ResearchEvent event = new ResearchEvent(task.id(), type, message, Instant.now());
        task.addEvent(event);
        events.accept(event);
    }
}
