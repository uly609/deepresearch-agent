package community.deepresearch;

import community.deepresearch.agent.ResearchWorkflow;
import community.deepresearch.model.ResearchEvent;
import community.deepresearch.model.ResearchReport;
import community.deepresearch.model.ResearchTask;
import community.deepresearch.store.RunStore;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class ResearchWorkflowTest {
    private ResearchWorkflowTest() {
    }

    public static void main(String[] args) {
        RunStore store = new RunStore(Path.of("build", "test-runs"));
        ResearchWorkflow workflow = ResearchWorkflow.defaultWorkflow(store);
        ResearchTask task = new ResearchTask("比较 Spring AI、LangChain4j 和 LangGraph4j");
        List<ResearchEvent> events = new ArrayList<>();
        ResearchReport report = workflow.run(task, events::add);

        assertTrue("completed".equals(task.status()), "task should complete");
        assertTrue(task.sources().size() >= 6, "sources should be collected");
        assertTrue(task.citations().size() == task.sources().size(), "each source should be checked");
        assertTrue(report.markdown().contains("引用校验"), "report should include verification section");
        assertTrue(events.stream().anyMatch(event -> "run_finished".equals(event.type())), "run_finished event should be emitted");
        System.out.println("ResearchWorkflowTest passed");
    }

    private static void assertTrue(boolean value, String message) {
        if (!value) {
            throw new AssertionError(message);
        }
    }
}
