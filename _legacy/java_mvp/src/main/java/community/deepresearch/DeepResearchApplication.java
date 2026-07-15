package community.deepresearch;

import community.deepresearch.agent.ResearchWorkflow;
import community.deepresearch.store.RunStore;
import community.deepresearch.web.ResearchServer;
import java.nio.file.Path;
import java.util.concurrent.Executors;

public final class DeepResearchApplication {
    private DeepResearchApplication() {
    }

    public static void main(String[] args) throws Exception {
        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8080"));
        RunStore runStore = new RunStore(Path.of(".deepresearch", "runs"));
        ResearchWorkflow workflow = ResearchWorkflow.defaultWorkflow(runStore);
        ResearchServer server = new ResearchServer(port, workflow, runStore, Executors.newCachedThreadPool());
        server.start();
        System.out.println("DeepResearch Agent is running at http://localhost:" + port);
    }
}
