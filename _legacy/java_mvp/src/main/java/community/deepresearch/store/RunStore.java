package community.deepresearch.store;

import community.deepresearch.model.Citation;
import community.deepresearch.model.ResearchEvent;
import community.deepresearch.model.ResearchSource;
import community.deepresearch.model.ResearchTask;
import community.deepresearch.model.SourceScore;
import community.deepresearch.util.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class RunStore {
    private final Path root;
    private final Map<String, ResearchTask> tasks = new ConcurrentHashMap<>();

    public RunStore(Path root) {
        this.root = root;
    }

    public void put(ResearchTask task) {
        tasks.put(task.id(), task);
        writeTask(task);
    }

    public ResearchTask get(String id) {
        return tasks.get(id);
    }

    public void writeTask(ResearchTask task) {
        try {
            Path dir = runDir(task.id());
            Files.createDirectories(dir);
            Files.writeString(dir.resolve("state.json"), toJson(task), StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new IllegalStateException("failed to write task state", exception);
        }
    }

    public void writeReport(String taskId, String markdown) {
        try {
            Path dir = runDir(taskId);
            Files.createDirectories(dir);
            Files.writeString(dir.resolve("report.md"), markdown, StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new IllegalStateException("failed to write report", exception);
        }
    }

    public String readReport(String taskId) {
        try {
            return Files.readString(runDir(taskId).resolve("report.md"), StandardCharsets.UTF_8);
        } catch (IOException exception) {
            return "";
        }
    }

    private Path runDir(String taskId) {
        return root.resolve(taskId);
    }

    private String toJson(ResearchTask task) {
        StringBuilder json = new StringBuilder();
        json.append("{");
        field(json, "id", task.id()).append(",");
        field(json, "question", task.question()).append(",");
        field(json, "status", task.status()).append(",");
        field(json, "createdAt", task.createdAt().toString()).append(",");
        field(json, "updatedAt", task.updatedAt().toString()).append(",");
        json.append("\"plan\":");
        if (task.plan() == null) {
            json.append("null");
        } else {
            json.append("{");
            field(json, "question", task.plan().question()).append(",");
            json.append("\"questions\":[");
            for (int i = 0; i < task.plan().questions().size(); i++) {
                if (i > 0) {
                    json.append(",");
                }
                json.append(Json.quote(task.plan().questions().get(i)));
            }
            json.append("]}");
        }
        json.append(",");
        json.append("\"sources\":[");
        for (int i = 0; i < task.sources().size(); i++) {
            ResearchSource source = task.sources().get(i);
            if (i > 0) {
                json.append(",");
            }
            json.append("{");
            field(json, "title", source.title()).append(",");
            field(json, "type", source.type()).append(",");
            field(json, "url", source.url()).append(",");
            field(json, "snippet", source.snippet()).append(",");
            field(json, "publishedAt", source.publishedAt().toString());
            json.append("}");
        }
        json.append("],");
        json.append("\"scores\":{");
        int scoreIndex = 0;
        for (Map.Entry<String, SourceScore> entry : task.scores().entrySet()) {
            if (scoreIndex++ > 0) {
                json.append(",");
            }
            SourceScore score = entry.getValue();
            json.append(Json.quote(entry.getKey())).append(":{");
            json.append("\"authority\":").append(score.authority()).append(",");
            json.append("\"freshness\":").append(score.freshness()).append(",");
            json.append("\"evidence\":").append(score.evidence()).append(",");
            json.append("\"finalScore\":").append(score.finalScore()).append("}");
        }
        json.append("},");
        json.append("\"citations\":[");
        for (int i = 0; i < task.citations().size(); i++) {
            Citation citation = task.citations().get(i);
            if (i > 0) {
                json.append(",");
            }
            json.append("{");
            field(json, "claim", citation.claim()).append(",");
            field(json, "sourceUrl", citation.sourceUrl()).append(",");
            json.append("\"confidence\":").append(citation.confidence()).append(",");
            json.append("\"supports\":").append(citation.supports());
            json.append("}");
        }
        json.append("],");
        json.append("\"events\":[");
        for (int i = 0; i < task.events().size(); i++) {
            ResearchEvent event = task.events().get(i);
            if (i > 0) {
                json.append(",");
            }
            json.append("{");
            field(json, "taskId", event.taskId()).append(",");
            field(json, "type", event.type()).append(",");
            field(json, "message", event.message()).append(",");
            field(json, "createdAt", event.createdAt().toString());
            json.append("}");
        }
        json.append("]}");
        return json.toString();
    }

    private StringBuilder field(StringBuilder json, String name, String value) {
        return json.append(Json.quote(name)).append(":").append(Json.quote(value));
    }
}
