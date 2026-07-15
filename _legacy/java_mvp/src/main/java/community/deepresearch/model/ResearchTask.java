package community.deepresearch.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class ResearchTask {
    private final String id;
    private final String question;
    private final Instant createdAt;
    private Instant updatedAt;
    private String status;
    private ResearchPlan plan;
    private List<ResearchSource> sources;
    private Map<String, SourceScore> scores;
    private List<Citation> citations;
    private ResearchReport report;
    private final List<ResearchEvent> events;

    public ResearchTask(String question) {
        this.id = "research_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        this.question = question;
        this.createdAt = Instant.now();
        this.updatedAt = createdAt;
        this.status = "queued";
        this.sources = List.of();
        this.scores = Map.of();
        this.citations = List.of();
        this.events = new ArrayList<>();
    }

    public String id() {
        return id;
    }

    public String question() {
        return question;
    }

    public Instant createdAt() {
        return createdAt;
    }

    public Instant updatedAt() {
        return updatedAt;
    }

    public String status() {
        return status;
    }

    public ResearchPlan plan() {
        return plan;
    }

    public List<ResearchSource> sources() {
        return sources;
    }

    public Map<String, SourceScore> scores() {
        return scores;
    }

    public List<Citation> citations() {
        return citations;
    }

    public ResearchReport report() {
        return report;
    }

    public List<ResearchEvent> events() {
        return List.copyOf(events);
    }

    public void markRunning() {
        status = "running";
        updatedAt = Instant.now();
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    public void setPlan(ResearchPlan plan) {
        this.plan = plan;
    }

    public void setSources(List<ResearchSource> sources) {
        this.sources = List.copyOf(sources);
    }

    public void setScores(Map<String, SourceScore> scores) {
        this.scores = new LinkedHashMap<>(scores);
    }

    public void setCitations(List<Citation> citations) {
        this.citations = List.copyOf(citations);
    }

    public void addEvent(ResearchEvent event) {
        events.add(event);
    }

    public void finish(ResearchReport report) {
        this.report = report;
        this.status = "completed";
        this.updatedAt = Instant.now();
    }
}
