package community.deepresearch.model;

import java.time.Instant;

public record ResearchReport(String taskId, String markdown, Instant generatedAt) {
}
