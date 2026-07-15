package community.deepresearch.model;

import java.time.Instant;

public record ResearchEvent(String taskId, String type, String message, Instant createdAt) {
}
