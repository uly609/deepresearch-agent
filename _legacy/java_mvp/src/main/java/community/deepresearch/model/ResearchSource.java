package community.deepresearch.model;

import java.time.LocalDate;

public record ResearchSource(String title, String type, String url, String snippet, LocalDate publishedAt) {
}
