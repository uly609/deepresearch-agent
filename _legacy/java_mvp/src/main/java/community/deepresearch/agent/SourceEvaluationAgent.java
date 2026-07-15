package community.deepresearch.agent;

import community.deepresearch.model.ResearchSource;
import community.deepresearch.model.SourceScore;
import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SourceEvaluationAgent {
    public Map<String, SourceScore> score(List<ResearchSource> sources) {
        Map<String, SourceScore> scores = new LinkedHashMap<>();
        for (ResearchSource source : sources) {
            double authority = switch (source.type()) {
                case "official" -> 0.95;
                case "paper" -> 0.88;
                case "github" -> 0.82;
                default -> 0.68;
            };
            int age = Math.max(0, LocalDate.now().getYear() - source.publishedAt().getYear());
            double freshness = Math.max(0.35, 1.0 - age * 0.15);
            double evidence = source.snippet().length() > 80 ? 0.8 : 0.55;
            double finalScore = round(authority * 0.5 + freshness * 0.3 + evidence * 0.2);
            scores.put(source.url(), new SourceScore(authority, freshness, evidence, finalScore));
        }
        return scores;
    }

    private double round(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
