package community.deepresearch.model;

public record Citation(String claim, String sourceUrl, double confidence, boolean supports) {
}
