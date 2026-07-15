package community.deepresearch.agent;

import community.deepresearch.model.ResearchSource;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SearchAgent {
    private final List<SourceConnector> connectors;

    public SearchAgent(List<SourceConnector> connectors) {
        this.connectors = List.copyOf(connectors);
    }

    public List<ResearchSource> search(String query) {
        Map<String, ResearchSource> deduped = new LinkedHashMap<>();
        for (SourceConnector connector : connectors) {
            for (ResearchSource source : connector.search(query)) {
                deduped.putIfAbsent(source.url(), source);
            }
        }
        return new ArrayList<>(deduped.values());
    }
}
