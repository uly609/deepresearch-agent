package community.deepresearch.agent;

import community.deepresearch.model.ResearchSource;
import java.util.List;

public interface SourceConnector {
    List<ResearchSource> search(String query);
}
