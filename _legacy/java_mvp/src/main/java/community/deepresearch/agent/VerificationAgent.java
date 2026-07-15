package community.deepresearch.agent;

import community.deepresearch.model.Citation;
import community.deepresearch.model.ResearchSource;
import community.deepresearch.model.SourceScore;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class VerificationAgent {
    public List<Citation> verify(String question, List<ResearchSource> sources, Map<String, SourceScore> scores) {
        List<Citation> citations = new ArrayList<>();
        for (ResearchSource source : sources) {
            SourceScore score = scores.get(source.url());
            double confidence = score == null ? 0.5 : score.finalScore();
            String claim = inferClaim(question, source);
            boolean supports = confidence >= 0.6 && source.snippet().length() > 20;
            citations.add(new Citation(claim, source.url(), confidence, supports));
        }
        return citations;
    }

    private String inferClaim(String question, ResearchSource source) {
        if ("official".equals(source.type())) {
            return source.title() + " 可支持框架能力和官方路线判断";
        }
        if ("github".equals(source.type())) {
            return source.title() + " 可支持社区活跃度和工程生态判断";
        }
        if ("paper".equals(source.type())) {
            return source.title() + " 可支持 Agent 架构趋势和研究挑战判断";
        }
        return source.title() + " 可作为行业趋势旁证";
    }
}
