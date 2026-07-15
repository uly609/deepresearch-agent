package community.deepresearch.agent;

import community.deepresearch.model.ResearchPlan;
import java.util.ArrayList;
import java.util.List;

public final class ResearchPlanner {
    public ResearchPlan plan(String question) {
        List<String> questions = new ArrayList<>();
        questions.add("识别主题的核心概念、时间范围和判断标准：" + question);
        questions.add("检索官方文档、发布说明和标准协议支持情况：" + question);
        questions.add("检索 GitHub 活跃度、社区生态、issue/PR 迹象：" + question);
        questions.add("检索论文、技术博客和行业分析中的趋势判断：" + question);
        questions.add("比较能力边界、适用场景、风险与落地成本：" + question);
        return new ResearchPlan(question, questions);
    }
}
