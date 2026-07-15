# Eval 评估框架设计

这篇讲 `deepresearch/eval_harness.py`。

## 为什么需要 Eval

Agent 输出不是固定答案。

同一个问题，不同模型、不同搜索结果、不同 prompt，输出都可能不同。

所以不能只用传统单元测试判断质量。

Agent 更需要看：

- 有没有完成任务
- 找到多少来源
- 生成多少 claim
- 弱引用有多少
- 改代码以后有没有退步

## 当前 EvalHarness 做什么

当前 `EvalHarness` 会跑一组固定问题。

每个问题跑完后统计：

```text
completed
source_count
claim_count
weak_claim_count
score
```

这不是最终评估标准，只是第一版。

## 为什么普通运行和 Eval 要分开

普通运行是给用户看：

```text
针对一个问题生成报告
```

Eval 是给开发者看：

```text
Agent 整体能力有没有稳定
```

一句话：

> 普通运行看结果，Eval 看能力。

## 面试话术

可以这样讲：

> 我加了 Eval Harness，是因为 Agent 的输出不是固定答案，不能只靠一次 demo 判断质量。当前评估会用固定题集统计任务完成、来源数量、claim 数量和弱引用数量。后续会继续扩展引用准确率、来源多样性、工具调用成功率和失败原因统计。

