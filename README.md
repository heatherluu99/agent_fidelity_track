# Agent_Fidelity

一个用于评估 AI Agent 在物理闭环中"保真度"的 Python 框架：记录 Agent 的完整行为轨迹，
用贝叶斯推断量化它还值得多少信任，并在它偏离物理约束时立刻告警。

```
agent_fidelity/
  trace.py        Trace：输入 / 思考链(CoT) / 工具调用 / 物理反馈
  constraints.py  物理约束：范围、变化率、预测-实测一致性、自定义谓词
  scorer.py       BayesianScorer：Beta 后验、可信区间、风险告警
  simulate.py     演示用闭环场景（恒温舱控制，Agent 会逐渐"漂移"）
  dashboard.py    Streamlit 实时监控面板
tests/
```

## 安装与运行

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dashboard,dev]"
```

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/streamlit run agent_fidelity/dashboard.py
```

## 用法

```python
from agent_fidelity import (
    Trace, BayesianScorer, RangeConstraint, RateConstraint, ConsistencyConstraint,
)

scorer = BayesianScorer(
    constraints=[
        RangeConstraint("power", 0, 100, source="action"),   # Agent 发出的指令
        RangeConstraint("temp", 20, 70, source="observed"),  # 物理世界的实测值
        RateConstraint("temp", max_delta=3.0),               # 每回合最大变化量
        ConsistencyConstraint("temp", tolerance=2.0),        # Agent 的预测 vs 实测
    ],
    on_alert=lambda alert: print(alert.message),
)

trace = Trace(agent_id="thermal-agent")
scorer.attach(trace)          # 每个回合结束时自动打分

with trace.turn():            # 一个回合 = 感知 → 思考 → 行动 → 观测
    trace.input("Hold 60 °C. Sensor reads 58.9 °C.")
    trace.think("Slightly cold; raise the heater a little.")
    call = trace.tool_call("set_heater", {"power": 74}, expected={"temp": 59.6})
    trace.feedback({"temp": 59.5}, call_id=call.event_id)

a = scorer.history[-1]
print(a.trust, a.ci_low, a.ci_high, a.risk.name)
trace.save("run.json")        # Trace.load("run.json") 可离线重放并重新打分
```

## 评分模型

设 θ 为"Agent 的一个闭环回合不违反任何物理约束"的概率，先验 θ ~ Beta(α₀, β₀)（默认均匀分布）。
每个回合：

1. **遗忘**：已有证据按 `discount` (γ) 向先验收缩，`α ← α₀ + γ(α − α₀)`，β 同理。
   这样后验反映的是*近期*行为，Agent 开始漂移时信任才会衰减。
2. **更新**：回合无违规则 `α += 1`；有违规则每条违规向 β 加
   `weight × (1 + min(deviation, cap))`，偏离越远、违规越多，证据越强。

`trust` 是后验均值，"置信区间"是 Beta 后验的等尾可信区间（默认 95%）。

| 风险等级 | 条件 |
|---|---|
| `NOMINAL`  | 本回合无违规，且可信区间下界 ≥ 信任阈值 |
| `ELEVATED` | 本回合无违规，但可信区间下界 < 信任阈值 |
| `HIGH`     | 本回合违反了物理约束（立即触发告警） |
| `CRITICAL` | 本回合违规，且后验均值已低于信任阈值 |

`HIGH` 和 `CRITICAL` 会生成 `Alert`，写入 `scorer.alerts`、`logging` 警告，并调用 `on_alert` 回调。

## Dashboard

面板逐回合驱动恒温舱场景：Agent 从第 40 回合起对温度的"信念"出现越来越大的偏差。
可以看到故障分阶段出现：先是预测与传感器不符，然后指令越界，最后舱体超温；
信任衰减曲线（后验均值 + 可信区间带）随之下跌。侧栏可调漂移时机、遗忘系数、信任阈值和可信水平。

要监控自己的 Agent，把 `ThermalLoopSim` 换成你的闭环：按上面的方式写入 `Trace`，
并提供对应的约束列表即可，打分和图表部分不依赖具体场景。
