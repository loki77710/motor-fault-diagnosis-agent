# 电机故障诊断 Agent

基于 LangGraph 的多轮对话诊断 Agent，支持振动+电流双模态分析、情景记忆、Reflexion 自我批判。

## 核心指标
- Top-1 准确率：95%
- Macro F1：95.36%
- 上下文压缩率：75.84%
- 覆盖 8 类电机故障

## 架构
信号输入 → 多模态特征提取 → RAG检索 → 情景记忆 → 假设生成 → Reflexion审查 → 结论输出

## 快速开始
1. pip install -r requirements.txt
2. 创建 `.env` 文件，写入 `DEEPSEEK_API_KEY=sk-xxxx`
3. python agent_v2.py

## 文件说明
- `agent_v2.py`：主流程（LangGraph 状态机）
- `signal_generator.py`：振动/电流信号生成
- `fft_tool.py`：FFT + 电流频谱分析
- `episodic_memory.py`：历史案例存储与压缩
- `evaluate.py`：消融实验脚本
- `motor_manual.txt`：诊断知识库（31 条）
- `test_cases.json`：20 条测试用例

## 实验结果
| 指标 | 基线 | 改进 |
|------|:----:|:----:|
| Top-1 准确率 | 90% | 95% |
| Macro F1 | 91.25% | 95.36% |
| 上下文压缩率 | — | 75.84% |
