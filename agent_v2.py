import os
import re
import json
import contextlib
import io
import time
import chromadb
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from chromadb.utils import embedding_functions

from signal_generator import get_signals
from fft_tool import (
    fft_analysis, features_to_text,
    current_spectrum_analysis, current_features_to_text
)
import episodic_memory as em

load_dotenv()
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

REQUIRED_INFO = ["振动频率", "振动方向", "温度趋势"]
MAX_ASK = 2
MAX_REFLEXION = 2
MAX_INVALID = 2

REFLEXION_ENABLED = os.environ.get("REFLEXION_ENABLED", "true").lower() == "true"
EPISODIC_ENABLED = os.environ.get("EPISODIC_ENABLED", "true").lower() == "true"
COMPRESS_ENABLED = os.environ.get("COMPRESS_ENABLED", "true").lower() == "true"
QUIET_MODE = os.environ.get("QUIET_MODE", "false").lower() == "true"
EVAL_MODE = os.environ.get("EVAL_MODE", "false").lower() == "true"

# ================= RAG 知识库 =================
if not QUIET_MODE:
    print("✅ 正在加载电机故障知识库...")
ef = embedding_functions.ONNXMiniLM_L6_V2()
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("motor_manual")
except Exception:
    pass
collection = chroma_client.create_collection("motor_manual", embedding_function=ef)

with open("motor_manual.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 用正则切分所有条款类型
chunks = re.split(
    r'(?=\[(?:电机故障|正常状态|鉴别诊断|特征解读|典型案例|诊断决策|诊断流程|故障树|诊断精度|误诊预防|数据质量|复合故障|维护决策|记录与分析|诊断知识)条款)',
    text
)
chunks = [c.strip() for c in chunks if c.strip()]

for i, c in enumerate(chunks):
    collection.add(documents=[c], ids=[f"id_{i}"])
if not QUIET_MODE:
    print(f"✅ 知识库就绪，共 {len(chunks)} 条条款")
    print(f"✅ 情景记忆已加载（当前 {em._collection.count()} 条历史案例）")


class AgentState(TypedDict):
    user_input: str
    collected_info: dict
    missing_info: list
    ask_count: int
    invalid_reply_count: int
    agent_question: str
    device_id: str
    signal_fault: str
    manual_context: str
    episodic_context: str
    raw_episode_tokens: int
    compressed_episode_tokens: int
    fft_text: str
    fft_summary: str
    hypotheses: list
    conclusion: str
    reflexion_count: int
    reflexion_critique: str
    reflexion_passed: bool
    reflexion_actually_triggered: bool


def extract_info(state: AgentState):
    text = state["user_input"]
    if not QUIET_MODE:
        print(f"\n[提取信息] {text}")

    invalid_keywords = ["不知道", "没注意", "不清楚", "继续", "不确定",
                        "没测", "无法", "没感觉", "没什么", "没关注"]
    if any(kw in text for kw in invalid_keywords):
        if not QUIET_MODE:
            print(f"   ⚠️ 无效回答，跳过提取")
        return {"invalid_reply_count": state.get("invalid_reply_count", 0) + 1}

    prompt = f"""从以下描述中提取症状特征。未提到的填"未知"。
描述：{text}
输出JSON，字段：振动频率、振动方向、温度趋势、声音特征、电流特征、冲击特征。
只输出JSON。"""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, timeout=30)
    content = r.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()
    try:
        new_info = json.loads(content)
    except json.JSONDecodeError:
        new_info = {}

    collected = dict(state.get("collected_info", {}))
    before_count = len(collected)
    for k, v in new_info.items():
        if v and v != "未知":
            collected[k] = v
    after_count = len(collected)

    invalid_count = state.get("invalid_reply_count", 0)
    if after_count == before_count:
        invalid_count += 1
        if not QUIET_MODE:
            print(f"   ⚠️ 未提取到新信息（累计无效 {invalid_count} 次）")
    else:
        invalid_count = 0

    if not QUIET_MODE:
        print(f"   已收集：{collected}")
    return {"collected_info": collected, "invalid_reply_count": invalid_count}


def check_sufficiency(state: AgentState):
    if EVAL_MODE:
        return {"missing_info": []}

    if state.get("invalid_reply_count", 0) >= MAX_INVALID:
        if not QUIET_MODE:
            print(f"\n[判断] 用户连续{MAX_INVALID}次无法提供有效信息，跳过追问")
        return {"missing_info": []}

    collected = state.get("collected_info", {})
    missing = [k for k in REQUIRED_INFO if collected.get(k) in [None, "未知", ""]]
    ask_count = state.get("ask_count", 0)

    if not missing or ask_count >= MAX_ASK:
        if not QUIET_MODE:
            print(f"\n[判断] 信息充分（追问次数：{ask_count}）")
        return {"missing_info": []}
    if not QUIET_MODE:
        print(f"\n[判断] 缺失：{missing}")
    return {"missing_info": missing, "ask_count": ask_count + 1}


def ask_question(state: AgentState):
    prompt = f"""已收集：{state['collected_info']}
缺失：{state['missing_info']}
生成一句自然的追问，只问一个关键问题，20字以内。"""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, timeout=30)
    q = r.choices[0].message.content.strip()
    if not QUIET_MODE:
        print(f"\n[追问] {q}")
    return {"agent_question": q}


def fetch_and_analyze(state: AgentState):
    if not QUIET_MODE:
        print(f"\n[诊断] 调取{state['device_id']}信号...")
    t, vib, cur = get_signals(state.get("signal_fault", "normal"))
    vf = fft_analysis(vib)
    cf = current_spectrum_analysis(cur)
    combined = features_to_text(vf) + "\n" + current_features_to_text(cf)

    fft_summary = (f"1x={vf['1x']:.3f}, 2x={vf['2x']:.3f}, "
                   f"峭度={vf['峭度']:.2f}, 边带比={cf['边带比']:.3f}, "
                   f"谐波={cf['谐波能量']:.3f}, 100Hz比={cf['100Hz比']:.3f}")
    if not QUIET_MODE:
        print(f"   振动: 1x={vf['1x']:.3f} 2x={vf['2x']:.3f} 峭度={vf['峭度']:.2f}")
        print(f"   电流: 边带比={cf['边带比']:.4f} 谐波={cf['谐波能量']:.4f} 100Hz比={cf['100Hz比']:.4f}")
    return {"fft_text": combined, "fft_summary": fft_summary}


def retrieve_manual(state: AgentState):
    query = f"{state['fft_text']} 用户描述：{state['collected_info']}"
    results = collection.query(query_texts=[query], n_results=4)
    context = "\n\n".join(results['documents'][0])
    if not QUIET_MODE:
        print(f"\n[RAG] 检索到 {len(results['documents'][0])} 条手册条款")
    return {"manual_context": context}


def retrieve_episodes(state: AgentState):
    if not EPISODIC_ENABLED:
        return {"episodic_context": "", "raw_episode_tokens": 0, "compressed_episode_tokens": 0}

    query = f"症状：{state['collected_info']}"
    episodes = em.retrieve_similar_episodes(query, top_k=3)

    if not episodes:
        if not QUIET_MODE:
            print(f"\n[情景记忆] 无历史案例")
        return {"episodic_context": "", "raw_episode_tokens": 0, "compressed_episode_tokens": 0}

    if COMPRESS_ENABLED:
        compressed, raw_tok, comp_tok = em.compress_episodes(episodes, client)
        if not QUIET_MODE:
            print(f"\n[情景记忆] 检索到 {len(episodes)} 条历史案例")
            print(f"   原始 Token: {raw_tok}  →  压缩后 Token: {comp_tok}  (压缩率 {1-comp_tok/max(raw_tok,1):.0%})")
        return {
            "episodic_context": compressed,
            "raw_episode_tokens": raw_tok,
            "compressed_episode_tokens": comp_tok
        }
    else:
        raw = "\n---\n".join([f"{e['timestamp']} {e['device_id']}: {e['root_cause']}"
                              for e in episodes])
        tok = em.estimate_tokens(raw)
        if not QUIET_MODE:
            print(f"\n[情景记忆] 检索到 {len(episodes)} 条（未压缩，{tok} tokens）")
        return {
            "episodic_context": raw,
            "raw_episode_tokens": tok,
            "compressed_episode_tokens": tok
        }


def generate_hypotheses(state: AgentState):
    if not QUIET_MODE:
        print(f"\n[假设] 生成故障假设...")
    critique_hint = ""
    if state.get("reflexion_critique"):
        critique_hint = f"\n【上一轮被批判的问题】：{state['reflexion_critique']}\n请针对性修正。"

    episodic_hint = ""
    if state.get("episodic_context"):
        episodic_hint = f"\n【历史相似案例】：\n{state['episodic_context']}\n（仅供参考，不要直接照搬结论）"

    prompt = f"""你是电机故障诊断专家，可以同时利用振动和电流两种信号。

用户描述：{state['user_input']}
已收集症状：{state['collected_info']}

{state['fft_text']}

【维修手册相关条款】：
{state['manual_context']}
{episodic_hint}
{critique_hint}

诊断原则：
- 如果所有特征都在正常范围内（振动1x/2x/3x均<0.05，电流边带比<0.02，谐波<0.05，100Hz比<0.02，峭度≈3），必须输出"正常"
- 振动异常（1x/2x/峭度）→ 指向不平衡/对中/松动/偏心
- 电流2sf边带明显（边带比>0.1）→ 指向转子断条
- 电流谐波（3次/5次，谐波>0.5）→ 指向定子绕组故障
- 电流100Hz明显（100Hz比>0.1）→ 指向电压不平衡

候选故障（必须从这个列表选择）：
正常、转子不平衡、转子断条、气隙偏心、定子绕组故障、定子绕组绝缘老化、对中不良、机械松动、电压不平衡

严格输出JSON数组：
[{{"hypothesis":"名称","confidence":0.7,"reason":"依据"}}, ...]
不要其他文字。"""

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, timeout=30)
    content = r.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()
    try:
        hyps = json.loads(content)
    except json.JSONDecodeError:
        hyps = [{"hypothesis": "解析失败", "confidence": 0, "reason": content[:200]}]
    if not QUIET_MODE:
        print(f"   Top-1: {hyps[0]['hypothesis']} ({hyps[0]['confidence']})")
    return {"hypotheses": hyps, "reflexion_critique": ""}


def output_conclusion(state: AgentState):
    top = state["hypotheses"][0]
    prompt = f"""你是电机故障诊断专家。

用户描述：{state['user_input']}
症状：{state['collected_info']}
{state['fft_text']}
【手册条款】：
{state['manual_context']}

Top-1假设：{top['hypothesis']}（置信度{top['confidence']}）
依据：{top['reason']}

输出诊断结论：根因结论、2-3条证据链、1-2条维修建议，150字内。"""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, timeout=30)
    return {"conclusion": r.choices[0].message.content.strip()}


def reflexion_node(state: AgentState):
    if not REFLEXION_ENABLED:
        return {"reflexion_passed": True}
    count = state.get("reflexion_count", 0)
    if count >= MAX_REFLEXION:
        return {"reflexion_passed": True}
    if not QUIET_MODE:
        print(f"\n[Reflexion] 第{count+1}轮自我批判...")

    top = state["hypotheses"][0]
    prompt = f"""你是红队审查员，请给当前诊断打分。

当前诊断：{top['hypothesis']}（置信度{top['confidence']}）
判断依据：{top['reason']}
症状：{state['collected_info']}
{state['fft_text']}
【手册条款】：
{state['manual_context']}

打分标准（总分10分）：
- 结论与振动/电流特征一致（0-4分）
- 结论与手册条款匹配（0-3分）
- 证据链完整、无明显遗漏（0-3分）

评分规则：
- 8-10分：优秀，可直接输出
- 6-7分：基本合理，有小瑕疵但结论方向正确
- 0-5分：存在明显矛盾或遗漏了更匹配的故障，需回炉

输出JSON：
{{"score": 8, "critique": "低于6分时指出具体问题；否则填空"}}
只输出JSON。"""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, timeout=30)
    content = r.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"score": 8, "critique": ""}

    score = result.get("score", 8)
    passed = score >= 6

    if passed:
        if not QUIET_MODE:
            print(f"   评分 {score}/10 ✅ 通过审查")
        return {"reflexion_passed": True, "reflexion_count": count + 1}
    else:
        if not QUIET_MODE:
            print(f"   评分 {score}/10 ⚠️ 回炉：{result.get('critique', '')[:80]}...")
        return {
            "reflexion_passed": False,
            "reflexion_critique": result.get("critique", ""),
            "reflexion_count": count + 1,
            "reflexion_actually_triggered": True,
        }


def save_episode_node(state: AgentState):
    if not EPISODIC_ENABLED or QUIET_MODE:
        return {}
    hyps = state.get("hypotheses") or []
    if not hyps:
        return {}
    top = hyps[0]
    evidence = top.get("reason", "")
    try:
        ep_id = em.save_episode(
            device_id=state.get("device_id", "unknown"),
            symptoms=state.get("collected_info", {}),
            fft_summary=state.get("fft_summary", ""),
            root_cause=top.get("hypothesis", ""),
            confidence=top.get("confidence", 0),
            evidence=evidence,
            full_report=state.get("conclusion", "")
        )
        if not QUIET_MODE:
            print(f"\n[情景记忆] 已保存案例 #{ep_id}")
    except Exception as e:
        if not QUIET_MODE:
            print(f"\n[情景记忆] 保存失败：{e}")
    return {}


def finalize(state: AgentState):
    if not QUIET_MODE:
        print(f"\n{'='*50}")
        print(state["conclusion"])
        print('='*50)
    return state


# ================= 构建图 =================
builder = StateGraph(AgentState)
builder.add_node("extract", extract_info)
builder.add_node("check", check_sufficiency)
builder.add_node("ask", ask_question)
builder.add_node("analyze", fetch_and_analyze)
builder.add_node("retrieve_manual", retrieve_manual)
builder.add_node("retrieve_episodes", retrieve_episodes)
builder.add_node("hypothesize", generate_hypotheses)
builder.add_node("conclude", output_conclusion)
builder.add_node("reflexion", reflexion_node)
builder.add_node("finalize", finalize)
builder.add_node("save", save_episode_node)

builder.set_entry_point("extract")
builder.add_edge("extract", "check")


def route_after_check(state):
    if state.get("missing_info"):
        return "ask"
    return "analyze"


builder.add_conditional_edges("check", route_after_check,
    {"ask": "ask", "analyze": "analyze"})
builder.add_edge("ask", END)
builder.add_edge("analyze", "retrieve_manual")
builder.add_edge("retrieve_manual", "retrieve_episodes")
builder.add_edge("retrieve_episodes", "hypothesize")
builder.add_edge("hypothesize", "conclude")
builder.add_edge("conclude", "reflexion")


def route_after_reflexion(state):
    if state.get("reflexion_passed"):
        return "finalize"
    return "hypothesize"


builder.add_conditional_edges("reflexion", route_after_reflexion,
    {"finalize": "finalize", "hypothesize": "hypothesize"})
builder.add_edge("finalize", "save")
builder.add_edge("save", END)

graph = builder.compile()


def run_conversation(user_input, device_id, signal_fault,
                     simulated_replies=None, quiet=False):
    state = {
        "user_input": user_input, "device_id": device_id,
        "signal_fault": signal_fault, "collected_info": {},
        "ask_count": 0, "invalid_reply_count": 0,
        "agent_question": "", "missing_info": [],
        "reflexion_count": 0, "reflexion_critique": "",
        "reflexion_passed": False,
        "reflexion_actually_triggered": False,
        "episodic_context": "",
        "raw_episode_tokens": 0,
        "compressed_episode_tokens": 0,
        "fft_summary": "",
    }
    replies = list(simulated_replies or [])
    reply_idx = 0
    if not quiet:
        print(f"\n{'#'*60}\n用户：{user_input}\n{'#'*60}")

    for _ in range(MAX_ASK + 3):
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = graph.invoke(state)
        else:
            result = graph.invoke(state)
        for k in result:
            state[k] = result[k]
        if result.get("agent_question"):
            if replies and reply_idx < len(replies):
                reply = replies[reply_idx]
                reply_idx += 1
                if not quiet:
                    print(f"你: {reply}")
            else:
                reply = input("你: ").strip() or "没注意"
            state["user_input"] = reply
            state["agent_question"] = ""
            state["missing_info"] = []
        else:
            return state
    return state


def run_evaluation():
    with open("test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        start = time.time()
        state = run_conversation(
            case["user_input"], case["device_id"], case["signal_fault"],
            simulated_replies=None, quiet=True
        )
        elapsed = time.time() - start
        hyps = state.get("hypotheses") or [{"hypothesis": "", "confidence": 0}]
        top1 = hyps[0].get("hypothesis", "")
        top3 = [h.get("hypothesis", "") for h in hyps]

        manual_ctx = state.get("manual_context", "")
        retrieved_correct = case["expected_top1"] in manual_ctx

        results.append({
            "id": case["id"],
            "expected": case["expected_top1"],
            "top1": top1,
            "top3": top3,
            "top1_correct": top1 == case["expected_top1"],
            "top3_correct": case["expected_top1"] in top3,
            "retrieved_correct_in_top3": retrieved_correct,
            "ask_count": state.get("ask_count", 0),
            "reflexion_count": state.get("reflexion_count", 0),
            "reflexion_actually_triggered": state.get("reflexion_actually_triggered", False),
            "raw_episode_tokens": state.get("raw_episode_tokens", 0),
            "compressed_episode_tokens": state.get("compressed_episode_tokens", 0),
            "degraded": False,
            "elapsed": round(elapsed, 2),
        })

    output_file = os.environ.get("OUTPUT_FILE", "eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    top1_acc = sum(r["top1_correct"] for r in results) / n
    top3_acc = sum(r["top3_correct"] for r in results) / n
    avg_reflex = sum(r["reflexion_count"] for r in results) / n
    total_raw = sum(r["raw_episode_tokens"] for r in results)
    total_comp = sum(r["compressed_episode_tokens"] for r in results)

    print(f"\n{'='*50}")
    print(f"REFLEXION={REFLEXION_ENABLED} EPISODIC={EPISODIC_ENABLED} COMPRESS={COMPRESS_ENABLED}")
    print(f"测试用例数：{n}")
    print(f"Top-1 准确率：{top1_acc:.2%}")
    print(f"Top-3 准确率：{top3_acc:.2%}")
    print(f"平均反思轮数：{avg_reflex:.2f}")
    print(f"情景记忆 Token：原始 {total_raw} → 压缩后 {total_comp} "
          f"(压缩率 {1-total_comp/max(total_raw,1):.1%})")
    print(f"结果已写入：{output_file}")
    print('='*50)


if __name__ == "__main__":
    mode = os.environ.get("MODE", "test")
    if mode == "test":
        print("\n========== 第一次诊断（无历史） ==========")
        run_conversation(
            "3号电机振动大，跟转速同步，径向为主，温度高5度",
            "Motor_03", "unbalance"
        )

        print("\n========== 第二次诊断（有历史） ==========")
        run_conversation(
            "5号电机振动大，轴向振动也大，2倍频突出",
            "Motor_05", "misalignment",
            simulated_replies=["温度正常"]
        )

        print("\n========== 第三次诊断（相似症状 + 用户不给信息） ==========")
        run_conversation(
            "7号电机振动大，跟转速同步",
            "Motor_07", "unbalance",
            simulated_replies=["不知道", "不清楚"]
        )

    elif mode == "eval":
        run_evaluation()