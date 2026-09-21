import os
import json
import subprocess
from collections import defaultdict


NAME_MAPPING = {
    "定子绕组故障": "定子绕组故障",
    "定子故障": "定子绕组故障",
    "定子绕组短路": "定子绕组故障",
    "定子匝间短路": "定子绕组故障",
    "定子绕组匝间短路": "定子绕组故障",
    "绝缘老化": "定子绕组绝缘老化",
    "绕组老化": "定子绕组绝缘老化",
    "转子动不平衡": "转子不平衡",
    "动不平衡": "转子不平衡",
    "不对中": "对中不良",
    "联轴器不对中": "对中不良",
    "松动": "机械松动",
    "机械松动故障": "机械松动",
    "气隙不均": "气隙偏心",
    "偏心": "气隙偏心",
    "转子导条断裂": "转子断条",
    "导条断裂": "转子断条",
    "健康": "正常",
    "无故障": "正常",
    "设备正常": "正常",
}


def normalize_name(name):
    if not name:
        return name
    name = name.strip()
    if "（" in name:
        name = name.split("（")[0].strip()
    if "(" in name:
        name = name.split("(")[0].strip()
    return NAME_MAPPING.get(name, name)


def run_eval(reflexion_enabled, episodic_enabled, compress_enabled, output_file):
    env = os.environ.copy()
    env["MODE"] = "eval"
    env["REFLEXION_ENABLED"] = "true" if reflexion_enabled else "false"
    env["EPISODIC_ENABLED"] = "true" if episodic_enabled else "false"
    env["COMPRESS_ENABLED"] = "true" if compress_enabled else "false"
    env["OUTPUT_FILE"] = output_file
    env["QUIET_MODE"] = "false"
    env["EVAL_MODE"] = "true"

    print(f"\n{'='*60}")
    print(f"REFLEXION={reflexion_enabled} | EPISODIC={episodic_enabled} | COMPRESS={compress_enabled}")
    print('='*60)

    subprocess.run(["python", "agent_v2.py"], env=env, check=True)

    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results):
    n = len(results)
    for r in results:
        r["expected_norm"] = normalize_name(r["expected"])
        r["top1_norm"] = normalize_name(r["top1"])
        r["top3_norm"] = [normalize_name(h) for h in r["top3"]]
        r["top1_correct"] = r["top1_norm"] == r["expected_norm"]
        r["top3_correct"] = r["expected_norm"] in r["top3_norm"]

    top1_acc = sum(r["top1_correct"] for r in results) / n
    top3_acc = sum(r["top3_correct"] for r in results) / n

    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        per_class[r["expected_norm"]]["total"] += 1
        if r["top1_correct"]:
            per_class[r["expected_norm"]]["correct"] += 1
    per_class_recall = {cls: d["correct"] / d["total"] for cls, d in per_class.items()}
    macro_recall = sum(per_class_recall.values()) / len(per_class_recall)

    pred_class = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        pred_class[r["top1_norm"]]["total"] += 1
        if r["top1_correct"]:
            pred_class[r["top1_norm"]]["correct"] += 1
    per_class_precision = {cls: d["correct"] / d["total"] for cls, d in pred_class.items()}
    macro_precision = sum(per_class_precision.values()) / len(per_class_precision)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-6)

    rag_recall_at3 = sum(1 for r in results if r.get("retrieved_correct_in_top3")) / n

    avg_ask = sum(r.get("ask_count", 0) for r in results) / n
    reflex_triggered = sum(1 for r in results if r.get("reflexion_actually_triggered", False))
    reflex_trigger_rate = reflex_triggered / n
    avg_reflex = sum(r.get("reflexion_count", 0) for r in results) / n

    total_raw = sum(r.get("raw_episode_tokens", 0) for r in results)
    total_comp = sum(r.get("compressed_episode_tokens", 0) for r in results)
    compression_rate = 1 - total_comp / max(total_raw, 1)

    return {
        "Top-1 准确率": top1_acc,
        "Top-3 准确率": top3_acc,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "RAG Recall@3": rag_recall_at3,
        "平均追问轮数": avg_ask,
        "Reflexion 触发率": reflex_trigger_rate,
        "平均反思轮数": avg_reflex,
        "情景记忆原始Tokens": total_raw,
        "情景记忆压缩后Tokens": total_comp,
        "上下文压缩率": compression_rate,
        "per_class_recall": per_class_recall,
    }


def print_metrics(name, m):
    print(f"\n{'='*60}")
    print(f"【{name}】")
    print('='*60)
    for k in ["Top-1 准确率", "Top-3 准确率", "Macro Precision",
              "Macro Recall", "Macro F1", "RAG Recall@3"]:
        print(f"{k:<24}{m[k]:.2%}")
    print('-'*60)
    print(f"{'Reflexion 触发率':<24}{m['Reflexion 触发率']:.2%}")
    print(f"{'平均反思轮数':<24}{m['平均反思轮数']:.2f}")
    print('-'*60)
    print(f"{'情景记忆原始Tokens':<24}{m['情景记忆原始Tokens']}")
    print(f"{'情景记忆压缩后Tokens':<24}{m['情景记忆压缩后Tokens']}")
    print(f"{'上下文压缩率':<24}{m['上下文压缩率']:.2%}")
    print('-'*60)
    print("每类 Recall：")
    for cls, rec in sorted(m["per_class_recall"].items()):
        print(f"   {cls:<20}{rec:.2%}")


def compare(base_m, imp_m):
    print(f"\n{'='*60}")
    print("消融实验对比")
    print('='*60)
    print(f"{'指标':<24}{'基线':<16}{'改进':<16}{'变化'}")
    print('-'*60)
    keys = ["Top-1 准确率", "Top-3 准确率", "Macro F1", "RAG Recall@3",
            "Reflexion 触发率", "上下文压缩率"]
    for k in keys:
        b, i = base_m[k], imp_m[k]
        print(f"{k:<24}{b:<16.4f}{i:<16.4f}{i-b:+.4f}")


if __name__ == "__main__":
    # 基线：全部关掉
    baseline = run_eval(False, False, False, "eval_baseline.json")

    # 改进：全开
    improved = run_eval(True, True, True, "eval_improved.json")

    base_m = compute_metrics(baseline)
    imp_m = compute_metrics(improved)

    print_metrics("基线（无特色）", base_m)
    print_metrics("改进（全特色）", imp_m)
    compare(base_m, imp_m)

    with open("ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {k: v for k, v in base_m.items() if not isinstance(v, dict)},
            "improved": {k: v for k, v in imp_m.items() if not isinstance(v, dict)},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n摘要已保存到 ablation_summary.json")