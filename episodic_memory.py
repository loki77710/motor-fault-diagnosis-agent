import os
import json
import sqlite3
import chromadb
from datetime import datetime
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "episodes.db")
CHROMA_PATH = os.path.join(BASE_DIR, "episodes_chroma")

# ================= 初始化 =================
_ef = embedding_functions.ONNXMiniLM_L6_V2()
_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    _collection = _client.get_collection("episodes")
except Exception:
    _collection = _client.create_collection("episodes", embedding_function=_ef)


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device_id TEXT,
        symptoms TEXT,
        fft_summary TEXT,
        root_cause TEXT,
        confidence REAL,
        evidence TEXT,
        full_report TEXT
    )""")
    conn.commit()
    conn.close()


_init_db()


def estimate_tokens(text):
    """粗略估算 token 数（中文按字符数/2，英文按字符数/4）"""
    if not text:
        return 0
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - chinese
    return int(chinese / 1.5 + other / 4)


# ================= 保存 =================
def save_episode(device_id, symptoms, fft_summary, root_cause, confidence, evidence, full_report):
    """保存诊断情景到 SQLite + ChromaDB"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symptoms_str = json.dumps(symptoms, ensure_ascii=False)

    # 1. 存入 SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO episodes 
        (timestamp, device_id, symptoms, fft_summary, root_cause, confidence, evidence, full_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, device_id, symptoms_str, fft_summary, root_cause,
         float(confidence), evidence, full_report))
    conn.commit()
    ep_id = cur.lastrowid
    conn.close()

    # 2. 存入 ChromaDB（用症状描述作为检索文本）
    doc_text = f"症状：{symptoms_str} 诊断：{root_cause}"
    _collection.add(
        documents=[doc_text],
        metadatas=[{"episode_id": ep_id, "timestamp": ts,
                    "device_id": device_id, "root_cause": root_cause}],
        ids=[f"ep_{ep_id}"]
    )
    return ep_id


# ================= 检索 =================
def retrieve_similar_episodes(query_text, top_k=3):
    """检索相似历史案例（返回完整数据）"""
    try:
        total = _collection.count()
        if total == 0:
            return []
    except Exception:
        return []

    n = min(top_k, total)
    results = _collection.query(query_texts=[query_text], n_results=n)
    if not results['ids'] or not results['ids'][0]:
        return []

    episodes = []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for ep_id_str in results['ids'][0]:
        ep_id = int(ep_id_str.split("_")[1])
        row = conn.execute("SELECT * FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        if row:
            episodes.append(dict(row))
    conn.close()
    return episodes


# ================= 压缩 =================
def compress_episodes(episodes, client):
    """把完整案例压缩成结构化摘要，返回 (压缩文本, 原始tokens, 压缩后tokens)"""
    if not episodes:
        return "", 0, 0

    raw_text = ""
    for ep in episodes:
        raw_text += f"""
【{ep['timestamp']} | {ep['device_id']}】
症状：{ep['symptoms']}
FFT：{ep['fft_summary'][:200]}
结论：{ep['root_cause']}({ep['confidence']})
证据：{ep['evidence'][:150]}
---
"""
    raw_tokens = estimate_tokens(raw_text)

    # 用 LLM 压缩
    prompt = f"""把以下历史诊断案例压缩成结构化摘要，每条一行。
保留：时间、设备、症状关键特征、诊断结论、置信度。
丢弃：详细证据链、维修建议、推理过程。

原始案例：
{raw_text}

输出格式（每条一行，不要其他文字）：
- [时间] [设备]: 症状={{关键特征}} → 诊断={{结论}}({{置信度}})"""

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, timeout=30)
    compressed = r.choices[0].message.content.strip()
    compressed_tokens = estimate_tokens(compressed)

    return compressed, raw_tokens, compressed_tokens


if __name__ == "__main__":
    # 测试
    print(f"数据库路径：{DB_PATH}")
    print(f"当前案例数：{_collection.count()}")