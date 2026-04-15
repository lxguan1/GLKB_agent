"""
LoCoMo Evaluation for ModifiedMemory

Loads one LoCoMo conversation, adds each session as a single unchunked trajectory,
runs sleep_update, then evaluates retrieval metrics (QwG%, Coverage, MRR) per layer.
Adversarial questions are excluded.

Usage:
    python locomo_eval.py
"""

import os
os.environ.setdefault("PROMPT_MODE", "conversational")

import asyncio
import json
import re
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import time

from tqdm import tqdm
from modified_memory_system import (
    ModifiedMemory, embed, async_llm, async_embed,
    Concept4, Reflection4, TrajectorySummary4, Trajectory4,
    Persona, PersonaEntry, TaskRubric, ConnectionManager4, ConnStats,
    get_token_usage,
)
from config import async_client
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


# =============================================================================
# DATASET CONFIG
# =============================================================================
DATASET_PATH = "locomo10.json"
DATASET_URL  = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_MAP = {1: "multi_hop", 2: "temporal", 3: "open_domain",
                4: "single_hop", 5: "adversarial"}

# =============================================================================
# HYPERPARAMETERS  ← change experiment settings here
# =============================================================================
TOP_K             = 20    # reflections retrieved per question
FILTER_THRESHOLD  = 0.50  # cosine similarity threshold for embedding filter
MIN_PER_GROUP     = 0     # minimum items kept per reflection group after filter (0 = pure threshold)
MAX_PER_GROUP     = 5     # maximum items kept per reflection group after filter
MAX_TOTAL_ITEMS   = 60    # maximum total items passed to the answer LLM (0 = no limit)
DIAGNOSE_N        = 3     # questions shown per category in the diagnosis printout

# Directory where all result JSON files are written
RESULTS_DIR = "locomo_eval_results"

# Model pricing (USD per 1M tokens)
LLM_COST_INPUT_PER_M  = 0.150  # gpt-4o-mini input
LLM_COST_OUTPUT_PER_M = 0.600  # gpt-4o-mini output
EMBED_COST_PER_M      = 0.020  # text-embedding-3-small

# Directory where memory snapshots are stored
SNAPSHOTS_DIR = "locomo_mem_snapshots"

# Memory snapshot: set to a file path string to pin a specific snapshot,
# or leave as None to auto-derive the path from conversation_index
# (e.g. mem_snapshot_conv0.json).  If the file exists the ingestion /
# sleep_update step is skipped; if it does not exist it is created after
# sleep_update so subsequent runs jump straight to QA evaluation.
MEMORY_SNAPSHOT_PATH: Optional[str] = None
# When True, filter_items_by_embedding scores each reflection item using the
# query components (predicted_reflections) the same way _match_layer_reflections
# does — max over query texts, 60% semantic + 40% BM25 — instead of scoring
# against the raw input question.  Set to False to revert to original behaviour.
USE_RETRIEVAL_SCORES_FOR_FILTER = True
REFINE_REFLECTIONS_DURING_SLEEP_UPDATE = True  # whether to run the reflection-refinement step during sleep_update
SLEEP_UPDATE_QUESTIONS_PER_CHUNK=3  # how many questions to include in each chunk during sleep_update_async (higher = more fine-grained but more LLM calls)
# When True, each QA pair is also scored by an LLM-as-a-judge that reads the
# gold answer and the generated answer and returns CORRECT or WRONG.
LLM_AS_JUDGE = True
EVALUATE_EXTRACTION = False  # whether to check if gold reflections contain the answer (sufficiency)

# =============================================================================
# MEMORY SNAPSHOT  (save / load the built memory to skip re-ingestion)
# =============================================================================

def save_mem_snapshot(mem: "ModifiedMemory", dia_to_traj: Dict[str, str], path: str) -> None:
    """Serialise ModifiedMemory + dia_to_traj to a JSON file.

    Numpy embeddings are stored as plain float lists so the file is
    human-readable and portable.
    """

    def _arr(a):
        return a.tolist() if isinstance(a, np.ndarray) else (a or [])

    def _concept(c):
        return {"id": c.id, "text": c.text, "embedding": _arr(c.embedding),
                "reflection_ids": c.reflection_ids}

    def _reflection(r):
        return {"id": r.id, "reflection_list": r.reflection_list,
                "embedding": _arr(r.embedding),
                "concept_ids": r.concept_ids,
                "trajectory_summary_ids": r.trajectory_summary_ids,
                "item_embeddings": [e.tolist() for e in r.item_embeddings] if r.item_embeddings else []}

    def _traj_sum(ts):
        return {"id": ts.id, "text": ts.text, "embedding": _arr(ts.embedding),
                "reflection_id": ts.reflection_id, "trajectory_id": ts.trajectory_id}

    def _traj(t):
        return {"id": t.id, "chunk_text": t.chunk_text, "timestamp": t.timestamp,
                "source_id": t.source_id, "source_type": t.source_type,
                "summary_id": t.summary_id, "concept_ids": t.concept_ids}

    def _persona_entry(pe):
        return {"name": pe.name, "summary": pe.summary, "embedding": _arr(pe.embedding)}

    def _conn_mgr(cm):
        return {
            "connections": {k: list(v) for k, v in cm.connections.items()},
            "stats": {
                f"{k[0]}|||{k[1]}": {
                    "times_traversed": v.times_traversed,
                    "times_led_to_gold": v.times_led_to_gold,
                    "newly_added": v.newly_added,
                }
                for k, v in cm.stats.items()
            },
        }

    snapshot = {
        "concepts":    {cid: _concept(c)   for cid, c  in mem.concepts.items()},
        "reflections": {rid: _reflection(r) for rid, r  in mem.reflections.items()},
        "traj_sums":   {sid: _traj_sum(ts)  for sid, ts in mem.traj_sums.items()},
        "trajectories":{tid: _traj(t)       for tid, t  in mem.trajectories.items()},
        "persona": {
            "entries":      {n: _persona_entry(pe) for n, pe in mem.persona.entries.items()},
            "last_updated": mem.persona.last_updated,
        },
        "rubrics": {k: {"doc_type": v.doc_type, "instructions": v.instructions}
                    for k, v in mem.rubrics.items()},
        "conn_c2r":          _conn_mgr(mem.conn_c2r),
        "conn_r2s":          _conn_mgr(mem.conn_r2s),
        "source_registry":   mem.source_registry,
        "_docs_since_sleep":  mem._docs_since_sleep,
        "_convs_since_sleep": mem._convs_since_sleep,
        "dia_to_traj":       dia_to_traj,
    }

    with open(path, "w") as fh:
        json.dump(snapshot, fh)
    print(f"Memory snapshot saved → {path}")


def load_mem_snapshot(path: str):
    """Deserialise a snapshot produced by save_mem_snapshot.

    Returns (mem, dia_to_traj).
    """
    with open(path) as fh:
        snap = json.load(fh)

    mem = ModifiedMemory()

    mem.concepts = {
        cid: Concept4(
            id=d["id"], text=d["text"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            reflection_ids=d["reflection_ids"],
        )
        for cid, d in snap["concepts"].items()
    }

    mem.reflections = {
        rid: Reflection4(
            id=d["id"], reflection_list=d["reflection_list"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            concept_ids=d["concept_ids"],
            trajectory_summary_ids=d["trajectory_summary_ids"],
            item_embeddings=[np.array(e, dtype=np.float32) for e in d.get("item_embeddings", [])],
        )
        for rid, d in snap["reflections"].items()
    }

    mem.traj_sums = {
        sid: TrajectorySummary4(
            id=d["id"], text=d["text"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            reflection_id=d["reflection_id"],
            trajectory_id=d["trajectory_id"],
        )
        for sid, d in snap["traj_sums"].items()
    }

    mem.trajectories = {
        tid: Trajectory4(
            id=d["id"], chunk_text=d["chunk_text"], timestamp=d["timestamp"],
            source_id=d["source_id"], source_type=d["source_type"],
            summary_id=d.get("summary_id"),
            concept_ids=d.get("concept_ids", []),
        )
        for tid, d in snap["trajectories"].items()
    }

    mem.persona = Persona(
        entries={
            name: PersonaEntry(
                name=pe["name"], summary=pe["summary"],
                embedding=np.array(pe["embedding"], dtype=np.float32),
            )
            for name, pe in snap["persona"]["entries"].items()
        },
        last_updated=snap["persona"]["last_updated"],
    )

    mem.rubrics = {
        k: TaskRubric(doc_type=v["doc_type"], instructions=v["instructions"])
        for k, v in snap["rubrics"].items()
    }

    def _load_conn(d):
        cm = ConnectionManager4()
        cm.connections = defaultdict(set, {k: set(v) for k, v in d["connections"].items()})
        cm.stats = {}
        for key_str, sv in d["stats"].items():
            src, tgt = key_str.split("|||", 1)
            cm.stats[(src, tgt)] = ConnStats(
                times_traversed=sv["times_traversed"],
                times_led_to_gold=sv["times_led_to_gold"],
                newly_added=sv["newly_added"],
            )
        return cm

    mem.conn_c2r = _load_conn(snap["conn_c2r"])
    mem.conn_r2s = _load_conn(snap["conn_r2s"])

    mem.source_registry   = snap["source_registry"]
    mem._docs_since_sleep  = snap.get("_docs_since_sleep", [])
    mem._convs_since_sleep = snap.get("_convs_since_sleep", [])

    dia_to_traj: Dict[str, str] = snap["dia_to_traj"]

    print(f"Memory snapshot loaded ← {path}")
    print(f"  trajectories={len(mem.trajectories)}  reflections={len(mem.reflections)}"
          f"  concepts={len(mem.concepts)}  dia_ids={len(dia_to_traj)}")
    return mem, dia_to_traj


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset() -> List[Dict]:
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            return json.load(f)
    import requests
    print(f"Downloading LoCoMo dataset from {DATASET_URL} ...")
    r = requests.get(DATASET_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    with open(DATASET_PATH, "w") as f:
        json.dump(data, f)
    print(f"Saved to {DATASET_PATH}")
    return data


# =============================================================================
# CONVERSATION PARSING
# =============================================================================

def parse_sessions(conv_data: Dict) -> List[Dict]:
    """Return list of sessions: {session_id, timestamp, text, dia_ids}."""
    sessions = []
    # find all session_N keys (not _date_time)
    session_nums = sorted({
        int(k.split("_")[1])
        for k in conv_data
        if k.startswith("session_")
        and not k.endswith("_date_time")
        and k.split("_")[1].isdigit()
    })
    for n in session_nums:
        key = f"session_{n}"
        turns = conv_data.get(key)
        if not isinstance(turns, list) or not turns:
            continue
        timestamp = conv_data.get(f"session_{n}_date_time", "")
        lines, dia_ids = [], []
        for t in turns:
            speaker = t.get("speaker", "?")
            text = t.get("text", "").strip()
            dia_id = t.get("dia_id", "")
            if text:
                lines.append(f"{speaker}: {text}")
            if dia_id:
                dia_ids.append(dia_id)
        # Also keep per-turn data (parallel to lines) for episodic splitting
        turns_data = []
        for t in turns:
            spk  = t.get("speaker", "?")
            txt  = t.get("text", "").strip()
            did  = t.get("dia_id", "")
            if txt:
                turns_data.append({"line": f"{spk}: {txt}", "dia_id": did})
        sessions.append({
            "session_id": key,
            "timestamp":  timestamp,
            "text":       "\n".join(lines),
            "dia_ids":    dia_ids,
            "turns":      turns_data,   # [{"line": "...", "dia_id": "..."}]
        })
    return sessions


def parse_qa_pairs(conv_item: Dict) -> List[Dict]:
    """Return list of {question, answer, evidence (dia_ids), category}."""
    qa_pairs = []
    for qa in conv_item.get("qa", []):
        cat_raw = qa.get("category", 0)
        if isinstance(cat_raw, str):
            category = cat_raw
        else:
            category = CATEGORY_MAP.get(int(cat_raw), "unknown")
        # evidence is a list of dia_id strings (e.g. ["D1:3", "D2:7"])
        # some entries are semicolon-separated multi-ids, e.g. "D8:6; D9:17" — split them
        raw_evidence = qa.get("evidence", [])
        evidence = []
        for e in raw_evidence:
            if isinstance(e, str) and e:
                for part in e.split(";"):
                    part = part.strip()
                    if part:
                        evidence.append(part)
        qa_pairs.append({
            "question": qa.get("question", ""),
            "answer":   str(qa.get("answer", "")),
            "evidence": evidence,
            "category": category,
        })
    return qa_pairs


# =============================================================================
# SESSION INGESTION STRATEGIES
# =============================================================================

async def ingest_sessions_flat_async(mem: "ModifiedMemory",
                                     sessions: List[Dict]) -> Dict[str, str]:
    """Async version of ingest_sessions_flat: all add_content_async calls are batched."""
    dia_to_traj: Dict[str, str] = {}

    await asyncio.gather(*[
        mem.add_content_async(
            text=sess["text"],
            source_id=sess["session_id"],
            source_type="conversation_session",
            timestamp=sess["timestamp"],
            chunk=False,
        )
        for sess in sessions
    ])

    for sess in sessions:
        for dia_id in sess["dia_ids"]:
            dia_to_traj[dia_id] = sess["session_id"]
    return dia_to_traj


async def ingest_sessions_episodic_async(mem: "ModifiedMemory",
                                          sessions: List[Dict],
                                          min_turns_per_episode: int = 3,
                                          max_splits: int = 4) -> Dict[str, str]:
    """Async version of ingest_sessions_episodic.

    Phase A: episode-split LLM calls for all long sessions are batched.
    Phase B: all add_content_async calls are batched.
    """
    dia_to_traj: Dict[str, str] = {}

    short_sessions = [s for s in sessions if len(s["turns"]) < 2 * min_turns_per_episode]
    long_sessions  = [s for s in sessions if len(s["turns"]) >= 2 * min_turns_per_episode]

    # ── Phase A: batch episode-split LLM calls ────────────────────────────────
    async def _get_split(sess) -> Tuple[Dict, List[int]]:
        turns   = sess["turns"]
        n_turns = len(turns)
        sid     = sess["session_id"]
        numbered_block = "\n".join(f"T{i+1}. {t['line']}" for i, t in enumerate(turns))
        split_after: List[int] = []
        try:
            result = await async_llm(
                async_client,
                "You identify natural episode boundaries in conversations. Respond in JSON.",
                f"""You are given the turns of a single conversation session.

CONVERSATION TURNS:
{numbered_block}

Identify up to {max_splits} points where the conversation shifts to a genuinely new, independently meaningful topic or episode. Return the turn numbers AFTER which to split (1-based). For example, if turns 1–9 form one episode and turns 10–20 another, return [9].

Constraints:
- Each resulting episode must contain at least {min_turns_per_episode} turns.
- Only split at clear topic/subject changes — not mid-topic or at minor asides.
- If there are no natural divisions, return an empty list.
- Return at most {max_splits} split points. Do not feel pressured to use all {max_splits} if fewer would be more natural.

Return JSON with "split_after": list of 1-based turn numbers after which to split.""",
            )
            raw = result.get("split_after", [])
            if isinstance(raw, list):
                split_after = sorted(
                    int(x) for x in raw
                    if isinstance(x, (int, float)) and 1 <= int(x) < n_turns
                )
        except Exception as e:
            print(f"    Episode-split LLM call failed for {sid}: {e}")
        return sess, split_after

    print(f"  [async] batching {len(long_sessions)} episode-split LLM calls …")
    split_results = await asyncio.gather(*[_get_split(s) for s in long_sessions])

    # ── Phase B: collect all add_content_async coroutines ────────────────────
    add_tasks: List = []
    dia_maps:  List[Dict[str, str]] = []

    for sess in short_sessions:
        add_tasks.append(
            mem.add_content_async(
                text=sess["text"],
                source_id=sess["session_id"],
                source_type="conversation_session",
                timestamp=sess["timestamp"],
                chunk=False,
            )
        )
        dia_maps.append({t["dia_id"]: sess["session_id"]
                         for t in sess["turns"] if t["dia_id"]})

    for sess, split_after in split_results:
        turns     = sess["turns"]
        n_turns   = len(turns)
        sid       = sess["session_id"]
        timestamp = sess["timestamp"]

        boundaries = [0] + split_after + [n_turns]
        valid_boundaries = [0]
        for bp in boundaries[1:]:
            if bp - valid_boundaries[-1] >= min_turns_per_episode:
                valid_boundaries.append(bp)
        if valid_boundaries[-1] != n_turns:
            valid_boundaries[-1] = n_turns

        episodes = [
            turns[valid_boundaries[i]:valid_boundaries[i + 1]]
            for i in range(len(valid_boundaries) - 1)
        ]

        if len(episodes) == 1:
            add_tasks.append(
                mem.add_content_async(
                    text=sess["text"],
                    source_id=sid,
                    source_type="conversation_session",
                    timestamp=timestamp,
                    chunk=False,
                )
            )
            dia_maps.append({t["dia_id"]: sid for t in turns if t["dia_id"]})
        else:
            for ep_idx, ep_turns in enumerate(episodes):
                source_id = f"{sid}_ep{ep_idx + 1}"
                ep_text   = "\n".join(t["line"] for t in ep_turns)
                add_tasks.append(
                    mem.add_content_async(
                        text=ep_text,
                        source_id=source_id,
                        source_type="conversation_session",
                        timestamp=timestamp,
                        chunk=False,
                    )
                )
                dia_maps.append({t["dia_id"]: source_id for t in ep_turns if t["dia_id"]})
            print(f"    {sid}: split into {len(episodes)} episodes "
                  f"({[len(e) for e in episodes]} turns each)")

    print(f"  [async] batching {len(add_tasks)} add_content calls …")
    await asyncio.gather(*add_tasks)

    for dm in dia_maps:
        dia_to_traj.update(dm)
    return dia_to_traj


# =============================================================================
# QA HELPERS
# =============================================================================

# =============================================================================
# LLM-AS-A-JUDGE
# =============================================================================
_LLM_JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as \"label\".
"""


async def async_llm_as_judge(async_client, question: str, gold_answer: str,
                              generated_answer: str) -> bool:
    """Async version of llm_as_judge."""
    user_msg = _LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    try:
        raw = await async_llm(
            async_client,
            "You are a fair evaluator. Label answers as CORRECT or WRONG and respond in JSON.",
            user_msg,
        )
        if isinstance(raw, dict):
            label = str(raw.get("label", "")).strip().upper()
            return label == "CORRECT"
        try:
            parsed = json.loads(raw)
            label = str(parsed.get("label", "")).strip().upper()
            return label == "CORRECT"
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        upper = str(raw).upper()
        if "CORRECT" in upper and "WRONG" not in upper:
            return True
        if "WRONG" in upper and "CORRECT" not in upper:
            return False
        raise ValueError(f"LLM judge response unclear: {raw}")
    except Exception as e:
        print(f"  [async_llm_as_judge] call failed: {e}")
        return False


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and gold answer strings.
    Strips punctuation before tokenizing so 'LGBTQ+' matches 'LGBTQ'
    and 'self-care's' matches 'self-care'."""
    def normalize(s: str) -> List[str]:
        s = str(s).lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)  # strip all punctuation/special chars
        return s.split()
    pred_tokens = normalize(pred)
    gold_tokens = normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def filter_relevant_items(question: str, numbered_items: List[str],
                          persona_note: str) -> List[str]:
    """Ask the LLM which 1-based indices from numbered_items are relevant to the question.
    Returns the filtered subset in original order.  Falls back to the full list on failure."""
    if not numbered_items:
        return numbered_items
    numbered_block = "\n".join(f"{i+1}. {item}" for i, item in enumerate(numbered_items))
    try:
        result = llm(
            "You are a relevance filter for memory retrieval. Respond in JSON.",
            f"""Question: {question}{persona_note}

Memory items (numbered):
{numbered_block}

Which items contain information that is directly relevant to answering the question? \
Include items that state the fact explicitly OR from which the answer can be directly inferred. \
Exclude items about unrelated topics, even if they mention the same people.

Return JSON with "indices": a list of 1-based integers (e.g. [2, 5, 7])."""
        )
        indices = result.get("indices", [])
        if not isinstance(indices, list) or not indices:
            return numbered_items
        kept = []
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= len(numbered_items):
                kept.append(numbered_items[idx - 1])
        return kept if kept else numbered_items
    except Exception:
        return numbered_items


def filter_items_by_embedding(question: str,
                              groups: List[List[str]],
                              display_groups: List[List[str]] = None,
                              threshold: float = 0.40,
                              min_per_group: int = 1,
                              query_texts: Optional[List[str]] = None) -> List[str]:
    """Embedding-based relevance filter over per-reflection item groups.

    groups         — plain-text items used for similarity computation.
    display_groups — optional timestamped versions to return; must mirror groups.
                     If None, items from groups are returned.
    query_texts    — optional list of query strings (e.g. qc.predicted_reflections)
                     used instead of the raw question for per-item scoring.
                     Scoring mirrors _match_layer_reflections:
                       sem_sim  = max cosine similarity over all query_texts
                       bm25_sim = max normalised BM25 score over all query_texts
                       combined = 0.6 * sem_sim + 0.4 * bm25_sim
                     Each item is kept when combined >= threshold.

    For each group (when query_texts is None):
      - Compute cosine similarity between the question and each plain-text item.
      - Keep all items whose similarity >= threshold.
      - If fewer than min_per_group items survive, top-up with the highest-similarity
        items from that group (preserving original order).

    Falls back to returning all display items if embedding fails.
    """
    if display_groups is None:
        display_groups = groups
    if not groups:
        return []
    all_plain = [item for group in groups for item in group]
    all_display = [item for group in display_groups for item in group]
    if not all_plain:
        return []

    # ── query-components path: score per item using predicted_reflections ──────
    if query_texts:
        try:
            q_embs    = embed(query_texts)                           # (Q, D)
            item_embs = embed(all_plain)                            # (N, D)
            sem_sims  = np.max(cos_sim(q_embs, item_embs), axis=0)  # (N,)

            bm25_raw  = _bm25_score_items(query_texts, all_plain)   # (Q, N)
            bm25_sims = np.max(bm25_raw, axis=0)                    # (N,)
            bm25_max  = float(bm25_sims.max())
            bm25_norm = bm25_sims / bm25_max if bm25_max > 0 else bm25_sims

            item_scores = 0.6 * sem_sims + 0.4 * bm25_norm         # (N,)
        except Exception:
            return all_display

        result: List[str] = []
        idx = 0
        for group, disp_group in zip(groups, display_groups):
            n = len(group)
            group_scores = item_scores[idx: idx + n]
            idx += n

            above = [i for i, s in enumerate(group_scores) if s >= threshold]

            if len(above) < min_per_group:
                ranked = sorted(range(n), key=lambda i: group_scores[i], reverse=True)
                above = sorted(ranked[:min_per_group])

            for i in above:
                result.append(disp_group[i])
        return result

    # ── original path: embed question and compute per-item cosine similarity ──
    try:
        q_emb     = embed([question])           # (1, D)
        item_embs = embed(all_plain)            # (N, D)
        sims = cos_sim(q_emb, item_embs)[0]     # (N,)
    except Exception:
        return all_display

    result: List[str] = []
    idx = 0
    for group, disp_group in zip(groups, display_groups):
        n = len(group)
        group_sims = sims[idx: idx + n]
        idx += n

        above = [i for i, s in enumerate(group_sims) if s >= threshold]

        if len(above) < min_per_group:
            ranked = sorted(range(n), key=lambda i: group_sims[i], reverse=True)
            above = sorted(ranked[:min_per_group])  # restore original order

        for i in above:
            result.append(disp_group[i])

    return result


async def async_filter_items_by_embedding(question: str,
                                           groups: List[List[str]],
                                           display_groups: List[List[str]] = None,
                                           threshold: float = 0.40,
                                           min_per_group: int = 1,
                                           query_texts: Optional[List[str]] = None) -> List[str]:
    """Async version of filter_items_by_embedding — uses async_embed so embed
    calls are non-blocking.  Both embed calls in each code path are issued
    concurrently via asyncio.gather."""
    if display_groups is None:
        display_groups = groups
    if not groups:
        return []
    all_plain   = [item for group in groups for item in group]
    all_display = [item for group in display_groups for item in group]
    if not all_plain:
        return []

    if query_texts:
        try:
            q_embs, item_embs = await asyncio.gather(
                async_embed(query_texts),
                async_embed(all_plain),
            )
            sem_sims  = np.max(cos_sim(q_embs, item_embs), axis=0)

            bm25_raw  = _bm25_score_items(query_texts, all_plain)
            bm25_sims = np.max(bm25_raw, axis=0)
            bm25_max  = float(bm25_sims.max())
            bm25_norm = bm25_sims / bm25_max if bm25_max > 0 else bm25_sims

            item_scores = 0.6 * sem_sims + 0.4 * bm25_norm
        except Exception:
            return all_display

        result: List[str] = []
        idx = 0
        for group, disp_group in zip(groups, display_groups):
            n = len(group)
            group_scores = item_scores[idx: idx + n]
            idx += n
            above = [i for i, s in enumerate(group_scores) if s >= threshold]
            if len(above) < min_per_group:
                ranked = sorted(range(n), key=lambda i: group_scores[i], reverse=True)
                above = sorted(ranked[:min_per_group])
            if len(above) > MAX_PER_GROUP:
                above = sorted(sorted(above, key=lambda i: group_scores[i], reverse=True)[:MAX_PER_GROUP])
            for i in above:
                result.append(disp_group[i])
        if MAX_TOTAL_ITEMS > 0 and len(result) > MAX_TOTAL_ITEMS:
            result = result[:MAX_TOTAL_ITEMS]
        return result

    try:
        q_emb, item_embs = await asyncio.gather(
            async_embed([question]),
            async_embed(all_plain),
        )
        sims = cos_sim(q_emb, item_embs)[0]
    except Exception:
        return all_display

    result: List[str] = []
    idx = 0
    for group, disp_group in zip(groups, display_groups):
        n = len(group)
        group_sims = sims[idx: idx + n]
        idx += n
        above = [i for i, s in enumerate(group_sims) if s >= threshold]
        if len(above) < min_per_group:
            ranked = sorted(range(n), key=lambda i: group_sims[i], reverse=True)
            above = sorted(ranked[:min_per_group])
        if len(above) > MAX_PER_GROUP:
            above = sorted(sorted(above, key=lambda i: group_sims[i], reverse=True)[:MAX_PER_GROUP])
        for i in above:
            result.append(disp_group[i])
    if MAX_TOTAL_ITEMS > 0 and len(result) > MAX_TOTAL_ITEMS:
        result = result[:MAX_TOTAL_ITEMS]
    return result


def _bm25_score_items(queries: List[str], items: List[str]) -> np.ndarray:
    """Return a (len(queries), len(items)) matrix of BM25 scores.
    Mirrors ModifiedMemory._bm25_score_items — kept in sync manually."""
    def tokenize(s: str) -> List[str]:
        return re.sub(r"[^a-z0-9]", " ", s.lower()).split()

    tokenized_items = [tokenize(item) for item in items]
    N = len(items)
    df: Dict[str, int] = defaultdict(int)
    for tokens in tokenized_items:
        for t in set(tokens):
            df[t] += 1
    avgdl = float(np.mean([len(t) for t in tokenized_items])) if tokenized_items else 1.0
    k1, b = 1.5, 0.75

    scores = np.zeros((len(queries), N))
    for qi, query in enumerate(queries):
        q_tokens = tokenize(query)
        for di, doc_tokens in enumerate(tokenized_items):
            dl = len(doc_tokens)
            tf_map: Dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1
            score = 0.0
            for t in q_tokens:
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                idf = np.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            scores[qi, di] = score
    return scores


def cap_reflection_items(question: str,
                         groups: List[List[str]],
                         display_groups: List[List[str]] = None,
                         cap: int = TOP_K) -> List[str]:
    """Select the top-`cap` reflection items by combined similarity score, then
    return them grouped by their source reflection list.

    Scoring mirrors the retrieval pipeline: 60% semantic (embedding cosine) +
    40% BM25 (normalised), with max over query terms per item.

    Groups are ordered by the best (highest) combined score item they contribute
    to the selection (descending).  Within each group, items appear in their
    original reflection_list order.

    groups         — plain-text items used for similarity computation.
    display_groups — timestamped versions to return; must mirror groups.
                     If None, plain groups are returned.
    cap            — maximum total items returned.

    Falls back to returning the first `cap` display items if scoring fails.
    """
    if display_groups is None:
        display_groups = groups
    if not groups:
        return []
    all_plain   = [item for group in groups for item in group]
    all_display = [item for group in display_groups for item in group]
    if not all_plain:
        return []
    try:
        q_emb     = embed([question])             # (1, D)
        item_embs = embed(all_plain)              # (N, D)
        sem_sims  = cos_sim(q_emb, item_embs)[0]  # (N,) — cosine similarities

        bm25_raw  = _bm25_score_items([question], all_plain)  # (1, N)
        bm25_sims = bm25_raw[0]                               # (N,)
        bm25_max  = float(bm25_sims.max())
        bm25_norm = bm25_sims / bm25_max if bm25_max > 0 else bm25_sims

        combined = 0.6 * sem_sims + 0.4 * bm25_norm          # (N,)
    except Exception:
        return all_display[:cap]

    # Build flat index: (flat_idx, group_idx, within_group_idx, combined_score)
    flat: List[Tuple] = []
    flat_idx = 0
    for g_idx, group in enumerate(groups):
        for w_idx in range(len(group)):
            flat.append((flat_idx, g_idx, w_idx, float(combined[flat_idx])))
            flat_idx += 1

    # Select top-cap items globally by combined score
    top_items = sorted(flat, key=lambda x: x[3], reverse=True)[:cap]

    # Determine best (highest) combined score per group among selected items
    group_best_sim: Dict[int, float] = {}
    for _, g_idx, _, score in top_items:
        if g_idx not in group_best_sim or score > group_best_sim[g_idx]:
            group_best_sim[g_idx] = score

    # Order groups by their best combined score, descending
    ordered_groups = sorted(group_best_sim.keys(),
                            key=lambda g: group_best_sim[g], reverse=True)

    # Collect selected within-group indices per group
    group_selected: Dict[int, List[int]] = {g: [] for g in ordered_groups}
    for _, g_idx, w_idx, _ in top_items:
        group_selected[g_idx].append(w_idx)

    # Emit: for each group (in group-rank order) preserve original item order
    result: List[str] = []
    for g_idx in ordered_groups:
        for w_idx in sorted(group_selected[g_idx]):
            result.append(display_groups[g_idx][w_idx])

    return result


async def async_get_short_answer(mem: "ModifiedMemory", async_client,
                                  question: str, top_k: int = TOP_K,
                                  retrieved: Tuple = None,
                                  min_per_group: int = MIN_PER_GROUP
                                  ) -> Tuple[str, int, Dict]:
    """Async version of get_short_answer.

    Uses mem.async_retrieve (non-blocking LLM + embed) and async_llm for all
    answer generation calls.  The two-step trajectory fallback is also async but
    remains sequential within this coroutine (step 2 depends on step 1).
    """
    qc = None
    if retrieved is not None:
        mc, mr, ms, mt = retrieved
    else:
        mc, mr, ms, mt, qc = await mem.async_retrieve(question, top_k=top_k,
                                                        use_query_components=USE_RETRIEVAL_SCORES_FOR_FILTER)

    def refl_timestamp(rid: str) -> str:
        r = mem.reflections.get(rid)
        if r and r.trajectory_summary_ids:
            ts_obj = mem.traj_sums.get(r.trajectory_summary_ids[0])
            if ts_obj:
                t = mem.trajectories.get(ts_obj.trajectory_id)
                if t:
                    return t.timestamp
        return ""

    plain_groups:   List[List[str]] = []
    display_groups: List[List[str]] = []
    for rid in list(mr.keys())[:top_k]:
        r = mem.reflections.get(rid)
        if r:
            ts = refl_timestamp(rid)
            ts_tag = f"[{ts}] " if ts else ""
            plain_group   = list(r.reflection_list)
            display_group = [f"{ts_tag}{item}" for item in r.reflection_list]
            if plain_group:
                plain_groups.append(plain_group)
                display_groups.append(display_group)

    n_items_before_filter = sum(len(g) for g in plain_groups)

    all_plain   = [item for g in plain_groups   for item in g]
    all_display = [item for g in display_groups for item in g]

    _query_texts = (
        qc.predicted_reflections
        if USE_RETRIEVAL_SCORES_FOR_FILTER and qc is not None and qc.predicted_reflections
        else None
    )

    # Simple top-k by per-item score (no threshold filter)
    _gather_tasks = [async_embed(_query_texts or [question]), async_embed(all_plain)]
    if mem.persona.entries:
        _gather_tasks.append(async_embed([question]))
    _gather_results = await asyncio.gather(*_gather_tasks, return_exceptions=True)

    q_embs    = _gather_results[0] if not isinstance(_gather_results[0], Exception) else None
    item_embs = _gather_results[1] if not isinstance(_gather_results[1], Exception) else None
    _persona_q_arr = (
        (_gather_results[2][0] if not isinstance(_gather_results[2], Exception) else None)
        if mem.persona.entries else None
    )

    if q_embs is not None and item_embs is not None and all_plain:
        try:
            sem_mat      = cos_sim(q_embs, item_embs)                          # (Q, N)
            bm25_raw     = _bm25_score_items(_query_texts or [question], all_plain)  # (Q, N)
            # Normalise BM25 per query row so each sub-query's max = 1, then
            # combine with semantic per query component and take the max across
            # components.  This ensures an item scores high only when it is
            # jointly relevant (both semantically and lexically) to a single
            # sub-query, rather than picking up the semantic peak from one
            # sub-query and the BM25 peak from a different one.
            bm25_row_max  = bm25_raw.max(axis=1, keepdims=True)               # (Q, 1)
            bm25_row_max  = np.where(bm25_row_max > 0, bm25_row_max, 1.0)
            bm25_norm_mat = bm25_raw / bm25_row_max                           # (Q, N)
            combined_mat  = 0.6 * sem_mat + 0.4 * bm25_norm_mat              # (Q, N)
            item_scores   = np.max(combined_mat, axis=0)                      # (N,)

            # Build flat index → (group_idx, within_group_idx) mapping
            flat_to_group: List[Tuple[int, int]] = []
            for g_idx, group in enumerate(plain_groups):
                for w_idx in range(len(group)):
                    flat_to_group.append((g_idx, w_idx))

            # Select top MAX_TOTAL_ITEMS globally by score
            ranked_idx = sorted(range(len(all_display)), key=lambda i: item_scores[i], reverse=True)
            if MAX_TOTAL_ITEMS > 0:
                ranked_idx = ranked_idx[:MAX_TOTAL_ITEMS]

            # Re-group selected items by source reflection list, ordered by each group's best score
            group_best_score: Dict[int, float] = {}
            group_selected_widx: Dict[int, List[int]] = {}
            for flat_i in ranked_idx:
                g_idx, w_idx = flat_to_group[flat_i]
                score = float(item_scores[flat_i])
                if g_idx not in group_best_score or score > group_best_score[g_idx]:
                    group_best_score[g_idx] = score
                group_selected_widx.setdefault(g_idx, []).append(w_idx)

            ordered_groups = sorted(group_best_score.keys(),
                                    key=lambda g: group_best_score[g], reverse=True)
            filtered_lines = []
            for g_idx in ordered_groups:
                for w_idx in sorted(group_selected_widx[g_idx]):
                    filtered_lines.append(display_groups[g_idx][w_idx])
        except Exception:
            filtered_lines = all_display[:MAX_TOTAL_ITEMS] if MAX_TOTAL_ITEMS > 0 else all_display
    else:
        filtered_lines = all_display[:MAX_TOTAL_ITEMS] if MAX_TOTAL_ITEMS > 0 else all_display

    traj_blocks = []
    if len(filtered_lines) < 3:
        for tid in list(mt.keys())[:3]:
            t = mem.trajectories.get(tid)
            if t:
                ts_tag = f"[{t.timestamp}]\n" if t.timestamp else ""
                traj_blocks.append(f"{ts_tag}{t.chunk_text}")

    persona_note = ""
    if _persona_q_arr is not None:
        try:
            _q_emb = _persona_q_arr
            _best_name, _best_score = None, -1.0
            for _pname, _pentry in mem.persona.entries.items():
                if _pentry.embedding is not None and len(_pentry.embedding) > 0:
                    _score = float(cos_sim(_q_emb.reshape(1, -1),
                                           _pentry.embedding.reshape(1, -1))[0, 0])
                    if _score > _best_score:
                        _best_score, _best_name = _score, _pname
            if _best_name:
                _e = mem.persona.entries[_best_name]
                persona_note = f"\nUser context [{_best_name}]: {_e.summary}"
        except Exception:
            pass

    content_block = "\n".join(f"- {line}" for line in filtered_lines)
    if traj_blocks:
        content_block += "\n\nSupporting passages:\n" + "\n---\n".join(traj_blocks)
    if not content_block:
        content_block = "(no context retrieved)"

    result = await async_llm(
        async_client,
        "You answer questions from conversation memory. Be extremely concise — "
        "match the style of these example answers: '17 March 2000', 'Straight woman', "
        "'mental health', 'Likely no'. "
        "Each memory bullet has a [timestamp]. Use it to resolve relative time expressions: "
        "'last year' means the year before the timestamp, 'next month' means the month after, etc. "
        "Respond in JSON.",
        f"""Question: {question}{persona_note}

Retrieved memory (with timestamps):
{content_block}

Instructions:
1. If a bullet contains a relative time expression (last week, two weeks ago, next month, etc.), \
resolve it to a concrete date/period using the [timestamp] on that bullet.
2. Reason across all bullets together — the answer may require combining information from \
multiple bullets (e.g. a location mentioned in one bullet and a person in another).
3. Answer in 5 words or fewer using the resolved, concrete value.
4. Always provide your best answer based on the available evidence. Only say "unknown" if \
there is genuinely zero relevant signal across all bullets.
Return JSON with "answer": your short answer string.""",
    )
    answer = result.get("answer", "")
    answer = answer if isinstance(answer, str) else str(answer)

    primary_was_unknown = answer.strip().lower() == "unknown"
    fallback_used   = False
    fallback_helped = False
    if primary_was_unknown:
        seen_sids: set = set()
        candidate_sums: List[Dict] = []
        for rid in list(mr.keys())[:top_k]:
            for sid in mem.conn_r2s.get(rid):
                if sid in seen_sids or sid not in mem.traj_sums:
                    continue
                seen_sids.add(sid)
                ts_obj = mem.traj_sums[sid]
                traj   = mem.trajectories.get(ts_obj.trajectory_id)
                ts_tag = f"[{traj.timestamp}] " if traj and traj.timestamp else ""
                candidate_sums.append({"sid": sid, "ts_tag": ts_tag, "text": ts_obj.text})

        if candidate_sums:
            fallback_used = True
            numbered_block = "\n".join(
                f"{i+1}. {c['ts_tag']}{c['text']}"
                for i, c in enumerate(candidate_sums)
            )
            try:
                sel_result = await async_llm(
                    async_client,
                    "You are a relevance filter for memory retrieval. Respond in JSON.",
                    f"""Question: {question}{persona_note}

Trajectory summaries (numbered):
{numbered_block}

Which summaries are most likely to contain information needed to answer the question? \
Return JSON with "indices": a list of 1-based integers.""",
                )
                sel_indices = sel_result.get("indices", [])
                if not isinstance(sel_indices, list):
                    sel_indices = []
                selected = [
                    candidate_sums[i - 1]
                    for i in sel_indices
                    if isinstance(i, int) and 1 <= i <= len(candidate_sums)
                ]
            except Exception:
                selected = candidate_sums

            if not selected:
                selected = candidate_sums

            traj_passages = []
            for c in selected:
                ts_obj = mem.traj_sums.get(c["sid"])
                if ts_obj:
                    traj = mem.trajectories.get(ts_obj.trajectory_id)
                    if traj:
                        ts_tag = f"[{traj.timestamp}]\n" if traj.timestamp else ""
                        traj_passages.append(f"{ts_tag}{traj.chunk_text}")

            if traj_passages:
                fallback_block = "\n---\n".join(traj_passages)
                try:
                    fb_result = await async_llm(
                        async_client,
                        "You answer questions from conversation memory. Be extremely concise — "
                        "match the style of these example answers: '17 March 2000', 'Straight woman', "
                        "'mental health', 'Likely no'. "
                        "Each passage has a [timestamp]. Use it to resolve relative time expressions. "
                        "Respond in JSON.",
                        f"""Question: {question}{persona_note}

Conversation passages (with timestamps):
{fallback_block}

Instructions:
1. Resolve any relative time expressions using the [timestamp] of the passage they appear in.
2. Answer in 5 words or fewer using the resolved, concrete value.
3. Only say "unknown" if there is genuinely zero relevant signal across all passages.
Return JSON with "answer": your short answer string.""",
                    )
                    fb_answer = fb_result.get("answer", "")
                    fb_answer = fb_answer if isinstance(fb_answer, str) else str(fb_answer)
                    if fb_answer.strip():
                        answer = fb_answer
                        if answer.strip().lower() != "unknown":
                            fallback_helped = True
                except Exception:
                    pass

    return answer, len(filtered_lines), {
        "used": fallback_used,
        "helped": fallback_helped,
        "primary_unknown": primary_was_unknown,
        "n_items_before_filter": n_items_before_filter,
    }


# =============================================================================
# EXTRACTION SUFFICIENCY
# =============================================================================

async def async_check_reflection_sufficiency(
    async_client,
    mem: "ModifiedMemory",
    question: str,
    gold_answer: str,
    gold_traj_ids: List[str],
    category: str = "",
) -> Dict:
    """Async version of check_reflection_sufficiency."""
    gold_sets     = mem.get_gold_for_trajectories(gold_traj_ids)
    gold_refl_ids = gold_sets.get("reflections", set())

    refl_lines = []
    for rid in gold_refl_ids:
        r = mem.reflections.get(rid)
        if r:
            for item in r.reflection_list:
                refl_lines.append(f"- {item}")

    if not refl_lines:
        return {"sufficient": False, "reason": "no reflections extracted from gold trajectories"}

    is_temporal = (category.lower() == "temporal")

    session_dates_block = ""
    if is_temporal:
        session_dates = []
        for tid in gold_traj_ids:
            traj = mem.trajectories.get(tid)
            if traj and traj.timestamp:
                session_dates.append(traj.timestamp[:10])
        if session_dates:
            unique_dates = sorted(set(session_dates))
            session_dates_block = (
                "\nSession date(s) when the conversation took place: "
                + ", ".join(unique_dates)
                + "\n"
            )

    if is_temporal:
        temporal_note = (
            "\nIMPORTANT — this is a TEMPORAL question. "
            "Conversations use relative time expressions (e.g. 'next Saturday', 'last week', "
            "'tomorrow') rather than absolute dates, so reflections are NOT expected to contain "
            "exact timestamps. Judge sufficiency on whether the event or activity is captured; "
            "the time reference is sufficient as long as it is consistent with the session dates "
            "provided above (if any). Do NOT mark insufficient solely because a precise date is "
            "absent from the reflections.\n"
        )
    else:
        temporal_note = ""

    result = await async_llm(
        async_client,
        "You assess whether a set of memory reflections contains enough information "
        "to answer a question. Respond in JSON.",
        f"""Question: {question}
Expected answer: {gold_answer}
{session_dates_block}
Reflections extracted from the relevant conversation turns:
{chr(10).join(refl_lines)}
{temporal_note}
Do these reflections contain the key facts needed to correctly answer the question? \
Answer true only if the specific answer (or information from which it can be directly inferred) \
is present; false if key details are absent or too vague. \
Return JSON: {{"sufficient": true or false, "reason": "one sentence describing what is present or missing"}}""",
    )
    return {
        "sufficient": bool(result.get("sufficient", False)),
        "reason":     str(result.get("reason", "")),
    }


async def async_evaluate_extraction_sufficiency(
    async_client,
    mem: "ModifiedMemory",
    questions: List[Dict],
    dia_to_traj: Dict[str, str],
) -> Dict:
    """Async version of evaluate_extraction_sufficiency: all LLM calls are batched."""
    tasks      : List = []
    task_idxs  : List[int] = []

    for i, qd in enumerate(questions):
        gold_traj_ids = list({
            tid
            for d in qd["evidence"] if d in dia_to_traj
            for tid in mem.source_registry.get(dia_to_traj[d], [])
        })
        if not gold_traj_ids:
            qd["_sufficiency"] = {"sufficient": None, "reason": "no gold trajectories found"}
        else:
            tasks.append(async_check_reflection_sufficiency(
                async_client, mem, qd["question"], qd["answer"], gold_traj_ids,
                category=qd.get("category", ""),
            ))
            task_idxs.append(i)

    results = await asyncio.gather(*tasks)
    for i, result in zip(task_idxs, results):
        questions[i]["_sufficiency"] = result

    evaluated    = [qd for qd in questions
                    if "_sufficiency" in qd and qd["_sufficiency"]["sufficient"] is not None]
    overall_vals = [qd["_sufficiency"]["sufficient"] for qd in evaluated]

    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for qd in evaluated:
        by_cat[qd["category"]].append(qd["_sufficiency"]["sufficient"])

    return {
        "overall": {
            "rate": float(np.mean(overall_vals)) if overall_vals else 0.0,
            "n":    len(overall_vals),
        },
        "per_category": {
            cat: {"rate": float(np.mean(vals)), "n": len(vals)}
            for cat, vals in by_cat.items()
        },
    }


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnose_answers(mem: "ModifiedMemory", questions: List[Dict],
                     top_k: int = TOP_K, n_per_category: int = DIAGNOSE_N, min_per_group: int = MIN_PER_GROUP):
    """Print the full chain for a sample of questions to diagnose answer generation."""
    from collections import defaultdict
    by_cat = defaultdict(list)
    for qd in questions:
        by_cat[qd["category"]].append(qd)

    for cat, qs in sorted(by_cat.items()):
        print(f"\n{'='*65}")
        print(f"  DIAGNOSIS — category: {cat}  (showing {n_per_category})")
        print(f"{'='*65}")
        # sort by F1 ascending to show worst cases first
        sample = sorted(qs, key=lambda q: q.get("_f1", 1.0))[:n_per_category]
        for qd in sample:
            mc, mr, ms, mt = qd.get("_retrieved_layers") or (({}, {}, {}, {}))

            # build the same context as get_short_answer
            def refl_ts(rid):
                r = mem.reflections.get(rid)
                if r and r.trajectory_summary_ids:
                    ts_obj = mem.traj_sums.get(r.trajectory_summary_ids[0])
                    if ts_obj:
                        t = mem.trajectories.get(ts_obj.trajectory_id)
                        if t: return t.timestamp
                return ""

            plain_groups:   List[List[str]] = []
            display_groups: List[List[str]] = []
            for rid in list(mr.keys())[:top_k]:
                r = mem.reflections.get(rid)
                if r:
                    ts = refl_ts(rid)
                    tag = f"[{ts}] " if ts else ""
                    plain_group   = list(r.reflection_list)
                    display_group = [f"{tag}{item}" for item in r.reflection_list]
                    if plain_group:
                        plain_groups.append(plain_group)
                        display_groups.append(display_group)

            all_display = [item for g in display_groups for item in g]
            # filtered = cap_reflection_items(qd["question"], plain_groups,
            #                                  display_groups=display_groups,
            #                                  cap=top_k)
            filtered = filter_items_by_embedding(qd["question"], plain_groups, display_groups=display_groups, threshold = FILTER_THRESHOLD, min_per_group=MIN_PER_GROUP)
            filtered_set = set(filtered)

            traj_blocks = []
            if len(filtered) < 3:
                for tid in list(mt.keys())[:3]:
                    t = mem.trajectories.get(tid)
                    if t:
                        tag = f"[{t.timestamp}] " if t.timestamp else ""
                        traj_blocks.append(f"  {tag}{t.chunk_text[:200]}")

            print(f"\n  Q:    {qd['question']}")
            print(f"  Gold: {qd['answer']}")
            print(f"  Pred: {qd.get('_pred', '(not generated)')}")
            print(f"  F1:   {qd.get('_f1', '?'):.3f}")
            print(f"  --- retrieved context ({len(all_display)} items → {len(filtered)} after cap) ---")
            for item in all_display:
                marker = "  ✓" if item in filtered_set else "  ✗"
                print(f"  {marker} {item}")
            for block in traj_blocks[:2]:
                print(f"    {block}")
            print()


# =============================================================================
# DIAGNOSTIC AGGREGATION
# =============================================================================

def compute_diagnostics(questions: List[Dict]) -> Dict:
    """Aggregate per-question diagnostic fields into a structured summary.

    Requires each qd to have been enriched by async_evaluate_retrieval and
    the Phase-3 processing loop (fields: _retrieval, _gold_traj_count,
    _gold_traj_ret_count, _pred, _f1, _judge_correct, _primary_unknown,
    _fallback_used, _fallback_helped, _n_items_before_filter,
    _n_items_after_filter).
    """
    diag: Dict = {}

    # ------------------------------------------------------------------
    # 1. Retrieval-to-answer 2×2 matrix
    #    RG = gold reflection retrieved, RB = not retrieved
    #    CG = answer correct (judge), CW = answer wrong
    # ------------------------------------------------------------------
    matrix_keys = ("RG_CG", "RG_CW", "RB_CG", "RB_CW")
    overall_mat: Dict[str, int] = {k: 0 for k in matrix_keys}
    mat_by_cat:  Dict[str, Dict[str, int]] = defaultdict(lambda: {k: 0 for k in matrix_keys})
    for qd in questions:
        if "_retrieval" not in qd or "_judge_correct" not in qd:
            continue
        refl_row = qd["_retrieval"].get("reflections")
        if refl_row is None:
            continue
        rg  = "RG" if refl_row.get("hit", 0) else "RB"
        cg  = "CG" if qd["_judge_correct"] else "CW"
        key = f"{rg}_{cg}"
        overall_mat[key]              += 1
        mat_by_cat[qd["category"]][key] += 1
    diag["retrieval_answer_matrix"] = {
        "overall":      dict(overall_mat),
        "per_category": {cat: dict(v) for cat, v in mat_by_cat.items()},
        "legend": {
            "RG_CG": "gold retrieved & answer correct",
            "RG_CW": "gold retrieved but answer wrong (synthesis failure)",
            "RB_CG": "gold not retrieved but answer correct (lucky/other)",
            "RB_CW": "gold not retrieved & answer wrong (retrieval failure)",
        },
    }

    # ------------------------------------------------------------------
    # 2. "Unknown" rate and fallback effectiveness
    # ------------------------------------------------------------------
    unk_by_cat: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "primary_unknown": 0, "fallback_used": 0,
        "fallback_helped": 0, "final_unknown": 0,
    })
    for qd in questions:
        cat = qd.get("category", "unknown")
        unk_by_cat[cat]["n"]               += 1
        unk_by_cat[cat]["primary_unknown"] += int(qd.get("_primary_unknown", False))
        unk_by_cat[cat]["fallback_used"]   += int(qd.get("_fallback_used",   False))
        unk_by_cat[cat]["fallback_helped"] += int(qd.get("_fallback_helped", False))
        unk_by_cat[cat]["final_unknown"]   += int(qd.get("_pred", "").strip().lower() == "unknown")
    diag["unknown_analysis"] = {}
    for cat, v in unk_by_cat.items():
        n = v["n"]
        fb_used = v["fallback_used"]
        diag["unknown_analysis"][cat] = {
            "n": n,
            "primary_unknown_rate": v["primary_unknown"] / n if n else 0.0,
            "fallback_used_rate":   fb_used / n if n else 0.0,
            "fallback_help_rate":   v["fallback_helped"] / fb_used if fb_used else 0.0,
            "final_unknown_rate":   v["final_unknown"] / n if n else 0.0,
        }

    # ------------------------------------------------------------------
    # 3. Reflection item filter aggressiveness
    # ------------------------------------------------------------------
    filter_by_cat: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "before_total": 0, "after_total": 0, "zero_after": 0,
        "wrong_and_zero": 0,
    })
    for qd in questions:
        cat    = qd.get("category", "unknown")
        before = qd.get("_n_items_before_filter", 0)
        after  = qd.get("_n_items_after_filter",  0)
        wrong  = not qd.get("_judge_correct", True)
        filter_by_cat[cat]["n"]            += 1
        filter_by_cat[cat]["before_total"] += before
        filter_by_cat[cat]["after_total"]  += after
        filter_by_cat[cat]["zero_after"]   += int(after == 0)
        filter_by_cat[cat]["wrong_and_zero"] += int(after == 0 and wrong)
    diag["filter_stats"] = {}
    for cat, v in filter_by_cat.items():
        n = v["n"]
        diag["filter_stats"][cat] = {
            "n": n,
            "mean_items_before": v["before_total"] / n if n else 0.0,
            "mean_items_after":  v["after_total"]  / n if n else 0.0,
            "zero_after_rate":   v["zero_after"]   / n if n else 0.0,
            "wrong_given_zero_after": (
                v["wrong_and_zero"] / v["zero_after"] if v["zero_after"] else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # 4. Multi-hop evidence completeness
    # ------------------------------------------------------------------
    def _mh_stats(qs: List[Dict]) -> Dict:
        if not qs:
            return {"n": 0}
        all_ret   = [qd["_gold_traj_ret_count"] == qd["_gold_traj_count"] for qd in qs]
        frac_ret  = [qd["_gold_traj_ret_count"] / qd["_gold_traj_count"]  for qd in qs]
        judge_ok  = [qd["_judge_correct"] for qd in qs if "_judge_correct" in qd]
        return {
            "n": len(qs),
            "all_gold_retrieved_rate":  float(np.mean(all_ret)),
            "mean_fraction_retrieved":  float(np.mean(frac_ret)),
            "judge_accuracy":           float(np.mean(judge_ok)) if judge_ok else None,
        }

    mh_qs = [qd for qd in questions
             if qd.get("category") == "multi_hop" and "_gold_traj_count" in qd]
    diag["multihop_completeness"] = {
        "1_gold_traj":     _mh_stats([qd for qd in mh_qs if qd["_gold_traj_count"] == 1]),
        "2plus_gold_traj": _mh_stats([qd for qd in mh_qs if qd["_gold_traj_count"] >  1]),
        "overall":         _mh_stats(mh_qs),
    }

    # ------------------------------------------------------------------
    # 5. Temporal F1=0 but judge=CORRECT (format mismatch)
    # ------------------------------------------------------------------
    temp_qs = [qd for qd in questions if qd.get("category") == "temporal"]
    if temp_qs:
        fmt_mismatch = [qd for qd in temp_qs
                        if qd.get("_f1", 1.0) == 0.0 and qd.get("_judge_correct")]
        diag["temporal_format_mismatch"] = {
            "n_temporal":            len(temp_qs),
            "n_f1_zero":             sum(1 for qd in temp_qs if qd.get("_f1", 1.0) == 0.0),
            "n_f1_zero_judge_correct": len(fmt_mismatch),
            "examples": [
                {"question": qd["question"], "gold": qd["answer"], "pred": qd.get("_pred", "")}
                for qd in fmt_mismatch[:5]
            ],
        }

    # ------------------------------------------------------------------
    # 6. Concept failure propagation
    #    When concept retrieval misses, how often does reflection retrieval
    #    still succeed (compensated via direct similarity)?
    # ------------------------------------------------------------------
    cp_by_cat: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "c_hit_r_hit": 0, "c_miss_r_hit": 0, "c_miss_r_miss": 0,
    })
    for qd in questions:
        if "_retrieval" not in qd:
            continue
        r = qd["_retrieval"]
        c_row  = r.get("concepts")
        rf_row = r.get("reflections")
        if c_row is None or rf_row is None:
            continue
        cat   = qd.get("category", "overall")
        c_hit = bool(c_row.get("hit", 0))
        r_hit = bool(rf_row.get("hit", 0))
        cp_by_cat[cat]["n"] += 1
        if c_hit and r_hit:
            cp_by_cat[cat]["c_hit_r_hit"]   += 1
        elif not c_hit and r_hit:
            cp_by_cat[cat]["c_miss_r_hit"]  += 1
        elif not c_hit and not r_hit:
            cp_by_cat[cat]["c_miss_r_miss"] += 1
    diag["concept_propagation"] = {}
    for cat, v in cp_by_cat.items():
        n_miss = v["c_miss_r_hit"] + v["c_miss_r_miss"]
        diag["concept_propagation"][cat] = {
            "n": v["n"],
            "n_concept_miss": n_miss,
            "refl_hit_given_concept_miss": (
                v["c_miss_r_hit"] / n_miss if n_miss else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # 7. Answer length: generated vs gold (word count)
    # ------------------------------------------------------------------
    len_by_cat: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "gold_words": 0, "pred_words": 0,
        "gold_gt5_n": 0, "wrong_gold_gt5": 0,
    })
    for qd in questions:
        if "_pred" not in qd:
            continue
        cat       = qd.get("category", "unknown")
        gold_len  = len(qd["answer"].split())
        pred_len  = len(qd.get("_pred", "").split())
        wrong     = not qd.get("_judge_correct", True)
        len_by_cat[cat]["n"]          += 1
        len_by_cat[cat]["gold_words"] += gold_len
        len_by_cat[cat]["pred_words"] += pred_len
        if gold_len > 5:
            len_by_cat[cat]["gold_gt5_n"] += 1
            if wrong:
                len_by_cat[cat]["wrong_gold_gt5"] += 1
    diag["answer_length"] = {}
    for cat, v in len_by_cat.items():
        n      = v["n"]
        gt5_n  = v["gold_gt5_n"]
        diag["answer_length"][cat] = {
            "n": n,
            "mean_gold_words": v["gold_words"] / n if n else 0.0,
            "mean_pred_words": v["pred_words"] / n if n else 0.0,
            "pct_gold_gt5_words": gt5_n / n if n else 0.0,
            "wrong_rate_when_gold_gt5": (
                v["wrong_gold_gt5"] / gt5_n if gt5_n else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # 8. Multi-hop retrieval gap analysis
    #    For each missing gold trajectory in multi-hop questions, classify
    #    why it wasn't retrieved: no_reflection_link, no_concept_link,
    #    gate_closed (concept not matched), gate_open_crowded (ranked out).
    # ------------------------------------------------------------------
    cause_counts: Dict[str, int] = defaultdict(int)
    gate_open_sims: List[float] = []
    gate_closed_examples: List[Dict] = []
    for qd in questions:
        if qd.get("category") != "multi_hop":
            continue
        for det in qd.get("_multihop_gap", {}).get("details", []):
            cause = det.get("cause", "unknown")
            cause_counts[cause] += 1
            if cause == "gate_open_crowded" and det.get("max_query_to_gold_sim") is not None:
                gate_open_sims.append(det["max_query_to_gold_sim"])
            if cause == "gate_closed" and len(gate_closed_examples) < 5:
                gate_closed_examples.append({
                    "gold_concepts": det.get("gold_concept_texts", []),
                    "pred_concepts": det.get("pred_concept_texts", []),
                })
    total_missing = sum(cause_counts.values())
    diag["multihop_gap_analysis"] = {
        "total_missing_gold_trajs": total_missing,
        "cause_breakdown": {
            k: {"count": v, "pct": round(v / total_missing, 3) if total_missing else 0.0}
            for k, v in cause_counts.items()
        },
        "gate_open_crowded_sim": {
            "mean":   round(float(np.mean(gate_open_sims)), 4) if gate_open_sims else None,
            "lt_0.3":     sum(1 for s in gate_open_sims if s < 0.3),
            "0.3_to_0.5": sum(1 for s in gate_open_sims if 0.3 <= s < 0.5),
            "0.5_to_0.7": sum(1 for s in gate_open_sims if 0.5 <= s < 0.7),
            "gte_0.7":    sum(1 for s in gate_open_sims if s >= 0.7),
        } if gate_open_sims else None,
        "gate_closed_examples": gate_closed_examples,
    }

    return diag


# =============================================================================
# RETRIEVAL EVALUATION
# =============================================================================

async def async_evaluate_retrieval(
    mem: "ModifiedMemory",
    questions: List[Dict],
    dia_to_traj: Dict[str, str],
    top_k: int = TOP_K,
) -> Tuple[Dict, int]:
    """Async version of evaluate_retrieval: all mem.async_retrieve calls are batched."""
    layers = ["concepts", "reflections", "trajectory_summaries", "trajectories"]

    # Resolve gold sets and identify questions that can be evaluated
    eval_idxs:  List[int]  = []
    eval_golds: List[Dict] = []
    gold_traj_ids_list: List[List[str]] = []
    for i, qd in enumerate(questions):
        gold_traj_ids = list({
            tid
            for d in qd["evidence"] if d in dia_to_traj
            for tid in mem.source_registry.get(dia_to_traj[d], [])
        })
        if gold_traj_ids:
            eval_idxs.append(i)
            eval_golds.append(mem.get_gold_for_trajectories(gold_traj_ids))
            gold_traj_ids_list.append(gold_traj_ids)

    # Batch all async_retrieve calls
    retrieve_results = await asyncio.gather(*[
        mem.async_retrieve(questions[i]["question"], top_k=top_k)
        for i in eval_idxs
    ])

    # Compute per-question metrics and stash on qd
    n_eval = 0
    for i_idx, (gold_sets, ret_result, g_traj_ids) in enumerate(
            zip(eval_golds, retrieve_results, gold_traj_ids_list)):
        i   = eval_idxs[i_idx]
        qd  = questions[i]
        mc, mr, ms, mt, qc = ret_result
        retrieved = {
            "concepts":             list(mc.keys()),
            "reflections":          list(mr.keys()),
            "trajectory_summaries": list(ms.keys()),
            "trajectories":         list(mt.keys()),
        }
        row = {"category": qd["category"]}
        for layer in layers:
            gold = gold_sets.get(layer, set())
            ret  = retrieved[layer]
            if not gold:
                row[layer] = None
                continue
            hit   = int(any(r in gold for r in ret))
            found = sum(1 for r in ret if r in gold)
            rank  = next((k + 1 for k, r in enumerate(ret) if r in gold), None)
            row[layer] = {
                "hit":      hit,
                "coverage": found / len(gold),
                "mrr":      1.0 / rank if rank else 0.0,
            }
        # Gold trajectory completeness (useful for multi-hop analysis)
        gold_traj_set = gold_sets.get("trajectories", set())
        ret_traj_set  = set(retrieved["trajectories"])
        qd["_gold_traj_count"]     = len(gold_traj_set)
        qd["_gold_traj_ret_count"] = len(gold_traj_set & ret_traj_set)
        qd["_retrieval"]       = row
        qd["_retrieved_layers"] = (mc, mr, ms, mt)
        # Gap analysis: for multi-hop questions where not all gold was retrieved
        if (qd.get("category") == "multi_hop"
                and qd["_gold_traj_count"] > qd["_gold_traj_ret_count"]):
            mc_keys = set(mc.keys())
            mr_keys = set(mr.keys())
            # All reflection IDs reachable from any concept via conn_c2r (incl. learned edges)
            _all_linked_rids: set = set()
            for _rids in mem.conn_c2r.connections.values():
                _all_linked_rids.update(_rids)
            gap_details = []
            for tid in g_traj_ids:
                if tid in ret_traj_set:
                    continue
                traj = mem.trajectories.get(tid)
                sid  = traj.summary_id if traj else None
                ts   = mem.traj_sums.get(sid) if sid else None
                rid  = ts.reflection_id if ts else None
                refl = mem.reflections.get(rid) if rid else None
                if refl is None:
                    gap_details.append({"traj_id": tid, "cause": "no_reflection_link"})
                    continue
                if rid not in _all_linked_rids:
                    gap_details.append({"traj_id": tid, "cause": "no_concept_link",
                                        "reflection_id": rid})
                    continue
                gate_open = any(rid in mem.conn_c2r.get(cid) for cid in mc_keys)
                if not gate_open:
                    gap_details.append({
                        "traj_id": tid, "cause": "gate_closed",
                        "reflection_id": rid,
                        "gold_concept_texts": [mem.concepts[cid].text
                                               for cid in refl.concept_ids
                                               if cid in mem.concepts],
                        "pred_concept_texts": qc.concept_texts if qc else [],
                    })
                else:
                    gap_details.append({
                        "traj_id": tid, "cause": "gate_open_crowded",
                        "reflection_id": rid,
                        "_gold_item_embs": refl.item_embeddings,
                        "_pred_refl_texts": qc.predicted_reflections if qc else [],
                    })
            qd["_multihop_gap"] = {"details": gap_details}
        n_eval += 1

    # Batch embed predicted_reflections for gate_open_crowded cases
    embed_batch: List[str] = []
    embed_tasks = []  # (det, start, end, gold_item_embs)
    for qd in questions:
        for det in qd.get("_multihop_gap", {}).get("details", []):
            if det.get("cause") != "gate_open_crowded":
                continue
            texts     = det.pop("_pred_refl_texts", [])
            gold_embs = det.pop("_gold_item_embs", [])
            if not texts or not gold_embs:
                det["max_query_to_gold_sim"] = None
                continue
            start = len(embed_batch)
            embed_batch.extend(texts)
            embed_tasks.append((det, start, len(embed_batch), gold_embs))
    if embed_batch:
        _all_embs = await async_embed(embed_batch)
        for det, start, end, gold_embs in embed_tasks:
            pred_embs = _all_embs[start:end]
            max_sim = max(
                float(cos_sim(pe.reshape(1, -1), ge.reshape(1, -1))[0, 0])
                for pe in pred_embs for ge in gold_embs
            )
            det["max_query_to_gold_sim"] = round(max_sim, 4)

    def aggregate(rows):
        metrics = {}
        for layer in layers:
            layer_rows = [r[layer] for r in rows if r.get(layer) is not None]
            if not layer_rows:
                metrics[layer] = {"QwG": 0.0, "coverage": 0.0, "MRR": 0.0, "n": 0}
            else:
                metrics[layer] = {
                    "QwG":      np.mean([r["hit"]      for r in layer_rows]),
                    "coverage": np.mean([r["coverage"] for r in layer_rows]),
                    "MRR":      np.mean([r["mrr"]      for r in layer_rows]),
                    "n":        len(layer_rows),
                }
        return metrics

    overall_metrics = aggregate([qd["_retrieval"] for qd in questions if "_retrieval" in qd])
    return overall_metrics, n_eval


# =============================================================================
# MAIN RUNNER
# =============================================================================

async def run_locomo_eval_async(conversation_index: int = 0, top_k: int = TOP_K,
                                 min_per_group: int = MIN_PER_GROUP,
                                 use_episodic_chunking: bool = False,
                                 memory_snapshot_path: Optional[str] = None,
                                 llm_judge: bool = LLM_AS_JUDGE):
    """Async version of run_locomo_eval.

    Parallelisation strategy
    ────────────────────────
    Phase 1 — Memory building:
      • ingest_sessions_flat_async / ingest_sessions_episodic_async
        – episode-split LLM calls batched; all add_content_async calls batched.
      • sleep_update_async – all per-stage LLM calls batched.

    Phase 2 — Retrieval evaluation:
      • async_evaluate_retrieval: all async_retrieve calls batched.

    Phase 3 — QA answer generation:
      • asyncio.gather over async_get_short_answer for every question.

    Phase 4 — LLM judge:
      • asyncio.gather over async_llm_as_judge for every question.

    Phase 5 — Score, aggregate, save: synchronous (unchanged logic).
    """
    print("=" * 65)
    print("  LoCoMo Evaluation — ModifiedMemory  [async]")
    print("=" * 65)

    # Resolve snapshot path
    if memory_snapshot_path is None:
        memory_snapshot_path = MEMORY_SNAPSHOT_PATH
    if memory_snapshot_path is None:
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        memory_snapshot_path = os.path.join(SNAPSHOTS_DIR, f"mem_snapshot_conv{conversation_index}.json")

    # 1. Load dataset
    dataset = load_dataset()
    print(f"Dataset has {len(dataset)} conversations; using index {conversation_index}.")
    conv_item = dataset[conversation_index]
    conv      = conv_item["conversation"]

    if os.path.exists(memory_snapshot_path):
        print(f"\nFound memory snapshot at '{memory_snapshot_path}' — skipping ingestion & sleep_update.")
        mem, dia_to_traj = load_mem_snapshot(memory_snapshot_path)
    else:
        # 2. Parse sessions
        sessions = parse_sessions(conv)
        print(f"Found {len(sessions)} sessions.")

        # 3. Async ingestion
        mem = ModifiedMemory()
        if use_episodic_chunking:
            print("Ingestion strategy: episodic (LLM-based session splitting)  [async]")
            dia_to_traj = await ingest_sessions_episodic_async(mem, sessions)
        else:
            print("Ingestion strategy: flat (one trajectory per session)  [async]")
            dia_to_traj = await ingest_sessions_flat_async(mem, sessions)

        print(f"Ingested {len(mem.trajectories)} trajectories; "
              f"mapped {len(dia_to_traj)} dialogue turns.")

        # 4. sleep_update (async — all LLM calls batched)
        print("Running sleep_update_async …")
        await mem.sleep_update_async(use_query_components=USE_RETRIEVAL_SCORES_FOR_FILTER, refine_reflections=REFINE_REFLECTIONS_DURING_SLEEP_UPDATE, n_questions_per_chunk=SLEEP_UPDATE_QUESTIONS_PER_CHUNK)

        # 4b. Validate memory integrity
        mem.validate_integrity()

        # 4c. Save snapshot
        save_mem_snapshot(mem, dia_to_traj, memory_snapshot_path)

    # 5. Parse QA pairs
    qa_pairs = parse_qa_pairs(conv_item)
    non_adv  = [q for q in qa_pairs if q["category"] != "adversarial"]
    print(f"QA pairs: {len(qa_pairs)} total, {len(non_adv)} non-adversarial.")

    # ── Phase 2: batch retrieval evaluation ───────────────────────────────────
    print(f"\nPhase 2: evaluating retrieval (top_k={top_k}) …  [async]")
    category_list = sorted({q["category"] for q in non_adv})
    overall_metrics, n_eval = await async_evaluate_retrieval(mem, non_adv, dia_to_traj, top_k)

    # Split per-question results by category
    layers = ["concepts", "reflections", "trajectory_summaries", "trajectories"]
    per_category: Dict[str, Dict] = {}
    for cat in category_list:
        cat_rows = [q["_retrieval"] for q in non_adv
                    if q.get("_retrieval") and q["_retrieval"]["category"] == cat]
        def agg(rows, layer):
            layer_rows = [r[layer] for r in rows if r.get(layer) is not None]
            if not layer_rows:
                return {"QwG": 0.0, "coverage": 0.0, "MRR": 0.0, "n": 0}
            return {
                "QwG":      float(np.mean([r["hit"]      for r in layer_rows])),
                "coverage": float(np.mean([r["coverage"] for r in layer_rows])),
                "MRR":      float(np.mean([r["mrr"]      for r in layer_rows])),
                "n":        len(layer_rows),
            }
        per_category[cat] = {
            "metrics": {l: agg(cat_rows, l) for l in layers},
            "n":       len(cat_rows),
        }

    # Print retrieval table
    header = f"{'Layer':<24} {'QwG':>8} {'Coverage':>10} {'MRR':>8} {'N':>6}"
    print("\n" + "=" * 65)
    print("Overall retrieval metrics")
    print("-" * 65)
    print(header)
    print("-" * 65)
    for layer in layers:
        m = overall_metrics[layer]
        print(f"{layer:<24} {m['QwG']:>8.3f} {m['coverage']:>10.3f} {m['MRR']:>8.3f} {m['n']:>6}")
    print("=" * 65)

    for cat in category_list:
        print(f"\n  Category: {cat}")
        print("  " + "-" * 63)
        print("  " + header)
        print("  " + "-" * 63)
        for layer in layers:
            m = per_category[cat]["metrics"][layer]
            print(f"  {layer:<24} {m['QwG']:>8.3f} {m['coverage']:>10.3f} {m['MRR']:>8.3f} {m['n']:>6}")

    suff_stats = {}
    if EVALUATE_EXTRACTION:
        print(f"\nPhase 2b: checking extraction sufficiency …  [async]")
        suff_stats = await async_evaluate_extraction_sufficiency(
            async_client, mem, non_adv, dia_to_traj
        )
        print("\n" + "=" * 65)
        print("Extraction sufficiency  (reflections contain answer?)")
        print("-" * 65)
        s_ov = suff_stats["overall"]
        print(f"{'Overall':<24} {s_ov['rate']:>8.1%}   (n={s_ov['n']})")
        print("-" * 65)
        for cat in sorted(suff_stats["per_category"]):
            s = suff_stats["per_category"][cat]
            print(f"{cat:<24} {s['rate']:>8.1%}   (n={s['n']})")
        print("=" * 65)

        insuff_by_cat: Dict[str, List[Dict]] = defaultdict(list)
        for qd in non_adv:
            if "_sufficiency" in qd and not qd["_sufficiency"]["sufficient"]:
                insuff_by_cat[qd["category"]].append(qd)
        if insuff_by_cat:
            print("\nSample insufficient cases (up to 3 per category):")
            for cat in sorted(insuff_by_cat):
                print(f"\n  [{cat}]")
                for qd in insuff_by_cat[cat][:3]:
                    print(f"    Q:      {qd['question']}")
                    print(f"    Gold:   {qd['answer']}")
                    print(f"    Reason: {qd['_sufficiency']['reason']}")

    # ── Phase 3: batch QA answer generation ───────────────────────────────────
    print(f"\nPhase 3: generating answers for {len(non_adv)} questions …  [async]")
    answer_results = await asyncio.gather(*[
        async_get_short_answer(
            mem, async_client, qd["question"], top_k=top_k,
            retrieved=qd.get("_retrieved_layers"), min_per_group=min_per_group,
        )
        for qd in non_adv
    ], return_exceptions=True)

    f1_scores: Dict[str, List[float]] = defaultdict(list)
    all_f1: List[float] = []
    item_counts: List[int] = []
    n_fallback_used   = 0
    n_fallback_helped = 0

    for qd, res in zip(non_adv, answer_results):
        if isinstance(res, Exception):
            print(f"  [QA] answer failed for '{qd['question'][:60]}': {res}")
            qd["_pred"] = ""
            qd["_f1"]   = 0.0
            all_f1.append(0.0)
            f1_scores[qd["category"]].append(0.0)
            item_counts.append(0)
            continue
        pred, n_items, fb_info = res
        f1 = compute_f1(pred, qd["answer"])
        qd["_pred"] = pred
        qd["_f1"]   = f1
        qd["_n_items_before_filter"] = fb_info.get("n_items_before_filter", 0)
        qd["_n_items_after_filter"]  = n_items
        qd["_primary_unknown"]       = fb_info.get("primary_unknown", False)
        qd["_fallback_used"]         = fb_info["used"]
        qd["_fallback_helped"]       = fb_info["helped"]
        all_f1.append(f1)
        f1_scores[qd["category"]].append(f1)
        item_counts.append(n_items)
        if fb_info["used"]:
            n_fallback_used += 1
        if fb_info["helped"]:
            n_fallback_helped += 1

    print("\n" + "=" * 65)
    print("QA F1 scores")
    print("-" * 65)
    print(f"{'Overall':<24} {np.mean(all_f1):>8.3f}   (n={len(all_f1)})")
    print(f"{'Avg items to generator':<24} {np.mean(item_counts):>8.1f}")
    print(f"{'Traj fallback used':<24} {n_fallback_used:>8}   (helped: {n_fallback_helped})")
    print("-" * 65)
    for cat in sorted(f1_scores):
        scores = f1_scores[cat]
        print(f"{cat:<24} {np.mean(scores):>8.3f}   (n={len(scores)})")
    print("=" * 65)

    # Save F1=0 cases
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zero_f1_cases = [
        {
            "category": qd["category"],
            "question": qd["question"],
            "gold":     qd["answer"],
            "pred":     qd.get("_pred", ""),
        }
        for qd in non_adv
        if qd.get("_f1", 1.0) == 0.0
    ][:50]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    zero_f1_path = os.path.join(RESULTS_DIR, f"zero_f1_cases_{timestamp}.json")
    with open(zero_f1_path, "w") as f:
        json.dump(zero_f1_cases, f, indent=2)
    print(f"F1=0 cases ({len(zero_f1_cases)}) saved to {zero_f1_path}")

    diagnose_answers(mem, non_adv, top_k=top_k, n_per_category=3,
                     min_per_group=min_per_group)

    # ── Phase 4: batch LLM judge ───────────────────────────────────────────────
    judge_scores: Dict[str, List[bool]] = defaultdict(list)
    all_judge: List[bool] = []
    if llm_judge:
        print(f"\nPhase 4: running LLM judge for {len(non_adv)} questions …  [async]")
        judge_results = await asyncio.gather(*[
            async_llm_as_judge(
                async_client,
                question=qd["question"],
                gold_answer=qd["answer"],
                generated_answer=qd.get("_pred", ""),
            )
            for qd in non_adv
        ], return_exceptions=True)

        for qd, jv in zip(non_adv, judge_results):
            correct = bool(jv) if not isinstance(jv, Exception) else False
            qd["_judge_correct"] = correct
            all_judge.append(correct)
            judge_scores[qd["category"]].append(correct)

        acc_overall = sum(all_judge) / len(all_judge) if all_judge else 0.0
        print("\n" + "=" * 65)
        print("LLM-as-a-judge accuracy")
        print("-" * 65)
        print(f"{'Overall':<24} {acc_overall:>8.1%}   (n={len(all_judge)})")
        print("-" * 65)
        for cat in sorted(judge_scores):
            cat_scores = judge_scores[cat]
            acc = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0
            print(f"{cat:<24} {acc:>8.1%}   (n={len(cat_scores)})")
        print("=" * 65)

    # ── Phase 5: save results ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"locomo_eval_results_{timestamp}.json")
    usage       = get_token_usage()
    prompt_tok  = usage["prompt_tokens"]
    compl_tok   = usage["completion_tokens"]
    llm_calls   = usage["llm_calls"]
    embed_tok   = usage["embed_tokens"]
    embed_calls = usage["embed_calls"]
    cost_input  = prompt_tok / 1_000_000 * LLM_COST_INPUT_PER_M
    cost_output = compl_tok  / 1_000_000 * LLM_COST_OUTPUT_PER_M
    cost_embed  = embed_tok  / 1_000_000 * EMBED_COST_PER_M
    cost_total  = cost_input + cost_output + cost_embed

    print(f"\n{'─' * 65}")
    print(f"API usage:")
    print(f"  LLM ({llm_calls} calls):")
    print(f"    Input tokens:  {prompt_tok:,}  (${cost_input:.4f})")
    print(f"    Output tokens: {compl_tok:,}  (${cost_output:.4f})")
    print(f"  Embeddings ({embed_calls} calls):")
    print(f"    Tokens:        {embed_tok:,}  (${cost_embed:.4f})")
    print(f"  Total cost:    ${cost_total:.4f}")
    print(f"{'─' * 65}")

    output = {
        "timestamp":        timestamp,
        "conversation_idx": conversation_index,
        "top_k":            top_k,
        "n_evaluated":      n_eval,
        "overall":          {l: {k: float(v) for k, v in overall_metrics[l].items()} for l in layers},
        "per_category": {
            cat: {
                "n":      per_category[cat]["n"],
                "metrics": {l: {k: float(v) for k, v in per_category[cat]["metrics"][l].items()}
                            for l in layers},
            }
            for cat in category_list
        },
        "qa_f1": {
            "overall":                  float(np.mean(all_f1)) if all_f1 else 0.0,
            "n":                        len(all_f1),
            "avg_items_to_generator":   float(np.mean(item_counts)) if item_counts else 0.0,
            "trajectory_fallback": {
                "used":   n_fallback_used,
                "helped": n_fallback_helped,
            },
            "per_category": {cat: {"mean_f1": float(np.mean(v)), "n": len(v)}
                             for cat, v in f1_scores.items()},
        },
        "llm_judge": (
            {
                "enabled": LLM_AS_JUDGE,
                "overall_accuracy": float(sum(all_judge) / len(all_judge)) if all_judge else 0.0,
                "n": len(all_judge),
                "per_category": {
                    cat: {
                        "accuracy": float(sum(v) / len(v)) if v else 0.0,
                        "n": len(v),
                    }
                    for cat, v in judge_scores.items()
                },
                "per_question": [
                    {
                        "question": qd["question"],
                        "gold":     qd["answer"],
                        "pred":     qd.get("_pred", ""),
                        "category": qd["category"],
                        "correct":  qd.get("_judge_correct"),
                    }
                    for qd in non_adv if "_judge_correct" in qd
                ],
            }
            if llm_judge
            else {"enabled": False}
        ),
        "extraction_sufficiency": {
            "overall": suff_stats.get("overall", None),
            "per_category": suff_stats.get("per_category", {}),
            "per_question": [
                {
                    "question":  qd["question"],
                    "answer":    qd["answer"],
                    "category":  qd["category"],
                    "sufficient": qd["_sufficiency"]["sufficient"],
                    "reason":     qd["_sufficiency"]["reason"],
                }
                for qd in non_adv if "_sufficiency" in qd
            ],
        },
        "diagnostics": compute_diagnostics(non_adv),
        "api_cost": {
            "llm_calls":       llm_calls,
            "prompt_tokens":   prompt_tok,
            "completion_tokens": compl_tok,
            "embed_calls":     embed_calls,
            "embed_tokens":    embed_tok,
            "cost_input_usd":  round(cost_input,  6),
            "cost_output_usd": round(cost_output, 6),
            "cost_embed_usd":  round(cost_embed,  6),
            "cost_total_usd":  round(cost_total,  6),
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output, non_adv


# =============================================================================
# ENTRY POINT
# =============================================================================


async def run_all_locomo_eval_async(
    n_conversations: int = 10,
    top_k: int = TOP_K,
    min_per_group: int = MIN_PER_GROUP,
    use_episodic_chunking: bool = False,
    llm_judge: bool = LLM_AS_JUDGE,
):
    """Async version of run_all_locomo_eval.

    Conversations are evaluated sequentially (one at a time) so that per-conversation
    snapshots and logs remain predictable.  All parallelism is *within* each
    conversation (ingestion, retrieval, QA, judge) via run_locomo_eval_async.
    """
    dataset = load_dataset()
    n_conversations = min(n_conversations, len(dataset))
    print(f"\nRunning async evaluation over {n_conversations} conversation(s).")

    per_conv_outputs: List[Dict] = []
    all_questions:    List[Dict] = []
    failed_conversations: List[Dict] = []

    for conv_idx in range(n_conversations):
        print(f"\n{'#' * 65}")
        print(f"# Conversation {conv_idx + 1} / {n_conversations}")
        print(f"{'#' * 65}")
        try:
            result, non_adv = await run_locomo_eval_async(
                conversation_index=conv_idx,
                top_k=top_k,
                min_per_group=min_per_group,
                use_episodic_chunking=use_episodic_chunking,
                llm_judge=llm_judge,
            )
            per_conv_outputs.append(result)
            all_questions.extend(non_adv)
        except Exception as exc:
            print(f"  [ERROR] conversation {conv_idx} failed: {exc}")
            failed_conversations.append({"index": conv_idx, "error": str(exc)})
            continue

    if not all_questions:
        print("No questions collected — nothing to aggregate.")
        return

    # ── Aggregate (same logic as run_all_locomo_eval) ─────────────────────────
    print(f"\n{'=' * 65}")
    print(f"AGGREGATE RESULTS  ({n_conversations} conversations, "
          f"{len(all_questions)} questions)")
    print(f"{'=' * 65}")

    layers   = ["concepts", "reflections", "trajectory_summaries", "trajectories"]
    all_cats = sorted({qd["category"] for qd in all_questions})

    def _agg_retrieval(questions: List[Dict], layer: str) -> Dict:
        rows = [
            qd["_retrieval"][layer]
            for qd in questions
            if "_retrieval" in qd and qd["_retrieval"].get(layer) is not None
        ]
        if not rows:
            return {"QwG": 0.0, "coverage": 0.0, "MRR": 0.0, "n": 0}
        return {
            "QwG":      float(np.mean([r["hit"]      for r in rows])),
            "coverage": float(np.mean([r["coverage"] for r in rows])),
            "MRR":      float(np.mean([r["mrr"]      for r in rows])),
            "n":        len(rows),
        }

    overall_retrieval = {l: _agg_retrieval(all_questions, l) for l in layers}
    header = f"{'Layer':<24} {'QwG':>8} {'Coverage':>10} {'MRR':>8} {'N':>6}"
    print("\nOverall retrieval metrics")
    print("-" * 65)
    print(header)
    print("-" * 65)
    for layer in layers:
        m = overall_retrieval[layer]
        print(f"{layer:<24} {m['QwG']:>8.3f} {m['coverage']:>10.3f} {m['MRR']:>8.3f} {m['n']:>6}")
    print("=" * 65)

    ret_by_cat = {}
    for cat in all_cats:
        cat_qs = [qd for qd in all_questions if qd.get("category") == cat]
        ret_by_cat[cat] = {l: _agg_retrieval(cat_qs, l) for l in layers}
        print(f"\n  Category: {cat}  (n={len(cat_qs)})")
        print("  " + "-" * 63)
        print("  " + header)
        print("  " + "-" * 63)
        for layer in layers:
            m = ret_by_cat[cat][layer]
            print(f"  {layer:<24} {m['QwG']:>8.3f} {m['coverage']:>10.3f} {m['MRR']:>8.3f} {m['n']:>6}")

    all_f1 = [qd["_f1"] for qd in all_questions if "_f1" in qd]
    f1_by_cat: Dict[str, List[float]] = defaultdict(list)
    for qd in all_questions:
        if "_f1" in qd:
            f1_by_cat[qd["category"]].append(qd["_f1"])

    if all_f1:
        print(f"\n{'=' * 65}")
        print("QA F1 scores (aggregate)")
        print("-" * 65)
        print(f"{'Overall':<24} {np.mean(all_f1):>8.3f}   (n={len(all_f1)})")
        print("-" * 65)
        for cat in sorted(f1_by_cat):
            scores = f1_by_cat[cat]
            print(f"{cat:<24} {np.mean(scores):>8.3f}   (n={len(scores)})")
        print("=" * 65)

    all_judge = [qd["_judge_correct"] for qd in all_questions if "_judge_correct" in qd]
    judge_by_cat: Dict[str, List[bool]] = defaultdict(list)
    for qd in all_questions:
        if "_judge_correct" in qd:
            judge_by_cat[qd["category"]].append(qd["_judge_correct"])

    if all_judge:
        acc = sum(all_judge) / len(all_judge)
        print(f"\n{'=' * 65}")
        print("LLM-as-a-judge accuracy (aggregate)")
        print("-" * 65)
        print(f"{'Overall':<24} {acc:>8.1%}   (n={len(all_judge)})")
        print("-" * 65)
        for cat in sorted(judge_by_cat):
            v = judge_by_cat[cat]
            print(f"{cat:<24} {sum(v)/len(v):>8.1%}   (n={len(v)})")
        print("=" * 65)

    agg_usage       = get_token_usage()
    agg_prompt_tok  = agg_usage["prompt_tokens"]
    agg_compl_tok   = agg_usage["completion_tokens"]
    agg_llm_calls   = agg_usage["llm_calls"]
    agg_embed_tok   = agg_usage["embed_tokens"]
    agg_embed_calls = agg_usage["embed_calls"]
    agg_cost_input  = agg_prompt_tok / 1_000_000 * LLM_COST_INPUT_PER_M
    agg_cost_output = agg_compl_tok  / 1_000_000 * LLM_COST_OUTPUT_PER_M
    agg_cost_embed  = agg_embed_tok  / 1_000_000 * EMBED_COST_PER_M
    agg_cost_total  = agg_cost_input + agg_cost_output + agg_cost_embed

    print(f"\n{'─' * 65}")
    print(f"API usage (all {n_conversations} conversation(s)):")
    print(f"  LLM ({agg_llm_calls} calls):")
    print(f"    Input tokens:  {agg_prompt_tok:,}  (${agg_cost_input:.4f})")
    print(f"    Output tokens: {agg_compl_tok:,}  (${agg_cost_output:.4f})")
    print(f"  Embeddings ({agg_embed_calls} calls):")
    print(f"    Tokens:        {agg_embed_tok:,}  (${agg_cost_embed:.4f})")
    print(f"  Total cost:    ${agg_cost_total:.4f}")
    print(f"{'─' * 65}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_output = {
        "timestamp":          timestamp,
        "n_conversations":    n_conversations,
        "n_questions_total":  len(all_questions),
        "retrieval": {
            "overall": overall_retrieval,
            "per_category": ret_by_cat,
        },
        "qa_f1": {
            "overall":      float(np.mean(all_f1)) if all_f1 else 0.0,
            "n":            len(all_f1),
            "per_category": {
                cat: {"mean_f1": float(np.mean(v)), "n": len(v)}
                for cat, v in f1_by_cat.items()
            },
        },
        "llm_judge": (
            {
                "enabled": True,
                "overall_accuracy": float(sum(all_judge) / len(all_judge)) if all_judge else 0.0,
                "n": len(all_judge),
                "per_category": {
                    cat: {"accuracy": float(sum(v) / len(v)) if v else 0.0, "n": len(v)}
                    for cat, v in judge_by_cat.items()
                },
            }
            if all_judge
            else {"enabled": False}
        ),
        "per_conversation":      per_conv_outputs,
        "failed_conversations":  failed_conversations,
        "diagnostics":           compute_diagnostics(all_questions),
        "api_cost": {
            "llm_calls":         agg_llm_calls,
            "prompt_tokens":     agg_prompt_tok,
            "completion_tokens": agg_compl_tok,
            "embed_calls":       agg_embed_calls,
            "embed_tokens":      agg_embed_tok,
            "cost_input_usd":    round(agg_cost_input,  6),
            "cost_output_usd":   round(agg_cost_output, 6),
            "cost_embed_usd":    round(agg_cost_embed,  6),
            "cost_total_usd":    round(agg_cost_total,  6),
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agg_path = os.path.join(RESULTS_DIR, f"locomo_eval_all_{n_conversations}convs_{timestamp}.json")
    with open(agg_path, "w") as f:
        json.dump(agg_output, f, indent=2)
    print(f"\nAggregate results saved to {agg_path}")

    if failed_conversations:
        print(f"\n{'!' * 65}")
        print(f"WARNING: {len(failed_conversations)} conversation(s) failed:")
        for fc in failed_conversations:
            print(f"  - Conversation index {fc['index']}: {fc['error']}")
        print(f"{'!' * 65}")
    else:
        print(f"All {n_conversations} conversation(s) completed successfully.")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(run_all_locomo_eval_async(use_episodic_chunking=True))
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal evaluation time: {elapsed:.2f} seconds")
