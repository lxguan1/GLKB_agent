"""
Modified Memory System — 4-Layer Architecture

Memory Storage (4 layers):
    Concepts → Reflections → TrajectorySummaries → Trajectories

Memory Utilization:
    PersonaMemory   — single summary of user, updated from conversations during sleep
    TaskRubricLib   — per-document-type instructions for synthesis

Extraction:  add_content(text, source_id, source_type)
Sleep:       sleep_update() — connection learning (top_gold_only) + persona + rubric refresh
Query:       query(question) — 4-layer retrieval + 2-stage synthesis (rubric select → format)
Eval:        evaluate(questions) — questions_with_gold, coverage, MRR per layer
"""

import os
os.environ.setdefault("PROMPT_MODE", "medical")

import json, uuid, asyncio, re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime, timezone
from tqdm import tqdm

# Lazy imports for slow packages — loaded on first use to keep MCP server startup fast
def cos_sim(a, b):
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    return _cos_sim(a, b)

def _get_tiktoken_encoding(name: str):
    import tiktoken as _tiktoken
    return _tiktoken.get_encoding(name)

from config import client, async_client, LLM_MODEL, EMBEDDING_MODEL

# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Concept4:
    id: str
    text: str
    embedding: np.ndarray
    reflection_ids: List[str] = field(default_factory=list)

@dataclass
class Reflection4:
    id: str
    reflection_list: List[str]
    embedding: np.ndarray          # embed of joined list
    concept_ids: List[str] = field(default_factory=list)
    trajectory_summary_ids: List[str] = field(default_factory=list)
    item_embeddings: List[np.ndarray] = field(default_factory=list)  # per-item embeddings

@dataclass
class TrajectorySummary4:
    id: str
    text: str
    embedding: np.ndarray
    reflection_id: str
    trajectory_id: str

@dataclass
class Trajectory4:
    id: str
    chunk_text: str
    timestamp: str
    source_id: str
    source_type: str               # "document" | "conversation"
    summary_id: Optional[str] = None
    concept_ids: List[str] = field(default_factory=list)

@dataclass
class PersonaEntry:
    name: str
    summary: str
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class Persona:
    entries: Dict[str, PersonaEntry] = field(default_factory=dict)
    last_updated: str = ""

@dataclass
class TaskRubric:
    doc_type: str
    instructions: str

@dataclass
class QueryComponents4:
    concept_texts: List[str]
    predicted_reflections: List[str]
    predicted_summary: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Embedding / Similarity Utilities
# ─────────────────────────────────────────────────────────────────────────────

def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.array([])
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts, encoding_format="float")
    return np.array([item.embedding for item in resp.data])

# ── Per-call timing infrastructure (opt-in via enable_timing()) ──────────────
# Timing individual awaits is accurate: perf_counter before/after a single
# `await` measures the real round-trip latency for that specific API call.
# Wrapping a gather() would be inaccurate because other coroutines run during it.
_TIMING_ENABLED: bool = False
_timing_log: list = []  # entries: ("llm"|"embed", n_items, duration_sec)

# Token usage tracking
_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
                      "embed_tokens": 0, "embed_calls": 0}

def reset_token_usage() -> None:
    global _token_usage
    _token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
                    "embed_tokens": 0, "embed_calls": 0}

def get_token_usage() -> dict:
    return dict(_token_usage)

def enable_timing() -> None:
    global _TIMING_ENABLED, _timing_log
    _TIMING_ENABLED = True
    _timing_log = []

def disable_timing() -> None:
    global _TIMING_ENABLED
    _TIMING_ENABLED = False

def get_timing_summary() -> dict:
    """Aggregate _timing_log into a summary dict suitable for display/storage."""
    llm_durs   = [d for op, n, d in _timing_log if op == "llm"]
    embed_durs = [d for op, n, d in _timing_log if op == "embed"]
    embed_texts = [n for op, n, d in _timing_log if op == "embed"]
    def _stats(durs):
        if not durs:
            return {"n": 0, "total_s": 0.0, "avg_s": 0.0, "min_s": 0.0, "max_s": 0.0}
        return {
            "n":       len(durs),
            "total_s": round(sum(durs), 2),
            "avg_s":   round(sum(durs) / len(durs), 2),
            "min_s":   round(min(durs), 2),
            "max_s":   round(max(durs), 2),
        }
    s = _stats(llm_durs)
    e = _stats(embed_durs)
    e["total_texts"] = sum(embed_texts)
    return {"llm": s, "embed": e}

async def async_embed(texts: List[str], _max_retries: int = 4, _base_delay: float = 2.0) -> np.ndarray:
    if not texts:
        return np.array([])
    # OpenAI limits embedding requests to 2048 texts per call — split and gather
    _BATCH_SIZE = 2048
    if len(texts) > _BATCH_SIZE:
        chunks = [texts[i:i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
        results = await asyncio.gather(*[async_embed(c, _max_retries, _base_delay) for c in chunks])
        return np.concatenate(results, axis=0)
    import openai as _oai
    _retryable = (_oai.RateLimitError, _oai.APITimeoutError,
                  _oai.APIConnectionError, _oai.InternalServerError)
    if _TIMING_ENABLED:
        import time as _t; _t0 = _t.perf_counter()
    for _attempt in range(_max_retries):
        try:
            resp = await async_client.embeddings.create(model=EMBEDDING_MODEL, input=texts, encoding_format="float")
            if _TIMING_ENABLED:
                _timing_log.append(("embed", len(texts), _t.perf_counter() - _t0))
            if resp.usage:
                _token_usage["embed_tokens"] += resp.usage.total_tokens
                _token_usage["embed_calls"] += 1
            return np.array([item.embedding for item in resp.data])
        except _retryable as _e:
            if _attempt == _max_retries - 1:
                raise
            _delay = _base_delay * (2 ** _attempt)
            print(f"    [async_embed] transient error ({type(_e).__name__}), retry {_attempt + 1}/{_max_retries - 1} in {_delay:.0f}s", file=__import__('sys').stderr)
            await asyncio.sleep(_delay)
        except Exception as _e:
            import sys, traceback
            print(f"    [async_embed] UNEXPECTED ERROR: {type(_e).__name__}: {_e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise


def top_k_sim(query_emb: np.ndarray, candidate_embs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if len(candidate_embs) == 0:
        return []
    sims = cos_sim(query_emb.reshape(1, -1), candidate_embs)[0]
    idx = np.argsort(sims)[-k:][::-1]
    return [(int(i), float(sims[i])) for i in idx]

def llm(system: str, user: str, json_mode=True) -> dict:
    kwargs = dict(model=LLM_MODEL,
                  messages=[{"role": "system", "content": system},
                             {"role": "user", "content": user}])
    if "gpt-5" not in LLM_MODEL:
        kwargs["temperature"] = 0.7
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content
    return json.loads(text) if json_mode else text


async def async_llm(async_client, system: str, user: str, json_mode: bool = True,
                    _max_retries: int = 4, _base_delay: float = 2.0):
    """Async LLM call with exponential-backoff retry on transient API errors."""
    import openai as _oai
    _retryable = (_oai.RateLimitError, _oai.APITimeoutError,
                  _oai.APIConnectionError, _oai.InternalServerError)
    kwargs = dict(model=LLM_MODEL,
                  messages=[{"role": "system", "content": system},
                             {"role": "user", "content": user}])
    if "gpt-5" not in LLM_MODEL:
        kwargs["temperature"] = 0.7
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if _TIMING_ENABLED:
        import time as _t; _t0 = _t.perf_counter()
    for _attempt in range(_max_retries):
        try:
            resp = await async_client.chat.completions.create(**kwargs)
            if _TIMING_ENABLED:
                _timing_log.append(("llm", 1, _t.perf_counter() - _t0))
            if resp.usage:
                _token_usage["prompt_tokens"] += resp.usage.prompt_tokens
                _token_usage["completion_tokens"] += resp.usage.completion_tokens
                _token_usage["llm_calls"] += 1
            text = resp.choices[0].message.content
            return json.loads(text) if json_mode else text
        except _retryable as _e:
            if _attempt == _max_retries - 1:
                raise
            _delay = _base_delay * (2 ** _attempt)
            print(f"    [async_llm] transient error ({type(_e).__name__}), retry {_attempt + 1}/{_max_retries - 1} in {_delay:.0f}s")
            await asyncio.sleep(_delay)

# ─────────────────────────────────────────────────────────────────────────────
# Simplified Connection Manager  (top_gold_only, no forgetting)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConnStats:
    times_traversed: int = 0
    times_led_to_gold: int = 0
    newly_added: bool = False

    @property
    def utility(self):
        return self.times_led_to_gold / self.times_traversed if self.times_traversed else 0.0

class ConnectionManager4:
    def __init__(self):
        self.connections: Dict[str, Set[str]] = defaultdict(set)
        self.stats: Dict[Tuple[str, str], ConnStats] = {}

    def add(self, src: str, tgt: str, new: bool = False):
        if tgt not in self.connections[src]:
            self.connections[src].add(tgt)
            self.stats[(src, tgt)] = ConnStats(newly_added=new)

    def get(self, src: str) -> Set[str]:
        return self.connections.get(src, set())

    def record(self, src: str, tgt: str, led_to_gold: bool):
        key = (src, tgt)
        if key not in self.stats:
            self.stats[key] = ConnStats()
        self.stats[key].times_traversed += 1
        if led_to_gold:
            self.stats[key].times_led_to_gold += 1

    def learn_top_gold_only(self, matched_src_ids: Set[str],
                             gold_tgt_ids: Set[str],
                             all_tgt_scores: Dict[str, float]):
        """top_gold_only: connect matched sources to the single highest-scored gold target.
        Uses scores for ALL gold targets (score=0 if not in candidate set), matching
        the original top_gold_only strategy which uses gold_scores_per_layer."""
        if not gold_tgt_ids or not matched_src_ids:
            return
        # Pick gold target with highest score regardless of whether it was in top-k
        best_gold = max(gold_tgt_ids, key=lambda tid: all_tgt_scores.get(tid, 0.0))
        for src in matched_src_ids:
            self.add(src, best_gold, new=True)

    def total_connections(self):
        return sum(len(v) for v in self.connections.values())

# ─────────────────────────────────────────────────────────────────────────────
# Modified Memory
# ─────────────────────────────────────────────────────────────────────────────

class ModifiedMemory:

    CONCEPT_THRESHOLD    = 0.75
    REFLECTION_THRESHOLD = 0.85
    CHUNK_SIZE           = 600
    CHUNK_OVERLAP        = 100

    def __init__(self):
        self.concepts:    Dict[str, Concept4]          = {}
        self.reflections: Dict[str, Reflection4]       = {}
        self.traj_sums:   Dict[str, TrajectorySummary4]= {}
        self.trajectories:Dict[str, Trajectory4]       = {}
        self.persona  = Persona()
        self.rubrics: Dict[str, TaskRubric] = {
            "general_qa": TaskRubric(
                doc_type="general_qa",
                instructions=(
                    "Answer the question directly and concisely. "
                    "State the key fact or finding first, then provide supporting detail "
                    "as needed. Use plain prose; avoid unnecessary bullet lists unless "
                    "enumerating distinct items. Cite specific evidence from the retrieved "
                    "content where available. If the information is incomplete, say so briefly."
                ),
            )
        }  # doc_type -> rubric

        # layer-pair connection managers
        self.conn_c2r = ConnectionManager4()  # concepts → reflections
        self.conn_r2s = ConnectionManager4()  # reflections → traj_summaries

        # tracking
        self.source_registry: Dict[str, List[str]] = {}  # source_id -> [traj_ids]
        self._docs_since_sleep:  List[str] = []  # source_ids of new docs
        self._convs_since_sleep: List[str] = []  # source_ids of new conversations

        self._tokenizer = _get_tiktoken_encoding("cl100k_base")

    # ── chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> List[str]:
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.CHUNK_SIZE:
            return [text]
        chunks, start = [], 0
        while start < len(tokens):
            end = start + self.CHUNK_SIZE
            chunk_text = self._tokenizer.decode(tokens[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            if end >= len(tokens):
                break
            start = end - self.CHUNK_OVERLAP
        return chunks

    # ── LLM extraction calls ─────────────────────────────────────────────────

    def _extract_components(self, chunk_text: str) -> dict:
        """Single LLM call → concept_texts, reflection_list, trajectory_summary."""
        return llm(
            "You extract structured memory components from text. Respond in JSON.",
            f"""Extract memory components from the following text chunk.

TEXT:
{chunk_text}

Return JSON with:
- "concept_texts": list of 2-5 short topic labels (1-4 words each) that categorise this chunk
- "reflection_list": list of 5-10 specific factual statements capturing ALL key information from this chunk. Be comprehensive — include specific names, dates, places, quantities, decisions, and personal details. Each statement must be self-contained and understandable without the original text. Preserve precise details (e.g. exact dates, locations, names) rather than generalising them.
- "trajectory_summary": one concise sentence (≤30 words) summarising what this chunk is about

Return ONLY the JSON."""
        )

    def _detect_doc_type(self, text_sample: str) -> str:
        """Infer document type from a text sample.
        Prefers standard types; allows a new snake_case type if none fit."""
        standard_types = ["research_paper", "literature_review", "clinical_report",
                          "book_chapter", "conversation", "technical_report"]
        result = llm(
            "You classify documents. Respond in JSON.",
            f"""What type of document is this?

Preferred types: {standard_types}

Use one of the preferred types if it fits well. If none are a good match, invent a concise snake_case label (e.g. "grant_application", "case_study", "protocol").
Do NOT use "other".

TEXT SAMPLE (first 2000 chars):
{text_sample[:2000]}

Return ONLY JSON like {{"doc_type": "research_paper"}}."""
        )
        detected = result.get("doc_type", "").strip().lower().replace(" ", "_").replace("-", "_")
        return detected if detected and detected != "other" else "research_paper"

    def _extract_rubric(self, text_sample: str, doc_type: str) -> str:
        """Extract a concise rubric (output format + considerations) for this document type."""
        result = llm(
            "You write concise task rubrics. Respond in JSON.",
            f"""Based on the structure of this {doc_type}, write a short rubric (≤240 words) that tells a system how to format and organise a response when answering questions about this type of document.

TEXT SAMPLE:
{text_sample[:1200]}

Return JSON with "instructions": the rubric text.
Focus on: output structure, level of detail, key considerations for this doc type."""
        )
        return result.get("instructions", "")

    def _extract_persona_update(self, conversations: List[str],
                                current_entries: Dict[str, PersonaEntry]) -> Dict[str, PersonaEntry]:
        """Update per-participant persona summaries (with embeddings) from recent conversations."""
        conv_text = "\n\n---\n\n".join(conversations[:10])

        current_dict = {name: entry.summary for name, entry in current_entries.items()}
        current_fmt = json.dumps(current_dict, indent=2) if current_dict else "(empty)"

        result = llm(
            "You maintain concise persona summaries for every participant in a conversation. Respond in JSON.",
            f"""Current persona summaries (keyed by participant name, may be empty):
{current_fmt}

Recent conversations:
{conv_text}

Instructions:
1. Identify every distinct participant / speaker in the conversations (e.g. "Alice", "Bob", "User", "Assistant", or any name/role that appears).
2. For each participant, write or update a persona summary that captures their interests, background, preferences, personality, and communication style as revealed by the conversations.
3. Each individual summary must be at most 100 words.
4. Only update a participant's summary if the new conversations add meaningful information.
5. Preserve participants from the current summaries even if they do not appear in the new conversations.

Return JSON with a single key "participants" whose value is an object mapping each participant's name to their updated summary string.
Example shape: {{"participants": {{"Alice": "...", "Bob": "..."}}}}"""
        )

        updated: dict = result.get("participants", {})
        if not isinstance(updated, dict) or not updated:
            return current_entries  # Fallback: keep existing

        # Merge: LLM-updated entries win; participants absent from this batch are preserved
        merged_text = {**current_dict, **updated}

        # Embed each participant as "{name}: {summary}" so similarity can be computed at query time
        new_entries: Dict[str, PersonaEntry] = {}
        embed_inputs = [f"{name}: {summary}" for name, summary in merged_text.items()]
        try:
            embs = embed(embed_inputs)
            for i, (name, summary) in enumerate(merged_text.items()):
                new_entries[name] = PersonaEntry(name=name, summary=summary, embedding=embs[i])
        except Exception:
            # Fallback: store entries without updated embeddings
            for name, summary in merged_text.items():
                existing = current_entries.get(name)
                new_entries[name] = PersonaEntry(
                    name=name, summary=summary,
                    embedding=existing.embedding if existing is not None else np.array([])
                )
        return new_entries

    # ── layer merging ─────────────────────────────────────────────────────────

    def _get_or_create_concept(self, text: str, refl_id: str, emb: Optional[np.ndarray] = None) -> str:
        if emb is None:
            emb = embed([text])[0]
        if self.concepts:
            cids = list(self.concepts.keys())
            cembs = np.array([self.concepts[c].embedding for c in cids])
            matches = top_k_sim(emb, cembs, k=1)
            if matches and matches[0][1] >= self.CONCEPT_THRESHOLD:
                cid = cids[matches[0][0]]
                if refl_id not in self.concepts[cid].reflection_ids:
                    self.concepts[cid].reflection_ids.append(refl_id)
                return cid
        cid = str(uuid.uuid4())
        self.concepts[cid] = Concept4(id=cid, text=text, embedding=emb, reflection_ids=[refl_id])
        return cid

    def _merge_reflection_lists(self, existing: List[str], new: List[str], max_items: int = 10) -> List[str]:
        """LLM-based merge: deduplicate semantically, preserve specificity, cap at max_items."""
        existing_fmt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(existing))
        new_fmt      = "\n".join(f"{i+1}. {s}" for i, s in enumerate(new))
        try:
            result = llm(
                "You merge lists of factual statements. Respond in JSON.",
                f"""Merge these two lists of factual statements into one consolidated list (at most {max_items} items).

Existing statements:
{existing_fmt}

New statements:
{new_fmt}

Rules:
- Combine statements that say the same thing (keep the more specific/complete version)
- Preserve all distinct facts
- Each statement must be self-contained and understandable without context
- Return at most {max_items} statements

Return JSON with "merged": list of statement strings."""
            )
            merged = result.get("merged", [])
            if merged and isinstance(merged, list):
                return merged[:max_items]
        except Exception:
            pass
        # fallback: keep existing, append truly new ones up to max
        return (existing + [s for s in new if s not in existing])[:max_items]

    def _get_or_create_reflection(self, refl_list: List[str], ts_id: str,
                                   emb: Optional[np.ndarray] = None,
                                   item_embs: Optional[List[np.ndarray]] = None) -> str:
        joined = " ".join(refl_list)
        if emb is None:
            emb = embed([joined])[0]
        if self.reflections:
            rids = list(self.reflections.keys())
            rembs = np.array([self.reflections[r].embedding for r in rids])
            matches = top_k_sim(emb, rembs, k=1)
            if matches and matches[0][1] >= self.REFLECTION_THRESHOLD:
                rid = rids[matches[0][0]]
                r = self.reflections[rid]
                merged = self._merge_reflection_lists(r.reflection_list, refl_list)
                r.reflection_list = merged
                # Embed joined text + all individual items in one call
                merged_all = embed([" ".join(merged)] + merged)
                r.embedding = merged_all[0]
                r.item_embeddings = list(merged_all[1:])
                if ts_id not in r.trajectory_summary_ids:
                    r.trajectory_summary_ids.append(ts_id)
                return rid
        rid = str(uuid.uuid4())
        self.reflections[rid] = Reflection4(id=rid, reflection_list=refl_list,
                                            embedding=emb, trajectory_summary_ids=[ts_id],
                                            item_embeddings=list(item_embs) if item_embs else [])
        return rid

    # ── public: add content ───────────────────────────────────────────────────

    def add_content(self, text: str, source_id: str, source_type: str = "document",
                    timestamp: Optional[str] = None, chunk: bool = True):
        """Chunk text, extract components, build layers. source_type: 'document'|'conversation'.
        timestamp: ISO 8601 string; defaults to now if not provided.
        chunk: if False, treat the entire text as a single chunk (useful for pre-segmented content
               such as conversation sessions where splitting would break dia_id mappings)."""
        print(f"  Adding {source_type} '{source_id}'...")
        chunks = self._chunk(text) if chunk else [text]
        traj_ids = []
        ts = timestamp if timestamp else datetime.now(timezone.utc).isoformat()

        for i, chunk_text in enumerate(tqdm(chunks, desc=f"  Extracting {source_id}", leave=False)):
            try:
                comp = self._extract_components(chunk_text)
            except Exception as e:
                print(f"    Chunk {i}: extraction failed ({e}), skipping")
                continue

            concept_texts  = comp.get("concept_texts") or []
            refl_list      = comp.get("reflection_list") or []
            traj_summary   = comp.get("trajectory_summary") or chunk_text[:80]

            if not isinstance(concept_texts, list):
                print(f"    Chunk {i}: concept_texts is not a list ({type(concept_texts).__name__}), skipping")
                continue
            if not isinstance(refl_list, list):
                print(f"    Chunk {i}: reflection_list is not a list ({type(refl_list).__name__}), skipping")
                continue

            if not refl_list:
                continue

            # bottom-up: Trajectory → TrajectorySummary → Reflection → Concepts
            traj_id = str(uuid.uuid4())
            traj = Trajectory4(id=traj_id, chunk_text=chunk_text, timestamp=ts,
                               source_id=source_id, source_type=source_type)

            ts_id = str(uuid.uuid4())
            ts_emb = embed([traj_summary])[0]
            traj_sum = TrajectorySummary4(id=ts_id, text=traj_summary,
                                         embedding=ts_emb, reflection_id="", trajectory_id=traj_id)

            rid = self._get_or_create_reflection(refl_list, ts_id)
            traj_sum.reflection_id = rid
            traj.summary_id = ts_id

            cids = []
            for ct in concept_texts:
                cid = self._get_or_create_concept(ct, rid)
                if cid not in self.reflections[rid].concept_ids:
                    self.reflections[rid].concept_ids.append(cid)
                if cid not in traj.concept_ids:
                    traj.concept_ids.append(cid)
                cids.append(cid)

            # initialise connections from structure
            self.conn_r2s.add(rid, ts_id)
            for cid in cids:
                self.conn_c2r.add(cid, rid)

            self.trajectories[traj_id] = traj
            self.traj_sums[ts_id] = traj_sum
            traj_ids.append(traj_id)

        self.source_registry[source_id] = traj_ids

        if source_type == "document":
            self._docs_since_sleep.append(source_id)
        else:
            self._convs_since_sleep.append(source_id)

        print(f"    → {len(traj_ids)} chunks, "
              f"{len(self.concepts)} concepts, {len(self.reflections)} reflections")

    async def add_content_async(self, text: str, source_id: str, source_type: str = "document",
                                timestamp: Optional[str] = None, chunk: bool = True):
        """Async version of add_content: all chunk LLM extractions fire concurrently.

        Fires one _async_llm call per chunk in a single asyncio.gather, then processes
        results synchronously so shared-state updates (concepts, reflections) remain
        thread-safe under the asyncio single-thread model.
        """
        print(f"  Adding {source_type} '{source_id}'...")
        chunks = self._chunk(text) if chunk else [text]
        ts = timestamp if timestamp else datetime.now(timezone.utc).isoformat()

        _system = "You extract structured memory components from text. Respond in JSON."
        _user_tmpl = (
            "Extract memory components from the following text chunk.\n\n"
            "TEXT:\n{chunk_text}\n\n"
            "Return JSON with:\n"
            '- "concept_texts": list of 2-5 short topic labels (1-4 words each) that categorise this chunk\n'
            '- "reflection_list": list of 5-10 specific factual statements capturing ALL key information '
            "from this chunk. Be comprehensive — include specific names, dates, places, quantities, "
            "decisions, and personal details. Each statement must be self-contained and understandable "
            "without the original text. Preserve precise details (e.g. exact dates, locations, names) "
            "rather than generalising them.\n"
            '- "trajectory_summary": one concise sentence (≤30 words) summarising what this chunk is about\n\n'
            "Return ONLY the JSON."
        )

        comps = await asyncio.gather(
            *[async_llm(async_client, _system, _user_tmpl.format(chunk_text=c))
              for c in chunks],
            return_exceptions=True,
        )

        # ── Pass 1: validate and register all texts that need embedding ──────
        valid_chunks = []  # (chunk_text, concept_texts, refl_list, traj_summary, traj_id, ts_id)
        embed_texts: List[str] = []
        text_to_idx: Dict[str, int] = {}

        def _register(t: str) -> None:
            if t not in text_to_idx:
                text_to_idx[t] = len(embed_texts)
                embed_texts.append(t)

        for i, (chunk_text, comp) in enumerate(zip(chunks, comps)):
            if isinstance(comp, Exception):
                print(f"    Chunk {i}: extraction failed ({comp}), skipping")
                continue

            concept_texts = comp.get("concept_texts") or []
            refl_list     = comp.get("reflection_list") or []
            traj_summary  = comp.get("trajectory_summary") or chunk_text[:80]

            if not isinstance(concept_texts, list):
                print(f"    Chunk {i}: concept_texts is not a list "
                      f"({type(concept_texts).__name__}), skipping")
                continue
            if not isinstance(refl_list, list):
                print(f"    Chunk {i}: reflection_list is not a list "
                      f"({type(refl_list).__name__}), skipping")
                continue
            if not refl_list:
                continue

            traj_id = str(uuid.uuid4())
            ts_id   = str(uuid.uuid4())
            _register(traj_summary)
            _register(" ".join(refl_list))
            for item in refl_list:
                _register(item)
            for ct in concept_texts:
                _register(ct)
            valid_chunks.append((chunk_text, concept_texts, refl_list,
                                 traj_summary, traj_id, ts_id))

        # ── Single batched embedding call ─────────────────────────────────────
        all_embs = await async_embed(embed_texts)

        def _get_emb(t: str) -> np.ndarray:
            return all_embs[text_to_idx[t]]

        # ── Pass 2: build memory structures with pre-computed embeddings ──────
        traj_ids = []
        for chunk_text, concept_texts, refl_list, traj_summary, traj_id, ts_id in valid_chunks:
            traj = Trajectory4(id=traj_id, chunk_text=chunk_text, timestamp=ts,
                               source_id=source_id, source_type=source_type)
            traj_sum = TrajectorySummary4(
                id=ts_id, text=traj_summary, embedding=_get_emb(traj_summary),
                reflection_id="", trajectory_id=traj_id,
            )
            rid = self._get_or_create_reflection(
                refl_list, ts_id,
                emb=_get_emb(" ".join(refl_list)),
                item_embs=[_get_emb(item) for item in refl_list],
            )
            traj_sum.reflection_id = rid
            traj.summary_id = ts_id

            cids = []
            for ct in concept_texts:
                cid = self._get_or_create_concept(ct, rid, emb=_get_emb(ct))
                if cid not in self.reflections[rid].concept_ids:
                    self.reflections[rid].concept_ids.append(cid)
                if cid not in traj.concept_ids:
                    traj.concept_ids.append(cid)
                cids.append(cid)

            self.conn_r2s.add(rid, ts_id)
            for cid in cids:
                self.conn_c2r.add(cid, rid)

            self.trajectories[traj_id] = traj
            self.traj_sums[ts_id]      = traj_sum
            traj_ids.append(traj_id)

        self.source_registry[source_id] = traj_ids

        if source_type == "document":
            self._docs_since_sleep.append(source_id)
        else:
            self._convs_since_sleep.append(source_id)

        print(f"    → {len(traj_ids)} chunks, "
              f"{len(self.concepts)} concepts, {len(self.reflections)} reflections")

    # ── sleep update ──────────────────────────────────────────────────────────

    def _refine_reflections(self):
        """Re-extraction pass: for every reflection, collect ALL source trajectory
        chunks that contributed to it, then ask the LLM to produce a more
        comprehensive, detail-preserving list.  Updates reflection_list in place.
        Iterates reflections (not trajectories) so each reflection is processed
        exactly once, even when it was merged from multiple trajectories.
        Runs before connection learning so improved facts propagate downstream."""
        print("  Refining reflections from raw chunks...")
        updated = 0
        for rid in tqdm(list(self.reflections.keys()), desc="  Refining reflections", leave=False):
            r = self.reflections[rid]

            # Collect the chunk text of every trajectory linked to this reflection
            chunk_texts = []
            for ts_id in r.trajectory_summary_ids:
                ts_obj = self.traj_sums.get(ts_id)
                if ts_obj:
                    traj = self.trajectories.get(ts_obj.trajectory_id)
                    if traj and traj.chunk_text:
                        chunk_texts.append(traj.chunk_text)

            if not chunk_texts:
                continue

            source_block = "\n\n---\n\n".join(chunk_texts)
            current = "\n".join(f"- {item}" for item in r.reflection_list)
            try:
                result = llm(
                    "You extract dense factual memory from conversation text. Respond in JSON.",
                    f"""Produce the most comprehensive and specific set of factual statements from the conversation below.

CONVERSATION CHUNK(S):
{source_block}

PREVIOUSLY EXTRACTED REFLECTIONS (for reference — may be incomplete or imprecise):
{current}

Using the conversation as the authoritative source, write a new reflection list. Use the previous reflections as a guide for what has already been noticed, but rewrite and expand freely. Prioritise specificity and completeness over brevity — every detail that could answer a question about the conversation should be captured. Each statement must be fully self-contained and use names rather than pronouns. Maximum 10 statements.

Return JSON with "reflection_list": list of statement strings."""
                )
                new_list = result.get("reflection_list", [])
                if new_list and isinstance(new_list, list):
                    r.reflection_list = new_list[:10]
                    r.embedding = embed([" ".join(r.reflection_list)])[0]
                    updated += 1
            except Exception as e:
                print(f"    Refinement failed for reflection {rid}: {e}")
        print(f"    Refined {updated}/{len(self.reflections)} reflections.")

    def _add_supplementary_reflections(self):
        """Supplementary-extraction pass: for every reflection, re-read ALL source
        trajectory chunks that contributed to it and ask the LLM to produce up to
        10 *additional* factual statements that the original reflection list missed.
        The new statements are appended to the existing reflection_list (duplicates
        intentionally avoided via the LLM prompt).  This is additive-only — existing
        statements are never removed or rewritten.
        Runs before connection learning so the extra facts propagate downstream."""
        print("  Adding supplementary reflections from raw chunks...")
        updated = 0
        for rid in tqdm(list(self.reflections.keys()), desc="  Supplementary reflections", leave=False):
            r = self.reflections[rid]

            # Collect the chunk text of every trajectory linked to this reflection
            chunk_texts = []
            for ts_id in r.trajectory_summary_ids:
                ts_obj = self.traj_sums.get(ts_id)
                if ts_obj:
                    traj = self.trajectories.get(ts_obj.trajectory_id)
                    if traj and traj.chunk_text:
                        chunk_texts.append(traj.chunk_text)

            if not chunk_texts:
                continue

            source_block = "\n\n---\n\n".join(chunk_texts)
            current = "\n".join(f"- {item}" for item in r.reflection_list)
            try:
                result = llm(
                    "You extract dense factual memory from conversation text. Respond in JSON.",
                    f"""You are given a conversation chunk and a list of facts already extracted from it.

CONVERSATION CHUNK(S):
{source_block}

ALREADY EXTRACTED FACTS (do NOT repeat these):
{current}

Your task: identify information present in the conversation that is NOT captured by the already-extracted facts above. Write up to 10 new, fully self-contained factual statements that fill those gaps. Each statement must use names rather than pronouns and must be specific enough to answer a question about the conversation. If there are no meaningful gaps, return an empty list.

Return JSON with "supplementary_list": list of new statement strings."""
                )
                new_items = result.get("supplementary_list", [])
                if new_items and isinstance(new_items, list):
                    r.reflection_list = r.reflection_list + [s for s in new_items[:10] if s]
                    r.embedding = embed([" ".join(r.reflection_list)])[0]
                    updated += 1
            except Exception as e:
                print(f"    Supplementary extraction failed for reflection {rid}: {e}")
        print(f"    Added supplementary statements to {updated}/{len(self.reflections)} reflections.")

    async def _add_supplementary_reflections_async(self):
        """Async version: all per-reflection LLM calls fire concurrently."""
        print("  Adding supplementary reflections from raw chunks...")
        rid_list = list(self.reflections.keys())

        async def _process_one(rid: str):
            r = self.reflections[rid]
            chunk_texts = []
            for ts_id in r.trajectory_summary_ids:
                ts_obj = self.traj_sums.get(ts_id)
                if ts_obj:
                    traj = self.trajectories.get(ts_obj.trajectory_id)
                    if traj and traj.chunk_text:
                        chunk_texts.append(traj.chunk_text)
            if not chunk_texts:
                return rid, []
            source_block = "\n\n---\n\n".join(chunk_texts)
            current      = "\n".join(f"- {item}" for item in r.reflection_list)
            result = await async_llm(
                async_client,
                "You extract dense factual memory from conversation text. Respond in JSON.",
                f"You are given a conversation chunk and a list of facts already extracted from it.\n\n"
                f"CONVERSATION CHUNK(S):\n{source_block}\n\n"
                f"ALREADY EXTRACTED FACTS (do NOT repeat these):\n{current}\n\n"
                "Your task: identify information present in the conversation that is NOT captured by "
                "the already-extracted facts above. Write up to 10 new, fully self-contained factual "
                "statements that fill those gaps. Each statement must use names rather than pronouns "
                "and must be specific enough to answer a question about the conversation. If there are "
                "no meaningful gaps, return an empty list.\n\n"
                'Return JSON with "supplementary_list": list of new statement strings.',
            )
            new_items = result.get("supplementary_list", [])
            return rid, new_items if isinstance(new_items, list) else []

        all_results = await asyncio.gather(
            *[_process_one(rid) for rid in rid_list],
            return_exceptions=True,
        )

        # Collect valid updates, then batch-embed all changed reflection texts
        pending: List[Tuple[str, List[str]]] = []  # (rid, valid_new_items)
        for result in all_results:
            if isinstance(result, Exception):
                continue
            rid, new_items = result
            if new_items and isinstance(new_items, list):
                valid_new = [s for s in new_items[:10] if s]
                if valid_new:
                    pending.append((rid, valid_new))

        joined_texts = [
            " ".join(self.reflections[rid].reflection_list + new_items)
            for rid, new_items in pending
        ]
        # Deduplicate new individual items for a single combined embed call
        all_new_items_flat: List[str] = []
        new_item_idx: Dict[str, int] = {}
        for _rid, new_items in pending:
            for item in new_items:
                if item not in new_item_idx:
                    new_item_idx[item] = len(all_new_items_flat)
                    all_new_items_flat.append(item)

        combined_texts = joined_texts + all_new_items_flat
        combined_embs = await async_embed(combined_texts)
        joined_embs   = combined_embs[:len(joined_texts)]
        item_embs_arr = combined_embs[len(joined_texts):]

        updated = 0
        for i, (rid, new_items) in enumerate(pending):
            r = self.reflections[rid]
            r.reflection_list = r.reflection_list + new_items
            r.embedding = joined_embs[i]
            r.item_embeddings = r.item_embeddings + [item_embs_arr[new_item_idx[item]] for item in new_items]
            updated += 1
        print(f"    Added supplementary statements to {updated}/{len(rid_list)} reflections.")

    def _enrich_reflections(self):
        """Two-pass enrichment for every reflection.

        Pass 1 — selective update: the LLM reviews the source chunks and the
        existing reflection_list together and identifies which individual statements
        are missing important details.  Only those statements are rewritten in-place
        with the added detail; others are left untouched.

        Pass 2 — supplementary additions: using the now-updated reflection_list as
        the 'already known' baseline, the LLM extracts up to 10 brand-new statements
        that cover information still not represented.

        Runs before connection learning so all enriched facts propagate downstream."""
        print("  Enriching reflections (update existing + add new)...")
        updated = 0
        for rid in tqdm(list(self.reflections.keys()), desc="  Enriching reflections", leave=False):
            r = self.reflections[rid]

            # Collect source chunk texts for this reflection
            chunk_texts = []
            for ts_id in r.trajectory_summary_ids:
                ts_obj = self.traj_sums.get(ts_id)
                if ts_obj:
                    traj = self.trajectories.get(ts_obj.trajectory_id)
                    if traj and traj.chunk_text:
                        chunk_texts.append(traj.chunk_text)

            if not chunk_texts:
                continue

            source_block = "\n\n---\n\n".join(chunk_texts)
            changed = False

            # ── Pass 1: selectively update existing statements ─────────────
            numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(r.reflection_list))
            try:
                result = llm(
                    "You extract dense factual memory from conversation text. Respond in JSON.",
                    f"""You are given a conversation and a numbered list of facts already extracted from it.

CONVERSATION CHUNK(S):
{source_block}

EXISTING FACTS (0-indexed):
{numbered}

Your task: identify which of the existing facts above are missing important specific details that are present in the conversation (e.g. names, numbers, dates, outcomes left vague). For each such fact, provide a rewritten version that incorporates the missing detail while preserving what was already correct. Do not rewrite facts that are already sufficiently specific. If no facts need updating, return an empty list.

Return JSON with "updates": list of objects with "index" (int, 0-based) and "updated_statement" (str).

Example output structure: {{"updates": [{{"index": ..., "updated_statement": "..."}}]}}"""
                )
                updates = result.get("updates", [])
                if updates and isinstance(updates, list):
                    for entry in updates:
                        idx = entry.get("index")
                        new_stmt = entry.get("updated_statement", "")
                        if isinstance(idx, int) and 0 <= idx < len(r.reflection_list) and new_stmt:
                            r.reflection_list[idx] = new_stmt
                            changed = True
            except Exception as e:
                print(f"    Pass-1 update failed for reflection {rid}: {e}")

            # ── Pass 2: add new statements not yet covered ─────────────────
            current = "\n".join(f"- {item}" for item in r.reflection_list)
            try:
                result = llm(
                    "You extract dense factual memory from conversation text. Respond in JSON.",
                    f"""You are given a conversation and the current list of facts extracted from it (some may have just been updated).

CONVERSATION CHUNK(S):
{source_block}

CURRENT FACTS (do NOT repeat these):
{current}

Your task: identify information present in the conversation that is still NOT captured by the current facts above. Write up to 10 new, fully self-contained factual statements that fill those gaps. Each statement must use names rather than pronouns and be specific enough to answer a question about the conversation. If there are no meaningful gaps, return an empty list.

Return JSON with "supplementary_list": list of new statement strings."""
                )
                new_items = result.get("supplementary_list", [])
                if new_items and isinstance(new_items, list):
                    r.reflection_list = r.reflection_list + [s for s in new_items[:10] if s]
                    changed = True
            except Exception as e:
                print(f"    Pass-2 addition failed for reflection {rid}: {e}")

            if changed:
                r.embedding = embed([" ".join(r.reflection_list)])[0]
                updated += 1

        print(f"    Enriched {updated}/{len(self.reflections)} reflections.")

    def sleep_update(self, n_questions_per_chunk: int = 3, top_k: int = 5,
                     refine_reflections: bool = True, build_rubrics: bool = True):
        """Connection learning (top_gold_only) + persona + rubric refresh.
        refine_reflections: if True (default), run a re-extraction pass over all
        trajectories before connection learning to improve reflection quality.
        build_rubrics: if False, skip rubric detection/extraction (saves ~2 LLM calls
        per document; useful when rubrics are not used downstream)."""
        print("\n[Sleep Update]")

        # 0. Optional reflection refinement pass (before connection learning)
        if refine_reflections:
            #self._refine_reflections()
            self._add_supplementary_reflections()

        # 1. Connection learning over all trajectories
        self._run_connection_learning(n_questions_per_chunk, top_k)

        # 2. Persona update from new conversations
        if self._convs_since_sleep:
            print(f"  Updating persona from {len(self._convs_since_sleep)} conversation(s)...")
            conv_texts = []
            for src_id in self._convs_since_sleep:
                for tid in self.source_registry.get(src_id, []):
                    if tid in self.trajectories:
                        conv_texts.append(self.trajectories[tid].chunk_text)
            if conv_texts:
                self.persona.entries = self._extract_persona_update(conv_texts, self.persona.entries)
                self.persona.last_updated = datetime.now(timezone.utc).isoformat()
                for _name, _entry in self.persona.entries.items():
                    print(f"    Persona [{_name}]: {_entry.summary}")

        # 3. Rubric creation / refresh for new documents
        if build_rubrics and self._docs_since_sleep:
            print(f"  Building rubrics for {len(self._docs_since_sleep)} new document(s)...")
            for src_id in self._docs_since_sleep:
                traj_ids = self.source_registry.get(src_id, [])
                if not traj_ids:
                    continue
                sample_text = " ".join(
                    self.trajectories[tid].chunk_text
                    for tid in traj_ids[:3] if tid in self.trajectories
                )
                doc_type = self._detect_doc_type(sample_text)
                new_inst = self._extract_rubric(sample_text, doc_type)
                if new_inst:
                    if doc_type not in self.rubrics:
                        self.rubrics[doc_type] = TaskRubric(doc_type=doc_type, instructions=new_inst)
                        print(f"    New rubric created: '{doc_type}'")
                    else:
                        self.rubrics[doc_type].instructions = new_inst
                        print(f"    Rubric refreshed: '{doc_type}'")

        self._docs_since_sleep.clear()
        self._convs_since_sleep.clear()
        print("  Sleep update done.")

    async def sleep_update_async(self, n_questions_per_chunk: int = 3, top_k: int = 5,
                                 refine_reflections: bool = True, build_rubrics: bool = True,
                                 use_query_components: bool = True):
        """Async version of sleep_update: LLM calls within every stage are batched.
        build_rubrics: if False, skip rubric detection/extraction (saves ~2 LLM calls
        per document; useful when rubrics are not used downstream).
        use_query_components: if False, skip LLM query-component generation during
        connection learning and use the raw question text directly."""
        print("\n[Sleep Update]")

        if refine_reflections:
            await self._add_supplementary_reflections_async()

        await self._run_connection_learning_async(n_questions_per_chunk, top_k,
                                                  use_query_components=use_query_components)

        # Persona update (synchronous; only relevant for conversation sources)
        if self._convs_since_sleep:
            print(f"  Updating persona from {len(self._convs_since_sleep)} conversation(s)...")
            conv_texts = []
            for src_id in self._convs_since_sleep:
                for tid in self.source_registry.get(src_id, []):
                    if tid in self.trajectories:
                        conv_texts.append(self.trajectories[tid].chunk_text)
            if conv_texts:
                self.persona.entries = self._extract_persona_update(conv_texts, self.persona.entries)
                self.persona.last_updated = datetime.now(timezone.utc).isoformat()
                for _name, _entry in self.persona.entries.items():
                    print(f"    Persona [{_name}]: {_entry.summary}")

        if build_rubrics and self._docs_since_sleep:
            await self._build_rubrics_async()

        self._docs_since_sleep.clear()
        self._convs_since_sleep.clear()
        print("  Sleep update done.")

    async def _build_rubrics_async(self):
        """Batch all doc-type detection + rubric extraction calls for new documents."""
        print(f"  Building rubrics for {len(self._docs_since_sleep)} new document(s)...")
        standard_types = ["research_paper", "literature_review", "clinical_report",
                          "book_chapter", "conversation", "technical_report"]

        # Collect sample texts per source
        src_samples: Dict[str, str] = {}
        for src_id in self._docs_since_sleep:
            traj_ids = self.source_registry.get(src_id, [])
            if traj_ids:
                src_samples[src_id] = " ".join(
                    self.trajectories[tid].chunk_text
                    for tid in traj_ids[:3] if tid in self.trajectories
                )

        if not src_samples:
            return

        async def _detect_type(src_id: str, sample: str):
            result = await async_llm(
                async_client,
                "You classify documents. Respond in JSON.",
                f"What type of document is this?\n\n"
                f"Preferred types: {standard_types}\n\n"
                "Use one of the preferred types if it fits well. If none are a good match, "
                'invent a concise snake_case label (e.g. "grant_application", "case_study", "protocol").\n'
                'Do NOT use "other".\n\n'
                f"TEXT SAMPLE (first 2000 chars):\n{sample[:2000]}\n\n"
                f'Return ONLY JSON like {{"doc_type": "research_paper"}}.',
            )
            detected = result.get("doc_type", "").strip().lower().replace(" ", "_").replace("-", "_")
            return src_id, detected if detected and detected != "other" else "research_paper"

        type_results = await asyncio.gather(
            *[_detect_type(s, t) for s, t in src_samples.items()],
            return_exceptions=True,
        )
        src_types: Dict[str, str] = {}
        for r in type_results:
            if not isinstance(r, Exception):
                src_types[r[0]] = r[1]

        async def _extract_rubric_a(src_id: str, doc_type: str):
            sample = src_samples[src_id]
            result = await async_llm(
                async_client,
                "You write concise task rubrics. Respond in JSON.",
                f"Based on the structure of this {doc_type}, write a short rubric (≤240 words) "
                "that tells a system how to format and organise a response when answering questions "
                "about this type of document.\n\n"
                f"TEXT SAMPLE:\n{sample[:1200]}\n\n"
                'Return JSON with "instructions": the rubric text.\n'
                "Focus on: output structure, level of detail, key considerations for this doc type.",
            )
            return src_id, doc_type, result.get("instructions", "")

        rubric_results = await asyncio.gather(
            *[_extract_rubric_a(s, t) for s, t in src_types.items()],
            return_exceptions=True,
        )

        for r in rubric_results:
            if isinstance(r, Exception):
                continue
            src_id, doc_type, new_inst = r
            if new_inst:
                if doc_type not in self.rubrics:
                    self.rubrics[doc_type] = TaskRubric(doc_type=doc_type, instructions=new_inst)
                    print(f"    New rubric created: '{doc_type}'")
                else:
                    self.rubrics[doc_type].instructions = new_inst
                    print(f"    Rubric refreshed: '{doc_type}'")

    async def _run_connection_learning_async(self, n_q: int, top_k: int,
                                              use_query_components: bool = True):
        """Async version: question-gen and query-component calls are all batched.
        use_query_components: if False, skip LLM query-component generation and use
        the raw question as both concept_texts and predicted_reflections."""
        print(f"  Running connection learning ({n_q} q/chunk)...")
        traj_ids = list(self.trajectories.keys())

        # Stage 1 — generate questions for every trajectory concurrently
        async def _gen_qs(tid: str):
            traj = self.trajectories[tid]
            try:
                result = await async_llm(
                    async_client,
                    "You generate factual questions from text. Respond in JSON.",
                    f"Generate {n_q} specific factual questions answerable from this text.\n\n"
                    f"TEXT:\n{traj.chunk_text}\n\n"
                    'Return JSON with "questions": list of question strings.',
                )
                return tid, result.get("questions", [])[:n_q]
            except Exception:
                return tid, []

        q_results = await asyncio.gather(
            *[_gen_qs(tid) for tid in traj_ids],
            return_exceptions=True,
        )

        # Build flat (tid, question) list + per-trajectory gold-reflection sets
        all_pairs: List[Tuple[str, str]] = []
        gold_r_map: Dict[str, Set[str]] = {}
        for result in q_results:
            if isinstance(result, Exception):
                continue
            tid, questions = result
            traj   = self.trajectories[tid]
            gold_r: Set[str] = set()
            if traj.summary_id and traj.summary_id in self.traj_sums:
                gold_r.add(self.traj_sums[traj.summary_id].reflection_id)
            gold_r_map[tid] = gold_r
            for q in questions:
                all_pairs.append((tid, q))

        # Stage 2 — generate query components for every question concurrently
        if use_query_components:
            async def _gen_qc(tid: str, q: str):
                try:
                    result = await async_llm(
                        async_client,
                        "You generate memory search components. Respond in JSON.",
                        "For this question, generate search components for a hierarchical memory system.\n\n"
                        f"Question: {q}\n\n"
                        "Return JSON with:\n"
                        '- "concept_texts": list of 2-4 short topic labels (1-4 words) relevant to this question\n'
                        '- "predicted_reflections": list of 3-5 diverse statements, each targeting a different aspect or event relevant to this question\n'
                        '- "predicted_summary": one sentence describing a chunk that would answer this question',
                    )
                    concept_texts         = result.get("concept_texts")
                    predicted_reflections = result.get("predicted_reflections")
                    predicted_summary     = result.get("predicted_summary")
                    if not isinstance(concept_texts, list) or not concept_texts:
                        concept_texts = [q[:40]]
                    if not isinstance(predicted_reflections, list) or not predicted_reflections:
                        predicted_reflections = [q]
                    if predicted_summary is not None and not isinstance(predicted_summary, str):
                        predicted_summary = None
                    return tid, QueryComponents4(
                        concept_texts=concept_texts,
                        predicted_reflections=predicted_reflections,
                        predicted_summary=predicted_summary,
                    )
                except Exception:
                    return tid, QueryComponents4(concept_texts=[q[:40]], predicted_reflections=[q])

            qc_results = await asyncio.gather(
                *[_gen_qc(tid, q) for tid, q in all_pairs],
                return_exceptions=True,
            )
        else:
            qc_results = [
                (tid, QueryComponents4(concept_texts=[q], predicted_reflections=[q]))
                for tid, q in all_pairs
            ]

        # ── Pre-batch all query embeddings before Stage 3 ────────────────────
        unique_ct: List[str] = []; ct_idx: Dict[str, int] = {}
        unique_pr: List[str] = []; pr_idx: Dict[str, int] = {}
        unique_ri: List[str] = []; ri_idx: Dict[str, int] = {}
        for res in qc_results:
            if isinstance(res, Exception):
                continue
            _, qc = res
            for t in qc.concept_texts:
                if t not in ct_idx:
                    ct_idx[t] = len(unique_ct); unique_ct.append(t)
            for t in qc.predicted_reflections:
                if t not in pr_idx:
                    pr_idx[t] = len(unique_pr); unique_pr.append(t)
        for r in self.reflections.values():
            for item in r.reflection_list:
                if item not in ri_idx:
                    ri_idx[item] = len(unique_ri); unique_ri.append(item)

        # Single embed call: ct + pr + ri concatenated — one round-trip instead of three
        _all_embed_texts = unique_ct + unique_pr + unique_ri
        _all_embs = await async_embed(_all_embed_texts) if _all_embed_texts else np.array([])
        n_ct, n_pr = len(unique_ct), len(unique_pr)
        ct_embs_arr = _all_embs[:n_ct]
        pr_embs_arr = _all_embs[n_ct:n_ct + n_pr]
        ri_embs_arr = _all_embs[n_ct + n_pr:]
        ct_emb_map: Dict[str, np.ndarray] = {t: ct_embs_arr[i] for t, i in ct_idx.items()}
        pr_emb_map: Dict[str, np.ndarray] = {t: pr_embs_arr[i] for t, i in pr_idx.items()}
        ri_emb_map: Dict[str, np.ndarray] = {t: ri_embs_arr[i] for t, i in ri_idx.items()}

        # Stage 3 — connection learning using pre-batched embeddings (no sync embed calls)
        for (tid, _q), result in zip(all_pairs, qc_results):
            if isinstance(result, Exception):
                continue
            _, qc = result
            gold_r = gold_r_map.get(tid, set())
            qc_ct_embs = (np.array([ct_emb_map[t] for t in qc.concept_texts])
                          if qc.concept_texts else None)
            qc_pr_embs = (np.array([pr_emb_map[t] for t in qc.predicted_reflections])
                          if qc.predicted_reflections else None)
            mc, mc_all = self._match_layer_concepts(qc.concept_texts, top_k, embs=qc_ct_embs)
            mr, mr_all = self._match_layer_reflections(
                qc.predicted_reflections, set(mc.keys()), top_k,
                embs=qc_pr_embs, text_emb_cache=ri_emb_map,
            )
            for cid in mc:
                for rid in self.conn_c2r.get(cid):
                    self.conn_c2r.record(cid, rid, rid in gold_r)
            self.conn_c2r.learn_top_gold_only(set(mc), gold_r, mr_all)

        print(f"    Connections after learning — "
              f"C→R: {self.conn_c2r.total_connections()}, "
              f"R→S: {self.conn_r2s.total_connections()} (static — R→S learning disabled)")

    def _run_connection_learning(self, n_q: int, top_k: int):
        print(f"  Running connection learning ({n_q} q/chunk)...")
        traj_ids = list(self.trajectories.keys())
        for tid in tqdm(traj_ids, desc="  Connection learning", leave=False):
            traj = self.trajectories[tid]
            questions = self._generate_questions(traj.chunk_text, n_q)
            gold_r  = set()
            if traj.summary_id and traj.summary_id in self.traj_sums:
                gold_r.add(self.traj_sums[traj.summary_id].reflection_id)

            for q in questions:
                qc = self._generate_query_components(q)
                mc, mc_all = self._match_layer_concepts(qc.concept_texts, top_k)
                mr, mr_all = self._match_layer_reflections(qc.predicted_reflections,
                                                            set(mc.keys()), top_k)

                # record traversals for existing C→R connections
                for cid in mc:
                    for rid in self.conn_c2r.get(cid):
                        self.conn_c2r.record(cid, rid, rid in gold_r)

                # learn: top_gold_only — connect matched concepts to highest-scored gold reflection
                self.conn_c2r.learn_top_gold_only(set(mc), gold_r, mr_all)

        print(f"    Connections after learning — "
              f"C→R: {self.conn_c2r.total_connections()}, "
              f"R→S: {self.conn_r2s.total_connections()} (static — R→S learning disabled)")

    def _generate_questions(self, chunk_text: str, n: int) -> List[str]:
        try:
            result = llm(
                "You generate factual questions from text. Respond in JSON.",
                f"""Generate {n} specific factual questions answerable from this text.

TEXT:
{chunk_text}

Return JSON with "questions": list of question strings."""
            )
            return result.get("questions", [])[:n]
        except Exception:
            return []

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _generate_query_components(self, question: str) -> QueryComponents4:
        try:
            result = llm(
                "You generate memory search components. Respond in JSON.",
                f"""For this question, generate search components for a hierarchical memory system.

Question: {question}

Return JSON with:
- "concept_texts": list of 2-4 short topic labels (1-4 words) relevant to this question
- "predicted_reflections": list of 3-5 diverse statements, each targeting a different aspect or event relevant to this question
- "predicted_summary": one sentence describing a chunk that would answer this question"""
            )
            concept_texts        = result.get("concept_texts")
            predicted_reflections = result.get("predicted_reflections")
            predicted_summary    = result.get("predicted_summary")

            if not isinstance(concept_texts, list) or not concept_texts:
                if concept_texts is not None:
                    print(f"    [query_components] concept_texts invalid ({type(concept_texts).__name__!r}: {concept_texts!r}), using fallback")
                concept_texts = [question[:40]]

            if not isinstance(predicted_reflections, list) or not predicted_reflections:
                if predicted_reflections is not None:
                    print(f"    [query_components] predicted_reflections invalid ({type(predicted_reflections).__name__!r}: {predicted_reflections!r}), using fallback")
                predicted_reflections = [question]

            if predicted_summary is not None and not isinstance(predicted_summary, str):
                print(f"    [query_components] predicted_summary invalid ({type(predicted_summary).__name__!r}: {predicted_summary!r}), using None")
                predicted_summary = None

            return QueryComponents4(
                concept_texts=concept_texts,
                predicted_reflections=predicted_reflections,
                predicted_summary=predicted_summary,
            )
        except Exception:
            return QueryComponents4(concept_texts=[question[:40]],
                                    predicted_reflections=[question])

    def _match_layer_concepts(self, texts: List[str], top_k: int,
                               embs: Optional[np.ndarray] = None,
                               ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns (top_k_scores, all_scores). all_scores covers every concept for gold lookup."""
        if not texts or not self.concepts:
            return {}, {}
        if embs is None:
            embs = embed(texts)
        cids = list(self.concepts.keys())
        cembs = np.array([self.concepts[c].embedding for c in cids])
        # max similarity across all query texts for each concept
        sims = np.max(cos_sim(embs, cembs), axis=0)
        all_scores = {cids[i]: float(sims[i]) for i in range(len(cids))}
        idx = np.argsort(sims)[-top_k:][::-1]
        top_scores = {cids[i]: float(sims[i]) for i in idx}
        return top_scores, all_scores

    def _bm25_score_items(self, queries: List[str], items: List[str]) -> np.ndarray:
        """Return a (len(queries), len(items)) matrix of BM25 scores."""
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

    def _match_layer_reflections(self, pred_refls: List[str],
                                  candidate_cids: Set[str], top_k: int,
                                  embs: Optional[np.ndarray] = None,
                                  text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
                                  ) -> Tuple[Dict[str, float], Dict[str, float]]:
        # candidates: reflections reachable from candidate concepts
        cand_rids: Set[str] = set()
        for cid in candidate_cids:
            cand_rids |= self.conn_c2r.get(cid)
        if not cand_rids or not pred_refls:
            return {}, {}

        # per-item embedding similarity with top-2 average aggregation
        if embs is None:
            embs = embed(pred_refls)
        rid_list = list(cand_rids)
        all_items, item_to_rid = [], []
        for rid in rid_list:
            for item in self.reflections[rid].reflection_list:
                all_items.append(item)
                item_to_rid.append(rid)
        if not all_items:
            return {}, {}
        # Use stored per-item embeddings when available (avoids any embed call)
        can_use_stored = all(
            len(self.reflections[rid].item_embeddings) == len(self.reflections[rid].reflection_list)
            for rid in rid_list
        )
        if can_use_stored:
            stored: List[np.ndarray] = []
            for rid in rid_list:
                stored.extend(self.reflections[rid].item_embeddings)
            item_embs = np.array(stored)
        elif text_emb_cache is not None:
            cached = [text_emb_cache.get(item) for item in all_items]
            item_embs = (np.array(cached) if all(e is not None for e in cached)
                         else embed(all_items))
        else:
            item_embs = embed(all_items)
        sem_sims = np.max(cos_sim(embs, item_embs), axis=0)  # (n_items,)

        # BM25 scores per item: max over pred_refls, then top-2 avg per reflection
        bm25_raw = self._bm25_score_items(pred_refls, all_items)  # (n_queries, n_items)
        bm25_sims = np.max(bm25_raw, axis=0)  # (n_items,) — best query match per item
        # normalise BM25 to [0, 1]
        bm25_max = float(bm25_sims.max())
        bm25_norm = bm25_sims / bm25_max if bm25_max > 0 else bm25_sims

        # combined score: 60% semantic + 40% BM25
        combined = 0.6 * sem_sims + 0.4 * bm25_norm

        # aggregate per reflection: top-2 average of combined scores
        rid_scores: Dict[str, List[float]] = defaultdict(list)
        for s, rid in zip(combined, item_to_rid):
            rid_scores[rid].append(float(s))
        scores = {}
        for rid, s_list in rid_scores.items():
            top2 = sorted(s_list, reverse=True)[:2]
            scores[rid] = float(np.mean(top2))
        # top-k
        top_rids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        result = {r: scores[r] for r in top_rids}
        return result, scores

    def _match_layer_summaries(self, pred_summary: Optional[str],
                                candidate_rids: Set[str], top_k: int,
                                emb: Optional[np.ndarray] = None,
                                ) -> Tuple[Dict[str, float], Dict[str, float]]:
        cand_sids: Set[str] = set()
        for rid in candidate_rids:
            cand_sids |= self.conn_r2s.get(rid)
        if not cand_sids:
            return {}, {}
        if not pred_summary:
            all_scores = {sid: 1.0 for sid in cand_sids}
            top = list(cand_sids)[:top_k]
            return {s: 1.0 for s in top}, all_scores
        if emb is None:
            emb = embed([pred_summary])[0]
        sid_list = list(cand_sids)
        sembs = np.array([self.traj_sums[s].embedding for s in sid_list if s in self.traj_sums])
        valid = [s for s in sid_list if s in self.traj_sums]
        if not valid:
            return {}, {}
        sims = cos_sim(emb.reshape(1, -1), sembs)[0]
        scores = {valid[i]: float(sims[i]) for i in range(len(valid))}
        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return {s: scores[s] for s in top}, scores

    def validate_integrity(self) -> Dict[str, List[str]]:
        """Check structural invariants of the memory store.

        Returns a dict of {check_name: [list of problem descriptions]}.
        An empty list for a check means it passed. Prints a summary.

        Invariants checked:
        - Every reflection has a non-empty reflection_list
        - Every reflection is reachable from at least one concept via conn_c2r
        - Every concept links to at least one reflection via conn_c2r
        - Every trajectory_summary has a valid trajectory_id and reflection_id
        - Every reflection has at least one trajectory_summary_id pointing to an existing traj_sum
        - Every concept has a non-empty text and an embedding
        - No dangling conn_c2r edges (references to missing concept or reflection IDs)
        """
        issues: Dict[str, List[str]] = {
            "empty_reflection_list":      [],
            "reflection_unreachable":     [],
            "concept_no_reflection_edge": [],
            "traj_sum_bad_trajectory":    [],
            "traj_sum_bad_reflection":    [],
            "reflection_bad_traj_sum":    [],
            "concept_empty_text":         [],
            "dangling_conn_c2r":          [],
        }

        # reflections reachable from concepts via conn_c2r
        reachable_rids: Set[str] = set()
        for cid, tgt_set in self.conn_c2r.connections.items():
            reachable_rids |= tgt_set

        for rid, r in self.reflections.items():
            if not r.reflection_list:
                issues["empty_reflection_list"].append(rid)
            if rid not in reachable_rids:
                issues["reflection_unreachable"].append(rid)
            live_ts = [ts for ts in r.trajectory_summary_ids if ts in self.traj_sums]
            if not live_ts:
                issues["reflection_bad_traj_sum"].append(
                    f"{rid}: trajectory_summary_ids={r.trajectory_summary_ids!r}")

        for cid, concept in self.concepts.items():
            if not concept.text or not concept.text.strip():
                issues["concept_empty_text"].append(cid)
            if not self.conn_c2r.get(cid):
                issues["concept_no_reflection_edge"].append(cid)

        for ts_id, ts in self.traj_sums.items():
            if ts.trajectory_id not in self.trajectories:
                issues["traj_sum_bad_trajectory"].append(
                    f"{ts_id}: trajectory_id={ts.trajectory_id!r}")
            if ts.reflection_id not in self.reflections:
                issues["traj_sum_bad_reflection"].append(
                    f"{ts_id}: reflection_id={ts.reflection_id!r}")

        for cid, tgt_set in self.conn_c2r.connections.items():
            if cid not in self.concepts:
                issues["dangling_conn_c2r"].append(f"src concept {cid!r} missing")
            for rid in tgt_set:
                if rid not in self.reflections:
                    issues["dangling_conn_c2r"].append(f"tgt reflection {rid!r} missing (src={cid!r})")

        # summary
        total = sum(len(v) for v in issues.values())
        print(f"\n[validate_integrity] {len(self.concepts)} concepts, "
              f"{len(self.reflections)} reflections, "
              f"{len(self.traj_sums)} traj_sums, "
              f"{len(self.trajectories)} trajectories")
        if total == 0:
            print("  All checks PASSED.")
        else:
            print(f"  {total} issue(s) found:")
            for check, problems in issues.items():
                if problems:
                    print(f"  [{check}] {len(problems)} problem(s):")
                    for p in problems[:5]:   # show up to 5 examples
                        print(f"    - {p}")
                    if len(problems) > 5:
                        print(f"    … and {len(problems) - 5} more")
        return issues

    async def async_retrieve(self, question: str, top_k: int = 5,
                              use_query_components: bool = True
                              ) -> Tuple[Dict, Dict, Dict, Dict, QueryComponents4]:
        """Async version of retrieve: async LLM for query components + batched async embeds.

        _match_layer_reflections uses stored per-item embeddings (populated at ingest time),
        so no embed call is needed at query time for the memory side.
        use_query_components: if False, skip LLM query-component generation and use
        the raw question as both concept_texts and predicted_reflections.
        """
        if use_query_components:
            result = await async_llm(
                async_client,
                "You generate memory search components. Respond in JSON.",
                f"For this question, generate search components for a hierarchical memory system.\n\n"
                f"Question: {question}\n\n"
                "Return JSON with:\n"
                '- "concept_texts": list of 2-4 short topic labels (1-4 words) relevant to this question\n'
                '- "predicted_reflections": list of 3-5 diverse statements, each targeting a different aspect or event relevant to this question\n'
                '- "predicted_summary": one sentence describing a chunk that would answer this question',
            )

            try:
                concept_texts         = result.get("concept_texts")
                predicted_reflections = result.get("predicted_reflections")
                predicted_summary     = result.get("predicted_summary")
                if not isinstance(concept_texts, list) or not concept_texts:
                    concept_texts = [question[:40]]
                if not isinstance(predicted_reflections, list) or not predicted_reflections:
                    predicted_reflections = [question]
                if predicted_summary is not None and not isinstance(predicted_summary, str):
                    predicted_summary = None
            except Exception:
                concept_texts, predicted_reflections, predicted_summary = [question], [question], None
        else:
            concept_texts, predicted_reflections, predicted_summary = [question], [question], None

        qc = QueryComponents4(concept_texts=concept_texts,
                              predicted_reflections=predicted_reflections,
                              predicted_summary=predicted_summary)

        # Batch embed all query-side texts in one async call
        batch = list(concept_texts) + list(predicted_reflections)
        summary_idx = None
        if predicted_summary:
            summary_idx = len(batch)
            batch.append(predicted_summary)
        all_embs = await async_embed(batch)
        ct_embs = all_embs[:len(concept_texts)]
        pr_embs = all_embs[len(concept_texts):len(concept_texts) + len(predicted_reflections)]
        s_emb   = all_embs[summary_idx] if summary_idx is not None else None

        mc, _ = self._match_layer_concepts(concept_texts, top_k, embs=ct_embs)
        mr, _ = self._match_layer_reflections(predicted_reflections, set(mc), top_k, embs=pr_embs)
        ms, _ = self._match_layer_summaries(predicted_summary, set(mr), top_k, emb=s_emb)
        mt = {self.traj_sums[sid].trajectory_id: 1.0
              for sid in ms if sid in self.traj_sums}
        return mc, mr, ms, mt, qc

    def retrieve(self, question: str, top_k: int = 5
                  ) -> Tuple[Dict, Dict, Dict, Dict, QueryComponents4]:
        """Return (concept_scores, reflection_scores, summary_scores, traj_scores, qc)."""
        qc = self._generate_query_components(question)
        mc, _ = self._match_layer_concepts(qc.concept_texts, top_k)
        mr, _ = self._match_layer_reflections(qc.predicted_reflections, set(mc), top_k)
        ms, _ = self._match_layer_summaries(qc.predicted_summary, set(mr), top_k)
        # get trajectories from summaries
        mt = {self.traj_sums[sid].trajectory_id: 1.0
              for sid in ms if sid in self.traj_sums}
        return mc, mr, ms, mt, qc

    # ── synthesis ─────────────────────────────────────────────────────────────

    def _select_rubric(self, question: str) -> TaskRubric:
        """Always returns a rubric; falls back to general_qa."""
        if len(self.rubrics) == 1:
            return next(iter(self.rubrics.values()))
        rubric_types = list(self.rubrics.keys())
        try:
            result = llm(
                "You select the most appropriate rubric type. Respond in JSON.",
                f"""Question: {question}

Available rubric types: {rubric_types}

Which rubric type best fits this question? Use "general_qa" for simple factual questions that don't need structured document-style output. Return JSON with "selected": one of the rubric type strings."""
            )
            sel = result.get("selected", "general_qa")
            return self.rubrics.get(sel) or self.rubrics["general_qa"]
        except Exception:
            return self.rubrics["general_qa"]

    def query(self, question: str, top_k: int = 5) -> str:
        """Full pipeline: retrieve → select rubric → synthesise."""
        mc, mr, ms, mt, qc = self.retrieve(question, top_k)

        # gather content
        reflections_text = []
        for rid in mr:
            r = self.reflections.get(rid)
            if r:
                reflections_text.extend(r.reflection_list)

        traj_texts = []
        for tid in list(mt)[:3]:
            t = self.trajectories.get(tid)
            if t:
                traj_texts.append(t.chunk_text)

        rubric = self._select_rubric(question)
        rubric_instructions = rubric.instructions

        persona_note = ""
        if self.persona.entries:
            try:
                _q_emb = embed([question])[0]
                _best_name, _best_score = None, -1.0
                for _pname, _pentry in self.persona.entries.items():
                    if _pentry.embedding is not None and len(_pentry.embedding) > 0:
                        _score = float(cos_sim(_q_emb.reshape(1, -1), _pentry.embedding.reshape(1, -1))[0, 0])
                        if _score > _best_score:
                            _best_score, _best_name = _score, _pname
                if _best_name:
                    _e = self.persona.entries[_best_name]
                    persona_note = f"\nUser context [{_best_name}]: {_e.summary}"
            except Exception:
                pass

        content_block = "\n".join(f"- {r}" for r in reflections_text[:12])
        if traj_texts:
            content_block += "\n\nSupporting passages:\n" + \
                             "\n---\n".join(traj_texts)

        result = llm(
            "You synthesise information from a memory system into a helpful answer. Respond in JSON.",
            f"""Question: {question}{persona_note}

Retrieved memory items:
{content_block}

Output instructions: {rubric_instructions}

Synthesise an answer. Return JSON with "answer": the synthesised response as a plain string."""
        )
        raw = result.get("answer", "No relevant information found.")
        # Guard: LLM sometimes returns a nested dict instead of a plain string
        answer = raw if isinstance(raw, str) else json.dumps(raw)

        return answer

    # ── evaluation ───────────────────────────────────────────────────────────

    def get_gold_items(self, source_id: str) -> Dict[str, Set[str]]:
        """Gold layer items for all trajectories from a source (document-level)."""
        gold: Dict[str, Set[str]] = {k: set() for k in
                                      ["concepts", "reflections", "trajectory_summaries", "trajectories"]}
        for tid in self.source_registry.get(source_id, []):
            traj = self.trajectories.get(tid)
            if not traj:
                continue
            gold["trajectories"].add(tid)
            if traj.summary_id:
                gold["trajectory_summaries"].add(traj.summary_id)
                if traj.summary_id in self.traj_sums:
                    gold["reflections"].add(self.traj_sums[traj.summary_id].reflection_id)
            gold["concepts"].update(traj.concept_ids)
        return gold

    def get_gold_for_trajectory(self, trajectory_id: str) -> Dict[str, Set[str]]:
        """Gold layer items for a single trajectory (chunk-level)."""
        gold: Dict[str, Set[str]] = {k: set() for k in
                                      ["concepts", "reflections", "trajectory_summaries", "trajectories"]}
        traj = self.trajectories.get(trajectory_id)
        if not traj:
            return gold
        gold["trajectories"].add(trajectory_id)
        if traj.summary_id:
            gold["trajectory_summaries"].add(traj.summary_id)
            if traj.summary_id in self.traj_sums:
                gold["reflections"].add(self.traj_sums[traj.summary_id].reflection_id)
        gold["concepts"].update(traj.concept_ids)
        return gold

    def get_gold_for_trajectories(self, trajectory_ids: List[str]) -> Dict[str, Set[str]]:
        """Union of gold layer items across multiple trajectories (e.g. multi-hop evidence)."""
        merged: Dict[str, Set[str]] = {k: set() for k in
                                        ["concepts", "reflections", "trajectory_summaries", "trajectories"]}
        for tid in trajectory_ids:
            for layer, ids in self.get_gold_for_trajectory(tid).items():
                merged[layer].update(ids)
        return merged

    def evaluate(self, questions: List[Dict], top_k: int = 5) -> Dict:
        """
        questions: list of dicts with 'question' and 'source_id' keys.
        Returns metrics: questions_with_gold, coverage, MRR per layer.
        """
        layers = ["concepts", "reflections", "trajectory_summaries"]
        counters = {layer: {"total_gold": 0, "retrieved_gold": 0,
                            "qwg": 0, "rr": []} for layer in layers}
        n = len(questions)

        for qd in tqdm(questions, desc="Evaluating"):
            q   = qd["question"]
            # Use per-chunk gold if trajectory_id is available, else fall back to document-level
            if "trajectory_id" in qd:
                gold = self.get_gold_for_trajectory(qd["trajectory_id"])
            else:
                gold = self.get_gold_items(qd["source_id"])
            mc, mr, ms, mt, _ = self.retrieve(q, top_k)

            layer_results = {
                "concepts":              (mc, gold["concepts"]),
                "reflections":           (mr, gold["reflections"]),
                "trajectory_summaries":  (ms, gold["trajectory_summaries"]),
            }
            for layer, (matched_dict, gold_ids) in layer_results.items():
                matched_ids = set(matched_dict.keys())
                hit_ids = matched_ids & gold_ids
                counters[layer]["total_gold"]    += len(gold_ids)
                counters[layer]["retrieved_gold"]  += len(hit_ids)
                if hit_ids:
                    counters[layer]["qwg"] += 1
                    ordered = sorted(matched_dict, key=matched_dict.get, reverse=True)
                    for rank, item_id in enumerate(ordered, 1):
                        if item_id in gold_ids:
                            counters[layer]["rr"].append(1.0 / rank)
                            break

        metrics = {}
        for layer in layers:
            c = counters[layer]
            metrics[layer] = {
                "questions_with_gold":     c["qwg"],
                "questions_with_gold_pct": round(100 * c["qwg"] / n, 1) if n else 0,
                "coverage":   round(c["retrieved_gold"] / c["total_gold"], 3) if c["total_gold"] else 0,
                "mrr":        round(float(np.mean(c["rr"])), 3) if c["rr"] else 0.0,
                "total_gold": c["total_gold"],
                "retrieved_gold": c["retrieved_gold"],
            }
        return metrics

# ─────────────────────────────────────────────────────────────────────────────
# Async question generation (for evaluation dataset)
# ─────────────────────────────────────────────────────────────────────────────

async def _gen_questions_async(async_client, chunk_text: str, source_id: str, n: int) -> List[Dict]:
    prompt = f"""Generate {n} specific factual questions answerable from this text.

TEXT:
{chunk_text}

Return JSON with "questions": list of question strings."""
    try:
        create_kwargs = dict(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        if "gpt-5" not in LLM_MODEL:
            create_kwargs["temperature"] = 0.7
        resp = await async_client.chat.completions.create(**create_kwargs)
        qs = json.loads(resp.choices[0].message.content).get("questions", [])
        return [{"question": q, "source_id": source_id} for q in qs[:n]]
    except Exception:
        return []

async def generate_eval_questions(mem: ModifiedMemory, n_per_chunk: int = 2) -> List[Dict]:
    tasks = []
    traj_ids = []
    for tid, traj in mem.trajectories.items():
        tasks.append(_gen_questions_async(async_client, traj.chunk_text,
                                          traj.source_id, n_per_chunk))
        traj_ids.append(tid)
    results = await asyncio.gather(*tasks)
    # Attach trajectory_id so evaluate() can use per-chunk gold
    out = []
    for tid, batch in zip(traj_ids, results):
        for q in batch:
            q["trajectory_id"] = tid
            out.append(q)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic conversation data
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_CONVERSATION = """User: I've been reading these three papers on cancer immunology. The T-cell exhaustion paper was really interesting — I didn't know PD-1 upregulation was so central to the immune evasion mechanism.
Assistant: Yes, PD-1 is a key checkpoint receptor. When T cells are chronically stimulated — as in tumours — they upregulate PD-1, which suppresses their effector function. The paper likely also discussed TIM-3 and LAG-3 as co-inhibitory receptors.
User: Exactly. It also mentioned that exhausted T cells have a distinct epigenetic state that limits their response to checkpoint blockade.
Assistant: That's a critical point. The TOX transcription factor drives that epigenetic programme. Even when PD-1 is blocked therapeutically, the epigenetic state constrains how well T cells recover.
User: The second paper was on tumour microenvironment metabolic competition. I found the section on lactate and pH surprising.
Assistant: Tumour cells preferentially use aerobic glycolysis — the Warburg effect — producing lactate that acidifies the TME. This suppresses T cell activity and promotes regulatory T cell infiltration.
User: Right. And the third paper tied it together with data on combination therapies — anti-PD-1 plus metabolic interventions.
Assistant: Combination approaches targeting both immune checkpoints and metabolic barriers seem more promising than either alone. The synergy comes from restoring T cell function while also clearing the hostile metabolic environment.
User: I work in a wet lab focusing on flow cytometry-based immune profiling. These papers are directly relevant to my current project on characterising exhaustion markers.
Assistant: That context is really useful. For your work, the exhaustion scoring by co-expression of PD-1, TIM-3, and LAG-3 would be directly applicable, along with the metabolic markers like GLUT1 for glycolytic activity.
User: Exactly my plan. I prefer papers that include mechanistic data alongside clinical correlates.
Assistant: Noted — you value mechanistic depth paired with translational relevance. I'll keep that in mind for future literature recommendations."""

# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def load_from_cache(cache_path: str, mem: "ModifiedMemory") -> List[str]:
    """Load all four memory layers from an existing HierarchicalMemory cache directly
    into *mem*, preserving embeddings and structural connections.  Returns the list
    of distinct source_ids loaded (each is also appended to mem._docs_since_sleep so
    that sleep_update will build rubrics for them)."""
    print(f"Loading memory components from: {cache_path}")
    with open(cache_path) as f:
        state = json.load(f)

    ext_docs: Dict[str, dict] = state.get("external_documents", {})

    # ── helper: derive a clean source_id from external_document_ids ──────────
    def _source_id(ext_ids: List[str]) -> str:
        for eid in ext_ids:
            identifier = ext_docs.get(eid, {}).get("identifier", "")
            if identifier:
                # Strip chunk suffix (e.g. "paper::chunk0" → "paper")
                base = identifier.split("/")[-1].split(".")[0]
                base = base.split("::")[0]
                return base[:40]
        return "cache_unknown"

    # ── 1. Concepts ───────────────────────────────────────────────────────────
    for cid, cdata in state.get("concepts", {}).items():
        emb = np.array(cdata["embedding"], dtype=np.float32)
        # direct_reflection_ids is the cache's equivalent of our reflection_ids
        refl_ids = cdata.get("direct_reflection_ids") or cdata.get("process_summary_ids", [])
        # filter stale/empty ids
        refl_ids = [r for r in refl_ids if r]
        mem.concepts[cid] = Concept4(
            id=cid,
            text=cdata["text"],
            embedding=emb,
            reflection_ids=refl_ids,
        )

    # ── 2. Reflections ────────────────────────────────────────────────────────
    for rid, rdata in state.get("reflections", {}).items():
        emb = np.array(rdata["embedding"], dtype=np.float32)
        mem.reflections[rid] = Reflection4(
            id=rid,
            reflection_list=rdata.get("reflection_list", []),
            embedding=emb,
            concept_ids=rdata.get("concept_ids", []),
            trajectory_summary_ids=rdata.get("trajectory_summary_ids", []),
        )

    # ── 3. Trajectory Summaries ───────────────────────────────────────────────
    for sid, sdata in state.get("trajectory_summaries", {}).items():
        emb = np.array(sdata["embedding"], dtype=np.float32)
        mem.traj_sums[sid] = TrajectorySummary4(
            id=sid,
            text=sdata["text"],
            embedding=emb,
            reflection_id=sdata.get("reflection_id", ""),
            trajectory_id=sdata.get("trajectory_id", ""),
        )

    # ── 4. Trajectories ───────────────────────────────────────────────────────
    src_to_trajs: Dict[str, List[str]] = defaultdict(list)
    for tid, tdata in state.get("trajectories", {}).items():
        chunk_text = tdata.get("data", {}).get("chunk_text", "")
        if not chunk_text:
            continue
        src_id = _source_id(tdata.get("external_document_ids", []))
        mem.trajectories[tid] = Trajectory4(
            id=tid,
            chunk_text=chunk_text,
            timestamp=tdata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            source_id=src_id,
            source_type="document",
            summary_id=tdata.get("summary_id"),
            concept_ids=tdata.get("concept_ids", []),
        )
        src_to_trajs[src_id].append(tid)

    # ── 5. Source registry + _docs_since_sleep ────────────────────────────────
    seen_sources: List[str] = []
    for src_id, traj_ids in src_to_trajs.items():
        mem.source_registry[src_id] = traj_ids
        mem._docs_since_sleep.append(src_id)
        seen_sources.append(src_id)

    # ── 6. Structural connections (concept→reflection, reflection→traj_sum) ───
    for cid, concept in mem.concepts.items():
        for rid in concept.reflection_ids:
            if rid in mem.reflections:
                mem.conn_c2r.add(cid, rid)
    for rid, refl in mem.reflections.items():
        for sid in refl.trajectory_summary_ids:
            if sid in mem.traj_sums:
                mem.conn_r2s.add(rid, sid)

    print(f"  Loaded: {len(mem.concepts)} concepts, {len(mem.reflections)} reflections, "
          f"{len(mem.traj_sums)} traj_summaries, {len(mem.trajectories)} trajectories")
    print(f"  Sources ({len(seen_sources)}): {seen_sources}")
    print(f"  Connections — C→R: {mem.conn_c2r.total_connections()}, "
          f"R→S: {mem.conn_r2s.total_connections()}")
    return seen_sources


def run_experiment():
    print("=" * 80)
    print("MODIFIED MEMORY SYSTEM — EXPERIMENT")
    print("LLM model:", LLM_MODEL)
    print("=" * 80)

    CACHE_PATH = "memory_cache_2024.11.20.623809v2.full_jci-132-156243_s42255-024-01139-z_call.json"
    OUTPUT_PATH = f"modified_memory_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # ── Step 1: Load memory components from existing cache ───────────────────
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: cache not found at {CACHE_PATH}")
        return
    mem = ModifiedMemory()
    print("\n[Step 1: Load from Cache]")
    sources = load_from_cache(CACHE_PATH, mem)

    # ── Step 2: Add synthetic conversation ───────────────────────────────────
    print("\n[Step 2: Add Synthetic Conversation]")
    mem.add_content(SYNTHETIC_CONVERSATION, source_id="conv_cancer_immunology",
                    source_type="conversation")

    print(f"\nMemory state: {len(mem.concepts)} concepts, "
          f"{len(mem.reflections)} reflections, "
          f"{len(mem.traj_sums)} traj_summaries, "
          f"{len(mem.trajectories)} trajectories")
    print(f"Rubrics: {list(mem.rubrics.keys())}")
    if mem.persona.entries:
        for _pname, _pentry in mem.persona.entries.items():
            print(f"Persona [{_pname}]: {_pentry.summary}")
    else:
        print("Persona: (empty)")

    # ── Step 3: Sleep update ───────────────────────────────────────────────────
    print("\n[Step 3: Sleep Update]")
    mem.sleep_update(n_questions_per_chunk=2, top_k=5)

    # ── Step 4: Generate eval questions ────────────────────────────────────────
    print("\n[Step 4: Generate Eval Questions]")
    eval_questions = asyncio.run(generate_eval_questions(mem, n_per_chunk=2))
    print(f"  Generated {len(eval_questions)} eval questions")

    # ── Step 5: Evaluate ───────────────────────────────────────────────────────
    print("\n[Step 5: Evaluate]")
    metrics = mem.evaluate(eval_questions, top_k=5)

    print("\nEvaluation Results:")
    for layer, m in metrics.items():
        print(f"  {layer:25s}  QwG={m['questions_with_gold']:3d}/{len(eval_questions)} "
              f"({m['questions_with_gold_pct']:5.1f}%)  "
              f"Coverage={m['coverage']:.3f}  MRR={m['mrr']:.3f}")

    # ── Step 6: Demo synthesis ─────────────────────────────────────────────────
    print("\n[Step 6: Demo Synthesis]")
    demo_qs = [
        "What is the role of PD-1 in T-cell exhaustion within tumours?",
        "How does metabolic competition in the tumour microenvironment suppress immunity?",
        "What combination therapy strategies were evaluated?",
    ]
    demo_answers = []
    for q in demo_qs:
        print(f"\n  Q: {q}")
        ans = mem.query(q)
        print(f"  A: {ans[:200]}{'...' if len(ans) > 200 else ''}")
        demo_answers.append({"question": q, "answer": ans})

    # ── Save results ────────────────────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "memory_stats": {
            "concepts": len(mem.concepts),
            "reflections": len(mem.reflections),
            "trajectory_summaries": len(mem.traj_sums),
            "trajectories": len(mem.trajectories),
            "rubric_types": list(mem.rubrics.keys()),
            "persona": {name: entry.summary for name, entry in mem.persona.entries.items()},
            "connections_c2r": mem.conn_c2r.total_connections(),
            "connections_r2s": mem.conn_r2s.total_connections(),
        },
        "eval_questions_count": len(eval_questions),
        "metrics": metrics,
        "demo_synthesis": demo_answers,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
