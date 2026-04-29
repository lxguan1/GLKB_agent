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

import json, sys, uuid, asyncio, re, random
import numpy as np
import tiktoken
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm

from config import client, async_client, LLM_MODEL, EMBEDDING_MODEL

# ─────────────────────────────────────────────────────────────────────────────
# Sleep-update hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

HARD_NEGATIVE_NOISE_THRESHOLD   = 0.3   # min mr_all score to count a retrieved reflection as a hard negative
MIN_PAIR_FLAGS_FOR_REEXTRACTION = 2     # a (gold, confuser) pair must be flagged ≥ this many times to be included
MAX_CONFUSERS_PER_GOLD          = 3     # max confuser episodes listed in a single contrastive re-extraction prompt

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

# Token usage tracking
_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
                      "embed_tokens": 0, "embed_calls": 0}

def reset_token_usage() -> None:
    global _token_usage
    _token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
                    "embed_tokens": 0, "embed_calls": 0}

def get_token_usage() -> dict:
    return dict(_token_usage)

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
    for _attempt in range(_max_retries):
        try:
            resp = await async_client.embeddings.create(model=EMBEDDING_MODEL, input=texts, encoding_format="float")
            if resp.usage:
                _token_usage["embed_tokens"] += resp.usage.total_tokens
                _token_usage["embed_calls"] += 1
            return np.array([item.embedding for item in resp.data])
        except _retryable as _e:
            if _attempt == _max_retries - 1:
                raise
            _delay = _base_delay * (2 ** _attempt)
            print(f"    [async_embed] transient error ({type(_e).__name__}), retry {_attempt + 1}/{_max_retries - 1} in {_delay:.0f}s")
            await asyncio.sleep(_delay)


def top_k_sim(query_emb: np.ndarray, candidate_embs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if len(candidate_embs) == 0:
        return []
    sims = cos_sim(query_emb.reshape(1, -1), candidate_embs)[0]
    idx = np.argsort(sims)[-k:][::-1]
    return [(int(i), float(sims[i])) for i in idx]

_TEMPORAL_SIGNAL_RE = re.compile(
    r'\b('
    # Month names
    r'january|february|march|april|may|june|july|august|september|october|november|december'
    # Day names
    r'|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    # Year-like numbers
    r'|\b(19|20)\d{2}\b'
    # Relative / positional time words
    r'|yesterday|today|tonight|tomorrow'
    r'|last|next|ago|earlier|later|recently|before|after|during|when|while'
    r'|week|month|year|day|hour|morning|afternoon|evening|night'
    r'|date|time|period|season|summer|winter|spring|autumn|fall'
    r')\b',
    re.IGNORECASE,
)

def _has_temporal_signal(text: str) -> bool:
    """Broad check — returns True if the text contains ANY possible temporal language.
    Used as a cheap pre-filter before the LLM call; errs on the side of inclusion."""
    return bool(_TEMPORAL_SIGNAL_RE.search(text))

def _parse_dt(s: str) -> Optional[datetime]:
    """Parse an ISO datetime string, stripping timezone for naive comparison."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except Exception:
        return None

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
    for _attempt in range(_max_retries):
        try:
            resp = await async_client.chat.completions.create(**kwargs)
            if resp.usage:
                _token_usage["prompt_tokens"] += resp.usage.prompt_tokens
                _token_usage["completion_tokens"] += resp.usage.completion_tokens
                _token_usage["llm_calls"] += 1
            text = resp.choices[0].message.content
            if text is None:
                if _attempt == _max_retries - 1:
                    raise ValueError("LLM returned None content after all retries")
                _delay = _base_delay * (2 ** _attempt)
                print(f"    [async_llm] None content, retry {_attempt + 1}/{_max_retries - 1} in {_delay:.0f}s", file=sys.stderr)
                await asyncio.sleep(_delay)
                continue
            return json.loads(text) if json_mode else text
        except _retryable as _e:
            if _attempt == _max_retries - 1:
                raise
            _delay = _base_delay * (2 ** _attempt)
            print(f"    [async_llm] transient error ({type(_e).__name__}), retry {_attempt + 1}/{_max_retries - 1} in {_delay:.0f}s")
            await asyncio.sleep(_delay)

# ─────────────────────────────────────────────────────────────────────────────
# Shared synthesis helpers  (used by both eval and sleep grading)
# ─────────────────────────────────────────────────────────────────────────────

async def async_synthesize_from_bullets(
    async_client,
    question: str,
    content_block: str,
    persona_note: str = "",
) -> str:
    """Generate a short answer from a bullet-list content block."""
    result = await async_llm(
        async_client,
        "You answer questions from conversation memory. "
        "Use the EXACT specific detail from the most relevant bullet — prefer the precise name, "
        "quote, or phrase over a vague generalization. "
        "Each memory bullet has a [timestamp]. Use it to resolve relative time expressions: "
        "'last year' means the year before the timestamp, 'next month' means the month after, etc. "
        "Respond in JSON.",
        f"Question: {question}{persona_note}\n\n"
        f"Retrieved memory (with timestamps):\n{content_block}\n\n"
        "Instructions:\n"
        "1. If a bullet contains a relative time expression (last week, two weeks ago, next month, etc.), "
        "resolve it to a concrete date/period using the [timestamp] on that bullet.\n"
        "2. When the question specifies a particular time period, event, or context, prioritize bullets "
        "whose timestamps match that period. Do not use information from unrelated sessions.\n"
        "3. Reason across the relevant bullets — the answer may require combining information from "
        "multiple bullets. If the question asks for a list of things, activities, or people, enumerate "
        "ALL instances found across the bullets, not just the first one.\n"
        "4. Answer as concisely as possible using the exact specific detail. Be brief, but never "
        "sacrifice the precise fact for brevity (e.g. prefer \"guinea pig\" over \"a pet\", "
        "\"like being in a fairy tale\" over \"magical feeling\").\n"
        "5. Always provide your best answer based on the available evidence. Only say \"unknown\" if "
        "there is genuinely zero relevant signal across all bullets.\n"
        'Return JSON with "answer": your short answer string.',
    )
    answer = result.get("answer", "")
    return answer if isinstance(answer, str) else str(answer)


async def async_synthesize_from_passages(
    async_client,
    question: str,
    passages_block: str,
    persona_note: str = "",
) -> str:
    """Generate a short answer from raw conversation passages (fallback path)."""
    result = await async_llm(
        async_client,
        "You answer questions from conversation memory. "
        "Use the EXACT specific detail from the most relevant passage — prefer the precise "
        "name, quote, or phrase over a vague generalization. "
        "Each passage has a [timestamp]. Use it to resolve relative time expressions. "
        "Respond in JSON.",
        f"Question: {question}{persona_note}\n\n"
        f"Conversation passages (with timestamps):\n{passages_block}\n\n"
        "Instructions:\n"
        "1. Resolve any relative time expressions using the [timestamp] of the passage they appear in.\n"
        "2. When the question specifies a particular time period or event, use ONLY passages from "
        "sessions matching that period.\n"
        "3. Answer as concisely as possible using the exact specific detail. Be brief, but never "
        "sacrifice the precise fact for brevity. If the question asks for a list of things, activities, "
        "or people, enumerate ALL instances found across the passages, not just the first one.\n"
        "4. Only say \"unknown\" if there is genuinely zero relevant signal across all passages.\n"
        'Return JSON with "answer": your short answer string.',
    )
    answer = result.get("answer", "")
    return answer if isinstance(answer, str) else str(answer)


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

        # diagnostics populated by sleep_update_async; persisted to snapshot
        self.last_sleep_stats: Dict = {}

        # Adaptive retrieval hyperparameter — updated each sleep cycle.
        # Initial value is conservative; sleep adapts it based on gold reflection ranks.
        self._adapted_top_k: int = 5

        # Diagnostic: number of queries where temporal filtering was triggered.
        # Should be 0 on the LoCoMo benchmark (no explicit-date questions).
        try:
            self._tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

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
2. For each participant, write or update a persona summary that captures their identity, interests, background, preferences, personality, and communication style as revealed by the conversations.
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
            "Extract memory components from the following text chunk.\n"
            f"Session timestamp: {ts}\n\n"
            "TEXT:\n{chunk_text}\n\n"
            "Return JSON with:\n"
            '- "concept_texts": list of 2-5 short topic labels (1-4 words each) that categorise this chunk\n'
            '- "reflection_list": list of 5-10 specific factual statements capturing ALL key information '
            "from this chunk. Be comprehensive — include specific names, dates, places, quantities, "
            "decisions, and personal details. Each statement must be self-contained and understandable "
            "without the original text. Preserve precise details (e.g. exact dates, locations, names) "
            "rather than generalising them. For any relative time expression in the text "
            "(e.g., 'last year', 'three years ago', 'next month'), include both the relative form "
            "and the resolved absolute date using the session timestamp "
            "(e.g., 'started surfing in 2018 (five years before the session timestamp)').\n"
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

    # ── sleep update ──────────────────────────────────────────────────────────

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

    async def sleep_update_async(self, n_questions_per_chunk: int = 1,
                                 refine_reflections: bool = False, build_rubrics: bool = True,
                                 use_query_components: bool = True,
                                 n_temporal_per_traj: int = 1,
                                 n_multihop_pairs_per_traj: int = 1):
        """Async version of sleep_update: LLM calls within every stage are batched.
        n_questions_per_chunk: single-hop questions per trajectory.
        n_temporal_per_traj: temporal questions per trajectory.
        n_multihop_pairs_per_traj: multi-hop trajectory pairs sampled per trajectory.
        build_rubrics: if False, skip rubric detection/extraction.
        use_query_components: if False, skip LLM query-component generation during
        connection learning and use the raw question text directly."""
        print("\n[Sleep Update]")

        if refine_reflections:
            await self._add_supplementary_reflections_async()

        await self._run_connection_learning_async(
            n_single_hop=n_questions_per_chunk,
            n_temporal=n_temporal_per_traj,
            n_multihop_pairs=n_multihop_pairs_per_traj,
            top_k=self._adapted_top_k,
            use_query_components=use_query_components,
        )

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

    async def _generate_categorized_questions_async(
        self,
        n_single_hop: int,
        n_temporal: int,
        n_multihop_pairs: int,
    ) -> List[Dict]:
        """Generate single-hop, temporal, and multi-hop training questions concurrently.

        Returns a flat list of question dicts, each with keys:
            question, gold_answer, expected_facts, category, source_traj_ids
        """
        traj_ids = list(self.trajectories.keys())
        all_questions: List[Dict] = []

        # ── Single-hop: one trajectory, n_single_hop questions ───────────────
        async def _gen_single_hop(tid: str) -> List[Dict]:
            traj = self.trajectories[tid]
            try:
                result = await async_llm(
                    async_client,
                    "You generate factual training questions from conversation text. Respond in JSON.",
                    f"Generate {n_single_hop} specific factual question(s) from this conversation episode.\n\n"
                    f"EPISODE:\n{traj.chunk_text}\n\n"
                    "Each question must be answerable from this episode alone.\n"
                    "Questions must ask about WHAT, WHO, or HOW — not WHEN or time-related aspects.\n"
                    "Do NOT use temporal qualifiers such as 'recently', 'last time', 'before', 'after', or 'when'.\n"
                    "For each question provide:\n"
                    '- "question": the question text (do not reference "the episode" or "the text")\n'
                    '- "gold_answer": a concise correct answer (1-3 sentences)\n'
                    '- "expected_facts": list of 1-3 key facts from the episode needed to answer\n\n'
                    'Return JSON: {"questions": [{"question": "...", "gold_answer": "...", "expected_facts": ["..."]}]}',
                )
                qs = result.get("questions", [])[:n_single_hop]
                for q in qs:
                    q["category"] = "single_hop"
                    q["source_traj_ids"] = [tid]
                return qs
            except Exception as error:
                print(f"    [_gen_single_hop] traj {tid[:8]}: {type(error).__name__}: {error}", file=sys.stderr)
                return []

        # ── Temporal: time-anchored questions per trajectory ─────────────────
        async def _gen_temporal(tid: str) -> List[Dict]:
            traj = self.trajectories[tid]
            ts_str = f" (session timestamp: {traj.timestamp})" if traj.timestamp else ""
            try:
                result = await async_llm(
                    async_client,
                    "You generate time-based training questions from conversation text. Respond in JSON.",
                    f"Generate {n_temporal} temporal question(s) from this conversation episode{ts_str}.\n\n"
                    f"EPISODE:\n{traj.chunk_text}\n\n"
                    "Each question must name a specific event or fact and ask WHEN it occurred — "
                    "the date or time must appear only in the answer, never in the question itself.\n"
                    "Good examples: 'What date did Alice receive her test results?', "
                    "'When did Bob start his new job?'\n"
                    "Each question must be answerable from this episode.\n"
                    "For each question provide:\n"
                    '- "question": the question text (do not reference "the episode" or "the text")\n'
                    '- "gold_answer_relative": the answer expressed as a relative time '
                    '(e.g. "last Saturday", "three weeks ago", "yesterday")\n'
                    '- "gold_answer_absolute": the answer expressed as an absolute date/time '
                    'resolved using the session timestamp (e.g. "Saturday 14 October 2023"). '
                    'Always provide both forms.\n'
                    '- "expected_facts": list of 1-3 key facts from the episode needed to answer\n\n'
                    'Return JSON: {"questions": [{"question": "...", "gold_answer_relative": "...", '
                    '"gold_answer_absolute": "...", "expected_facts": ["..."]}]}',
                )
                qs = result.get("questions", [])[:n_temporal]
                for q in qs:
                    q["category"] = "temporal"
                    q["source_traj_ids"] = [tid]
                    # Carry both forms; gold_answer is relative for display/logging
                    q["gold_answer"] = q.get("gold_answer_relative", "")
                return qs
            except Exception:
                print(f"    [_gen_temporal] traj {tid[:8]}: {type(error).__name__}: {error}", file=sys.stderr)
                return []

        # Run single-hop and temporal concurrently across all trajectories
        sh_results, tmp_results = await asyncio.gather(
            asyncio.gather(*[_gen_single_hop(tid) for tid in traj_ids], return_exceptions=True),
            asyncio.gather(*[_gen_temporal(tid) for tid in traj_ids], return_exceptions=True),
        )
        for r in sh_results:
            if not isinstance(r, Exception):
                all_questions.extend(r)
        for r in tmp_results:
            if not isinstance(r, Exception):
                all_questions.extend(r)

        # ── Multi-hop: sampled trajectory pairs, preferring concept-ID overlap ──
        if n_multihop_pairs > 0 and len(traj_ids) >= 2:
            # Build concept → trajectory map for overlap detection
            concept_to_trajs: Dict[str, List[str]] = defaultdict(list)
            for tid in traj_ids:
                for cid in self.trajectories[tid].concept_ids:
                    concept_to_trajs[cid].append(tid)

            pair_set: Set[Tuple[str, str]] = set()
            for tid in traj_ids:
                # Find trajectories that share at least one concept (same people/topics)
                related: Set[str] = set()
                for cid in self.trajectories[tid].concept_ids:
                    for other in concept_to_trajs[cid]:
                        if other != tid:
                            related.add(other)
                # Prefer related trajectories; fall back to random if none found
                candidates = list(related) if related else [t for t in traj_ids if t != tid]
                sampled = random.sample(candidates, min(n_multihop_pairs, len(candidates)))
                for other in sampled:
                    pair_set.add((min(tid, other), max(tid, other)))  # canonical order

            async def _gen_multihop(tid_a: str, tid_b: str) -> List[Dict]:
                traj_a = self.trajectories[tid_a]
                traj_b = self.trajectories[tid_b]
                try:
                    result = await async_llm(
                        async_client,
                        "You generate multi-hop training questions from conversation episodes. Respond in JSON.",
                        "Generate 1 question that requires information from BOTH episodes to answer correctly.\n\n"
                        f"EPISODE A:\n{traj_a.chunk_text}\n\n"
                        f"EPISODE B:\n{traj_b.chunk_text}\n\n"
                        "The question must be impossible to answer from either episode alone — "
                        "the answer must combine facts from both.\n"
                        "Provide:\n"
                        '- "question": the question text (do not reference "the episode" or "the text")\n'
                        '- "gold_answer": a concise correct answer drawing on both episodes\n'
                        '- "expected_facts": list of 2-4 key facts needed, note which episode each comes from\n\n'
                        'Return JSON: {"questions": [{"question": "...", "gold_answer": "...", "expected_facts": ["..."]}]}',
                    )
                    qs = result.get("questions", [])[:1]
                    for q in qs:
                        q["category"] = "multi_hop"
                        q["source_traj_ids"] = [tid_a, tid_b]
                    return qs
                except Exception:
                    print(f"    [_gen_multi_hop] traj {tid[:8]}: {type(error).__name__}: {error}", file=sys.stderr)
                    return []

            mh_results = await asyncio.gather(
                *[_gen_multihop(a, b) for a, b in pair_set],
                return_exceptions=True,
            )
            for r in mh_results:
                if not isinstance(r, Exception):
                    all_questions.extend(r)

        return all_questions

    async def _run_connection_learning_async(self, n_single_hop: int = 3,
                                              n_temporal: int = 1,
                                              n_multihop_pairs: int = 1,
                                              top_k: int = 5,
                                              use_query_components: bool = True):
        """Async version: question-gen and query-component calls are all batched.
        Generates single-hop, temporal, and multi-hop training questions, then
        processes them in curriculum order (single_hop → temporal → multi_hop).
        use_query_components: if False, skip LLM query-component generation and use
        the raw question as both concept_texts and predicted_reflections."""
        print(f"  Running connection learning ({n_single_hop} single-hop, "
              f"{n_temporal} temporal, {n_multihop_pairs} multi-hop pairs per traj)...")

        # Stage 1 — generate categorized questions for all trajectories
        all_qdicts = await self._generate_categorized_questions_async(
            n_single_hop, n_temporal, n_multihop_pairs
        )

        # Attach gold reflection IDs to each question dict (union over all source trajs)
        for qd in all_qdicts:
            gold_r: Set[str] = set()
            for tid in qd["source_traj_ids"]:
                traj = self.trajectories.get(tid)
                if traj and traj.summary_id and traj.summary_id in self.traj_sums:
                    gold_r.add(self.traj_sums[traj.summary_id].reflection_id)
            qd["_gold_r"] = gold_r

        if not all_qdicts:
            print("    No questions generated; skipping connection learning.")
            return

        # Stage 2 — generate query components for every question concurrently
        question_texts = [qd["question"] for qd in all_qdicts]

        if use_query_components:
            async def _gen_qc(q: str):
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
                    return QueryComponents4(
                        concept_texts=concept_texts,
                        predicted_reflections=predicted_reflections,
                        predicted_summary=predicted_summary,
                    )
                except Exception:
                    return QueryComponents4(concept_texts=[q[:40]], predicted_reflections=[q])

            qc_results = await asyncio.gather(
                *[_gen_qc(q) for q in question_texts],
                return_exceptions=True,
            )
        else:
            qc_results = [
                QueryComponents4(concept_texts=[q], predicted_reflections=[q])
                for q in question_texts
            ]

        # ── Pre-batch all query embeddings before Stage 3 ────────────────────
        unique_ct: List[str] = []; ct_idx: Dict[str, int] = {}
        unique_pr: List[str] = []; pr_idx: Dict[str, int] = {}
        unique_ri: List[str] = []; ri_idx: Dict[str, int] = {}
        for res in qc_results:
            if not isinstance(res, QueryComponents4):
                continue
            qc = res
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

        # Single embed call: ct + pr + ri concatenated — one round-trip
        _all_embed_texts = unique_ct + unique_pr + unique_ri
        _all_embs = await async_embed(_all_embed_texts) if _all_embed_texts else np.array([])
        n_ct, n_pr = len(unique_ct), len(unique_pr)
        ct_emb_map: Dict[str, np.ndarray] = {t: _all_embs[i] for t, i in ct_idx.items()}
        pr_emb_map: Dict[str, np.ndarray] = {t: _all_embs[n_ct + i] for t, i in pr_idx.items()}
        ri_emb_map: Dict[str, np.ndarray] = {t: _all_embs[n_ct + n_pr + i] for t, i in ri_idx.items()}

        # Stage 3 — connection learning in curriculum order
        CURRICULUM_ORDER = ["single_hop", "temporal", "multi_hop"]
        ordered = sorted(
            zip(all_qdicts, qc_results),
            key=lambda x: CURRICULUM_ORDER.index(x[0].get("category", "single_hop"))
            if x[0].get("category", "single_hop") in CURRICULUM_ORDER else len(CURRICULUM_ORDER),
        )

        # Reverse map: reflection_id → trajectory_id (for confuser identification)
        rid_to_tid: Dict[str, str] = {}
        for ts in self.traj_sums.values():
            if ts.reflection_id:
                rid_to_tid[ts.reflection_id] = ts.trajectory_id

        # Phase 2: pair confusion counter accumulated across all questions this sleep cycle
        pair_flag_counts: Dict[tuple, int] = {}  # (gold_traj_id, confuser_traj_id) → count

        # Counters for hard negative logging summary
        _hn_total = 0       # total hard negatives seen across all questions
        _hn_q_count = 0     # questions that had at least one hard negative

        # ── Pass 1: retrieval + Phase 2 HN identification + Phase 3 grading ──
        # Store per-question retrieval data so Pass 2 (credit assignment) can
        # access it after grades are available.
        _retrieval_data: List[tuple] = []  # (qd, qc, mc, mr, mr_all, hard_neg_rids)
        _grade_jobs: List[tuple] = []       # (qd, coroutine)
        _needed_ks: List[Optional[int]] = []  # min top_k to retrieve all gold, per question

        for qd, qc_or_exc in ordered:
            if not isinstance(qc_or_exc, QueryComponents4):
                continue
            qc = qc_or_exc
            gold_r = qd["_gold_r"]
            qc_ct_embs = (np.array([ct_emb_map[t] for t in qc.concept_texts if t in ct_emb_map])
                          if qc.concept_texts else None)
            qc_pr_embs = (np.array([pr_emb_map[t] for t in qc.predicted_reflections if t in pr_emb_map])
                          if qc.predicted_reflections else None)
            mc, mc_all = self._match_layer_concepts(qc.concept_texts, self._adapted_top_k, embs=qc_ct_embs)
            mr, mr_all = self._match_layer_reflections(
                qc.predicted_reflections, set(mc.keys()), self._adapted_top_k,
                embs=qc_pr_embs, text_emb_cache=ri_emb_map,
            )

            # Track gold reflection rank for adaptive top_k update
            _needed_ks.append(self._gold_retrieval_rank(qd["_gold_r"], mc_all, mr_all))

            # Phase 3: queue grading coroutine
            _grade_jobs.append((qd, self._grade_answer_async(
                question             = qd["question"],
                category             = qd.get("category", "single_hop"),
                mr                   = mr,
                query_texts          = qc.predicted_reflections,
                gold_answer          = qd.get("gold_answer", ""),
                gold_r               = gold_r,
                source_traj_ids      = qd.get("source_traj_ids", []),
                gold_answer_relative = qd.get("gold_answer_relative"),
                gold_answer_absolute = qd.get("gold_answer_absolute"),
            )))

            # Phase 2: hard negative identification
            hard_neg_rids: Set[str] = {
                rid for rid in mr
                if rid not in gold_r
                and mr_all.get(rid, 0.0) > HARD_NEGATIVE_NOISE_THRESHOLD
            }
            if hard_neg_rids:
                _hn_total += len(hard_neg_rids)
                _hn_q_count += 1
                ambiguous_pairs: List[tuple] = [
                    (cid, hard_neg_rid)
                    for cid in mc
                    for hard_neg_rid in hard_neg_rids
                    if hard_neg_rid in self.conn_c2r.get(cid)
                ]
                gold_retrieved = bool(gold_r & set(mr))
            # Record traversals on existing connections (safe unconditionally)
            for cid in mc:
                for rid in self.conn_c2r.get(cid):
                    self.conn_c2r.record(cid, rid, rid in gold_r)

            _retrieval_data.append((qd, qc, mc, mr, mr_all, hard_neg_rids))

        # ── Phase 3: batch grading ────────────────────────────────────────────
        grade_results = await asyncio.gather(
            *[coro for _, coro in _grade_jobs],
            return_exceptions=True,
        )
        # Build grade result index keyed by position for Pass 2
        _grades: List[Optional[Dict]] = []
        _grade_by_cat: Dict[str, Dict[str, int]] = {}
        for (qd, _), gr in zip(_grade_jobs, grade_results):
            if isinstance(gr, Exception):
                _grades.append(None)
                continue
            _grades.append(gr)
            cat = qd.get("category", "single_hop")
            if cat not in _grade_by_cat:
                _grade_by_cat[cat] = {"correct": 0, "partial": 0, "wrong": 0,
                                      "gold_in_top": 0, "total": 0}
            c = _grade_by_cat[cat]
            c["total"] += 1
            if gr["correct"]:
                c["correct"] += 1
            elif gr["partial"]:
                c["partial"] += 1
            else:
                c["wrong"] += 1
            if gr["gold_in_top_items"]:
                c["gold_in_top"] += 1
        print("    Grading summary by category:")
        for cat, c in _grade_by_cat.items():
            tot = c["total"] or 1
            print(
                f"      {cat}: correct={c['correct']}/{tot} "
                f"partial={c['partial']}/{tot} wrong={c['wrong']}/{tot} "
                f"gold_in_top={c['gold_in_top']}/{tot}"
            )

        # ── Pass 2: credit assignment ─────────────────────────────────────────
        _detail_reextract_tids: List[str] = []   # Phase 5: LOSSY EXTRACTION / mismatch flags
        _ca_counts: Dict[str, int] = {
            "reinforce": 0, "pipeline_failure": 0, "conflation": 0,
            "lossy_extraction": 0, "wrong_path_gated": 0,
            "wrong_path_mismatch": 0, "missing_connection": 0,
        }

        def _credit_assign(
            gold_r_sub:  Set[str],
            gold_tid:    str,
            mc:          Dict[str, float],
            mr:          Dict[str, float],
            mr_all:      Dict[str, float],
            hard_neg_rids: Set[str],
            hop_correct: bool,
            gold_in_top: bool,          # whether gold items for THIS sub-problem are in filtered set
            qc:          "QueryComponents4",
        ):
            """Credit assignment tree for a single sub-problem (one hop or a single/temporal question)."""
            if hop_correct:
                _ca_counts["reinforce"] += 1
                return  # reinforcement already done via record() in Pass 1

            gold_retrieved = bool(gold_r_sub & set(mr))

            if gold_retrieved:
                if not gold_in_top:
                    _ca_counts["pipeline_failure"] += 1
                elif hard_neg_rids:
                    _ca_counts["conflation"] += 1
                    for hard_neg_rid in hard_neg_rids:
                        confuser_tid = rid_to_tid.get(hard_neg_rid)
                        if confuser_tid and confuser_tid != gold_tid:
                            key = (gold_tid, confuser_tid)
                            pair_flag_counts[key] = pair_flag_counts.get(key, 0) + 1
                    # Gold WAS retrieved — reinforce it so it wins over confusers next time
                    gold_rid = next(iter(gold_r_sub & set(mr)), None)
                    if gold_rid:
                        for cid in mc:
                            self.conn_c2r.add(cid, gold_rid, new=True)
                    # Also queue detail re-extraction: confusers may be present but the
                    # gold reflection's content may itself be the synthesis bottleneck
                    _detail_reextract_tids.append(gold_tid)
                else:
                    _ca_counts["lossy_extraction"] += 1
                    _detail_reextract_tids.append(gold_tid)

            else:  # gold not retrieved
                if hard_neg_rids:
                    for hard_neg_rid in hard_neg_rids:
                        confuser_tid = rid_to_tid.get(hard_neg_rid)
                        if confuser_tid and confuser_tid != gold_tid:
                            key = (gold_tid, confuser_tid)
                            pair_flag_counts[key] = pair_flag_counts.get(key, 0) + 1

                    # Gate 1: would the gold reflection score above the k-th retrieved?
                    gold_rid = max(gold_r_sub,
                                   key=lambda rid: mr_all.get(rid, 0.0),
                                   default=None)
                    gold_r_obj = self.reflections.get(gold_rid) if gold_rid else None
                    pr_embs = [pr_emb_map[t] for t in qc.predicted_reflections if t in pr_emb_map]
                    kth_score = min(mr_all[rid] for rid in mr) if mr else 0.0

                    gold_estimated_score = 0.0
                    if gold_r_obj and gold_r_obj.item_embeddings and pr_embs:
                        pr_arr   = np.array(pr_embs)
                        item_arr = np.array(gold_r_obj.item_embeddings)
                        gold_estimated_score = float(cos_sim(pr_arr, item_arr).max())
                    elif gold_rid and gold_rid in ri_emb_map:
                        pr_arr = np.array(pr_embs) if pr_embs else None
                        if pr_arr is not None:
                            gold_estimated_score = float(
                                cos_sim(pr_arr, ri_emb_map[gold_rid].reshape(1, -1)).max()
                            )

                    if gold_estimated_score > kth_score:
                        _ca_counts["wrong_path_gated"] += 1
                        for cid in mc:
                            cid_routes_to_confuser = any(
                                hn_rid in self.conn_c2r.get(cid) for hn_rid in hard_neg_rids
                            )
                            if not cid_routes_to_confuser:
                                self.conn_c2r.add(cid, gold_rid, new=True)
                    else:
                        _ca_counts["wrong_path_mismatch"] += 1
                        _detail_reextract_tids.append(gold_tid)

                else:
                    _ca_counts["missing_connection"] += 1
                    self.conn_c2r.learn_top_gold_only(set(mc), gold_r_sub, mr_all)

        for i, (qd, qc, mc, mr, mr_all, hard_neg_rids) in enumerate(_retrieval_data):
            gold_r   = qd["_gold_r"]
            category = qd.get("category", "single_hop")
            gr       = _grades[i]

            answer_correct       = gr["correct"]           if gr else False
            partial_hops_correct = gr["partial_hops_correct"] if gr else []
            filtered_plain       = gr["filtered_plain"]    if gr else []

            if category == "multi_hop":
                for tid in qd.get("source_traj_ids", []):
                    traj = self.trajectories.get(tid)
                    if not traj or not traj.summary_id:
                        continue
                    ts_obj = self.traj_sums.get(traj.summary_id)
                    if not ts_obj:
                        continue
                    gold_rid_hop = ts_obj.reflection_id
                    gold_r_hop   = {gold_rid_hop}
                    hop_correct  = tid in partial_hops_correct or answer_correct

                    # Per-hop gold_in_top: are this hop's items in the filtered set?
                    hop_r_obj    = self.reflections.get(gold_rid_hop)
                    hop_gold_in_top = bool(
                        hop_r_obj and set(hop_r_obj.reflection_list) & set(filtered_plain)
                    )
                    gold_tid = rid_to_tid.get(gold_rid_hop, tid)

                    _credit_assign(
                        gold_r_sub    = gold_r_hop,
                        gold_tid      = gold_tid,
                        mc            = mc,
                        mr            = mr,
                        mr_all        = mr_all,
                        hard_neg_rids = hard_neg_rids,
                        hop_correct   = hop_correct,
                        gold_in_top   = hop_gold_in_top,
                        qc            = qc,
                    )
            else:
                gold_rid_single = next(iter(gold_r), None)
                gold_tid = rid_to_tid.get(gold_rid_single, "") if gold_rid_single else ""
                gold_in_top = bool(
                    gold_rid_single
                    and self.reflections.get(gold_rid_single)
                    and set(self.reflections[gold_rid_single].reflection_list) & set(filtered_plain)
                )
                _credit_assign(
                    gold_r_sub    = gold_r,
                    gold_tid      = gold_tid,
                    mc            = mc,
                    mr            = mr,
                    mr_all        = mr_all,
                    hard_neg_rids = hard_neg_rids,
                    hop_correct   = answer_correct,
                    gold_in_top   = gold_in_top,
                    qc            = qc,
                )

        print(
            f"    Hard negatives summary: {_hn_q_count} questions had hard negatives "
            f"({_hn_total} total hard neg reflections across all questions)"
        )
        if pair_flag_counts:
            top_pairs = sorted(pair_flag_counts.items(), key=lambda x: -x[1])[:5]
            for (gtid, ctid), cnt in top_pairs:
                print(f"      pair_flag ({gtid[:8]}…, {ctid[:8]}…) × {cnt}")
        # Persist sleep diagnostics for snapshot / comparison
        self.last_sleep_stats = {
            "credit_assignment": dict(_ca_counts),
            "training_grades":   {cat: dict(v) for cat, v in _grade_by_cat.items()},
        }
        print(f"    Credit assignment branches: {_ca_counts}")

        # ── Adaptive top_k update ─────────────────────────────────────────────
        MAX_TOP_K_CAP = 30
        wrong_needed_ks = [
            k for k, (qd, _), gr in zip(_needed_ks, _grade_jobs, grade_results)
            if k is not None
            and isinstance(gr, dict)
            and not gr.get("correct", False)
        ]
        if wrong_needed_ks:
            new_top_k = int(np.percentile(wrong_needed_ks, 80))
            new_top_k = min(max(new_top_k, 1), MAX_TOP_K_CAP)
            self._adapted_top_k = new_top_k
            print(
                f"    Adaptive top_k updated → {self._adapted_top_k} "
                f"(n_wrong={len(wrong_needed_ks)}, "
            )
        else:
            print(f"    Adaptive top_k: unchanged ({self._adapted_top_k}) — no wrong questions with rank data")
        self.last_sleep_stats["adapted_top_k"] = self._adapted_top_k

        # ── Phase 5: batched re-extraction ───────────────────────────────────
        await self._contrastive_reextract_async(
            pair_flag_counts,
            min_flags            = MIN_PAIR_FLAGS_FOR_REEXTRACTION,
            max_confusers_per_gold = MAX_CONFUSERS_PER_GOLD,
        )
        await self._detail_reextract_async(_detail_reextract_tids)
        # ─────────────────────────────────────────────────────────────────────
        print(f"    Connections after learning — "
              f"C→R: {self.conn_c2r.total_connections()}, "
              f"R→S: {self.conn_r2s.total_connections()} (static — R→S learning disabled)")

    async def _append_reflection_items_async(
        self, rid: str, new_items: List[str]
    ) -> bool:
        """Strictly additive: append new_items to a reflection's list and recompute embeddings.
        Returns True if items were actually appended."""
        r = self.reflections.get(rid)
        if not r or not new_items:
            return False
        valid = [s for s in new_items if s and s not in r.reflection_list]
        if not valid:
            return False
        joined_text = " ".join(r.reflection_list + valid)
        all_texts   = [joined_text] + valid
        try:
            embs = await async_embed(all_texts)
        except Exception:
            return False
        r.reflection_list = r.reflection_list + valid
        r.embedding       = embs[0]
        r.item_embeddings = r.item_embeddings + list(embs[1:])
        return True

    async def _contrastive_reextract_async(
        self,
        pair_flag_counts: Dict[tuple, int],
        min_flags: int = 2,
        max_confusers_per_gold: int = 3,
    ) -> None:
        """Additive contrastive re-extraction, grouped per gold trajectory.

        Groups all qualifying (gold, confuser) pairs by gold trajectory, then makes one
        LLM call per gold listing up to max_confusers_per_gold confusers (highest-count
        first). This removes the per-pair count ceiling of the old per-pair design.
        """
        # Group qualifying pairs by gold_traj_id, highest-count confusers first
        gold_to_confusers: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for (gold_tid, confuser_tid), cnt in pair_flag_counts.items():
            if cnt >= min_flags:
                gold_to_confusers[gold_tid].append((confuser_tid, cnt))

        if not gold_to_confusers:
            return

        # For each gold, keep only the top-N confusers by count
        for gold_tid in gold_to_confusers:
            gold_to_confusers[gold_tid].sort(key=lambda x: -x[1])
            gold_to_confusers[gold_tid] = gold_to_confusers[gold_tid][:max_confusers_per_gold]

        print(f"    [Phase 5] Contrastive re-extraction: {len(gold_to_confusers)} gold trajectories")

        async def _one_gold(gold_tid: str, confusers: List[Tuple[str, int]]) -> Optional[Tuple[str, List[str]]]:
            gold_traj = self.trajectories.get(gold_tid)
            if not gold_traj:
                return None
            gold_ts = self.traj_sums.get(gold_traj.summary_id or "")
            if not gold_ts:
                return None
            gold_r = self.reflections.get(gold_ts.reflection_id)
            if not gold_r:
                return None

            # Build confuser blocks (skip any that can't be resolved)
            confuser_blocks = []
            for i, (confuser_tid, _) in enumerate(confusers, start=1):
                ct = self.trajectories.get(confuser_tid)
                if ct:
                    label = chr(ord("B") + i - 1)   # B, C, D …
                    confuser_blocks.append(f"Episode {label}:\n{ct.chunk_text}")
            if not confuser_blocks:
                return None

            existing    = "\n".join(f"- {item}" for item in gold_r.reflection_list)
            confuser_section = "\n\n".join(confuser_blocks)
            n_confusers = len(confuser_blocks)
            try:
                result = await async_llm(
                    async_client,
                    "You extract discriminative memory statements. Respond in JSON.",
                    f"The following conversation episode is being confused with "
                    f"{n_confusers} other episode(s) during retrieval.\n\n"
                    f"Episode A (improve this one):\n{gold_traj.chunk_text}\n\n"
                    f"Confusing episode(s):\n{confuser_section}\n\n"
                    f"Existing reflections for Episode A:\n{existing}\n\n"
                    "Generate 2-3 NEW reflection statements for Episode A that capture facts "
                    "or details unique to Episode A and absent or clearly different in ALL of "
                    "the confusing episodes above. "
                    "Do not repeat anything already in the existing reflections. "
                    "Each statement must be fully self-contained (use names, not pronouns).\n\n"
                    'Return JSON with "new_reflections": list of statement strings.',
                )
                new_items = result.get("new_reflections", [])
                if not isinstance(new_items, list):
                    return None
                return gold_ts.reflection_id, [s for s in new_items[:3] if s]
            except Exception:
                return None

        tasks = [_one_gold(gold_tid, confusers)
                 for gold_tid, confusers in gold_to_confusers.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        appended = 0
        for res in results:
            if isinstance(res, Exception) or res is None:
                continue
            rid, new_items = res
            if await self._append_reflection_items_async(rid, new_items):
                appended += 1
        print(f"      Appended new items to {appended}/{len(gold_to_confusers)} reflections.")

    async def _detail_reextract_async(
        self,
        traj_ids: List[str],
    ) -> None:
        """Additive detail re-extraction for LOSSY EXTRACTION and query-content mismatch cases.

        Generates 2-3 new reflection statements capturing facts not yet in the existing list.
        """
        # Deduplicate traj_ids
        seen: Set[str] = set()
        unique_tids = [tid for tid in traj_ids if not (tid in seen or seen.add(tid))]  # type: ignore[func-returns-value]

        if not unique_tids:
            return

        print(f"    [Phase 5] Detail re-extraction: {len(unique_tids)} trajectories")

        async def _one_traj(tid: str) -> Optional[Tuple[str, List[str]]]:
            traj = self.trajectories.get(tid)
            if not traj:
                return None
            ts = self.traj_sums.get(traj.summary_id or "")
            if not ts:
                return None
            r = self.reflections.get(ts.reflection_id)
            if not r:
                return None
            existing = "\n".join(f"- {item}" for item in r.reflection_list)
            try:
                result = await async_llm(
                    async_client,
                    "You extract dense factual memory from conversation text. Respond in JSON.",
                    f"Here is a conversation episode and its current memory reflections.\n\n"
                    f"Episode:\n{traj.chunk_text}\n\n"
                    f"Existing reflections:\n{existing}\n\n"
                    "Generate 2-3 NEW reflection statements capturing specific facts, details, "
                    "or nuances present in the episode that are not yet captured above. "
                    "Do not repeat anything already listed. "
                    "Each statement must be fully self-contained (use names, not pronouns).\n\n"
                    'Return JSON with "new_reflections": list of statement strings.',
                )
                new_items = result.get("new_reflections", [])
                if not isinstance(new_items, list):
                    return None
                return ts.reflection_id, [s for s in new_items[:3] if s]
            except Exception:
                return None

        tasks   = [_one_traj(tid) for tid in unique_tids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        appended = 0
        for res in results:
            if isinstance(res, Exception) or res is None:
                continue
            rid, new_items = res
            if await self._append_reflection_items_async(rid, new_items):
                appended += 1
        print(f"      Appended new items to {appended}/{len(unique_tids)} reflections.")

    async def _grade_answer_async(
        self,
        question: str,
        category: str,
        mr: Dict[str, float],
        query_texts: List[str],
        gold_answer: str,
        gold_r: Set[str],
        source_traj_ids: List[str],
        max_items: int = 60,
        gold_answer_relative: Optional[str] = None,
        gold_answer_absolute: Optional[str] = None,
    ) -> Dict:
        """Grade a training question against its gold answer.

        Replicates the eval pipeline's item selection (semantic + BM25, top max_items)
        so that `gold_in_top_items` faithfully reflects what the answer LLM would see.

        Returns:
            correct              – bool
            partial              – bool  (multi-hop: some hops correct, not all)
            partial_hops_correct – List[str]  source_traj_ids that were answerable
            gold_in_top_items    – bool  (gold reflection's items present in filtered set)
        """
        # ── 1. Assemble items from retrieved reflections (with timestamps) ────
        plain_items:   List[str] = []
        display_items: List[str] = []
        gold_item_set: Set[str]  = set()

        for rid in mr:
            r = self.reflections.get(rid)
            if not r:
                continue
            ts = ""
            if r.trajectory_summary_ids:
                ts_obj = self.traj_sums.get(r.trajectory_summary_ids[0])
                if ts_obj:
                    t = self.trajectories.get(ts_obj.trajectory_id)
                    if t:
                        ts = t.timestamp
            ts_tag = f"[{ts}] " if ts else ""
            for item in r.reflection_list:
                plain_items.append(item)
                display_items.append(f"{ts_tag}{item}")
                if rid in gold_r:
                    gold_item_set.add(item)

        # ── 2. Per-item scoring (semantic + BM25) — matches eval pipeline ─────
        filtered_display: List[str]
        if plain_items:
            try:
                q_embs, item_embs_arr = await asyncio.gather(
                    async_embed(query_texts),
                    async_embed(plain_items),
                )
                sem_mat      = cos_sim(q_embs, item_embs_arr)               # (Q, N)
                bm25_raw     = self._bm25_score_items(query_texts, plain_items)
                bm25_row_max = bm25_raw.max(axis=1, keepdims=True)
                bm25_row_max = np.where(bm25_row_max > 0, bm25_row_max, 1.0)
                bm25_norm    = bm25_raw / bm25_row_max
                combined     = 0.6 * sem_mat + 0.4 * bm25_norm             # (Q, N)
                item_scores  = np.max(combined, axis=0)                    # (N,)
                ranked       = sorted(range(len(plain_items)),
                                      key=lambda i: item_scores[i], reverse=True)
                if max_items > 0:
                    ranked = ranked[:max_items]
                filtered_display = [display_items[i] for i in ranked]
                filtered_plain   = [plain_items[i]   for i in ranked]
            except Exception:
                filtered_display = display_items[:max_items] if max_items > 0 else display_items
                filtered_plain   = plain_items[:max_items]   if max_items > 0 else plain_items
        else:
            filtered_display = []
            filtered_plain   = []

        gold_in_top_items = bool(gold_item_set & set(filtered_plain))

        # ── 3. Generate short answer ──────────────────────────────────────────
        content_block = "\n".join(f"- {line}" for line in filtered_display) or "(no context retrieved)"
        try:
            generated_answer = await async_synthesize_from_bullets(
                async_client, question, content_block,
            )
        except Exception:
            generated_answer = ""

        # ── 4. Judge ──────────────────────────────────────────────────────────
        # For temporal questions, concatenate relative / absolute into one gold
        # string so the judge accepts either form in a single call.
        if category == "temporal" and gold_answer_relative and gold_answer_absolute:
            effective_gold = f"{gold_answer_relative} / {gold_answer_absolute}"
        else:
            effective_gold = gold_answer

        try:
            raw = await async_llm(
                async_client,
                "You are a fair evaluator. Label answers as CORRECT or WRONG and respond in JSON.",
                "Your task is to label an answer to a question as CORRECT or WRONG.\n\n"
                f"Question: {question}\n"
                f"Gold answer: {effective_gold}\n"
                f"Generated answer: {generated_answer}\n\n"
                "Be generous: if the generated answer conveys the same fact or date as the gold "
                "answer (even in different format or with extra words), label it CORRECT. "
                "For dates, 'May 7th' and '7 May' are the same. "
                "Relative and absolute dates referring to the same point in time are the same.\n\n"
                'Return JSON with "label": "CORRECT" or "WRONG".',
            )
            correct = str(raw.get("label", "")).strip().upper() == "CORRECT"
        except Exception:
            correct = False

        # ── 5. Multi-hop partial credit ───────────────────────────────────────
        partial_hops_correct: List[str] = []
        if category == "multi_hop" and not correct:
            # For each source trajectory, check if gold reflection items for that hop
            # are in the filtered set — a proxy for whether that hop was answerable.
            for tid in source_traj_ids:
                traj = self.trajectories.get(tid)
                if not traj or not traj.summary_id:
                    continue
                ts_obj = self.traj_sums.get(traj.summary_id)
                if not ts_obj:
                    continue
                hop_rid = ts_obj.reflection_id
                hop_r   = self.reflections.get(hop_rid)
                if not hop_r:
                    continue
                hop_items = set(hop_r.reflection_list)
                if hop_items & set(filtered_plain):
                    partial_hops_correct.append(tid)

        partial = bool(partial_hops_correct) and not correct

        return {
            "correct":              correct,
            "partial":              partial,
            "partial_hops_correct": partial_hops_correct,
            "gold_in_top_items":    gold_in_top_items,
            "generated_answer":     generated_answer,  # for logging
            "filtered_plain":       filtered_plain,    # for per-hop gold_in_top check in Phase 4
        }

    def _gold_retrieval_rank(
        self,
        gold_rids: Set[str],
        mc_all: Dict[str, float],
        mr_all: Dict[str, float],
    ) -> Optional[int]:
        """Return the minimum top_k needed to retrieve all gold reflections.

        Uses already-computed mc_all (all concept scores) and mr_all (all reflection
        scores within the current concept-gated candidates) — no extra retrieval.

        Returns None if all gold reflections have no concept connections (gate_closed
        cases where increasing top_k cannot help).
        """
        needed_ks = []
        for gold_rid in gold_rids:
            connected_cids = [
                cid for cid, rids in self.conn_c2r.connections.items()
                if gold_rid in rids
            ]
            if not connected_cids:
                continue  # gate_closed — top_k can't fix this

            best_concept_score = max(mc_all.get(cid, 0.0) for cid in connected_cids)
            # 1-indexed rank: number of concepts scoring strictly higher + 1
            concept_rank = sum(1 for s in mc_all.values() if s > best_concept_score) + 1

            if gold_rid in mr_all:
                gold_refl_score = mr_all[gold_rid]
                refl_rank = sum(1 for s in mr_all.values() if s > gold_refl_score) + 1
                needed_ks.append(max(concept_rank, refl_rank))
            else:
                # Gold concept wasn't matched at current top_k — use concept rank as proxy
                needed_ks.append(concept_rank)

        return max(needed_ks) if needed_ks else None

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

    def _get_reflection_timestamp(self, rid: str) -> Optional[datetime]:
        """Follow reflection → trajectory_summary → trajectory to get a naive datetime."""
        r = self.reflections.get(rid)
        if not r or not r.trajectory_summary_ids:
            return None
        ts_obj = self.traj_sums.get(r.trajectory_summary_ids[0])
        if not ts_obj:
            return None
        t = self.trajectories.get(ts_obj.trajectory_id)
        if not t:
            return None
        return _parse_dt(t.timestamp)

    async def _async_extract_temporal_range(
        self, question: str
    ) -> Optional[Tuple[datetime, datetime]]:
        """Returns (start, end) ONLY when a date range is directly derivable from the question.

        Named events without explicit dates (e.g. "Paris trip") return None — the system
        cannot derive a date range for those without external knowledge.
        """
        if not _has_temporal_signal(question):
            return None
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            result = await async_llm(
                async_client,
                "You extract date ranges from questions. Respond in JSON.",
                f"""Today's date: {today}
Question: {question}

Can a precise date range be derived from this question?

Set confidence to "high" ONLY when the question contains a time reference specific enough to compute an unambiguous date range:
- An explicit calendar date or month ("on March 15", "in October 2023", "January 3rd")
- A countable relative unit from today ("last Tuesday", "two weeks ago", "last month", "3 days ago")

Set confidence to "low" for ALL of the following:
- Vague or non-countable expressions ("recently", "a while ago", "before", "after", "at some point")
- Named events without a literal date ("the Paris trip", "the birthday party", "last summer", "our vacation")
- Implicit timing ("when they were doing X", "during the discussion about Y", "at the time of Z")
- General, preference, or factual questions with no temporal anchor

When confidence is "high", provide a GENEROUS range (±1 week for a specific date, full month ±2 weeks for a month reference).

Return JSON: {{"confidence": "high" or "low", "start_date": "YYYY-MM-DD" or null, "end_date": "YYYY-MM-DD" or null}}"""
            )
            if result.get("confidence") != "high":
                return None
            start_str = result.get("start_date")
            end_str = result.get("end_date")
            if not start_str or not end_str:
                return None
            return (datetime.fromisoformat(start_str), datetime.fromisoformat(end_str))
        except Exception:
            return None

    async def async_retrieve(self, question: str, top_k: Optional[int] = None,
                              use_query_components: bool = True
                              ) -> Tuple[Dict, Dict, Dict, Dict, QueryComponents4]:
        """Async version of retrieve: async LLM for query components + batched async embeds.

        _match_layer_reflections uses stored per-item embeddings (populated at ingest time),
        so no embed call is needed at query time for the memory side.
        use_query_components: if False, skip LLM query-component generation and use
        the raw question as both concept_texts and predicted_reflections.
        """
        if top_k is None:
            top_k = self._adapted_top_k
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
        mr_top, mr_all = self._match_layer_reflections(predicted_reflections, set(mc), top_k, embs=pr_embs)

        # Temporal re-ranking: only fires when a date range is derivable from the question
        temporal_range = await self._async_extract_temporal_range(question)
        if temporal_range:
            start_dt, end_dt = temporal_range
            for rid in mr_all:
                ts = self._get_reflection_timestamp(rid)
                in_range = ts is not None and start_dt <= ts <= end_dt
                mr_all[rid] = 0.7 * mr_all[rid] + 0.3 * (1.0 if in_range else 0.0)
            top_rids = sorted(mr_all, key=mr_all.get, reverse=True)[:top_k]
            mr = {r: mr_all[r] for r in top_rids}
        else:
            mr = mr_top

        ms, _ = self._match_layer_summaries(predicted_summary, set(mr), top_k, emb=s_emb)
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

    async def async_query(self, question: str, top_k: int = 5,
                          max_items: int = 60,
                          use_query_components: bool = False) -> str:
        """Full pipeline: retrieve → per-item hybrid scoring → synthesise → trajectory fallback."""
        mc, mr, ms, mt, qc = await self.async_retrieve(question, top_k,
                                                        use_query_components=use_query_components)

        # ── Build flat item lists with timestamp tags ─────────────────────────
        def _refl_timestamp(rid: str) -> str:
            r = self.reflections.get(rid)
            if r and r.trajectory_summary_ids:
                ts_obj = self.traj_sums.get(r.trajectory_summary_ids[0])
                if ts_obj:
                    t = self.trajectories.get(ts_obj.trajectory_id)
                    if t:
                        return t.timestamp
            return ""

        plain_groups:   List[List[str]] = []
        display_groups: List[List[str]] = []
        for rid in list(mr.keys())[:top_k]:
            r = self.reflections.get(rid)
            if r:
                ts = _refl_timestamp(rid)
                ts_tag = f"[{ts}] " if ts else ""
                plain_groups.append(list(r.reflection_list))
                display_groups.append([f"{ts_tag}{item}" for item in r.reflection_list])

        all_plain   = [item for g in plain_groups   for item in g]
        all_display = [item for g in display_groups for item in g]

        # ── Per-item hybrid scoring (semantic + BM25) ─────────────────────────
        query_texts = (qc.predicted_reflections
                       if qc is not None and qc.predicted_reflections else [question])

        _gather_tasks = [async_embed(query_texts), async_embed(all_plain)]
        if self.persona.entries:
            _gather_tasks.append(async_embed([question]))
        _results = await asyncio.gather(*_gather_tasks, return_exceptions=True)

        q_embs    = _results[0] if not isinstance(_results[0], Exception) else None
        item_embs = _results[1] if not isinstance(_results[1], Exception) else None
        _persona_emb = (
            (_results[2][0] if not isinstance(_results[2], Exception) else None)
            if self.persona.entries else None
        )

        filtered_lines = all_display
        filtered_plain = all_plain
        if q_embs is not None and item_embs is not None and all_plain:
            try:
                sem_mat      = cos_sim(q_embs, item_embs)                              # (Q, N)
                bm25_raw     = self._bm25_score_items(query_texts, all_plain)          # (Q, N)
                bm25_row_max = bm25_raw.max(axis=1, keepdims=True)
                bm25_row_max = np.where(bm25_row_max > 0, bm25_row_max, 1.0)
                combined_mat = 0.6 * sem_mat + 0.4 * (bm25_raw / bm25_row_max)        # (Q, N)
                item_scores  = np.max(combined_mat, axis=0)                            # (N,)

                flat_to_group: List[Tuple[int, int]] = [
                    (g_idx, w_idx)
                    for g_idx, g in enumerate(plain_groups)
                    for w_idx in range(len(g))
                ]
                ranked_idx = sorted(range(len(all_display)),
                                    key=lambda i: item_scores[i], reverse=True)
                if max_items > 0:
                    ranked_idx = ranked_idx[:max_items]

                group_best: Dict[int, float] = {}
                group_widx: Dict[int, List[int]] = {}
                for fi in ranked_idx:
                    g_idx, w_idx = flat_to_group[fi]
                    sc = float(item_scores[fi])
                    if g_idx not in group_best or sc > group_best[g_idx]:
                        group_best[g_idx] = sc
                    group_widx.setdefault(g_idx, []).append(w_idx)

                ordered = sorted(group_best, key=group_best.get, reverse=True)
                filtered_lines, filtered_plain = [], []
                for g_idx in ordered:
                    for w_idx in sorted(group_widx[g_idx]):
                        filtered_lines.append(display_groups[g_idx][w_idx])
                        filtered_plain.append(plain_groups[g_idx][w_idx])
            except Exception:
                filtered_lines = all_display[:max_items] if max_items > 0 else all_display
                filtered_plain = all_plain[:max_items]   if max_items > 0 else all_plain

        # ── Trajectory fallback when items are sparse ─────────────────────────
        traj_blocks = []
        if len(filtered_lines) < 3:
            for tid in list(mt.keys())[:3]:
                t = self.trajectories.get(tid)
                if t:
                    ts_tag = f"[{t.timestamp}]\n" if t.timestamp else ""
                    traj_blocks.append(f"{ts_tag}{t.chunk_text}")

        # ── Persona note ──────────────────────────────────────────────────────
        persona_note = ""
        if _persona_emb is not None:
            try:
                _best_name, _best_score = None, -1.0
                for _pname, _pentry in self.persona.entries.items():
                    if _pentry.embedding is not None and len(_pentry.embedding) > 0:
                        _score = float(cos_sim(
                            _persona_emb.reshape(1, -1), _pentry.embedding.reshape(1, -1)
                        )[0, 0])
                        if _score > _best_score:
                            _best_score, _best_name = _score, _pname
                if _best_name:
                    persona_note = f"\nUser context [{_best_name}]: {self.persona.entries[_best_name].summary}"
            except Exception:
                pass

        # ── Primary synthesis ─────────────────────────────────────────────────
        content_block = "\n".join(f"- {line}" for line in filtered_lines)
        if traj_blocks:
            content_block += "\n\nSupporting passages:\n" + "\n---\n".join(traj_blocks)
        if not content_block:
            content_block = "(no context retrieved)"

        answer = await async_synthesize_from_bullets(
            async_client, question, content_block, persona_note
        )

        # ── Trajectory fallback when primary returns unknown ──────────────────
        if answer.strip().lower() == "unknown":
            seen_sids: Set[str] = set()
            candidate_sums: List[Dict] = []
            for rid in list(mr.keys())[:top_k]:
                for sid in self.conn_r2s.get(rid):
                    if sid in seen_sids or sid not in self.traj_sums:
                        continue
                    seen_sids.add(sid)
                    ts_obj = self.traj_sums[sid]
                    traj   = self.trajectories.get(ts_obj.trajectory_id)
                    ts_tag = f"[{traj.timestamp}] " if traj and traj.timestamp else ""
                    candidate_sums.append({"sid": sid, "ts_tag": ts_tag, "text": ts_obj.text})

            if candidate_sums:
                numbered = "\n".join(
                    f"{i+1}. {c['ts_tag']}{c['text']}" for i, c in enumerate(candidate_sums)
                )
                try:
                    sel = await async_llm(
                        async_client,
                        "You are a relevance filter for memory retrieval. Respond in JSON.",
                        f"Question: {question}{persona_note}\n\nTrajectory summaries:\n{numbered}\n\n"
                        "Which summaries are most likely to contain information needed to answer the question? "
                        'Return JSON with "indices": a list of 1-based integers.',
                    )
                    indices = sel.get("indices", [])
                    selected = [candidate_sums[i - 1] for i in indices
                                if isinstance(i, int) and 1 <= i <= len(candidate_sums)]
                    if not selected:
                        selected = candidate_sums
                except Exception:
                    selected = candidate_sums

                passages = []
                for c in selected:
                    ts_obj = self.traj_sums.get(c["sid"])
                    if ts_obj:
                        traj = self.trajectories.get(ts_obj.trajectory_id)
                        if traj:
                            ts_tag = f"[{traj.timestamp}]\n" if traj.timestamp else ""
                            passages.append(f"{ts_tag}{traj.chunk_text}")
                if passages:
                    try:
                        fb = await async_synthesize_from_passages(
                            async_client, question, "\n---\n".join(passages), persona_note
                        )
                        if fb.strip():
                            answer = fb
                    except Exception:
                        pass

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
        """Sync wrapper around async_evaluate for backwards compatibility."""
        return asyncio.run(self.async_evaluate(questions, top_k))

    async def async_evaluate(self, questions: List[Dict], top_k: int = 5,
                              concurrency: int = 8) -> Dict:
        """
        questions: list of dicts with 'question' and 'source_id' keys.
        Returns metrics: questions_with_gold, coverage, MRR per layer.
        """
        layers = ["concepts", "reflections", "trajectory_summaries"]
        counters = {layer: {"total_gold": 0, "retrieved_gold": 0,
                            "qwg": 0, "rr": []} for layer in layers}
        n = len(questions)
        sem = asyncio.Semaphore(concurrency)

        async def _eval_one(qd: Dict) -> Dict:
            async with sem:
                q = qd["question"]
                mc, mr, ms, mt, _ = await self.async_retrieve(q, top_k)
                return {"qd": qd, "mc": mc, "mr": mr, "ms": ms}

        results = await asyncio.gather(*[_eval_one(qd) for qd in questions])

        for res in results:
            qd = res["qd"]
            if "trajectory_id" in qd:
                gold = self.get_gold_for_trajectory(qd["trajectory_id"])
            else:
                gold = self.get_gold_items(qd["source_id"])
            layer_results = {
                "concepts":             (res["mc"], gold["concepts"]),
                "reflections":          (res["mr"], gold["reflections"]),
                "trajectory_summaries": (res["ms"], gold["trajectory_summaries"]),
            }
            for layer, (matched_dict, gold_ids) in layer_results.items():
                matched_ids = set(matched_dict.keys())
                hit_ids = matched_ids & gold_ids
                counters[layer]["total_gold"]     += len(gold_ids)
                counters[layer]["retrieved_gold"] += len(hit_ids)
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


# =============================================================================
# SQLite persistence
# =============================================================================

import sqlite3
from collections import defaultdict as _defaultdict


def _blob(a: np.ndarray) -> bytes:
    return a.astype(np.float32).tobytes()


def _emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def save_to_sqlite(db_path: str, mem: "ModifiedMemory") -> None:
    """Serialize a ModifiedMemory instance to a SQLite file (overwrites if exists)."""
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE concepts (
            id TEXT PRIMARY KEY, text TEXT, embedding BLOB, reflection_ids TEXT
        );
        CREATE TABLE reflections (
            id TEXT PRIMARY KEY, reflection_list TEXT, embedding BLOB,
            concept_ids TEXT, trajectory_summary_ids TEXT
        );
        CREATE TABLE reflection_item_embeddings (
            reflection_id TEXT, idx INTEGER, embedding BLOB,
            PRIMARY KEY (reflection_id, idx)
        );
        CREATE TABLE traj_sums (
            id TEXT PRIMARY KEY, text TEXT, embedding BLOB,
            reflection_id TEXT, trajectory_id TEXT
        );
        CREATE TABLE trajectories (
            id TEXT PRIMARY KEY, chunk_text TEXT, timestamp TEXT,
            source_id TEXT, source_type TEXT, summary_id TEXT, concept_ids TEXT
        );
        CREATE TABLE persona_entries (
            name TEXT PRIMARY KEY, summary TEXT, embedding BLOB
        );
        CREATE TABLE rubrics (
            doc_type TEXT PRIMARY KEY, instructions TEXT
        );
        CREATE TABLE connections (
            table_name TEXT, src TEXT, tgt TEXT,
            times_traversed INTEGER, times_led_to_gold INTEGER, newly_added INTEGER,
            PRIMARY KEY (table_name, src, tgt)
        );
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY, value TEXT
        );
    """)

    cur.executemany("INSERT INTO concepts VALUES (?,?,?,?)", [
        (c.id, c.text, _blob(c.embedding), json.dumps(c.reflection_ids))
        for c in mem.concepts.values()
    ])
    cur.executemany("INSERT INTO reflections VALUES (?,?,?,?,?)", [
        (r.id, json.dumps(r.reflection_list), _blob(r.embedding),
         json.dumps(r.concept_ids), json.dumps(r.trajectory_summary_ids))
        for r in mem.reflections.values()
    ])
    cur.executemany("INSERT INTO reflection_item_embeddings VALUES (?,?,?)", [
        (r.id, idx, _blob(e))
        for r in mem.reflections.values()
        for idx, e in enumerate(r.item_embeddings or [])
    ])
    cur.executemany("INSERT INTO traj_sums VALUES (?,?,?,?,?)", [
        (ts.id, ts.text, _blob(ts.embedding), ts.reflection_id, ts.trajectory_id)
        for ts in mem.traj_sums.values()
    ])
    cur.executemany("INSERT INTO trajectories VALUES (?,?,?,?,?,?,?)", [
        (t.id, t.chunk_text, t.timestamp, t.source_id, t.source_type,
         t.summary_id, json.dumps(t.concept_ids))
        for t in mem.trajectories.values()
    ])
    cur.executemany("INSERT INTO persona_entries VALUES (?,?,?)", [
        (pe.name, pe.summary, _blob(pe.embedding))
        for pe in mem.persona.entries.values()
    ])
    cur.executemany("INSERT INTO rubrics VALUES (?,?)", [
        (v.doc_type, v.instructions) for v in mem.rubrics.values()
    ])

    def _insert_conn(name: str, cm: ConnectionManager4) -> None:
        cur.executemany("INSERT INTO connections VALUES (?,?,?,?,?,?)", [
            (name, src, tgt, s.times_traversed, s.times_led_to_gold, int(s.newly_added))
            for (src, tgt), s in cm.stats.items()
        ])

    _insert_conn("conn_c2r", mem.conn_c2r)
    _insert_conn("conn_r2s", mem.conn_r2s)

    cur.executemany("INSERT INTO metadata VALUES (?,?)", {
        "source_registry":      json.dumps(mem.source_registry),
        "_docs_since_sleep":    json.dumps(mem._docs_since_sleep),
        "_convs_since_sleep":   json.dumps(mem._convs_since_sleep),
        "last_sleep_stats":     json.dumps(mem.last_sleep_stats),
        "_adapted_top_k":       json.dumps(mem._adapted_top_k),
        "persona_last_updated": mem.persona.last_updated,
    }.items())

    con.commit()
    con.close()


def load_from_sqlite(db_path: str) -> "ModifiedMemory":
    """Deserialize a ModifiedMemory instance from a SQLite file."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    mem = ModifiedMemory()

    mem.concepts = {
        row[0]: Concept4(id=row[0], text=row[1], embedding=_emb(row[2]),
                         reflection_ids=json.loads(row[3]))
        for row in cur.execute("SELECT id, text, embedding, reflection_ids FROM concepts")
    }

    item_embs = _defaultdict(list)
    for rid, idx, blob in cur.execute(
        "SELECT reflection_id, idx, embedding "
        "FROM reflection_item_embeddings ORDER BY reflection_id, idx"
    ):
        item_embs[rid].append(_emb(blob))

    mem.reflections = {
        row[0]: Reflection4(
            id=row[0], reflection_list=json.loads(row[1]), embedding=_emb(row[2]),
            concept_ids=json.loads(row[3]), trajectory_summary_ids=json.loads(row[4]),
            item_embeddings=item_embs.get(row[0], []),
        )
        for row in cur.execute(
            "SELECT id, reflection_list, embedding, concept_ids, trajectory_summary_ids "
            "FROM reflections"
        )
    }

    mem.traj_sums = {
        row[0]: TrajectorySummary4(
            id=row[0], text=row[1], embedding=_emb(row[2]),
            reflection_id=row[3], trajectory_id=row[4],
        )
        for row in cur.execute(
            "SELECT id, text, embedding, reflection_id, trajectory_id FROM traj_sums"
        )
    }

    mem.trajectories = {
        row[0]: Trajectory4(
            id=row[0], chunk_text=row[1], timestamp=row[2],
            source_id=row[3], source_type=row[4],
            summary_id=row[5], concept_ids=json.loads(row[6]),
        )
        for row in cur.execute(
            "SELECT id, chunk_text, timestamp, source_id, source_type, summary_id, concept_ids "
            "FROM trajectories"
        )
    }

    meta = dict(cur.execute("SELECT key, value FROM metadata").fetchall())

    mem.persona = Persona(
        entries={
            row[0]: PersonaEntry(name=row[0], summary=row[1], embedding=_emb(row[2]))
            for row in cur.execute("SELECT name, summary, embedding FROM persona_entries")
        },
        last_updated=meta.get("persona_last_updated", ""),
    )

    mem.rubrics = {
        row[0]: TaskRubric(doc_type=row[0], instructions=row[1])
        for row in cur.execute("SELECT doc_type, instructions FROM rubrics")
    }

    def _load_conn(name: str) -> ConnectionManager4:
        cm = ConnectionManager4()
        for src, tgt, tt, tlg, na in cur.execute(
            "SELECT src, tgt, times_traversed, times_led_to_gold, newly_added "
            "FROM connections WHERE table_name=?", (name,)
        ):
            cm.connections[src].add(tgt)
            cm.stats[(src, tgt)] = ConnStats(
                times_traversed=tt, times_led_to_gold=tlg, newly_added=bool(na),
            )
        return cm

    mem.conn_c2r = _load_conn("conn_c2r")
    mem.conn_r2s = _load_conn("conn_r2s")

    mem.source_registry    = json.loads(meta.get("source_registry", "{}"))
    mem._docs_since_sleep  = json.loads(meta.get("_docs_since_sleep", "[]"))
    mem._convs_since_sleep = json.loads(meta.get("_convs_since_sleep", "[]"))
    mem.last_sleep_stats   = json.loads(meta.get("last_sleep_stats", "{}"))
    mem._adapted_top_k     = json.loads(meta.get("_adapted_top_k", "5"))

    con.close()
    return mem


# =============================================================================
# ConversationMemory — agent-facing API
# =============================================================================

class ConversationMemory:
    """
    Drop-in memory interface for agent systems.

    Quick start
    -----------
        mem = await ConversationMemory.create("my_agent.db")

        await mem.add_document(text, doc_id="paper_001")

        mem.add_turn("User", "What are the risks?")
        mem.add_turn("Agent", "The main risks are ...")
        await mem.flush_session(session_id="sess_001")

        await mem.add_conversation(turns, session_id="sess_002")

        answer = await mem.answer("What are the main risks?")

        await mem.consolidate()

        await mem.close()
    """

    def __init__(self, mem: ModifiedMemory, db_path: str) -> None:
        self._mem = mem
        self._db_path = db_path
        self._turn_buffers = _defaultdict(list)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, db_path: str = "agent_memory.db") -> "ConversationMemory":
        """Load from db_path if it exists, otherwise start fresh."""
        if os.path.exists(db_path):
            mem = load_from_sqlite(db_path)
        else:
            mem = ModifiedMemory()
        return cls(mem, db_path)

    def save(self) -> None:
        save_to_sqlite(self._db_path, self._mem)

    async def close(self) -> None:
        self.save()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def add_document(
        self,
        text: str,
        doc_id: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Ingest a document or KB article (auto-chunked)."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        await self._mem.add_content_async(
            text=text, source_id=doc_id,
            source_type="document", timestamp=ts, chunk=True,
        )

    def add_turn(
        self,
        speaker: str,
        content: str,
        session_id: str = "default",
    ) -> None:
        """Buffer a single live turn. Call flush_session() when the session ends."""
        self._turn_buffers[session_id].append(f"{speaker}: {content}")

    async def flush_session(
        self,
        session_id: str = "default",
        timestamp: Optional[str] = None,
    ) -> None:
        """Commit buffered turns for session_id to long-term memory."""
        lines = self._turn_buffers.pop(session_id, [])
        if not lines:
            return
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        await self._mem.add_content_async(
            text="\n".join(lines), source_id=session_id,
            source_type="conversation", timestamp=ts, chunk=False,
        )

    async def add_conversation(
        self,
        turns: List[Dict],
        session_id: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Ingest a completed session at once.

        turns: list of {"speaker": str, "content": str} dicts in order.
        """
        text = "\n".join(f"{t['speaker']}: {t['content']}" for t in turns)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        await self._mem.add_content_async(
            text=text, source_id=session_id,
            source_type="conversation", timestamp=ts, chunk=False,
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    async def answer(self, question: str, top_k: int = 5) -> str:
        return await self._mem.async_query(question, top_k=top_k)

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def consolidate(
        self,
        n_questions_per_chunk: int = 3,
        build_rubrics: bool = False,
    ) -> None:
        """Improve memory quality. Suggested cadence: every 5-10 sessions."""
        await self._mem.sleep_update_async(
            n_questions_per_chunk=n_questions_per_chunk,
            build_rubrics=build_rubrics,
        )
