# agent.py - Single-agent GLKB system with on-demand skill loading
#
# Replaces the previous multi-agent pipeline (QuestionRouter -> Parallel KG+Lit -> FinalAnswer)
# with a single LlmAgent that has all tools plus SkillToolset for loading
# detailed skill instructions on demand.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.skills import Skill
from google.adk.skills.models import Frontmatter, Resources
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools import FunctionTool
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# -----------------------------------------
# Logging (standard Python logging per ADK guide)
# -----------------------------------------

LOG_DIR = os.getenv(
    "AGENTS_LOG_DIR",
    os.path.join(os.path.dirname(__file__), "..", "agent_logs"),
)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "agent.log")),
        logging.StreamHandler(),
    ],
    force=True,  # Override ADK CLI's prior config
)

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# -----------------------------------------
# Tools (from tools.py)
# -----------------------------------------
from tools import glkb_tools, pubmed_tools

# -----------------------------------------
# Model
# -----------------------------------------
# Override via LLM_MODEL env var. Examples:
#   openai/gpt-5.2                          (default, requires OPENAI_API_KEY)
#   openrouter/openai/gpt-4o-mini:free      (requires OPENROUTER_API_KEY)
#   openrouter/<provider>/<model>:free      (any OpenRouter model)
_model_name = os.getenv("LLM_MODEL", "openai/gpt-5.2")
LLM_MODEL = LiteLlm(model=_model_name)

# -----------------------------------------
# Skill Loading Helper
# -----------------------------------------

def load_skill_from_directory(skill_dir: Path) -> Skill:
    """Load a skill from a directory containing SKILL.md and optional references/."""
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text()

    # Parse YAML frontmatter between --- markers
    parts = text.split("---", 2)
    if len(parts) >= 3:
        frontmatter_data = yaml.safe_load(parts[1])
        instructions = parts[2].strip()
    else:
        frontmatter_data = {"name": skill_dir.name, "description": ""}
        instructions = text

    frontmatter = Frontmatter(
        name=frontmatter_data.get("name", skill_dir.name),
        description=frontmatter_data.get("description", ""),
    )

    # Load references if they exist (dict[str, str]: name -> content)
    references = {}
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for ref_file in sorted(refs_dir.glob("*.md")):
            references[ref_file.stem] = ref_file.read_text()

    resources = Resources(references=references)

    return Skill(
        frontmatter=frontmatter,
        instructions=instructions,
        resources=resources,
    )

# -----------------------------------------
# Memory (LayerMem — direct integration)
# -----------------------------------------
# Set LAYERMEM_ENABLED=true in .env to enable.

# Hyperparameters
MEMORY_DB_PATH = os.getenv(
    "LAYERMEM_DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent_logs", "layermem.db")),
)
BUFFER_THRESHOLD_TOKENS = 800     # flush when buffered turns exceed this
CONSOLIDATE_EVERY_N_FLUSHES = 5   # run sleep_update every N flushes

_LAYERMEM_ENABLED = os.getenv("LAYERMEM_ENABLED", "false").lower() == "true"

mem = None
_session_id = f"session_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"
_buffer_tokens: int = 0
_flush_count: int = 0

if _LAYERMEM_ENABLED:
    _LAYERMEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "layerwise_memory"))
    sys.path.insert(0, _LAYERMEM_DIR)
    os.environ.setdefault("PROMPT_MODE", "conversational")

    from agent_memory import ConversationMemory, ModifiedMemory, load_from_sqlite

    _mem_inner = load_from_sqlite(MEMORY_DB_PATH) if os.path.exists(MEMORY_DB_PATH) else ModifiedMemory()
    mem = ConversationMemory(_mem_inner, MEMORY_DB_PATH)
    logger.info(f"LayerMem enabled (db: {MEMORY_DB_PATH})")
else:
    logger.info("LayerMem disabled (set LAYERMEM_ENABLED=true to enable)")


def _approx_tokens(text: str) -> int:
    return len(text) // 4


async def _trigger_flush(wait: bool = False) -> None:
    """Pop the turn buffer and ingest. Background by default; await if wait=True."""
    global _buffer_tokens, _flush_count
    lines = mem._turn_buffers.pop(_session_id, [])
    _buffer_tokens = 0
    if not lines:
        return
    content = "\n".join(lines)
    timestamp = datetime.now(timezone.utc).isoformat()
    _flush_count += 1
    source_id = f"{_session_id}_part{_flush_count}"
    ingestion = mem._mem.add_content_async(content, source_id, "conversation", timestamp, False)
    if wait:
        await ingestion
    else:
        asyncio.create_task(ingestion)
    if _flush_count % CONSOLIDATE_EVERY_N_FLUSHES == 0:
        asyncio.create_task(mem.consolidate())


async def _memory_after_agent_callback(callback_context) -> None:
    """Auto-buffer each turn and flush to LayerMem when threshold is reached."""
    global _buffer_tokens
    if not _LAYERMEM_ENABLED:
        return None

    user_text = ""
    if callback_context.user_content:
        user_text = " ".join(
            p.text for p in (getattr(callback_context.user_content, "parts", None) or [])
            if getattr(p, "text", None)
        )

    agent_text = ""
    for event in reversed(callback_context.session.events):
        if event.author == "GLKBAgent" and event.content:
            texts = [p.text for p in (event.content.parts or []) if getattr(p, "text", None)]
            if texts:
                agent_text = " ".join(texts)
                break

    if not user_text and not agent_text:
        return None

    turn_tokens = _approx_tokens(user_text + agent_text)

    # Flush before adding if threshold would be crossed
    if _buffer_tokens > 0 and _buffer_tokens + turn_tokens >= BUFFER_THRESHOLD_TOKENS:
        await _trigger_flush()

    logger.debug(f"Memory buffer | user={len(user_text)}chars agent={len(agent_text)}chars turn_tokens~{turn_tokens}")
    if user_text:
        mem.add_turn("User", user_text, session_id=_session_id)
    if agent_text:
        mem.add_turn("GLKBAgent", agent_text, session_id=_session_id)
    _buffer_tokens += turn_tokens

    # Flush if a single turn alone exceeds threshold
    if _buffer_tokens >= BUFFER_THRESHOLD_TOKENS:
        await _trigger_flush()

    return None


async def query_memory(question: str) -> dict:
    """Query long-term memory for relevant context from past sessions."""
    if not _LAYERMEM_ENABLED:
        return {"answer": "Memory is disabled. Set LAYERMEM_ENABLED=true to enable."}
    answer = await mem.answer(question)
    return {"answer": answer}


async def save_memory() -> dict:
    """Flush current buffer, consolidate memory, and persist to disk."""
    if not _LAYERMEM_ENABLED:
        return {"status": "disabled"}
    await _trigger_flush(wait=True)
    await mem.consolidate(n_questions_per_chunk=1)
    mem.save()
    return {"status": "ok", "path": MEMORY_DB_PATH}


class MemoryToolset(BaseToolset):
    """Exposes memory tools and flushes + saves on runner shutdown."""

    def __init__(self):
        super().__init__()

    async def get_tools(self, readonly_context: ReadonlyContext = None) -> list:
        return [FunctionTool(query_memory), FunctionTool(save_memory)]

    async def close(self) -> None:
        if not _LAYERMEM_ENABLED or mem is None:
            return
        await _trigger_flush(wait=True)
        mem.save()
        logger.info("Memory flushed and saved on runner close.")

# -----------------------------------------
# Load Skills
# -----------------------------------------
SKILLS_DIR = Path(__file__).parent / "skills"
kg_skill = load_skill_from_directory(SKILLS_DIR / "glkb_knowledge_graph")
lit_skill = load_skill_from_directory(SKILLS_DIR / "pubmed_reader")

logger.info(f"Loaded skill: {kg_skill.frontmatter.name}")
logger.info(f"Loaded skill: {lit_skill.frontmatter.name}")

# -----------------------------------------
# Base Instruction
# -----------------------------------------
BASE_INSTRUCTION = """
You are the GLKB biomedical QA assistant. The Genomic Literature Knowledge Base (GLKB) integrates over 263 million biomedical terms and more than 14.6 million biomedical relationships curated from 38 million PubMed abstracts and nine biomedical repositories.

You have access to skills for detailed workflows. Load them as needed:
- "glkb-knowledge-graph": Cypher query generation, schema navigation, vocabulary mapping for the GLKB Neo4j database
- "pubmed-reader": Article retrieval strategy using GLKB search and direct PubMed/PMC access

WORKFLOW:
1. Assess the question type:
   - KG-only (counts, lists, schema queries) -> load KG skill only
   - Needs biomedical explanation or evidence -> load both skills
   - Ambiguous -> load both skills
2. Load relevant skill(s) via load_skill and follow their instructions to query tools
3. Synthesize a grounded answer using the evidence gathered

IMPORTANT:
- You have access to the full conversation history. For follow-up questions, use context from previous exchanges.
- Filter results after each tool call to keep relevant and important items (e.g., high citations, high cooccurrences).
- If information is insufficient after querying, acknowledge limitations.
- Kindly refuse to answer questions that are not related to biomedical research, the GLKB database, or the GLKB agent system.

MEMORY WORKFLOW:
- At the start of each session, call query_memory to surface relevant prior context.
- Conversation turns are saved to memory automatically — do not attempt to save them manually.
- Call save_memory when the user ends the session or explicitly asks to save.

EVIDENCE AND CITATION WORKFLOW:
1. After gathering evidence from tools, identify the specific sentences or passages
   that directly support your answer.
2. For each article you will cite, call `cite_evidence` with:
   - pmid: the article's PubMed ID
   - quote: the EXACT sentence(s) from the tool output (abstract, full text, or
     KG evidence field) — do NOT paraphrase
   - context_type: "abstract", "fulltext", "kg_evidence", or "title"
3. You MUST call cite_evidence before referencing a PMID in your answer.
   Do not cite articles without registering evidence first.
   If you cannot find a specific supporting quote, the system will fall back to
   the article abstract automatically — but specific quotes are always preferred.
4. Then write your final answer, citing articles inline:
   [PMID](https://pubmed.ncbi.nlm.nih.gov/PMID)
   Example: "TP53 plays a key role in apoptosis [38743124](https://pubmed.ncbi.nlm.nih.gov/38743124)."
5. When summarizing database/graph structure results (not articles), do not cite.
6. Use markdown headers and bullet points for well-structured answers.
"""

# -----------------------------------------
# Root Agent
# -----------------------------------------
root_agent = LlmAgent(
    name="GLKBAgent",
    model=LLM_MODEL,
    description=(
        "GLKB biomedical QA agent that queries the Neo4j knowledge graph "
        "and retrieves PubMed literature to produce grounded, cited answers."
    ),
    instruction=BASE_INSTRUCTION,
    tools=[
        *glkb_tools,
        *pubmed_tools,
        SkillToolset(skills=[kg_skill, lit_skill]),
        MemoryToolset(),
    ],
    after_agent_callback=_memory_after_agent_callback,
)