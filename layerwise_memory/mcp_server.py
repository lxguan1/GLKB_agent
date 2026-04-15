"""
LayerMem MCP Server

Exposes the ModifiedMemory 4-layer hierarchical memory system as an MCP server.
The server holds a single ModifiedMemory instance in-process; all tool calls
operate on it. State is persisted via explicit save_snapshot calls.

Tools:
    add_content   — ingest text into memory
    query         — retrieve + answer a question (uses async_get_short_answer)
    sleep_update  — connection learning + persona + rubric refresh
    save_snapshot — persist memory to a JSON file
    get_status    — inspect current memory counts

Usage:
    python mcp_server.py

Configuration (environment variables):
    MCP_SNAPSHOT_PATH  — path to snapshot file (default: mem_snapshot_mcp.json)
    OPENAI_API_KEY     — required for real LLM/embedding calls
"""

import asyncio
import json
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from modified_memory_system import ModifiedMemory
from locomo_eval_modified import async_get_short_answer, save_mem_snapshot, load_mem_snapshot
from config import async_client

# ---------------------------------------------------------------------------
# Snapshot path
# ---------------------------------------------------------------------------

SNAPSHOT_PATH = os.getenv("MCP_SNAPSHOT_PATH", "mem_snapshot_mcp.json")

# ---------------------------------------------------------------------------
# Load or initialise memory at startup
# ---------------------------------------------------------------------------

if os.path.exists(SNAPSHOT_PATH):
    mem, _dia_to_traj = load_mem_snapshot(SNAPSHOT_PATH)
else:
    print(f"No snapshot found at '{SNAPSHOT_PATH}' — starting with empty memory.", file=sys.stderr)
    mem = ModifiedMemory()

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

app = Server("layerMem")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="add_content",
            description=(
                "Ingest a piece of text into the long-term memory system. "
                "The text is chunked, and each chunk is processed by an LLM to extract "
                "concepts (short topic labels), reflections (specific factual statements), "
                "and a trajectory summary. These form the 4-layer memory hierarchy used for retrieval.\n\n"
                "source_type controls how the content is treated downstream:\n"
                "- 'conversation': persona summaries for each speaker are updated during sleep_update.\n"
                "- 'document': a task rubric (output format instructions) is generated for the document "
                "type during sleep_update.\n\n"
                "Typical workflow: call add_content one or more times, then call sleep_update once "
                "to build connections and refresh metadata. Do not call sleep_update after every "
                "single add_content — batch ingestions first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The full text to store in memory.",
                    },
                    "source_id": {
                        "type": "string",
                        "description": (
                            "A unique identifier for this source (e.g. 'session_2024-03-01', "
                            "'paper_smith2023'). Used to group trajectories and look up gold items."
                        ),
                    },
                    "source_type": {
                        "type": "string",
                        "description": (
                            "'conversation' (default) or 'document'. "
                            "Conversations trigger persona updates during sleep_update; "
                            "documents trigger rubric generation."
                        ),
                        "default": "conversation",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": (
                            "ISO 8601 timestamp for this content (e.g. '2024-03-01T14:00:00Z'). "
                            "Stored with each trajectory and surfaced in query answers for temporal reasoning. "
                            "Defaults to the current UTC time if omitted."
                        ),
                    },
                    "chunk": {
                        "type": "boolean",
                        "description": (
                            "If true (default), long text is split into overlapping 600-token chunks before "
                            "extraction. Set to false for pre-segmented content (e.g. a single conversation "
                            "session) where splitting would break context."
                        ),
                        "default": True,
                    },
                },
                "required": ["text", "source_id"],
            },
        ),
        types.Tool(
            name="query",
            description=(
                "Retrieve relevant memory and answer a question. "
                "Uses a 4-layer retrieval pipeline (concepts → reflections → trajectory summaries → "
                "trajectories) combined with BM25 + semantic scoring to select the most relevant "
                "memory items, then calls an LLM to produce a concise answer.\n\n"
                "The answer style is short and conversational (typically 5 words or fewer), "
                "suited for factual recall questions such as 'Where did Alice grow up?' or "
                "'When did the study begin?'. Timestamps stored with each memory item are used "
                "to resolve relative time expressions (e.g. 'last week', 'two months ago').\n\n"
                "If the primary answer is 'unknown', the system automatically falls back to "
                "scanning raw trajectory passages before giving up.\n\n"
                "Returns: {\"answer\": \"...\", \"n_items_retrieved\": N}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer from memory.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Number of reflections to retrieve before scoring and filtering. "
                            "Higher values increase recall at the cost of more LLM context. Default 20."
                        ),
                        "default": 20,
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="sleep_update",
            description=(
                "Run the full post-ingestion update pipeline. This is an expensive operation "
                "(many LLM calls) and should be called once after a batch of add_content calls, "
                "not after every individual ingestion.\n\n"
                "Three stages run in sequence:\n"
                "1. Reflection refinement (optional): re-extracts supplementary facts from "
                "trajectories to improve reflection quality.\n"
                "2. Connection learning: generates synthetic questions for each trajectory, "
                "runs retrieval, and records connections between the memory layers that "
                "led to correct answers. These connections accelerate future retrieval.\n"
                "3. Persona update (conversations only): summarises each speaker's interests, "
                "background, and communication style from recent conversation sources.\n"
                "   Rubric generation (documents only): detects document type and writes "
                "output-format instructions used to shape query answers.\n\n"
                "Returns connection counts: {\"connections_c2r\": N, \"connections_r2s\": N}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n_questions_per_chunk": {
                        "type": "integer",
                        "description": (
                            "Number of synthetic questions generated per trajectory chunk during "
                            "connection learning. More questions improve connection coverage but "
                            "increase LLM cost. Default 3."
                        ),
                        "default": 3,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Retrieval depth used during connection learning. Default 5.",
                        "default": 5,
                    },
                    "refine_reflections": {
                        "type": "boolean",
                        "description": (
                            "If true (default), run a supplementary extraction pass over all "
                            "trajectories to add missing facts before connection learning."
                        ),
                        "default": True,
                    },
                    "build_rubrics": {
                        "type": "boolean",
                        "description": (
                            "If true (default), detect document type and generate output-format "
                            "rubrics for new document sources. Set to false to save LLM calls "
                            "when rubrics are not needed."
                        ),
                        "default": True,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="save_snapshot",
            description=(
                "Persist the entire memory state (all layers, connections, persona, rubrics) "
                "to a JSON file. The server does not auto-save — call this explicitly after "
                "sleep_update or at the end of a session to avoid losing ingested content. "
                "On next server startup the snapshot is loaded automatically if the path matches "
                f"the configured MCP_SNAPSHOT_PATH (currently '{SNAPSHOT_PATH}')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": f"File path to write. Defaults to '{SNAPSHOT_PATH}'.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_status",
            description=(
                "Return a summary of the current in-memory state: counts for each memory layer "
                "(trajectories, reflections, concepts, trajectory summaries), number of persona "
                "entries, list of rubric types, and how many sources are pending sleep_update. "
                "Useful for verifying that add_content succeeded or checking whether "
                "sleep_update needs to be called."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── add_content ──────────────────────────────────────────────────────────
    if name == "add_content":
        await mem.add_content_async(
            text=arguments["text"],
            source_id=arguments["source_id"],
            source_type=arguments.get("source_type", "conversation"),
            timestamp=arguments.get("timestamp"),
            chunk=arguments.get("chunk", True),
        )
        result = {
            "status": "ok",
            "source_id": arguments["source_id"],
            "trajectory_count": len(mem.source_registry.get(arguments["source_id"], [])),
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── query ─────────────────────────────────────────────────────────────────
    elif name == "query":
        answer, n_items, _fallback = await async_get_short_answer(
            mem,
            async_client,
            question=arguments["question"],
            top_k=arguments.get("top_k", 20),
        )
        result = {
            "answer": answer,
            "n_items_retrieved": n_items,
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── sleep_update ──────────────────────────────────────────────────────────
    elif name == "sleep_update":
        await mem.sleep_update_async(
            n_questions_per_chunk=arguments.get("n_questions_per_chunk", 3),
            top_k=arguments.get("top_k", 5),
            refine_reflections=arguments.get("refine_reflections", True),
            build_rubrics=arguments.get("build_rubrics", True),
        )
        result = {
            "status": "ok",
            "connections_c2r": mem.conn_c2r.total_connections(),
            "connections_r2s": mem.conn_r2s.total_connections(),
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── save_snapshot ─────────────────────────────────────────────────────────
    elif name == "save_snapshot":
        path = arguments.get("path", SNAPSHOT_PATH)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_mem_snapshot(mem, {}, path)
        result = {"status": "ok", "path": path}
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── get_status ────────────────────────────────────────────────────────────
    elif name == "get_status":
        result = {
            "trajectories":      len(mem.trajectories),
            "reflections":       len(mem.reflections),
            "concepts":          len(mem.concepts),
            "traj_sums":         len(mem.traj_sums),
            "persona_entries":   len(mem.persona.entries),
            "rubrics":           list(mem.rubrics.keys()),
            "docs_since_sleep":  len(mem._docs_since_sleep),
            "convs_since_sleep": len(mem._convs_since_sleep),
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
