# GLKB Agent

A biomedical question-answering system built with [Google ADK (Agent Development Kit)](https://google.github.io/adk-docs/) that queries the GLKB Neo4j knowledge graph (263M+ biomedical terms, 14.6M+ relationships) and retrieves PubMed literature to produce grounded, cited answers.

## Architecture

```
GLKBAgent (single LlmAgent, gpt-5.2)
  ├── GLKB Tools (always registered):
  │     get_database_schema, vocabulary_search, execute_cypher,
  │     article_search, cite_evidence
  ├── PubMed Tools (always registered):
  │     search_pubmed, fetch_abstract, get_fulltext,
  │     find_similar_articles, get_citing_articles, comprehensive_report
  └── SkillToolset (on-demand instruction loading):
        ├── glkb-knowledge-graph  → Cypher workflow, schema, query patterns
        └── pubmed-reader         → Article retrieval strategy, tool selection
```

The agent assesses each question, loads relevant skill instructions on-demand via `load_skill()`, uses the appropriate tools, and synthesizes a cited answer — all in one LLM session.

> **Note:** The previous multi-agent pipeline (QuestionRouterAgent → Parallel KG+Lit → EvidenceMergeAgent → FinalAnswerAgent) has been replaced by this single-agent design.

## Features

- **Single-agent with on-demand skills**: One `LlmAgent` handles routing, evidence gathering, and synthesis. Detailed workflow instructions are loaded from skill files only when needed, keeping the context window efficient.
- **Knowledge Graph Integration**: Queries GLKB Neo4j with auto-generated Cypher, including safety guards (write blocking, CartesianProduct rejection, 30s timeout, 500-row cap).
- **Literature Retrieval**: Searches GLKB-indexed articles and direct NCBI PubMed/PMC access with caching and adaptive rate limiting.
- **Evidence Grounding**: Agent must call `cite_evidence` with verbatim quotes before citing any PMID.
- **Conversation History**: Full multi-turn context available to the agent across exchanges.
- **REST API Service**: FastAPI service with SSE streaming, SQLite session persistence, and session rewind support.

## Project Structure

```
GLKB_agent/
├── my_agent/
│   ├── agent.py               # Single GLKBAgent definition + skill loading
│   ├── tools.py               # GLKB Neo4j tools and PubMed reader tools
│   ├── memory.py              # Memory tools (Mem0+Qdrant, currently inactive)
│   ├── run_async.py           # Direct async runner for single queries
│   ├── skills/
│   │   ├── glkb_knowledge_graph/
│   │   │   ├── SKILL.md                    # Cypher workflow instructions
│   │   │   └── references/
│   │   │       ├── schema.md               # Full GLKB schema reference
│   │   │       └── cypher-patterns.md      # Common query patterns
│   │   └── pubmed_reader/
│   │       ├── SKILL.md                    # Article retrieval strategy
│   │       └── references/
│   │           ├── pubmed-api.md           # PubMed API documentation
│   │           └── tool-reference.md       # Tool selection reference
│   └── scripts/
│       └── pubmed_reader/                  # NCBI E-utilities wrappers
│           ├── search_pubmed.py
│           ├── fetch_article.py
│           ├── fetch_fulltext.py
│           ├── find_citations.py
│           ├── find_similar.py
│           ├── comprehensive_report.py
│           └── utils/                      # Caching, rate limiting, validators
├── service/                   # FastAPI web service
│   ├── api.py                 # REST endpoints
│   ├── session_service.py     # Async SQLite session persistence
│   ├── runner.py              # ADK session bridging + rewind support
│   ├── models.py              # Pydantic v2 request/response models
│   └── requirements.txt       # Service dependencies
├── agent_logs/                # Runtime logs
│   └── agent.log
├── docs/                      # Design docs and guides
└── CLAUDE.md
```

## Prerequisites

- Python 3.11+
- Neo4j database with GLKB data
- OpenAI API key (for GPT-5.2 via LiteLLM)

## Installation

1. **Install Google ADK and dependencies**:
   ```bash
   pip install 'google-adk>=1.25.0' python-dotenv neo4j httpx litellm loguru pyyaml
   ```

2. **Install service dependencies** (for REST API):
   ```bash
   pip install -r service/requirements.txt
   ```

3. **Configure environment variables**:

   Create a `.env` file in the `my_agent/` directory:
   ```env
   # Neo4j Configuration
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=glkb

   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key

   # Optional: NCBI E-utilities (increases rate limit from 3 to 10 req/s)
   NCBI_API_KEY=your_ncbi_api_key
   NCBI_EMAIL=your_email@example.com

   # Optional: Custom log directory
   AGENTS_LOG_DIR=/path/to/logs
   ```

## Usage

### Option 1: ADK CLI (Development)

```bash
adk run my_agent
```

Example queries:
- "What is TP53?"
- "How many articles about CFTR were published since 2023?"
- "What genes are associated with diabetes?"

### Option 2: ADK Web UI

```bash
adk web my_agent --port 8080
```

### Option 3: Direct async runner

```bash
python my_agent/run_async.py --query "What is TP53?"
```

### Option 4: FastAPI REST Service (Production)

```bash
uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload

# Or directly (port 5001, debug logging):
python service/api.py
```

API Documentation available at: `http://localhost:8000/docs`

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/apps/{app}/users/{user}/sessions` | Create session |
| `GET` | `/apps/{app}/users/{user}/sessions` | List sessions |
| `GET` | `/apps/{app}/users/{user}/sessions/{id}` | Get session |
| `DELETE` | `/apps/{app}/users/{user}/sessions/{id}` | Delete session |
| `POST` | `/apps/{app}/users/{user}/sessions/{id}/chat` | Chat (sync) |
| `POST` | `/apps/{app}/users/{user}/sessions/{id}/chat/stream` | Chat (SSE stream) |
| `POST` | `/apps/{app}/users/{user}/sessions/{id}/rewind` | Rewind session to before an invocation |
| `GET` | `/apps/{app}/users/{user}/sessions/{id}/messages` | Get conversation history |
| `POST` | `/stream` | Simplified SSE endpoint (GLKB backend compatible) |

#### Example API Usage

```bash
# Create a session
SESSION=$(curl -s -X POST http://localhost:8000/apps/glkb/users/user1/sessions | jq -r '.id')

# Send a message (non-streaming)
curl -X POST "http://localhost:8000/apps/glkb/users/user1/sessions/${SESSION}/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is TP53?"}'

# Send a message (streaming via SSE)
curl -N -X POST "http://localhost:8000/apps/glkb/users/user1/sessions/${SESSION}/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "What genes are associated with breast cancer?"}'

# Rewind to before a specific invocation (for edit/regenerate)
curl -X POST "http://localhost:8000/apps/glkb/users/user1/sessions/${SESSION}/rewind" \
  -H "Content-Type: application/json" \
  -d '{"invocation_id": "inv_abc123"}'

# Get conversation history
curl "http://localhost:8000/apps/glkb/users/user1/sessions/${SESSION}/messages"
```

## Available Tools

### GLKB Knowledge Graph Tools

| Tool | Description |
|------|-------------|
| `get_database_schema` | Retrieve the full GLKB Neo4j schema |
| `vocabulary_search` | Search for biomedical concepts by name; includes OntologyMapping expansion |
| `execute_cypher` | Execute read-only Cypher queries (write ops blocked; cartesian products rejected) |
| `article_search` | Search GLKB-indexed PubMed articles by keywords or PubMed IDs with impact scoring |
| `cite_evidence` | Register a verbatim evidence quote from an article before citing it in the answer |

### PubMed / NCBI Tools

| Tool | Description |
|------|-------------|
| `search_pubmed` | Search PubMed via NCBI ESearch; supports date, author, journal filters and PubMed query syntax |
| `fetch_abstract` | Fetch abstract, metadata, MeSH terms, and keywords for a PMID |
| `get_fulltext` | Retrieve full-text sections from PMC Open Access articles (~3M available) |
| `find_similar_articles` | Find related papers via NCBI ELink (shared MeSH terms, citations, content similarity) |
| `get_citing_articles` | Find papers that cite a given PMID, sorted by date |
| `comprehensive_report` | Full single-article analysis: metadata, abstract, full text, similar papers, citation metrics |

## ADK Skills (On-Demand Instruction Loading)

Skills use ADK's `SkillToolset` for incremental context loading. Only the skill name and description are always present in the system prompt; full instructions and references are loaded on demand.

| Skill | When to load | Contents |
|-------|-------------|---------|
| `glkb-knowledge-graph` | Questions requiring KG queries | Cypher generation workflow, schema reference, common query patterns |
| `pubmed-reader` | Questions requiring literature evidence | Article retrieval strategy, tool selection guidance, PubMed API docs |

Skills are loaded from `my_agent/skills/` using `load_skill_from_directory()` in `agent.py` (ADK's `load_skill_from_dir` does not exist in 1.25.x).

## Knowledge Graph Schema

**Node Types:**
- `Article` — PubMed articles with metadata (pubmedid, title, abstract, pubdate, authors, n_citation, doi)
- `Journal` — Publication journals (title, impact_factor, issn)
- `Vocabulary` subtypes (connect to Articles via `ContainTerm`):
  - `Gene`, `DiseaseOrPhenotypicFeature`, `ChemicalEntity`, `SequenceVariant`, `MeshTerm`, `AnatomicalEntity`
- `Vocabulary` subtypes (ontology only, no Article links):
  - `Pathway`, `BiologicalProcess`, `CellularComponent`, `MolecularFunction`

**Key Relationship Types:**
- `ContainTerm` — Article → Vocabulary (article mentions a biomedical term)
- `Cite` — Article → Article
- `PublishedIn` — Article → Journal
- `GeneToDiseaseAssociation`, `GeneToGeneAssociation`, `GeneToPathwayAssociation`, `GeneToGoTermAssociation`
- `ChemicalOrDrugOrTreatmentToDiseaseOrPhenotypicFeatureAssociation`
- `VariantToGeneAssociation`, `VariantToDiseaseAssociation`
- `Cooccur` — Co-occurrence between vocabulary terms (with evidence list)
- `OntologyMapping` — Cross-references between ontologies
- `HierarchicalStructure` — Ontology parent/child relationships

**Full-text indexes:** `vocabulary_Names` (on `Vocabulary.name`), `article_Title` (on `Article.title`)

## Logging

Logs are written to `agent_logs/agent.log`:
```
[TOOL CALL] tool_name | Input: {...}
[TOOL RESULT] tool_name | Output: {...}
[TOOL ERROR] tool_name | Error: ...
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `NEO4J_URI` | Yes | Bolt URI (e.g., `bolt://host:7687`) |
| `NEO4J_USER` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `NEO4J_DATABASE` | Yes | Neo4j database name |
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-5.2 via LiteLLM |
| `NCBI_API_KEY` | No | NCBI API key — increases PubMed rate limit from 3 to 10 req/s |
| `NCBI_EMAIL` | No | Contact email for NCBI E-utilities (recommended for production) |
| `AGENTS_LOG_DIR` | No | Custom log directory (default: `agent_logs/`) |

## Model Configuration

Configured in `my_agent/agent.py`:
```python
LLM_MODEL = LiteLlm(model="openai/gpt-5.2")  # Single agent (GLKBAgent)
```

## Development

### Modifying the Agent

Edit `my_agent/agent.py` to:
- Change the model or base instruction
- Add or remove tools
- Add new skills to the `SkillToolset`

### Adding Tools

Edit `my_agent/tools.py`:
- Tool functions must be `async`
- Apply `@log_tool_call` decorator for logging
- Wrap with `FunctionTool()` for ADK
- Add to `glkb_tools` or `pubmed_tools` export lists

### Adding or Editing Skills

Edit files under `my_agent/skills/<skill_name>/`:
- `SKILL.md` — frontmatter (`name`, `description`) + instruction body
- `references/*.md` — loaded on-demand via `load_skill_resource()`

### Running Tests
```bash
adk eval my_agent path/to/eval_set.json
```
> No eval sets exist yet.

## Troubleshooting

**Neo4j Connection Failed**
- Verify `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`
- Ensure Neo4j is running and accessible

**OpenAI API Errors**
- Check `OPENAI_API_KEY` is valid and has quota

**Cypher Query Timeouts**
- Queries are killed after 30 seconds. Add stricter `WHERE` filters, use explicit node labels, or reduce `LIMIT`.

**PubMed Rate Limit Errors**
- Set `NCBI_API_KEY` in `.env` to raise the rate limit from 3 to 10 req/s.

**Import Errors**
- Ensure you are running from the repo root or that `my_agent/` is on the Python path.
- Ensure all dependencies are installed: `pip install 'google-adk>=1.25.0' python-dotenv neo4j httpx litellm loguru pyyaml`

**SkillToolset Not Found**
- Requires `google-adk>=1.25.0`. The `SkillToolset` API is experimental and subject to change.

## License

Internal use only — University of Michigan Medical School

## References

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [GLKB Knowledge Base](https://github.com/yuanhao96/GLKB)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/)
- [BioC PMC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/)
