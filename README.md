# Smart CS

Smart CS is a backend AI customer-service system built with FastAPI and LangGraph. It uses a supervisor to route each conversation to the right specialist agent, call business tools, retrieve knowledge with MySQL + Qdrant RAG, record traces and metrics, and escalate risky cases to human review.

This project is designed as an interview-ready multi-agent backend: compact enough to run locally, but complete enough to demonstrate production-style agent orchestration, tool calling, memory, retrieval, evaluation, and observability.

## Highlights

- FastAPI service for auth, chat, orders, tickets, tools, traces, metrics, and human review.
- LangGraph `Supervisor` for routing, agent handoff, compliance checks, and final response assembly.
- Specialist agents:
  - `TicketAgent` for refund, order, and ticket workflows.
  - `KnowledgeAgent` for knowledge-base question answering.
  - `ComplaintAgent` for complaint intake and escalation.
  - `ComplianceAgent` for response safety checks.
- Tool Calling through `ToolRegistry`, including order lookup, ticket creation, refund eligibility, RAG context building, trace recording, and human escalation.
- Three-layer memory:
  - Working memory for the current request.
  - Short-term chat history for the active session.
  - Long-term user profile and conversation summaries.
- RAG with MySQL chunk storage, Qdrant vector search, and BGE-M3 embeddings.
- Human Review queue for complaints, low-confidence routing, compliance failures, explicit human requests, and high-value cases.
- Trace and Metrics support for debugging and evaluating agent behavior.
- Lightweight Evaluation runner with intent, routing, tool, RAG, human-review, trace, and latency metrics.

## Architecture

```mermaid
flowchart TD
    Client[Client / API Caller] --> API[FastAPI]
    API --> Supervisor[LangGraph Supervisor]
    Supervisor --> Memory[Working / Short-term / Long-term Memory]
    Supervisor --> Router[Intent Router]
    Router --> Ticket[TicketAgent]
    Router --> Knowledge[KnowledgeAgent]
    Router --> Complaint[ComplaintAgent]
    Ticket --> Tools[ToolRegistry]
    Knowledge --> Tools
    Complaint --> Tools
    Tools --> Orders[Order Tool]
    Tools --> Tickets[Ticket Tool]
    Tools --> RAG[RAG Context Builder]
    RAG --> Qdrant[Qdrant Vector Store]
    RAG --> MySQL[MySQL Chunk Store]
    Supervisor --> Compliance[ComplianceAgent]
    Compliance --> Review[Human Review]
    Supervisor --> Trace[Trace]
    API --> Metrics[Metrics]
```

## Project Layout

```text
app/
  agents/          Ticket, knowledge, complaint, compliance, router, and base agents
  evaluation/      Evaluation cases, runner, and metrics
  memory/          Working, short-term, and long-term memory
  rag/             Loader, splitter, BGE embeddings, MySQL chunks, Qdrant vectors, retriever
  tools/           ToolRegistry and business tools
  main.py          FastAPI app and HTTP routes
  supervisor.py    LangGraph orchestration
data/
  knowledge_base/  Markdown knowledge base files used by RAG
tests/
  test_api_flow.py
  test_chat_flow.py
  test_evaluation.py
  test_rag.py
  test_qdrant_rag.py
```

## Local Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Configure `.env`:

```bash
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/smart_cs
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=smart_cs_chunks
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

Initialize the database:

```bash
alembic upgrade head
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## MCP

The existing HTTP adapter remains available at:

- `POST /mcp/tools/list`
- `POST /mcp/tools/call`

The stdio MCP server entrypoint is:

```bash
python -m app.mcp.stdio_server
```

It registers:

- `knowledge_rag_context(query: str, top_k: int = 3)`
- `order_query(order_id: str)`

`user_id` is read from `MCP_USER_ID`; when unset it defaults to `mcp-demo-user`.

Claude Desktop example:

```json
{
  "mcpServers": {
    "smart-cs": {
      "command": "python",
      "args": ["-m", "app.mcp.stdio_server"],
      "cwd": "C:\\Users\\15207\\Desktop\\ai 学习\\smart-cs",
      "env": {
        "MCP_USER_ID": "mcp-demo-user"
      }
    }
  }
}
```

Cursor example:

```json
{
  "mcpServers": {
    "smart-cs": {
      "command": "python",
      "args": ["-m", "app.mcp.stdio_server"],
      "cwd": "C:\\Users\\15207\\Desktop\\ai 学习\\smart-cs",
      "env": {
        "MCP_USER_ID": "mcp-demo-user"
      }
    }
  }
}
```

## Knowledge Base

The RAG knowledge base lives under `data/knowledge_base/`. The current import script can generate e-commerce customer-service Markdown files from the `rjac/e-commerce-customer-support-qa` dataset:

```bash
python scripts/import_ecommerce_customer_service.py
```

The retriever flow is:

1. Load Markdown files from `data/knowledge_base/`.
2. Split documents into chunks.
3. Generate BGE-M3 embeddings.
4. Store chunk text and metadata in MySQL.
5. Store vectors and chunk ids in Qdrant.
6. Retrieve top-k chunk ids from Qdrant.
7. Read full chunk content from MySQL.
8. Rerank and build the final RAG context.

## Tests

Run the default test suite:

```bash
pytest -v
```

Run Qdrant integration tests:

```bash
pytest -v -m integration
```

API flow tests require a running server and are skipped unless enabled:

```bash
RUN_API_FLOW_TESTS=1 pytest -v tests/test_api_flow.py
```

The Qdrant RAG tests cover:

- `MySQLChunkStore` writing and reading chunks.
- `QdrantVectorStore` writing and searching vectors.
- `SimpleRetriever` retrieving `refund.md` for `退款多久到账`.
- Returned content containing `3-5 个工作日`.

## Evaluation

Evaluation cases live in `app/evaluation/eval_cases.json`. The runner reports:

- `intent_accuracy`
- `agent_route_accuracy`
- `tool_selection_accuracy`
- `rag_source_hit_rate`
- `human_review_trigger_accuracy`
- `trace_consistency_accuracy`
- `avg_latency_ms`

Run it with:

```bash
python -m app.evaluation.runner
```

## API Surface

Main routes include:

- `GET /health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/users/me`
- `POST /api/chat`
- `POST /api/chat/stream`
- `GET /api/orders`
- `POST /api/orders`
- `GET /api/tickets`
- `GET /api/tickets/{ticket_id}`
- `PATCH /api/tickets/{ticket_id}/status`
- `GET /api/tools`
- `POST /api/tools/execute`
- `GET /api/traces`
- `GET /api/admin/reviews`
- `GET /api/metrics`

## Roadmap

- Add a first-class MCP-style tool service layer.
- Expand skill/workflow abstractions for refund, complaint, and knowledge QA scenarios.
- Add Docker Compose services for API and MySQL.
- Publish evaluation reports and operational dashboards.
