# Decisions & Assumptions Log

This document tracks all decisions and assumptions made during the planning and implementation of Query Alchemy.

---

## User Decisions

Decisions explicitly made by the user during the planning conversation.

| # | Decision | Context |
|---|----------|---------|
| 1 | **Python with FastAPI** | Technology stack selection |
| 2 | **Local demo only** | No deployment configuration needed (Docker, cloud, etc.) |
| 3 | **No authentication** | API endpoint is open for this exercise |
| 4 | **Rate limit: 1 request per second** | Protection against abuse |
| 5 | **Both unit and integration tests** | Unit tests with mocked LLM, integration tests with real API |
| 6 | **Simple error format** | `{error: string, code: number}` - no complex error schema |

---

## Assumptions from Requirements (problem.md)

Assumptions documented in the original problem specification.

| # | Assumption | Implication |
|---|------------|-------------|
| 1 | Web-based service | HTTP API, not CLI or library |
| 2 | Claude API connection available | User must provide `ANTHROPIC_API_KEY` |
| 3 | Stateless service | No database, no session persistence, each query is independent |
| 4 | Fixed JSON schema for output | All responses conform to the schema in problem.md |
| 5 | Parse without executing | Extract structure/intent only, no actual query execution |

---

## Technical Decisions (by Claude)

Technical implementation decisions made during planning.

### Architecture & Structure

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | `src/query_alchemy/` package layout | Standard Python package structure, allows `pip install -e .` |
| 2 | Separate modules: `main.py`, `models.py`, `parser.py`, `config.py` | Clear separation of concerns |
| 3 | `pydantic-settings` for configuration | Clean env var handling, validation, `.env` file support |

### Dependencies

| # | Decision | Rationale |
|---|----------|-----------|
| 4 | `slowapi` for rate limiting | Simple, well-maintained, integrates directly with FastAPI |
| 5 | `hatchling` as build backend | Modern, fast, recommended for new Python projects |
| 6 | `pytest-asyncio` for async test support | Standard for testing async FastAPI apps |
| 7 | `httpx` for test client | Required by FastAPI's TestClient for async support |

### API Design

| # | Decision | Rationale |
|---|----------|-----------|
| 8 | Single endpoint: `POST /parse` | Simple API surface, matches the single use case |
| 9 | Health check endpoint: `GET /health` | Standard practice for service monitoring |
| 10 | Query max length: 2000 characters | Reasonable limit to prevent abuse, fits most natural language queries |
| 11 | Whitespace trimming on input | Better UX, prevents accidental empty queries |

### LLM Integration

| # | Decision | Rationale |
|---|----------|-----------|
| 12 | Default model: `claude-sonnet-4-20250514` | Good balance of speed and quality for structured output |
| 13 | Max tokens: 2048 | Sufficient for JSON response, prevents runaway costs |
| 14 | Sync Anthropic client in async endpoint | Anthropic SDK is sync by default; wrapping adds complexity for minimal benefit in this demo |
| 15 | Include today's date in system prompt | Required for normalizing relative temporal expressions ("yesterday", "last week") |
| 16 | JSON-only response instruction | Ensures parseable output, reduces post-processing complexity |

### Error Handling

| # | Decision | Rationale |
|---|----------|-----------|
| 17 | Custom `ParseError` exception class | Clean separation between parsing errors and other exceptions |
| 18 | Specific HTTP status codes: 400 (validation), 429 (rate limit), 500 (server), 502 (API error), 503 (connection) | Standard HTTP semantics, helps client-side error handling |
| 19 | Log errors server-side, return generic messages to client | Security best practice, avoid leaking internal details |

### Testing

| # | Decision | Rationale |
|---|----------|-----------|
| 20 | Skip integration tests if no API key | Allows CI to run without secrets, developers can run locally |
| 21 | Factory fixture for mock responses | Flexible, reusable across different test cases |
| 22 | Separate test files by layer | `test_models.py`, `test_parser.py`, `test_api.py` - easier to maintain and run selectively |

---

## Out of Scope (Explicit)

Items explicitly excluded from this implementation.

| # | Item | Reason |
|---|------|--------|
| 1 | Authentication/authorization | User decision: keep open for exercise |
| 2 | Persistent storage | Requirement: stateless service |
| 3 | Deployment configuration | User decision: local demo only |
| 4 | Frontend/UI | Not in requirements |
| 5 | Response caching | Adds complexity, minimal benefit for demo |
| 6 | Multiple LLM providers | Not in requirements, keep simple |
| 7 | Query history/analytics | Requirement: stateless service |

---

## Open Questions (Resolved)

Questions that were asked and resolved during planning.

| # | Question | Resolution |
|---|----------|------------|
| 1 | Technology stack? | Python + FastAPI |
| 2 | Deployment target? | Local demo only |
| 3 | Authentication needed? | No |
| 4 | Rate limiting requirements? | 1 request per second |
| 5 | Testing approach? | Unit (mocked) + Integration (real API) |
| 6 | Error response format? | Simple `{error, code}` |

---

## Implementation Decisions (during build)

Decisions made during the actual implementation phase.

| # | Decision | Rationale |
|---|----------|-----------|
| 23 | Strip markdown code blocks from LLM response | Claude sometimes returns JSON wrapped in \`\`\`json ... \`\`\` blocks; parsing needs to handle this gracefully |
| 24 | Use `ConfigDict` instead of class-based `Config` in Pydantic models | Avoids deprecation warning in Pydantic V2 |
| 25 | Simple HTML/CSS/JS web interface served by FastAPI | User requested web interface; lightweight solution with no additional dependencies |

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-02-03 | Added implementation decisions during build | Claude |
| 2026-02-03 | Initial decisions log created | Claude |
