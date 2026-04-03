# Country Information AI Agent

LangGraph + FastAPI agent that answers natural language questions about countries
using live data from the REST Countries API.

## Architecture
User → POST /api/v1/ask
↓
[intent_node]   — LLM extracts country name + data fields
↓
[tool_node]     — Fetches live data from restcountries.com
↓
[synthesis_node] — LLM generates grounded natural language answer
↓
Response

## Setup
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Configure env
cp .env.example .env
# → Add your OPENAI_API_KEY to .env

# Run dev server
poetry run uvicorn app.main:app --reload
```

## Usage
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What currency does Japan use?"}'
```

## Run Tests
```bash
poetry run pytest tests/ -v
```

## Known Limitations

- One country per question (multi-country comparisons not supported)
- Depends on restcountries.com availability
- LLM parsing may fail on very ambiguous questions

Implementation Order

pyproject.toml → poetry install
.env from .env.example
app/core/config.py → app/core/logging.py
app/models/schemas.py
app/agent/state.py → app/agent/tools.py → app/agent/nodes.py → app/agent/graph.py
app/api/routes.py → app/main.py
tests/test_agent.py

Then run:
bashpoetry install
poetry run uvicorn app.main:app --reload