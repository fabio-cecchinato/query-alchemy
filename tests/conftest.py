import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_anthropic_response():
    """Factory for creating mock Anthropic responses."""
    def _create_response(json_content: str):
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json_content)]
        return mock_message
    return _create_response


@pytest.fixture
def sample_parse_result():
    return {
        "query": "Show me all calls from yesterday",
        "intent": "search",
        "action": "retrieve_calls",
        "entities": [
            {"type": "resource", "value": "calls"},
            {"type": "temporal", "value": "yesterday", "normalized": "2026-02-02"},
        ],
        "filters": {"date": "2026-02-02"},
        "syntax": {"verb": "show", "subject": "calls", "temporal": "yesterday"},
        "keywords": ["show", "calls", "yesterday"],
        "confidence": 0.92,
    }
