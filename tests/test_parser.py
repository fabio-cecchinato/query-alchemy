import pytest
import json
import asyncio
from unittest.mock import MagicMock
from anthropic import APIError, APIConnectionError
from query_alchemy.parser import QueryParser, ParseError


class TestQueryParser:
    def test_parse_success(self, mock_anthropic_response, sample_parse_result):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            json.dumps(sample_parse_result)
        )

        parser = QueryParser(client=mock_client)
        result = asyncio.run(parser.parse("Show me all calls from yesterday"))

        assert result.intent.value == "search"
        assert result.confidence == 0.92
        assert len(result.entities) == 2

    def test_parse_calls_anthropic_with_correct_params(self, mock_anthropic_response, sample_parse_result):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            json.dumps(sample_parse_result)
        )

        parser = QueryParser(client=mock_client)
        asyncio.run(parser.parse("test query"))

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "model" in call_kwargs
        assert "max_tokens" in call_kwargs
        assert "system" in call_kwargs
        assert "messages" in call_kwargs
        assert call_kwargs["messages"][0]["content"] == "Parse this query:\n\ntest query"

    def test_parse_invalid_json(self, mock_anthropic_response):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            "This is not valid JSON"
        )

        parser = QueryParser(client=mock_client)
        with pytest.raises(ParseError) as exc_info:
            asyncio.run(parser.parse("test query"))

        assert exc_info.value.code == 500
        assert "JSON" in exc_info.value.message

    def test_parse_schema_mismatch(self, mock_anthropic_response):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            json.dumps({"invalid": "schema"})
        )

        parser = QueryParser(client=mock_client)
        with pytest.raises(ParseError) as exc_info:
            asyncio.run(parser.parse("test query"))

        assert exc_info.value.code == 500
        assert "schema" in exc_info.value.message

    def test_parse_api_connection_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = APIConnectionError(request=MagicMock())

        parser = QueryParser(client=mock_client)
        with pytest.raises(ParseError) as exc_info:
            asyncio.run(parser.parse("test query"))

        assert exc_info.value.code == 503
        assert "connect" in exc_info.value.message.lower()

    def test_parse_api_error(self):
        mock_client = MagicMock()
        mock_error = MagicMock()
        mock_error.message = "Rate limit exceeded"
        mock_client.messages.create.side_effect = APIError(
            message="Rate limit exceeded",
            request=MagicMock(),
            body=None
        )

        parser = QueryParser(client=mock_client)
        with pytest.raises(ParseError) as exc_info:
            asyncio.run(parser.parse("test query"))

        assert exc_info.value.code == 502

    def test_parse_with_all_entity_types(self, mock_anthropic_response):
        full_result = {
            "query": "Find customers in New York with status active",
            "intent": "search",
            "action": "find_customers",
            "entities": [
                {"type": "customer", "value": "customers"},
                {"type": "location", "value": "New York"},
                {"type": "status", "value": "active"},
            ],
            "filters": {"location": "New York", "status": "active"},
            "syntax": {
                "verb": "find",
                "subject": "customers",
                "questionType": "none",
            },
            "keywords": ["find", "customers", "New York", "active"],
            "confidence": 0.88,
        }

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            json.dumps(full_result)
        )

        parser = QueryParser(client=mock_client)
        result = asyncio.run(parser.parse("Find customers in New York with status active"))

        assert result.intent.value == "search"
        assert len(result.entities) == 3
        assert result.entities[1].type.value == "location"
