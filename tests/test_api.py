import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from query_alchemy.main import app, limiter
from query_alchemy.models import QueryParseResult
from query_alchemy.parser import ParseError


@pytest.fixture
def client():
    return TestClient(app)


class TestParseEndpoint:
    def test_parse_success(self, client, sample_parse_result):
        with patch("query_alchemy.main.parser") as mock_parser:
            mock_parser.parse = AsyncMock(
                return_value=QueryParseResult.model_validate(sample_parse_result)
            )

            # Reset rate limiter for test
            limiter.reset()

            response = client.post(
                "/parse",
                json={"query": "Show me all calls from yesterday"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["intent"] == "search"
            assert data["confidence"] == 0.92

    def test_empty_query_error_format(self, client):
        limiter.reset()
        response = client.post("/parse", json={"query": ""})
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert data["code"] == 400

    def test_missing_query_error_format(self, client):
        limiter.reset()
        response = client.post("/parse", json={})
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data

    def test_whitespace_query(self, client):
        limiter.reset()
        response = client.post("/parse", json={"query": "   "})
        assert response.status_code == 400

    def test_rate_limiting_error_format(self, client, sample_parse_result):
        with patch("query_alchemy.main.parser") as mock_parser:
            mock_parser.parse = AsyncMock(
                return_value=QueryParseResult.model_validate(sample_parse_result)
            )

            # Reset rate limiter for test
            limiter.reset()

            # First request should succeed
            response1 = client.post(
                "/parse",
                json={"query": "test query"},
            )
            assert response1.status_code == 200

            # Second immediate request should be rate limited
            response2 = client.post(
                "/parse",
                json={"query": "test query"},
            )
            assert response2.status_code == 429
            data = response2.json()
            assert "error" in data
            assert data["code"] == 429

    def test_parse_error_handling(self, client):
        with patch("query_alchemy.main.parser") as mock_parser:
            mock_parser.parse = AsyncMock(
                side_effect=ParseError("Failed to parse AI response as JSON", 500)
            )

            limiter.reset()

            response = client.post(
                "/parse",
                json={"query": "test query"},
            )

            assert response.status_code == 500

    def test_api_connection_error_handling(self, client):
        with patch("query_alchemy.main.parser") as mock_parser:
            mock_parser.parse = AsyncMock(
                side_effect=ParseError("Failed to connect to AI service", 503)
            )

            limiter.reset()

            response = client.post(
                "/parse",
                json={"query": "test query"},
            )

            assert response.status_code == 503

    def test_internal_error(self, client):
        with patch("query_alchemy.main.parser") as mock_parser:
            mock_parser.parse = AsyncMock(side_effect=Exception("Unexpected error"))

            limiter.reset()

            response = client.post(
                "/parse",
                json={"query": "test query"},
            )

            assert response.status_code == 500


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestWebInterface:
    def test_root_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Query Alchemy" in response.text
