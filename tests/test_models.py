import pytest
from pydantic import ValidationError
from query_alchemy.models import (
    QueryRequest,
    QueryParseResult,
    Entity,
    EntityType,
    Intent,
    Syntax,
)


class TestQueryRequest:
    def test_valid_query(self):
        req = QueryRequest(query="Show me all calls")
        assert req.query == "Show me all calls"

    def test_strips_whitespace(self):
        req = QueryRequest(query="  Show me all calls  ")
        assert req.query == "Show me all calls"

    def test_empty_query_fails(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_whitespace_only_fails(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_max_length(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="a" * 2001)


class TestQueryParseResult:
    def test_valid_result(self, sample_parse_result):
        result = QueryParseResult.model_validate(sample_parse_result)
        assert result.intent == Intent.SEARCH
        assert result.confidence == 0.92

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            QueryParseResult(
                query="test",
                intent="search",
                action="test",
                entities=[],
                filters={},
                syntax=Syntax(),
                keywords=[],
                confidence=1.5,  # Invalid: > 1
            )

    def test_confidence_negative_fails(self):
        with pytest.raises(ValidationError):
            QueryParseResult(
                query="test",
                intent="search",
                action="test",
                entities=[],
                filters={},
                syntax=Syntax(),
                keywords=[],
                confidence=-0.1,  # Invalid: < 0
            )

    def test_invalid_intent(self):
        with pytest.raises(ValidationError):
            QueryParseResult(
                query="test",
                intent="invalid_intent",
                action="test",
                entities=[],
                filters={},
                syntax=Syntax(),
                keywords=[],
                confidence=0.5,
            )


class TestEntity:
    def test_valid_entity(self):
        entity = Entity(type="resource", value="calls")
        assert entity.type == EntityType.RESOURCE
        assert entity.value == "calls"

    def test_entity_with_all_fields(self):
        entity = Entity(
            type="condition",
            value=600,
            normalized=600,
            operator="greater_than",
            field="duration",
            unit="seconds",
            modifiers=["minimum"],
        )
        assert entity.field == "duration"
        assert entity.unit == "seconds"

    def test_invalid_entity_type(self):
        with pytest.raises(ValidationError):
            Entity(type="invalid_type", value="test")


class TestSyntax:
    def test_empty_syntax(self):
        syntax = Syntax()
        assert syntax.verb is None
        assert syntax.subject is None

    def test_syntax_with_question_type_alias(self):
        syntax = Syntax.model_validate({"questionType": "what", "verb": "show"})
        assert syntax.question_type.value == "what"
        assert syntax.verb == "show"

    def test_syntax_with_all_fields(self):
        syntax = Syntax(
            verb="show",
            subject="calls",
            question_type="what",
            temporal="yesterday",
            adjectives=["recorded"],
            conditions=["lasted more than 10 minutes"],
            target="customer",
        )
        assert syntax.adjectives == ["recorded"]
        assert len(syntax.conditions) == 1
