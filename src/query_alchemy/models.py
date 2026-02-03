from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(str, Enum):
    SEARCH = "search"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    LIST = "list"
    COUNT = "count"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"


class EntityType(str, Enum):
    RESOURCE = "resource"
    TEMPORAL = "temporal"
    CONDITION = "condition"
    METRIC = "metric"
    CUSTOMER = "customer"
    USER = "user"
    LOCATION = "location"
    STATUS = "status"


class Operator(str, Enum):
    EQUALS = "equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_OR_EQUAL = "greater_or_equal"
    LESS_OR_EQUAL = "less_or_equal"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    BETWEEN = "between"


class QuestionType(str, Enum):
    WHAT = "what"
    WHO = "who"
    WHEN = "when"
    WHERE = "where"
    WHY = "why"
    HOW = "how"
    NONE = "none"


class Entity(BaseModel):
    type: EntityType
    value: Any
    normalized: Any | None = None
    operator: Operator | None = None
    field: str | None = None
    unit: str | None = None
    modifiers: list[str] | None = None


class Syntax(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    verb: str | None = None
    subject: str | None = None
    question_type: QuestionType | None = Field(None, alias="questionType")
    temporal: str | None = None
    adjectives: list[str] | None = None
    conditions: list[str] | None = None
    target: str | None = None


class QueryParseResult(BaseModel):
    query: str
    intent: Intent
    action: str
    entities: list[Entity]
    filters: dict[str, Any]
    syntax: Syntax
    keywords: list[str]
    confidence: float = Field(ge=0, le=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class ErrorResponse(BaseModel):
    error: str
    code: int
