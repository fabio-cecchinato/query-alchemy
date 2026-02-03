# Original request
Exercise - Build service with structured output
## The challenge
- Build a minimal service that accepts queries and produces structured JSON output. Foundational for working with internal LLMs and API integrations.
## Requirements
- Accept natural language queries
- Return structured JSON with schema
- Include confidence scoring
- Handle edge cases gracefully
- Basic validation & error handling
- Documentation of approach
## Learning objectives
- Practice using AI tools in planning and implementation
- Understand structured output patterns for AI/LLM work
- Experience workflow: plan → build → test → validate

# My plan / interpretation
## Restating of the problem
Build a service that:
- Takes as input a sentence in natural language
- Returns as output a structured JSON (and schema) representing the input sentence
- Edge cases / errors must be handled (e.g. partial / empty sentences, prompt injection)
- All the choices made during the plan / implementation must be documented

## Assumptions / Documented choices
- The service will be web-based (web application)
- Connection to LLM is available (Claude via API key)
- There is no database / context (each query from the user is independent)
- JSON schema of the output will always be the same
- Parse the natural language query and extract its structure and intent into a standardized JSON schema, without actually executing the query
## Examples
Input: 

```
Show me all recorded calls from yesterday that lasted more than 10 minutes
```

Output:

```json
{
  "query": "Show me all recorded calls from yesterday that lasted more than 10 minutes",
  "intent": "search",
  "action": "retrieve_calls",
  "entities": [
    {
      "type": "resource",
      "value": "calls",
      "modifiers": ["recorded"]
    },
    {
      "type": "temporal",
      "value": "yesterday",
      "normalized": "2026-02-02"
    },
    {
      "type": "condition",
      "field": "duration",
      "operator": "greater_than",
      "value": 600,
      "unit": "seconds"
    }
  ],
  "filters": {
    "status": "recorded",
    "date": "2026-02-02",
    "minDuration": 600
  },
  "syntax": {
    "verb": "show",
    "subject": "calls",
    "adjectives": ["recorded"],
    "temporal": "yesterday",
    "conditions": ["lasted more than 10 minutes"]
  },
  "keywords": ["show", "recorded", "calls", "yesterday", "lasted", "minutes"],
  "confidence": 0.95
}
```

## JSON schema for output

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "Natural Language Query Parse Result",
  "required": ["query", "intent", "action", "entities", "filters", "syntax", "keywords", "confidence"],
  "properties": {
    "query": {
      "type": "string",
      "description": "The original natural language query"
    },
    "intent": {
      "type": "string",
      "description": "The primary intent of the query",
      "enum": ["search", "aggregate", "compare", "list", "count", "update", "delete", "create"]
    },
    "action": {
      "type": "string",
      "description": "The specific operation to perform (e.g., retrieve_calls, calculate_usage)"
    },
    "entities": {
      "type": "array",
      "description": "Extracted entities from the query",
      "items": {
        "type": "object",
        "required": ["type", "value"],
        "properties": {
          "type": {
            "type": "string",
            "description": "Entity category",
            "enum": ["resource", "temporal", "condition", "metric", "customer", "user", "location", "status"]
          },
          "value": {
            "description": "The extracted value (can be string, number, boolean, etc.)"
          },
          "normalized": {
            "description": "Optional normalized form of the value"
          },
          "operator": {
            "type": "string",
            "description": "Optional comparison operator",
            "enum": ["equals", "greater_than", "less_than", "greater_or_equal", "less_or_equal", "contains", "starts_with", "ends_with", "between"]
          },
          "field": {
            "type": "string",
            "description": "Optional field name this entity refers to"
          },
          "unit": {
            "type": "string",
            "description": "Optional unit of measurement"
          },
          "modifiers": {
            "type": "array",
            "description": "Optional modifiers or adjectives",
            "items": {
              "type": "string"
            }
          }
        }
      }
    },
    "filters": {
      "type": "object",
      "description": "Structured filters ready for query execution",
      "additionalProperties": true
    },
    "syntax": {
      "type": "object",
      "description": "Grammatical structure of the query",
      "properties": {
        "verb": {
          "type": "string",
          "description": "Main verb of the query"
        },
        "subject": {
          "type": "string",
          "description": "Main subject of the query"
        },
        "questionType": {
          "type": "string",
          "description": "Type of question (what, who, when, where, why, how)",
          "enum": ["what", "who", "when", "where", "why", "how", "none"]
        },
        "temporal": {
          "type": "string",
          "description": "Temporal expression if present"
        },
        "adjectives": {
          "type": "array",
          "description": "Adjectives modifying the subject",
          "items": {
            "type": "string"
          }
        },
        "conditions": {
          "type": "array",
          "description": "Conditional clauses",
          "items": {
            "type": "string"
          }
        },
        "target": {
          "type": "string",
          "description": "Target entity (e.g., customer name, user)"
        }
      }
    },
    "keywords": {
      "type": "array",
      "description": "Relevant keywords extracted from the query",
      "items": {
        "type": "string"
      }
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score of the parsing (0 to 1)",
      "minimum": 0,
      "maximum": 1
    }
  }
}
```
