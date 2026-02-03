from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .models import QueryRequest, QueryParseResult, ErrorResponse
from .parser import QueryParser, ParseError

# Get the path to static files
STATIC_DIR = Path(__file__).parent / "static"

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Query Alchemy",
    description="Natural language query parser using Claude API",
    version="0.1.0",
)
app.state.limiter = limiter

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Maximum 1 request per second.", "code": 429},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc.errors()[0]["msg"]), "code": 400},
    )


parser = QueryParser()


@app.post(
    "/parse",
    response_model=QueryParseResult,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
@limiter.limit("1/second")
async def parse_query(request: Request, body: QueryRequest) -> QueryParseResult:
    """Parse a natural language query into structured JSON output."""
    try:
        result = await parser.parse(body.query)
        return result
    except ParseError as e:
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Serve the web interface."""
    return FileResponse(STATIC_DIR / "index.html")
