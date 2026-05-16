from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.in_memory_session_store import SessionNotFoundError
from api.rest.schemas import (
    AnalyzeRequest,
    AskRequest,
    CreateSessionRequest,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    SessionResponse,
    TurnsResponse,
    TurnResponse,
)
from services.paperintel_service import PaperIntelService


def create_rest_app(*, service: PaperIntelService) -> FastAPI:
    app = FastAPI(title="PaperIntel API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(error="session_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def internal_error_handler(request, exc):  # noqa: ANN001
        error = ErrorResponse(
            error="internal_error",
            detail="An internal error occurred while processing the request.",
        )
        return JSONResponse(status_code=500, content=error.model_dump(mode="json"))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        status = service.health()
        response = HealthResponse.from_health_status(status)
        return JSONResponse(
            status_code=200 if status.healthy else 503,
            content=response.model_dump(mode="json"),
        )

    @app.post("/sessions", response_model=SessionResponse)
    async def create_session(payload: CreateSessionRequest):
        session = service.create_session(
            persona=payload.persona,
            original_query=payload.original_query,
        )
        return SessionResponse.from_session(session)

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        session = service.get_session(session_id)
        return SessionResponse.from_session(session)

    @app.get("/sessions/{session_id}/turns", response_model=TurnsResponse)
    async def list_turns(session_id: str, limit: int = 50):
        turns = service.list_turns(session_id, limit=limit)
        return TurnsResponse(
            turns=[TurnResponse.from_turn(turn) for turn in turns],
        )

    @app.post("/sessions/{session_id}/analyze", response_model=MessageResponse)
    async def analyze_paper(session_id: str, payload: AnalyzeRequest):
        result = service.analyze_paper(session_id, str(payload.paper_url))
        return MessageResponse.from_handler_result(result)

    @app.post("/sessions/{session_id}/ask", response_model=MessageResponse)
    async def ask_question(session_id: str, payload: AskRequest):
        result = service.ask_question(session_id, payload.question)
        return MessageResponse.from_handler_result(result)

    return app
