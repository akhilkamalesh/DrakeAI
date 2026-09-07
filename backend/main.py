"""FastAPI REST API Server for DrakeAI."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path so 'backend' is importable from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.db import check_db_health, close_db_pool, get_db_pool
from backend.embeddings import get_embedder
from backend.graph.state import AgentState
from backend.graph.workflow import chat_graph
from backend.models import ChatRequest, ChatResponse, SourceItem


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("drakeai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup warmup and graceful shutdown."""
    logger.info("Initializing DrakeAI backend...")
    # Initialize DB connection pool
    try:
        get_db_pool()
        logger.info("Database connection pool initialized.")
    except Exception as e:
        logger.error("Failed to initialize database pool: %s", e)

    # Pre-warm SentenceTransformer embedder in background thread
    try:
        await asyncio.to_thread(get_embedder)
        logger.info("SentenceTransformer model pre-warmed.")
    except Exception as e:
        logger.error("Failed to pre-warm embedder: %s", e)

    yield

    logger.info("Shutting down DrakeAI backend...")
    close_db_pool()


app = FastAPI(
    title="DrakeAI RAG Backend",
    description="FastAPI service for semantic lyric search and hybrid context graph retrieval across Drake's discography.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit and local frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint verifying database connectivity and model status."""
    db_ok = check_db_health()
    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "llm_provider": "gemini",
        "gemini_model": settings.GEMINI_MODEL,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.GEMINI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dim": settings.TARGET_EMBEDDING_DIM
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Orchestrates end-to-end conversation flow via LangGraph:
    Node 1 (Guardrail & Intent) -> Node 2 (Hybrid Retrieval) -> Node 3 (Reasoning Agent) -> Node 4 (Response Formatter)
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    initial_state: AgentState = {
        "prompt": request.prompt.strip(),
        "history": request.history,
        "is_relevant": False,
        "rejection_message": None,
        "semantic_query": None,
        "metadata_filters": {},
        "query_vector": None,
        "retrieved_context": [],
        "excluded_track_ids": [],
        "retry_count": 0,
        "is_vetted": False,
        "vetting_rationale": None,
        "reasoning_analysis": None,
        "document_rationales": {},
        "final_response": "",
        "sources": []
    }

    try:
        # Run graph in worker thread to prevent blocking the async event loop
        result: AgentState = await asyncio.to_thread(chat_graph.invoke, initial_state)

        # Parse sources into Pydantic models
        sources = [SourceItem(**s) for s in result.get("sources", [])]

        return ChatResponse(
            response=result.get("final_response", ""),
            sources=sources,
            reasoning=result.get("reasoning_analysis"),
            agent_trace=result.get("agent_trace")
        )

    except Exception as e:
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streams the agent thought process and pipeline breakdown in real time:
    query_analysis -> pulled_songs -> vetting_agent -> reasoning -> complete
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    initial_state: AgentState = {
        "prompt": request.prompt.strip(),
        "history": request.history,
        "is_relevant": False,
        "rejection_message": None,
        "semantic_query": None,
        "metadata_filters": {},
        "query_vector": None,
        "retrieved_context": [],
        "excluded_track_ids": [],
        "retry_count": 0,
        "is_vetted": False,
        "vetting_rationale": None,
        "reasoning_analysis": None,
        "document_rationales": {},
        "final_response": "",
        "sources": [],
        "extracted_keywords": [],
        "pulled_tracks": [],
        "vetting_decisions": [],
        "agent_trace": None
    }

    async def event_generator():
        try:
            async for step in chat_graph.astream(initial_state):
                for node_name, state_update in step.items():
                    if node_name == "guardrail_intent":
                        mf = state_update.get("metadata_filters", {}) or {}
                        af_filters = mf.get("audio_filters") or {}
                        af_targets = mf.get("audio_targets") or {}
                        event = {
                            "step": "query_analysis",
                            "prompt": request.prompt.strip(),
                            "keywords": state_update.get("extracted_keywords", []),
                            "semantic_query": state_update.get("semantic_query"),
                            "audio_feature_creation": {
                                "filters": {k: v for k, v in af_filters.items() if v is not None} if isinstance(af_filters, dict) else {},
                                "targets": {k: v for k, v in af_targets.items() if v is not None} if isinstance(af_targets, dict) else {},
                                "sort_by": mf.get("sort_by")
                            },
                            "metadata_limits": {
                                "limit": mf.get("limit", 3),
                                "personal_feel": mf.get("personal_feel"),
                                "release_year_before": mf.get("release_year_before"),
                                "release_year_after": mf.get("release_year_after"),
                                "album_name": mf.get("album_name")
                            },
                            "is_relevant": state_update.get("is_relevant", True)
                        }
                        yield json.dumps(event) + "\n"

                    elif node_name == "hybrid_retrieval":
                        event = {
                            "step": "pulled_songs",
                            "pulled_songs": state_update.get("pulled_tracks", [])
                        }
                        yield json.dumps(event) + "\n"

                    elif node_name == "vet_track":
                        event = {
                            "step": "vetting_agent",
                            "is_vetted": state_update.get("is_vetted", False),
                            "retry_count": state_update.get("retry_count", 0),
                            "primary_rationale": state_update.get("vetting_rationale"),
                            "decisions": state_update.get("vetting_decisions", [])
                        }
                        yield json.dumps(event) + "\n"

                    elif node_name == "reasoning_agent":
                        event = {
                            "step": "reasoning",
                            "thematic_analysis": state_update.get("reasoning_analysis")
                        }
                        yield json.dumps(event) + "\n"

                    elif node_name in ("response_formatter", "out_of_scope"):
                        event = {
                            "step": "complete",
                            "response": state_update.get("final_response", ""),
                            "sources": state_update.get("sources", []),
                            "agent_trace": state_update.get("agent_trace", {})
                        }
                        yield json.dumps(event) + "\n"
        except Exception as e:
            logger.error("Error in chat_stream_endpoint: %s", e, exc_info=True)
            yield json.dumps({"step": "error", "error": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
        app_dir=str(PROJECT_ROOT)
    )

