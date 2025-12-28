from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import logging

# /// Restaurant RAG (Stage 2)
from rag.restaurant_store import RestaurantVectorStore
from rag.restaurant_answer import safe_refusal, whatsapp_style_answer

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restaurant_support_rag")

# -------------------------
# Reliability knobs (Stage 2)
# -------------------------
TOP_K = 4

# /// For SentenceTransformer + FAISS IP (cosine-like), 0.30–0.45 is a practical range
# /// We choose 0.35 for a small FAQ so it answers when evidence is clearly relevant.
THRESHOLD = 0.28

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Restaurant WhatsApp Support RAG (Hallucination-Safe)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load restaurant index once at startup (Stage 1 -> Stage 2)
# -------------------------
store = RestaurantVectorStore()


class RagRequest(BaseModel):
    question: str


class RagResponse(BaseModel):
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)  # /// evidence ids (chunks)
    confidence: float = 0.0  # /// best retrieval score (0..1)
    reason: str | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float):
        # /// clamp to 0..1 for safety
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


@app.get("/")
async def root():
    return {"status": "ok", "message": "Restaurant WhatsApp Support RAG running"}


def refusal_response(best_score: float, reason: str) -> RagResponse:
    # /// Stage 2: refuse instead of guessing (WhatsApp-ready)
    logger.warning("Refusal | best=%.4f | reason=%s", best_score, reason)
    return RagResponse(
        answer=safe_refusal(),
        sources=[],
        confidence=0.0,
        reason=reason,
    )


@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(payload: RagRequest):
    question = payload.question.strip()
    if not question:
        return refusal_response(0.0, "Empty question")

    # /// Stage 2: retrieve evidence from restaurant index
    results = store.search(question, top_k=TOP_K)

    best_score = float(results[0]["score"]) if results else 0.0
    confidence = max(0.0, min(1.0, best_score))

    logger.info(
        "RAG request | q=%r | best=%.4f | top_k=%d",
        question,
        best_score,
        TOP_K,
    )

    # /// Stage 2 gate: no evidence => refuse (never guess)
    if not results:
        return refusal_response(0.0, "No supporting sources found")

    if best_score < THRESHOLD:
        return refusal_response(best_score, "Evidence below threshold")

    # /// Evidence exists: produce WhatsApp-ready grounded answer
    answer = whatsapp_style_answer(question, results)

    # /// Provide source ids for traceability (chunk ranks)
    sources = [f"restaurant_chunk_rank_{i}" for i in range(len(results))]

    return RagResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        reason=None,
    )
