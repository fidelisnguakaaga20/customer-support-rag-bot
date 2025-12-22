from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

import logging

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_rag")

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "data" / "support_faq.txt"

# -------------------------
# Reliability knobs
# -------------------------
TOP_K = 3

# You tuned this already for your small FAQ
THRESHOLD = 0.72  # /// best cosine-like similarity must be >= this

# /// Stage 3: evidence enforcement (quotes must be present)
REQUIRE_QUOTES = True

# /// Output hygiene
MAX_ANSWER_CHARS = 450


# -------------------------
# Data loading & chunking
# -------------------------
def load_faq_chunks(path: Path = FAQ_PATH) -> list[str]:
    """
    Chunking strategy:
      1) split by blank lines (best for Q/A blocks)
      2) fallback to line-based chunks if too few
      3) drop ultra-short noise chunks
    """
    text = path.read_text(encoding="utf-8").strip()

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    if len(chunks) < 6:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = [c for c in chunks if len(c) >= 20]
    return chunks


# -------------------------
# Embedding model (DistilBERT)
# -------------------------
print("Loading embedding model: distilbert-base-uncased")
embed_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
embed_model = AutoModel.from_pretrained("distilbert-base-uncased")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Return float32 embeddings (n, d) for a list of texts.
    Important:
      - normalize embeddings so FAISS inner product ~= cosine similarity
    """
    with torch.no_grad():
        encoded = embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = embed_model(**encoded)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


# -------------------------
# Build FAISS index at startup
# -------------------------
faq_chunks = load_faq_chunks()
faq_embeddings = embed_texts(faq_chunks)
dim = faq_embeddings.shape[1]

# normalized vectors + IP index => cosine-like similarity (higher is better)
index = faiss.IndexFlatIP(dim)
index.add(faq_embeddings)

print(f"FAISS index built over {len(faq_chunks)} FAQ chunks.")
faq_source_ids = [f"faq_chunk_{i}" for i in range(len(faq_chunks))]


# -------------------------
# Generation model (distilgpt2)
# -------------------------
print("Loading generation model: distilgpt2")
text_gen = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=80,
    do_sample=True,
    temperature=0.3,
)


# -------------------------
# Retrieval
# -------------------------
def retrieve(question: str, k: int = TOP_K):
    q_emb = embed_texts([question])
    scores, indices = index.search(q_emb, k=k)

    scores = scores[0].tolist()
    indices = indices[0].tolist()

    retrieved = []
    for rank, (i, s) in enumerate(zip(indices, scores)):
        if i == -1:
            continue
        retrieved.append(
            {
                "id": faq_source_ids[i],
                "text": faq_chunks[i],
                "score": float(s),  # cosine-like similarity
                "rank": rank,
            }
        )

    best = float(retrieved[0]["score"]) if len(retrieved) >= 1 else 0.0
    return retrieved, best


# -------------------------
# Answer cleaning
# -------------------------
def clean_answer(text: str) -> str:
    text = text.strip()

    # prevent prompt leakage
    for marker in ["CONTEXT:", "QUESTION:", "ANSWER:", "FINAL ANSWER", "QUOTES"]:
        # don't blindly remove QUOTES header because we need it for validation
        if marker != "QUOTES":
            text = text.replace(marker, "").strip()

    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip() + "..."

    return text


def rag_answer(question: str, context_chunks: list[str]) -> str:
    """
    Stage 3 prompt: force evidence quoting.
    If answer isn't in context => model must say the exact unknown phrase.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a customer support assistant.

RULES (STRICT):
- Use ONLY the provided CONTEXT.
- If the answer is not in the CONTEXT, output exactly:
I don't know based on the current information.
- Your final output MUST contain a section called QUOTES.
- QUOTES must contain 1–3 exact sentences copied from the CONTEXT that support the answer.

CONTEXT:
{context}

QUESTION: {question}

FINAL ANSWER (1-3 sentences):
QUOTES (copied from context):
"""

    raw = text_gen(prompt)[0]["generated_text"]

    # take only the tail after the last occurrence of our "FINAL ANSWER" anchor
    # (helps reduce repeated prompt)
    if "FINAL ANSWER" in raw:
        raw = raw.split("FINAL ANSWER", 1)[-1].strip()

    return clean_answer(raw)


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Customer Support RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RagRequest(BaseModel):
    question: str


# Stage 1: strict JSON response
class RagResponse(BaseModel):
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float):
        # Stage 4 partial: clamp to 0..1
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


@app.get("/")
async def root():
    return {"status": "ok", "message": "Customer Support RAG API running"}


def validation_fail(reason: str) -> RagResponse:
    # Stage 4: controlled failure response
    logger.warning("Validation failed: %s", reason)
    return RagResponse(answer=None, sources=[], confidence=0.0, reason=reason)


@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(payload: RagRequest):
    question = payload.question.strip()
    if not question:
        return validation_fail("Empty question")

    retrieved, best_score = retrieve(question, k=TOP_K)
    confidence = float(max(0.0, min(1.0, best_score)))
    sources = [r["id"] for r in retrieved]

    logger.info(
        "RAG request | q=%r | best=%.4f conf=%.4f sources=%s",
        question,
        best_score,
        confidence,
        sources,
    )

    # Stage 2 gate: no evidence => no generation
    if confidence < THRESHOLD or not retrieved:
        return RagResponse(
            answer=None,
            sources=[],
            confidence=0.0,
            reason="No supporting sources found",
        )

    # Generate with context
    context_chunks = [r["text"] for r in retrieved]
    answer_text = rag_answer(question, context_chunks)

    # If model explicitly says unknown => enforce null
    if answer_text.strip().lower().startswith("i don't know based on the current information"):
        return RagResponse(
            answer=None,
            sources=[],
            confidence=0.0,
            reason="No supporting sources found",
        )

    # -------------------------
    # Stage 3: Evidence-only enforcement
    # -------------------------
    if REQUIRE_QUOTES:
        # Must contain QUOTES section
        if "QUOTES" not in answer_text:
            return validation_fail("Missing QUOTES evidence")

        # Extract quotes section text after 'QUOTES'
        parts = answer_text.split("QUOTES", 1)
        quotes_section = parts[1] if len(parts) > 1 else ""

        # Verify at least one non-empty line in quotes exists verbatim in context
        context_joined = "\n".join(context_chunks)
        quote_lines = [ln.strip() for ln in quotes_section.splitlines() if ln.strip()]

        has_verbatim_evidence = any(q in context_joined for q in quote_lines)

        if not has_verbatim_evidence:
            return validation_fail("No quoted evidence found")

    # Stage 4: final schema safety (answer must be non-empty here)
    final_answer = answer_text.strip()
    if not final_answer:
        return validation_fail("Empty answer after generation")

    return RagResponse(
        answer=final_answer,
        sources=sources,
        confidence=confidence,
        reason=None,
    )
