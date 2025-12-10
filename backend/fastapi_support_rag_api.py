from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

# ---------- Paths ----------

BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "data" / "support_faq.txt"


# ---------- Data loading & splitting ----------

def load_faq_chunks(path: Path = FAQ_PATH):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Very simple split on periods
    raw_chunks = [c.strip() for c in text.split(".") if c.strip()]
    return raw_chunks


# ---------- Embedding model (DistilBERT) ----------

print("Loading embedding model: distilbert-base-uncased")
embed_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
embed_model = AutoModel.from_pretrained("distilbert-base-uncased")


def embed_texts(texts):
    """Return numpy embeddings (n, d) for a list of texts."""
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
    return embeddings.cpu().numpy()


# ---------- Build FAISS index once at startup ----------

faq_chunks = load_faq_chunks()
faq_embeddings = embed_texts(faq_chunks)
dim = faq_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(faq_embeddings)
print(f"FAISS index built over {len(faq_chunks)} FAQ chunks.")


# ---------- Generation model (distilgpt2) ----------

print("Loading generation model: distilgpt2")
text_gen = pipeline(
      "text-generation",
    model="distilgpt2",
    max_new_tokens=80,   # shorter answers
    do_sample=True,
    temperature=0.3,     # less random
)

def rag_answer(question: str) -> str:
    """
    Stateless RAG:
      1) embed question
      2) retrieve top FAQ chunks
      3) generate answer using those chunks as context
    """
    q_emb = embed_texts([question])
    scores, indices = index.search(q_emb, k=3)
    top_chunks = [faq_chunks[i] for i in indices[0]]
    context = "\n\n".join(top_chunks)

    prompt = f"""You are a helpful customer support assistant.

Use ONLY the following context to answer the question.
If the answer is not in the context, say clearly:
"I don't know based on the current information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    raw = text_gen(prompt)[0]["generated_text"]
    # keep only the part after the last "ANSWER:"
    if "ANSWER:" in raw:
        answer = raw.split("ANSWER:")[-1].strip()
    else:
        answer = raw.strip()
    return answer


# ---------- FastAPI app ----------

app = FastAPI(title="Customer Support RAG API")

# allow frontend at localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RagRequest(BaseModel):
    question: str  # <--- this MUST match frontend JSON key


class RagResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "Customer Support RAG API running"}


@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(payload: RagRequest):
    """
    Expects JSON:
      { "question": "..." }

    Returns JSON:
      { "answer": "..." }
    """
    answer = rag_answer(payload.question)
    return RagResponse(answer=answer)
