# Customer Support RAG Bot — Reliability-First (Local, FastAPI + FAISS)

Customer Support RAG API that answers questions from a company FAQ using **local models only**, with a **strict evidence-only policy**.

This project is built to demonstrate **production-grade RAG reliability**:
- No hallucinations
- Null instead of guessing
- Strict API contract
- Validation gate

No paid APIs. No cloud hosting. Runs fully on my laptop.

---

## 🎥 Demo (Loom + YouTube)
🎥 ** code: https://github.com/fidelisnguakaaga20/customer-support-rag-bot
- 🎥 **Loom:** https://www.loom.com/share/c9190d1c34054f3b84576e29ec832c67  
- 🎥 **YouTube (Unlisted):** https://youtu.be/QnsbyYltVDo  

### Demo shows:
1. Backend running locally
2. Swagger UI (`/docs`) used to query the API
3. A **valid FAQ question** returning an answer + sources + confidence
4. An **unknown question** returning `answer: null`
5. Explanation of the **evidence-only + validation gate**

---

## 🧠 What This Bot Does

- Loads a **support FAQ text file**
- Splits it into meaningful chunks
- Embeds chunks using **DistilBERT (local)**
- Indexes embeddings with **FAISS**
- On each question:
  - Retrieves relevant FAQ chunks
  - Computes retrieval confidence
  - **Generates an answer only if evidence exists**
  - Rejects answers without quoted evidence
- Exposes a **FastAPI `/rag` endpoint**
- Always returns **strict JSON** (never free-form text)

Everything runs **CPU-only** and offline.

---

## ✅ Reliability Guarantees (Core of This Project)

The API enforces **four reliability layers**:

1. **Strict JSON schema**
   ```json
   {
     "answer": "string | null",
     "sources": ["faq_chunk_0"],
     "confidence": 0.0,
     "reason": "string | null"
   }
Retrieval confidence gating

If similarity is below threshold → no generation

Evidence-only answers

Generated answers must include verbatim quotes from retrieved context

Validation gate

If evidence is missing or malformed → forced rejection:

json
Copy code
{
  "answer": null,
  "confidence": 0.0,
  "reason": "No quoted evidence found"
}
This guarantees the system never hallucinates.

🚀 How to Run (Local Only)
Backend (FastAPI + FAISS + local HF models)
bash
Copy code
cd backend
python -m pip install -r requirements.txt
python -m uvicorn fastapi_support_rag_api:app --reload
Backend runs at:

cpp
Copy code
http://127.0.0.1:8000
Swagger UI:

arduino
Copy code
http://127.0.0.1:8000/docs
Test via Swagger or curl
Known FAQ question

bash
Copy code
curl -s -X POST "http://127.0.0.1:8000/rag" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is your refund policy?\"}" | python -m json.tool
Unknown question (hallucination blocked)

bash
Copy code
curl -s -X POST "http://127.0.0.1:8000/rag" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"How do I integrate Stripe webhooks with your product?\"}" | python -m json.tool
📂 Proof Pack (Reliability Evidence)
pgsql
Copy code
backend/examples/
├── valid/
│   └── refund_policy.json
└── rejected/
    └── stripe_webhooks.json
These files document expected behavior for accepted vs rejected queries.

🗂 Project Structure
text
Copy code
customer-support-rag/
  backend/
    data/
      support_faq.txt
    examples/
      valid/
      rejected/
    fastapi_support_rag_api.py
    requirements.txt

  frontend/          # optional UI (not required for reliability demo)
  README.