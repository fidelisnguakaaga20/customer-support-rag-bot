````md
# Customer Support RAG Bot (Local, FastAPI + FAISS + Next.js)

Customer Support AI that answers questions from a FAQ using **local models only**.

This project fits into my **MASTER LLM ENGINEERING ROADMAP** as:

- Month 2 → RAG + FastAPI + Next.js
- Month 3 → Project 3: **Customer Support AI (UMA/Myaza style)**

No paid APIs, no cloud hosting — everything runs on my laptop.

---

## 🎥 Demo (Loom + YouTube)


- 🎥 Loom: https://www.loom.com/share/c75f498b4e834b7ea15ec52e9e56890d
- 🎥 YouTube (Unlisted): https://youtu.be/AH-ZbUXaoGY


The demo shows:

1. Backend starting: `uvicorn fastapi_support_rag_api:app --reload --port 8000`
2. Frontend running on `http://localhost:3000`
3. Asking questions like:
   - “What is your refund policy?”
   - “How long does shipping take?”
   - “Do you ship internationally?”
4. Short architecture walkthrough: **Next.js → FastAPI → FAISS → local HF model**

---

## 🧠 What This Bot Does

- Loads a small **FAQ text file** with support policies.
- Splits it into chunks.
- Embeds chunks with a **local DistilBERT encoder**.
- Indexes them in **FAISS** (vector search).
- On each question:
  - Finds the most relevant FAQ chunks.
  - Builds a prompt with instructions + context.
  - Uses **distilgpt2** locally to generate an answer.
- Exposes a **FastAPI `/rag` endpoint**.
- Provides a simple **Next.js chat UI** to talk to the bot.

Everything is **CPU-only** and runs offline.

---

## 🏗 Tech Stack

**Backend**

- Python 3.11+
- FastAPI
- Uvicorn
- HuggingFace Transformers
- sentence-transformers
- DistilBERT for embeddings
- DistilGPT-2 for generation
- FAISS (faiss-cpu) for vector search

**Frontend**

- Next.js
- TypeScript
- React
- Fetch-based client calling `POST /rag`

---

## 📂 Project Structure

```text
customer-support-rag/
  backend/
    data/
      support_faq.txt
    fastapi_support_rag_api.py
    requirements.txt
    .venv/            # local, ignored by git

  frontend/
    app/
      page.tsx        # chat UI
    package.json
    tsconfig.json
    next.config.mjs
    ...

  .gitignore
  README.md           # this file
  file-tree.txt
  README.md roadmap   # high-level roadmap notes
````

---

## 🚀 How to Run (Local Only)

### 1. Backend (FastAPI + FAISS + local HF models)

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate    # on Windows Git Bash

pip install -r requirements.txt

uvicorn fastapi_support_rag_api:app --reload --port 8000
```

You should see logs like:

* `Loading embedding model: distilbert-base-uncased`
* `FAISS index built over X FAQ chunks`
* `Uvicorn running on http://127.0.0.1:8000`

### 2. Frontend (Next.js chat UI)

In another terminal:

```bash
cd frontend
npm install      # first time only
npm run dev
```

Open:

* `http://localhost:3000`

Ask:

* `What is your refund policy?`
* `How long does shipping take?`
* `Do you ship internationally?`

---

## 📝 FAQ Data Source

The FAQ lives in:

```text
backend/data/support_faq.txt
```

Example content:

```text
Our refund policy: Customers can request a refund within 30 days of purchase.

Shipping usually takes between 3 and 5 business days.

If a product is damaged, we will send a replacement free of charge.

We do not ship internationally at this time.
```

To adapt this bot to another company, you only need to:

1. Edit `support_faq.txt`
2. Restart the backend

No code changes required.

---

## 🔗 How This Fits My LLM Roadmap

This project demonstrates:

* RAG over domain-specific text (support FAQ)
* Local embeddings + vector search (FAISS)
* Local generation (distilgpt2)
* FastAPI as LLM backend
* Next.js + TypeScript frontend integration

In my 3-month LLM engineering roadmap, this is:

* **Month 2 – Week 5–8**: RAG + FastAPI + Next.js (Customer Support AI)
* **Month 3 – Project 3** in the “10 Portfolio Projects” list.

Other completed projects in the same roadmap:

1. Embedding Search Engine (Chroma + SentenceTransformers)
2. Resume RAG Chatbot (PDF resume + RAG + FastAPI + Next.js)
3. ✅ Customer Support RAG Bot (this project)

```
```
