# NGU Restaurant — WhatsApp Customer Support RAG (No Hallucinations)

A **local, reliability-first RAG system** that helps restaurant staff reply to WhatsApp customers faster — **without guessing**.

✅ Answers only from approved restaurant info (menu + policies)  
✅ Refuses safely when info is missing  
✅ Never guesses prices, availability, or delivery areas  
✅ Hybrid Mode: **human-in-the-loop copy/paste** (no automation)

---

## 🎥 Demo (Loom + YouTube)

- **Code:** https://github.com/fidelisnguakaaga20/customer-support-rag-bot  
- **Loom:** https://www.loom.com/share/c9190d1c34054f3b84576e29ec832c67  
- **YouTube (Unlisted):** https://youtu.be/QnsbyYltVDo  

---

## 🧠 What This System Does (Restaurant Mode)

- Loads restaurant knowledge from:
  - `backend/knowledge_bases/restaurant/restaurant_faq.txt`
- Splits into chunks and embeds using local models
- Indexes embeddings with **FAISS**
- On each question:
  - Retrieves evidence from the restaurant index
  - Applies a strict threshold gate
  - Returns a **WhatsApp-ready reply** only when evidence exists
  - Otherwise returns a **safe refusal** or **follow-up question**

---

## ✅ Reliability Guarantees (Non-Negotiable)

This system enforces:

- **Evidence-only answers**
- **Refusal instead of guessing** (prices, delivery, payment rules, availability)
- **No invented menu items**
- **Short, polite, WhatsApp-ready tone**

---

## 📦 Project Structure

```text
customer-support-rag/
  backend/
    fastapi_support_rag_api.py
    build_restaurant_index.py
    rag/
      restaurant_answer.py
      restaurant_store.py
    indexes/
      restaurant/
        faiss.index
        chunks.json
    knowledge_bases/
      restaurant/
        restaurant_faq.txt
  frontend/                  # optional UI
  sales/                     # monetization assets
  proof/                     # screenshots for sales demos
  README.md
