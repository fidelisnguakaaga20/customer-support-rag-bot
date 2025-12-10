"use client";

import { FormEvent, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_SUPPORT_BACKEND_URL ?? "http://127.0.0.1:8000";

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput("");
    setError(null);

    // add user message
    setMessages((prev) => [...prev, { role: "user", content: question }]);

    try {
      setLoading(true);

      const res = await fetch(`${BACKEND_URL}/rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Backend error: ${res.status}`);
      }

      const data = await res.json();
      const answer: string = data.answer ?? String(data);

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: answer },
      ]);
    } catch (err: any) {
      console.error(err);
      setError("Could not reach support AI. Check backend is running.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I could not contact the support engine. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-50 flex flex-col">
      {/* Header */}
      <header className="w-full border-b border-slate-800 px-4 py-3 md:px-8 flex items-center justify-between">
        <div>
          <h1 className="text-lg md:text-2xl font-semibold">
            Customer Support RAG Bot
          </h1>
          <p className="text-xs md:text-sm text-slate-400">
            Ask about refunds, shipping, damaged products, and more.
          </p>
        </div>
        <span className="text-[10px] md:text-xs text-emerald-400 border border-emerald-500/30 rounded-full px-2 py-1">
          Project 3 · Local RAG
        </span>
      </header>

      {/* Chat area */}
      <section className="flex-1 flex justify-center px-2 py-3 md:px-4 md:py-4">
        <div className="w-full max-w-2xl flex flex-col rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm overflow-hidden">
          <div className="flex-1 overflow-y-auto px-3 py-3 md:px-4 md:py-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-xs md:text-sm text-slate-400 text-center mt-4">
                Start the conversation: try{" "}
                <span className="font-mono">
                  &quot;What is your refund policy?&quot;
                </span>
              </div>
            )}

            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`flex ${
                  m.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-3 py-2 text-xs md:text-sm leading-relaxed whitespace-pre-wrap ${
                    m.role === "user"
                      ? "bg-emerald-500 text-slate-950"
                      : "bg-slate-800 text-slate-50"
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}
          </div>

          {/* Error bar */}
          {error && (
            <div className="px-3 py-2 md:px-4 text-[11px] md:text-xs text-amber-300 bg-amber-900/40 border-t border-amber-700/40">
              {error}
            </div>
          )}

          {/* Input */}
          <form
            onSubmit={handleSubmit}
            className="border-t border-slate-800 px-2 py-2 md:px-3 md:py-3 bg-slate-900"
          >
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your question here…"
                className="flex-1 rounded-2xl bg-slate-950 border border-slate-700 px-3 py-2 text-xs md:text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
              <button
                type="submit"
                disabled={loading}
                className="rounded-2xl bg-emerald-500 text-slate-950 px-3 py-2 text-xs md:text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? "Thinking…" : "Send"}
              </button>
            </div>
          </form>
        </div>
      </section>
    </main>
  );
}
