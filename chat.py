"""
chat.py
-------
Simple CLI chat loop that:
- retrieves top-k documents using Retriever
- concatenates them as context
- calls generator to produce answer

Usage:
    python chat.py
"""

from retriever import Retriever
from generator import generate_answer

def build_context(results):
    # Join retrieved texts with separators and keep short excerpts
    parts = []
    for r in results:
        txt = r.get("text", "")
        # keep first 500 characters of each doc for context to avoid very long prompts
        parts.append(f"---\nFilename: {r.get('filename')}\n{txt[:500]}")
    return "\n\n".join(parts)

def main():
    print("RAG Chatbot (type 'exit' or 'quit' to stop)")
    retriever = Retriever()
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        # retrieve
        results = retriever.retrieve(q, k=3)
        context = build_context(results)
        # generate answer
        ans = generate_answer(context, q, max_length=150)
        print("\nBot:", ans)

if __name__ == "__main__":
    main()
