"""
generator.py
------------
Simple generation wrapper using Hugging Face transformers pipeline (causal LM).
This is intentionally minimal to keep the demo lightweight.

Note:
- The example uses 'gpt2' by default (small model). If you have a larger model
  or prefer an instruction-tuned model, change MODEL_NAME.

Requirements:
- transformers
- torch
"""

from transformers import pipeline

MODEL_NAME = "gpt2"  # small; replace with larger / instruction-tuned model if desired

print("Loading text-generation pipeline with model:", MODEL_NAME)
gen = pipeline("text-generation", model=MODEL_NAME, device=-1)  # device=-1 -> CPU

def generate_answer(context, question, max_length=150, top_k=50, temperature=0.7):
    """
    Create a prompt that includes the retrieved context and the user question,
    then ask the LM to continue / answer.

    Returns a short string (the LM response).
    """
    prompt = "Context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"
    out = gen(prompt, max_length=max_length, do_sample=True, top_k=top_k, temperature=temperature, num_return_sequences=1)
    text = out[0]["generated_text"]
    # The model returns the entire prompt + completion. We split to get the completion after "Answer:"
    if "Answer:" in text:
        return text.split("Answer:", 1)[1].strip()
    # fallback: return the tail portion
    return text[len(prompt):].strip()

if __name__ == "__main__":
    # tiny demo
    ctx = "Python is an interpreted, high-level programming language. It supports multiple paradigms."
    q = "Is Python statically typed?"
    ans = generate_answer(ctx, q, max_length=80)
    print("Answer:\n", ans)
