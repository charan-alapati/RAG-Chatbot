RAG Chatbot (Retrieval-Augmented Generation)
A compact Retrieval-Augmented Generation (RAG) demo:

ingest documents -> build vector index (FAISS)
retrieve nearest docs given a user query
generate an answer with a causal LM (Hugging Face transformers)
This is a small, educational demo intended to run locally on CPU.

Files
ingest.py - read docs/*.txt, build embeddings, save FAISS index + metadata
retriever.py - load index + metadata and run retrieval
generator.py - wrap Hugging Face text-generation pipeline (default: gpt2)
chat.py - simple CLI using retriever + generator
docs/ - directory where you should put .txt files (not included)
Quick start
Create a docs/ folder in the rag_chatbot/ directory and add a few .txt files. Example:
