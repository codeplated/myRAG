"""
RAG ingestion pipeline package.

Reads PDF books from the configured books directory, cleans and chunks text,
generates embeddings via a local Ollama model, and stores them in a pgvector
backed Postgres database.
"""

