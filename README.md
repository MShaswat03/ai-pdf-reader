# AI PDF Reader

A local AI-powered PDF question-answering app built with FastAPI, Ollama, embeddings, and semantic search.

## Features

- Upload a PDF
- Extract text from the PDF
- Split text into chunks
- Generate local embeddings using Ollama
- Retrieve relevant chunks using cosine similarity
- Generate answers using a local LLM
- Show source chunks with similarity scores and previews

## Tech Stack

- Python
- FastAPI
- Ollama
- nomic-embed-text
- llama3.2:1b
- pypdf
- NumPy
- HTML/CSS

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt