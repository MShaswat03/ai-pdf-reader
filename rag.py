import requests
import numpy as np
from pypdf import PdfReader


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text
            text += "\n\n"

    return text


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def get_embedding(text):
    url = "http://localhost:11434/api/embeddings"

    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }

    response = requests.post(url, json=payload)
    result = response.json()

    return result["embedding"]


def create_chunk_embeddings(chunks):
    chunk_embeddings = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        chunk_embeddings.append(embedding)

    return chunk_embeddings


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


def retrieve_top_chunks_by_embedding(chunks, chunk_embeddings, question, top_k=3):
    question_embedding = get_embedding(question)
    results = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_embedding = chunk_embeddings[index - 1]
        score = cosine_similarity(question_embedding, chunk_embedding)
        results.append((score, index, chunk))

    results.sort(reverse=True)

    filtered_results = []

    for score, index, chunk in results[:top_k]:
        if score >= 0.4:
            filtered_results.append((score, index, chunk))

    return filtered_results


def build_context(results):
    context = ""

    for score, index, chunk in results:
        context += f"[Chunk {index}]\n"
        context += chunk
        context += "\n\n"

    return context


def generate_answer(question, context):
    if context == "":
        return "I don't know based on the provided document."

    prompt = f"""
You are an AI assistant.

Answer directly using the context.
Do not say the context does not explicitly state the answer if the answer can be reasonably inferred from the context.
Keep the answer short.
If the answer is not in the context, say: I don't know based on the provided document.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def call_ollama(prompt):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, json=payload)
    result = response.json()

    return result["response"]


def answer_question(chunks, chunk_embeddings, question):
    results = retrieve_top_chunks_by_embedding(chunks, chunk_embeddings, question)

    if len(results) == 0:
        return {
            "answer": "No relevant chunks found.",
            "sources": []
        }

    context = build_context(results)
    prompt = generate_answer(question, context)
    answer = call_ollama(prompt)

    sources = []

    for score, index, chunk in results:
        sources.append({
            "chunk": index,
            "similarity": round(score, 3),
            "preview": chunk[:250] + "..."
        })

    return {
        "answer": answer,
        "sources": sources
    }