from rag import read_pdf
from rag import chunk_text
from rag import create_chunk_embeddings
from rag import answer_question


text = read_pdf("data/sample.pdf")
chunks = chunk_text(text)

print("Creating chunk embeddings...")
chunk_embeddings = create_chunk_embeddings(chunks)
print("Ready. Type exit to stop.")

while True:
    question = input("Ask a question: ")

    if question == "exit":
        break

    result = answer_question(chunks, chunk_embeddings, question)

    print("ANSWER:")
    print(result["answer"])

    print("SOURCES:")
    for source in result["sources"]:
        print("Chunk", source["chunk"], "- similarity:", source["similarity"])