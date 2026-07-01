import os
import shutil

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from rag import read_pdf
from rag import chunk_text
from rag import create_chunk_embeddings
from rag import answer_question


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


chunks = []
chunk_embeddings = []
uploaded_filename = None


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": None,
            "sources": [],
            "question": "",
            "uploaded_filename": uploaded_filename
        }
    )


@app.post("/upload")
def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    global chunks
    global chunk_embeddings
    global uploaded_filename

    if not pdf_file.filename.endswith(".pdf"):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": "Please upload a PDF file.",
                "sources": [],
                "question": "",
                "uploaded_filename": uploaded_filename
            }
        )

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{pdf_file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)

    uploaded_filename = pdf_file.filename

    text = read_pdf(file_path)
    chunks = chunk_text(text)

    print("Creating chunk embeddings...")
    chunk_embeddings = create_chunk_embeddings(chunks)
    print("PDF ready.")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": "PDF uploaded and processed. You can now ask questions.",
            "sources": [],
            "question": "",
            "uploaded_filename": uploaded_filename
        }
    )

@app.post("/ask")
def ask_question(request: Request, question: str = Form(...)):
    if len(chunks) == 0 or len(chunk_embeddings) == 0:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": "Please upload a PDF first.",
                "sources": [],
                "question": question,
                "uploaded_filename": uploaded_filename
            }
        )

    result = answer_question(chunks, chunk_embeddings, question)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": result["answer"],
            "sources": result["sources"],
            "question": question,
            "uploaded_filename": uploaded_filename
        }
    )