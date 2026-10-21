from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import shutil

from embeddings import get_embeddings
from vector_store import (
    create_vector_store,
    add_documents_to_vector_store,
    load_vector_store,
    list_persisted_documents,
    delete_documents_by_ids,
    clear_persisted_documents,
)
from chat_model import get_chat_model, chat_model_invoke

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI(
    title="Unified RAG API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GLOBAL_VECTOR_STORE = None
CHAT_HISTORY = []

# ----------------------------------
# UTIL FUNCTIONS
# ----------------------------------

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        return UnstructuredFileLoader(file_path).load()


def chunk_documents(docs, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

MAX_CHAT_HISTORY_MESSAGES = 10


def prune_chat_history(chat_history):
    # Keep only the most recent MAX_CHAT_HISTORY_MESSAGES records
    if len(chat_history) > MAX_CHAT_HISTORY_MESSAGES:
        return chat_history[-MAX_CHAT_HISTORY_MESSAGES:]
    return chat_history

# ----------------------------------
# HEALTH CHECK
# ----------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Unified RAG API"}

# ----------------------------------
# FEATURE 2: DOCUMENT INGESTION
# ----------------------------------
@app.post("/scrape/document")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = load_document(file_path)
        chunks = chunk_documents(docs)

        embeddings = get_embeddings()
        global GLOBAL_VECTOR_STORE
        if GLOBAL_VECTOR_STORE is None:
            GLOBAL_VECTOR_STORE = create_vector_store(
                embeddings=embeddings,
                documents=chunks
            )
        else:
            GLOBAL_VECTOR_STORE = add_documents_to_vector_store(
                GLOBAL_VECTOR_STORE,
                embeddings=embeddings,
                documents=chunks,
            )

        return {
            "message": "Document processed successfully",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ----------------------------------
# CHAT (WORKS FOR BOTH)
# ----------------------------------
@app.on_event("startup")
def on_startup():
    global GLOBAL_VECTOR_STORE
    embeddings = get_embeddings()
    GLOBAL_VECTOR_STORE = load_vector_store(embeddings)


@app.post("/chat")
def chat(query: str):
    if GLOBAL_VECTOR_STORE is None:
        raise HTTPException(status_code=404, detail="No vector store has been initialized yet")

    retriever = GLOBAL_VECTOR_STORE.as_retriever(search_kwargs={"k": 15})

    global CHAT_HISTORY
    CHAT_HISTORY.append({"role": "user", "content": query})
    CHAT_HISTORY = prune_chat_history(CHAT_HISTORY)

    chat_model = get_chat_model()
    answer = chat_model_invoke(chat_model, CHAT_HISTORY, retriever)

    CHAT_HISTORY.append({"role": "assistant", "content": answer})
    CHAT_HISTORY = prune_chat_history(CHAT_HISTORY)

    return {"answer": answer}


@app.get("/chunks")
def list_chunks():
    documents = list_persisted_documents()
    return {
        "chunks": [
            {
                "id": idx + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for idx, doc in enumerate(documents)
        ]
    }


@app.delete("/chunks")
def delete_chunks(ids: list[int] = Body(default=None, embed=True)):
    global GLOBAL_VECTOR_STORE
    embeddings = get_embeddings()

    if ids is None:
        clear_persisted_documents()
        GLOBAL_VECTOR_STORE = load_vector_store(embeddings)
        return {"message": "All chunks deleted"}

    remaining_docs = delete_documents_by_ids(ids)
    GLOBAL_VECTOR_STORE = load_vector_store(embeddings)

    return {
        "message": "Selected chunks deleted",
        "deleted_ids": ids,
        "remaining_count": len(remaining_docs),
    }
