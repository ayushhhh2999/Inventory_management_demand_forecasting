import os
import pickle

from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.documents import Document
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

PERSISTED_DOCS_FILE = os.path.join(os.path.dirname(__file__), "persisted_vector_store.pkl")


def save_documents(documents, file_path=PERSISTED_DOCS_FILE):
    serialized = [
        {"page_content": doc.page_content, "metadata": doc.metadata or {}}
        for doc in documents
    ]
    with open(file_path, "wb") as f:
        pickle.dump(serialized, f)


def load_documents(file_path=PERSISTED_DOCS_FILE):
    if not os.path.exists(file_path):
        return []

    with open(file_path, "rb") as f:
        serialized = pickle.load(f)

    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in serialized
    ]


def create_vector_store(embeddings, documents):
    persisted_docs = load_documents()
    combined_docs = persisted_docs + documents
    save_documents(combined_docs)

    vector_store = InMemoryVectorStore.from_documents(
        documents=combined_docs,
        embedding=embeddings,
    )
    return vector_store


def load_vector_store(embeddings):
    documents = load_documents()
    if documents:
        return InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
        )
    return InMemoryVectorStore(embeddings)


def add_documents_to_vector_store(vector_store, embeddings, documents):
    vector_store.add_documents(
        documents=documents,
        embedding=embeddings,
    )

    persisted_docs = load_documents()
    save_documents(persisted_docs + documents)
    return vector_store


def list_persisted_documents():
    return load_documents()


def clear_persisted_documents(file_path=PERSISTED_DOCS_FILE):
    if os.path.exists(file_path):
        os.remove(file_path)


def create_retriever_from_vector_store(vector_store, k):
    retriever = GraphRetriever(
        store=vector_store,
        edges=[("path", "path"), ("source", "source")],
        strategy=Eager(k=k, start_k=1, max_depth=2),
    )
    return retriever


def retrieve_documents(retriever, query):
    results = retriever.retrieve(query)
    return results



