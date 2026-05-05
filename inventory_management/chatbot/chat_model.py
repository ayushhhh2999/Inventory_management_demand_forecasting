from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------
# 1. Chat Model
# -------------------------------
def get_chat_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        reasoning_format="parsed",
        max_retries=2,
    )


# -------------------------------
# 2. Conversational RAG
# -------------------------------
def chat_model_invoke(chat_model, chat_history: list, retriever):
    """
    chat_history: List[{"role": "user" | "assistant", "content": str}]
    retriever: vector_store.as_retriever()
    """

    parser = StrOutputParser()

    def format_history(history):
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history
        )

    # 🔹 Query rewrite prompt
    query_rewrite_prompt = PromptTemplate(
        input_variables=["chat_history"],
        template="""
You are an expert at semantic search.

Given the following conversation history,
generate a single, concise search query that will retrieve
the most relevant informatio.

Conversation:
{chat_history}

Search Query:
"""
    )

    # 🔹 Final answer prompt
    final_prompt = PromptTemplate(
        input_variables=["chat_history", "context"],
        template="""
You are a helpful assistant analyzing a GitHub repository.

Conversation so far:
{chat_history}

Relevant repository context:
{context}

Using the context above, answer the user's latest question
clearly and accurately dont use any /n,-,# just generate plain text strictly.
Your answer should be based only on the provided context. If the context does not 
contain the information needed to answer, say "I don't know based on the provided information."
"""
    )

    # 🔹 Retrieval pipeline
    def retrieve_context(chat_history_text: str):
        search_query = (
            query_rewrite_prompt
            | chat_model
            | parser
        ).invoke({"chat_history": chat_history_text})

        docs = retriever.invoke(search_query)
        return "\n\n".join(d.page_content for d in docs)

    # 🔹 Final answer
    answer = (
        final_prompt
        | chat_model
        | parser
    ).invoke({
        "chat_history": format_history(chat_history),
        "context": retrieve_context(format_history(chat_history)),
    })

    return answer
