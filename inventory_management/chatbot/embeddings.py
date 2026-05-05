from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

load_dotenv()

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

embeddings = get_embeddings()
print(type(embeddings))
