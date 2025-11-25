import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_or_create_collection(
    name="travel_log_memory",
    embedding_function=openai_ef
)

def get_embedding(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

def save_document(doc_id: str, text: str):
    embedding = get_embedding(text)
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding]
    )

def query_similar(text: str, n_results: int = 3):
    results = collection.query(
        query_texts=[text],
        n_results=n_results
    )
    return results["documents"][0]

def generate_answer(question: str, context_chunks: list[str]):
    context = "\n".join(context_chunks)

    prompt = f"""
You are an assistant. Use the context to answer.

Context:
{context}

Question:
{question}
"""

    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )

    return res.choices[0].message.content
