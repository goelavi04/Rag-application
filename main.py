from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid

# ── Load keys ─────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")

# ── Load embedding model locally ──────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# ── Connect to Pinecone ───────────────────────────────
pc         = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-documents"

existing = [i.name for i in pc.list_indexes()]
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

pinecone_index = pc.Index(index_name)

# ── Connect to OpenRouter ─────────────────────────────
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ── FastAPI app ───────────────────────────────────────
app = FastAPI(title="RAG Application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request models ────────────────────────────────────
class DocumentRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# ── Helper: Split text into chunks ────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ── Route 1: Serve the HTML page ──────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

# ── Route 2: Upload and index a document ──────────────
@app.post("/upload")
async def upload_document(req: DocumentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    # Split into chunks
    chunks = chunk_text(req.text)

    # Generate embeddings for all chunks
    embeddings = embedder.encode(chunks).tolist()

    # Build vectors for Pinecone
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id":       str(uuid.uuid4()),
            "values":   embedding,
            "metadata": {"text": chunk}
        })

    # Store in Pinecone
    pinecone_index.upsert(vectors=vectors)

    return {
        "success": True,
        "chunks":  len(chunks),
        "message": f"Indexed {len(chunks)} chunks successfully"
    }

# ── Route 3: Ask a question ───────────────────────────
@app.post("/query")
async def query_document(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="No question provided")

    # Convert question to vector
    question_embedding = embedder.encode(req.question).tolist()

    # Search Pinecone for most relevant chunks
    results = pinecone_index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    # Extract relevant chunks above similarity threshold
    relevant_chunks = [
        match["metadata"]["text"]
        for match in results["matches"]
        if match["score"] > 0.1
    ]

    if not relevant_chunks:
        return {"answer": "I could not find relevant information in the uploaded documents to answer this question."}

    # Build context from retrieved chunks
    context = "\n\n".join(relevant_chunks)

    # Send to OpenRouter AI
    response = llm.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using ONLY the provided context. If the answer is not in the context, say so clearly."
            },
            {
                "role": "user",
                "content": f"""Context from documents:
{context}

Question: {req.question}

Answer based only on the context above:"""
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer":  answer,
        "sources": relevant_chunks
    }

# ── Route 4: Clear all documents ──────────────────────
@app.delete("/clear")
async def clear_documents():
    pinecone_index.delete(delete_all=True)
    return {"success": True, "message": "All documents cleared"}