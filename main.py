from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
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

# ── Connect to services ───────────────────────────────

# Load embedding model locally (free, no API needed)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
index_name = "rag-documents"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,         # all-MiniLM-L6-v2 produces 384-dimensional vectors
        metric="cosine",       # cosine similarity for comparing vectors
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

index = pc.Index(index_name)

# Connect to OpenRouter (uses OpenAI format)
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

# ── Request models (FastAPI uses Pydantic for validation) ──
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
async def index():
    with open("templates/index.html") as f:
        return f.read()

# ── Route 2: Upload and index a document ──────────────
@app.post("/upload")
async def upload_document(req: DocumentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    # Split document into chunks
    chunks = chunk_text(req.text)

    # Generate embeddings for all chunks at once
    embeddings = embedder.encode(chunks).tolist()

    # Store in Pinecone
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id":       str(uuid.uuid4()),
            "values":   embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)

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
    results = index.query(
        vector=question_embedding,
        top_k=3,                # Get top 3 most relevant chunks
        include_metadata=True
    )

    # Extract the relevant text chunks
    relevant_chunks = [
        match["metadata"]["text"]
        for match in results["matches"]
        if match["score"] > 0.3   # Only use chunks with >30% similarity
    ]

    if not relevant_chunks:
        return {"answer": "I could not find relevant information in the uploaded documents to answer this question."}

    # Build context from retrieved chunks
    context = "\n\n".join(relevant_chunks)

    # Send to OpenRouter AI with context
    response = llm.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
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
    index.delete(delete_all=True)
    return {"success": True, "message": "All documents cleared"}