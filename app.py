import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
import asyncpg

# Initialize the FastAPI core application
app = FastAPI(
    title="Production-Ready Local RAG Service",
    description="A fully offline data-private RAG API running Qwen-2.5, Ollama, and pgvector.",
    version="1.0.0"
)

# Configuration settings matching container network boundaries
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_db")
LLM_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# Define structured Pydantic models for incoming requests and API schemas
class DocumentSubmission(BaseModel):
    text: str = Field(..., min_length=10, description="The raw knowledge text chunk to be vectorized and stored.")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="The natural language question to ask the RAG engine.")

class QueryResponse(BaseModel):
    question: str
    answer: str
    context_used: List[str]

@app.on_event("startup")
async def startup_db_setup():
    """
    Executes immediately when FastAPI starts up. Establishes a database connection
    to activate the pgvector extension and build the target layout if missing.
    """
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Register the pgvector extension within the current database instance
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Build a robust table designed to house text chunks along with 768-dimensional embeddings
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_vault (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(768) NOT NULL
            );
        """)
        # Generate an HNSW index to optimize vector similarity query lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS hnsw_embedding_idx 
            ON knowledge_vault USING hnsw (embedding vector_cosine_ops);
        """)
        await conn.close()
    except Exception as e:
        print(f"[FATAL STARTUP ERROR] Could not initialize PostgreSQL/pgvector database: {e}")

async def get_ollama_embedding(text_chunk: str) -> List[float]:
    """
    Communicates with the Ollama service to translate a text block into a dense vector array.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_API_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text_chunk}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="Ollama embedding generation failed.")
            return response.json().get("embedding", [])
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama connection error: {exc}")

async def query_ollama_llm(prompt: str) -> str:
    """
    Sends a structured prompt directly to the local Qwen-2.5 instance and returns the output string.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="Ollama LLM generation failed.")
            return response.json().get("response", "").strip()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama LLM server unreachable: {exc}")

@app.post("/v1/ingest", status_code=201, summary="Ingest raw text chunk into vector store")
async def ingest_document(payload: DocumentSubmission, background_tasks: BackgroundTasks):
    """
    Ingests text snippets, generates embeddings via Ollama, and persists them into pgvector.
    """
    # Generate the vector array immediately
    embedding = await get_ollama_embedding(payload.text)
    
    if not embedding:
        raise HTTPException(status_code=500, detail="Generated embedding vector was empty.")

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Store both raw content and its mathematical representation inside the database
        await conn.execute(
            "INSERT INTO knowledge_vault (content, embedding) VALUES ($1, $2);",
            payload.text, embedding
        )
        await conn.close()
        return {"status": "success", "message": "Text embedding successfully indexed into pgvector vault."}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Database persistent write failed: {err}")

@app.post("/v1/query", response_model=QueryResponse, summary="Query the RAG pipeline")
async def query_rag_pipeline(payload: QueryRequest):
    """
    Converts incoming questions to vectors, fetches relevant contexts via cosine distance,
    and runs a context-augmented prompt through Qwen-2.5.
    """
    # Transform user inquiry into a matching vector space
    query_vector = await get_ollama_embedding(payload.question)
    
    context_chunks = []
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Query pgvector records sorted by standard Cosine distance metrics to extract top 3 matches
        records = await conn.fetch(
            """
            SELECT content FROM knowledge_vault 
            ORDER BY embedding <=> $1::vector 
            LIMIT 3;
            """,
            query_vector
        )
        await conn.close()
        context_chunks = [r['content'] for r in records]
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Database vector similarity search broke: {err}")

    # Fallback response check if the database does not contain matches or data
    if not context_chunks:
        system_context_str = "No verified system context available inside the secure local database."
    else:
        system_context_str = "\n---\n".join(context_chunks)

    # Build an explicit instructional prompt ensuring zero model hallucinations
    structured_prompt = (
        "You are a secure, enterprise-grade AI technical assistant. Answer the user's question accurately using ONLY "
        "the provided background context sections below. If the context does not contain the answer, state clearly "
        "that you do not possess that internal information. Do not mention external knowledge resources.\n\n"
        f"BACKGROUND CONTEXT:\n{system_context_str}\n\n"
        f"USER QUESTION: {payload.question}\n\n"
        "EXPERT VERIFIED ANSWER:"
    )

    # Fetch response from Qwen-2.5 engine
    generated_answer = await query_ollama_llm(structured_prompt)

    return QueryResponse(
        question=payload.question,
        answer=generated_answer,
        context_used=context_chunks
    )
