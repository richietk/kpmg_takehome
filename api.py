"""
FastAPI backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
from pathlib import Path
from retriever import DocumentRetriever
from pipeline import RAGPipeline
from ingest import load_finance_jsonl
import uvicorn
import requests

app = FastAPI(
    title="RAG Q&A demo API"
)

# Global pipeline instance
rag_pipeline: Optional[RAGPipeline] = None

class QueryRequest(BaseModel):
    """Request model for Q&A queries"""
    query: str
    k: int = 3
    use_reranking: bool = True


class Source(BaseModel):
    """Source document model"""
    title: str
    url: str


class QueryResponse(BaseModel):
    """Response model for Q&A queries"""
    query: str
    answer: str
    sources: List[Source]
    num_chunks_retrieved: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    pipeline_loaded: bool
    ollama_connected: bool


def load_pipeline():
    """initiatlize and load RAG pipeline"""
    global rag_pipeline

    vector_store_path = "data/vector_store"
    wiki_sample_path = "data/wiki_sample.json"

    # get ollama host from env. for demo, localhost is used
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")

    print("initializing retriever...")
    retriever = DocumentRetriever(use_reranker=True)

    # load vector store, or rebuild if missing
    if Path(vector_store_path).exists():
        print("Loading saved vector store...")
        retriever.load(vector_store_path)
    else:
        print(f"Vector store not found at {vector_store_path}")

        # check if processed data exists to rebuild from
        if not Path(wiki_sample_path).exists():
            print(f"{wiki_sample_path} not found. Converting finance dataset...")
            finance_jsonl_path = os.getenv("FINANCE_DATASET_PATH", "data/wikipedia_finance_trunc.jsonl")

            if not Path(finance_jsonl_path).exists():
                raise FileNotFoundError(
                    f"Finance dataset not found at {finance_jsonl_path}. "
                    f"Please provide the dataset or set FINANCE_DATASET_PATH environment variable."
                )

            load_finance_jsonl(input_path=finance_jsonl_path, output_path=wiki_sample_path)
            print(f"Converted finance dataset successfully")

        print(f"Rebuilding vector store from {wiki_sample_path}...")
        with open(wiki_sample_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        print(f"Indexing {len(articles)} articles...")
        retriever.index_documents(articles)
        retriever.save(vector_store_path)
        print(f"Vector store built and saved to {vector_store_path}")

    rag_pipeline = RAGPipeline(retriever, ollama_host=ollama_host)
    print("Pipeline ready")


@app.on_event("startup")
async def startup_event():
    """Load pipeline on startup"""
    try:
        load_pipeline()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Q&A API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    pipeline_loaded = rag_pipeline is not None

    # check ollama connection
    ollama_connected = False
    if pipeline_loaded:
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "localhost")
            requests.get(f"http://{ollama_host}:11434/api/tags", timeout=5)
            ollama_connected = True
        except:
            pass

    return HealthResponse(
        status="healthy" if (pipeline_loaded and ollama_connected) else "degraded",
        pipeline_loaded=pipeline_loaded,
        ollama_connected=ollama_connected
    )


@app.post("/answer", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """
    answer a question using the RAG pipeline

    - query: user question
    - k: number of docs to retrieve
    - use_reranking: bool to use cross-encoder reranking
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized."
        )

    try:
        result = rag_pipeline.answer(
            query=request.query,
            k=request.k,
            use_reranking=request.use_reranking
        )

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """Get system stats"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Count chunks in vector store. This is optional and used for DEBUG/INFO
    vector_store = rag_pipeline.retriever.vector_store
    num_chunks = vector_store.index.ntotal if vector_store else 0

    return {
        "total_chunks_indexed": num_chunks,
        "embedding_model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "llm_model": rag_pipeline.model_name
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
