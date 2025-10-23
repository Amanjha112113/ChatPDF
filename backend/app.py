import os
import uuid  # For creating unique document IDs
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
# Removed numpy, it's not needed for the Pinecone client
import google.generativeai as genai
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# --- 1. Initialization & Config ---

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    """
    print("--- Server Starting Up ---")
    
    # Initialize Gemini
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    print("Gemini Initialized.")

    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY not found in environment variables.")
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "chat-with-pdf"
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension for text-embedding-004
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    app.state.pinecone_index = pc.Index(index_name)
    print("Pinecone Initialized.")
    
    yield
    
    # --- Server Shutting Down ---
    print("--- Server Shutting Down ---")
    
app = FastAPI(lifespan=lifespan)

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# !!! REMOVED IN-MEMORY document_store !!!
# The server is now stateless. Your data-loss bug is fixed.

# --- 2. Pydantic Models ---

class UploadResponse(BaseModel):
    doc_id: str
    message: str
    num_chunks: int

class QueryRequest(BaseModel):
    doc_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[str]

# --- 3. RAG Pipeline Helper Functions ---

def extract_text_from_pdf(content: bytes) -> str:
    """Extracts text from PDF content using PyMuPDF (fitz)."""
    text = ""
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return [chunk for chunk in chunks if chunk.strip()]

def create_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks."""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response['embedding'] # Returns a simple list of lists
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {e}")

# --- 4. API Endpoints ---

# <-- FIXED: Removed the 'TextUpload' class

@app.post("/upload", response_model=UploadResponse)
# <-- FIXED: Changed endpoint to accept a File, not JSON
async def upload_pdf(request: Request, file: UploadFile = File(...)): 
    """
    Handles PDF upload, processing, chunking, embedding, and upserting to Pinecone.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        # <-- FIXED: Added logic to read and parse the file from the request
        content = await file.read()
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        doc_id = str(uuid.uuid4()) # This is our namespace

        # Upsert embeddings AND chunks to Pinecone
        vectors = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            vectors.append({
                "id": f"{doc_id}_{i}",
                "values": embedding,
                "metadata": {
                    # <-- FIXED: Storing the text chunk in the metadata
                    "text_chunk": chunk  
                }
            })
        
        # Batch upsert to Pinecone
        pinecone_index = request.app.state.pinecone_index
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            pinecone_index.upsert(vectors=batch, namespace=doc_id)
        
        # <-- FIXED: Removed the line that saved to the in-memory 'document_store'
        
        print(f"PDF processed and stored in Pinecone with namespace: {doc_id}")

        return UploadResponse(
            doc_id=doc_id,
            message="PDF processed and stored successfully",
            num_chunks=len(chunks)
        )
    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: Request, query_request: QueryRequest):
    """
    Answers a query by retrieving context from Pinecone and generating an answer.
    """
    
    # <-- FIXED: Removed logic that checked the in-memory 'document_store'
    pinecone_index = request.app.state.pinecone_index

    try:
        # 2. Embed the query
        query_embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query_request.query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = query_embedding_response['embedding']

        # 3. Search Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5, # Get top 5 relevant chunks
            namespace=query_request.doc_id,
            include_metadata=True # CRITICAL: This retrieves the text_chunk
        )

        # <-- FIXED: Get text chunks from metadata, not from the dead 'document_store'
        relevant_chunks = [match.metadata['text_chunk'] for match in results.matches]
        
        if not relevant_chunks:
            # This can happen if the doc_id is valid but the query finds no matches
            raise HTTPException(status_code=404, detail="No relevant context found for your query in this document.")

        context = "\n\n---\n\n".join(relevant_chunks)

        # 5. Generate response using Gemini
        model = genai.GenerativeModel("gemini-1.5-pro") 
        prompt = (
            f"You are a helpful assistant. Answer the following question based ONLY on the context provided below.\n"
            f"If the answer is not in the context, state that you cannot find the answer in the document.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query_request.query}\n\n"
            f"ANSWER:"
        )
        
        response = model.generate_content(prompt)

        return QueryResponse(
            answer=response.text,
            source_chunks=relevant_chunks
        )
    except Exception as e:
        # Catch Pinecone errors (e.g., namespace not found)
        if "namespace" in str(e):
             raise HTTPException(status_code=404, detail="Document ID not found. The session may have expired.")
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def read_root():
    """A simple endpoint to confirm the server is running."""
    return {"message": "Welcome to the Chat with PDF API! (Stateless)"}

# --- 6. Run the Server ---

if __name__ == "__main__":
    import uvicorn
    # PORT is set by Render
    port = int(os.getenv("PORT", 8000))
    # Use reload=True for development only
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
    