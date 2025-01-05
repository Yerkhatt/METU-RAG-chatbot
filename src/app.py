from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
import os

from src.llm import GroqAPI  # Direct import statement
from src.retriever import Retriever_SBERT  # Direct import statement
from src.schemas import QueryRequest  # Import QueryRequest

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve frontend at root
@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")

# Mount static files after the root endpoint
app.mount("/static", StaticFiles(directory="frontend"), name="static")

sbert_retriever = Retriever_SBERT()
LLM = GroqAPI()

MAX_TOKENS = 3800  # Target around 5000 tokens, being conservative

@app.get("/models")
async def get_models():
    return {"models": GroqAPI.AVAILABLE_MODELS}

@app.post("/query")
async def query(request: QueryRequest):
    print('Query request received!')
    
    # Extract core search query
    search_query = LLM.extract_search_query(request.query_text)
    print(f'Using search query: {search_query}')
    
    # Get initial documents and sort by score using extracted query
    retrieved_docs = sbert_retriever.retrieve(search_query)
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
    print(f'Retrieved {len(sorted_docs)} documents, checking relevance in parallel...')
    
    # Use original query for relevance checking and answer generation
    relevant_docs = LLM.batch_check_relevance(request.query_text, sorted_docs)
    print(f'Found {len(relevant_docs)} relevant documents')
    
    # Process relevant documents
    relevant_info = []
    total_words = 0
    
    for doc in relevant_docs:
        if extracted_info := LLM.extract_relevant_info(request.query_text, doc)[0]:
            word_count = len(extracted_info.split())
            
            if total_words + word_count > MAX_TOKENS:
                print(f'Token limit reached at {total_words} words')
                break
            
            relevant_info.append({
                'url': doc['url'],
                'extracted_info': extracted_info
            })
            total_words += word_count
            print(f'Added info from {doc["url"]} (words: {word_count})')
    
    print(f'Total words collected: {total_words}')
    print(f'Documents with relevant info: {len(relevant_info)}')
    
    if not relevant_info:
        return {"response": "I don't have enough relevant information to answer that question."}
    
    response = LLM.answer_with_context(request.query_text, relevant_info)
    return {"response": response}

@app.get("/prompts")
async def get_prompts():
    return {
        "relevance_prompt": LLM.RELEVANCE_PROMPT,
        "extract_info_prompt": LLM.EXTRACT_INFO_PROMPT,
        "answer_prompt": LLM.ANSWER_PROMPT,
        "extract_query_prompt": LLM.EXTRACT_QUERY_PROMPT
    }

@app.post("/prompts")
async def update_prompts(prompts: dict):
    try:
        # Update prompts if they exist in request
        if "relevance_prompt" in prompts:
            LLM.RELEVANCE_PROMPT = prompts["relevance_prompt"]
        if "extract_info_prompt" in prompts:
            LLM.EXTRACT_INFO_PROMPT = prompts["extract_info_prompt"]
        if "answer_prompt" in prompts:
            LLM.ANSWER_PROMPT = prompts["answer_prompt"]
        if "extract_query_prompt" in prompts:
            LLM.EXTRACT_QUERY_PROMPT = prompts["extract_query_prompt"]
        
        print("Prompts updated successfully")
        return {"status": "success", "message": "Prompts updated successfully"}
    except Exception as e:
        print(f"Error updating prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# test code
@app.post("/sbert_retrieve")  # Change endpoint to /sbert_retrieve for clarity
async def read_root(request: QueryRequest):
    retrieval_result = sbert_retriever.retrieve(request.query_text)
    return {"results": retrieval_result}  # Wrap the result in a dictionary to ensure JSON serializability

