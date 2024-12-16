import os

# Set a writable directory for Hugging Face cache and environment variables
hf_cache_dir = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from langchain.chains import RetrievalQA
import time

# Global variables
model = None
tokenizer = None
qa_pipeline = None
embed_model = None
qdrant = None

class Item(BaseModel):
    query: str

app = FastAPI()

# Mount static files from TestFolder
app.mount("/files", StaticFiles(directory="TestFolder"), name="files")

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, qa_pipeline, embed_model, qdrant
    
    print("🚀 Loading models....")
    start_time = time.perf_counter()
    
    # Load embedding model
    sentence_embedding_model_path = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    embed_model = HuggingFaceEmbeddings(
        model_name=sentence_embedding_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=hf_cache_dir,
    )

    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(path="qdrant/")
        qdrant = QdrantVectorStore(qdrant_client, "MyCollection", embed_model, distance="Dot")
    except Exception as e:
        print(f"❌ Error initializing Qdrant: {e}")

    # Load QA model
    model_path = "distilbert-base-cased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, cache_dir=hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=hf_cache_dir)

    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    end_time = time.perf_counter()
    print(f"✅ Models loaded successfully in {end_time - start_time:.2f} seconds.")

@app.on_event("shutdown")
async def shutdown_event():
    global model, tokenizer, qa_pipeline, embed_model, qdrant
    print("🚪 Shutting down the API and releasing model memory.")
    del model, tokenizer, qa_pipeline, embed_model, qdrant

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI"}

@app.post("/search")
def search(item: Item):
    print("Search endpoint")
    query = item.query
    
    search_result = qdrant.similarity_search(
        query=query, k=10
    )
    
    list_res = [
        {"id": i, "path": res.metadata.get("path"), "content": res.page_content}
        for i, res in enumerate(search_result)
    ]
    
    return list_res

@app.post("/ask_localai")
async def ask_localai(item: Item):
    query = item.query
    
    try:
        # First, get relevant documents
        docs = qdrant.similarity_search(query, k=3)
        
        # Combine the documents into a single context
        context = " ".join([doc.page_content for doc in docs])
        
        # Use the QA pipeline directly
        answer = qa_pipeline(
            question=query,
            context=context,
            max_length=512,
            max_answer_length=50,
            handle_long_sequences=True
        )
        
        return {
            "question": query,
            "answer": answer["answer"],
            "confidence": answer["score"],
            "source_documents": [
                {
                    "content": doc.page_content[:1000],
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}
