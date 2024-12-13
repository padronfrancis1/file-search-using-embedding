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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForQuestionAnswering
from langchain_community.llms import HuggingFacePipeline
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.schema import Document
import time
import torch

model = None
tokenizer = None
dolly_pipeline_hf = None
embed_model = None
qdrant = None
model_name_hf = None
text_generation_pipeline = None
qa_pipeline = None

class Item(BaseModel):
    query: str

app = FastAPI()
# app.mount("/TestFolder", StaticFiles(directory="./TestFolder"), name="TestFolder")

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, dolly_pipeline_hf, embed_model, qdrant, model_name_hf, text_generation_pipeline, qa_pipeline
    
    print("🚀 Loading model....")
    
    sentence_embedding_model_path = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    start_time = time.perf_counter()
    
    embed_model = HuggingFaceEmbeddings(
        model_name=sentence_embedding_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=hf_cache_dir,
    )

    try:
        qdrant_client = QdrantClient(path="/app/qdrant/meta.json")
        qdrant = QdrantVectorStore(qdrant_client, "MyCollection", embed_model, distance="Dot")
    except Exception as e:
        print(f"❌ Error initializing Qdrant: {e}")

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
    print(f"✅ Dolly model loaded successfully in {end_time - start_time:.2f} seconds.")

app.on_event("shutdown")
async def shutdown_event():
    global model, tokenizer, dolly_pipeline_hf
    print("🚪 Shutting down the API and releasing model memory.")
    del model, tokenizer, dolly_pipeline_hf, embed_model, qdrant, model_name_hf, text_generation_pipeline, qa_pipeline


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI"}

@app.post("/search")
def search(Item:Item):
    print("Search endpoint")
    query = Item.query
    
    search_result = qdrant.similarity_search(
        query=query, k=10
    )
    i = 0
    list_res = []
    for res in search_result:
        list_res.append({"id":i,"path":res.metadata.get("path"),"content":res.page_content})


    return list_res

@app.post("/ask_localai")
async def ask_localai(item: Item):
    query = item.query
    
    search_result = qdrant.similarity_search(query=query, k=3)
    if not search_result:
        return {"error": "No relevant results found for the query."}

    context = " ".join([res.page_content for res in search_result])
    if not context.strip():
        return {"error": "No relevant context found."}

    try:
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {query}\n"
            f"Answer concisely and only based on the context provided. Do not repeat the context or the question.\n"
            f"Answer:"
        )
        qa_result = qa_pipeline(question=query, context=context)
        answer = qa_result["answer"]

        return {
            "question": query,
            "answer": answer
        }
    except Exception as e:
        return {"error": "Failed to generate an answer."}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return {"item": item, "total_price": item.price + (item.tax or 0)}
