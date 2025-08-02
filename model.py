from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
import aiohttp
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain_pinecone import PineconeVectorStore
from fastapi import FastAPI, HTTPException,Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import os
from pinecone import Pinecone, ServerlessSpec
from starlette.concurrency import run_in_threadpool
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from typing import List
import tempfile
import asyncio
import concurrent.futures

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    session_id: str


embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14',max_tokens=300,  # Reduced from 400
    temperature=0.2,  # Reduced from 0.4 for faster processing
    request_timeout=15,  # Add timeout
    max_retries=1 )


pc =Pinecone()

index_name = "bajajhackrx"

if not pc.has_index(index_name):
    pc.create_index(name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws',region='us-east-1'))
    
index_store = pc.Index(index_name)
store = PineconeVectorStore(index=index_store,embedding=embedding_model)

template = PromptTemplate(template="""
You are helpfull assistant for conversational question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Give response in short conversational manner.
    Context: {context}\n\n
    Question : {question}""",input_variables=['context','question'],validate_template=True)

def full_context(doc):
    context_text = "\n\n".join(i.page_content for i in doc)
    return context_text

parallel_chain = RunnableParallel({
    'question':RunnablePassthrough(),
    'context':RunnableLambda(full_context)
})

API_TOKEN = "f799dd3c9ae79667d28623cf53c3683e115c2ebb26fff88fafc7bc55225c70d1"

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split("Bearer ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token")

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run(req: RunRequest, _: str = Depends(verify_token)):
    async with aiohttp.ClientSession() as session:
        async with session.get(req.documents) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Unable to fetch document")
            pdf_content = await response.read()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name
    
    # Use ThreadPoolExecutor for CPU-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Load and split documents in parallel
        loader = await run_in_threadpool(PyMuPDFLoader, tmp_path)
        docs = await run_in_threadpool(loader.load)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=200)
        
        chunks = await run_in_threadpool(splitter.split_documents, docs)

    # Create session and store embeddings
    session_id = str(uuid4())
    await run_in_threadpool(store.add_documents, chunks, namespace=session_id)

    # Create retriever
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 8, "namespace": session_id})

    async def process_single_question(question: str):
        retrieved_docs = retriever.invoke(question)
        context = full_context(retrieved_docs)
        prompt = template.invoke({'context': context, 'question': question})
        response = model.invoke(prompt)
        return response.content
    
    # Process all questions concurrently
    tasks = [process_single_question(q) for q in req.questions]
    results = await asyncio.gather(*tasks)
    
    return {"answers": results}