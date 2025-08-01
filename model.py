from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
from langchain_core.prompts import PromptTemplate
from tempfile import NamedTemporaryFile
import aiohttp
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from uuid import uuid4
from langchain_pinecone import PineconeVectorStore
from fastapi import FastAPI, HTTPException,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from pinecone import Pinecone, ServerlessSpec
from starlette.concurrency import run_in_threadpool
from langchain.retrievers import MultiQueryRetriever
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from typing import List
import tempfile
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
import concurrent.futures
from functools import partial

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

model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14',max_tokens=400,temperature=0.4)


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


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")
    if file.filename.lower().endswith(".pdf"):

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        abs_file_path = os.path.abspath(file_path)
        with open(abs_file_path, "wb") as f:
            f.write(await file.read())

        loader = await run_in_threadpool(PyMuPDFLoader,abs_file_path)
        docs = await run_in_threadpool(loader.load)
        if len(docs) < 10:
            splitter = RecursiveCharacterTextSplitter(separators='',chunk_size=100,chunk_overlap=20)
        elif len(docs) < 50:
            splitter = RecursiveCharacterTextSplitter(separators='',chunk_size=500,chunk_overlap=100)
        elif len(docs) <100 :
            splitter = RecursiveCharacterTextSplitter(separators='',chunk_size=750,chunk_overlap=150)
        else :
            splitter = RecursiveCharacterTextSplitter(separators='',chunk_size=1500,chunk_overlap=200)

        chunks = await run_in_threadpool(splitter.split_documents, docs)
        session_id = str(uuid4())
        await run_in_threadpool(store.add_documents, chunks,namespace=session_id)
        return {
            'status':"Upload Completed",
            'session_id':session_id
        }

    if file.filename.lower().endswith(".txt"):
        text = (await file.read()).decode("utf-8")
        from langchain_core.documents import Document
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "],chunk_size=1000,chunk_overlap=200)

        chunks = await run_in_threadpool(splitter.split_documents, docs)
        session_id = str(uuid4())
        await run_in_threadpool(store.add_documents, chunks,namespace=session_id)
        return {
            'status':"Upload Completed",
            'session_id':session_id
        }
    
    
@app.post("/qabot")
async def qabot(req:Query ):
    query = req.query
    session_id = req.session_id
    # session_id = session_id
    retriver = store.as_retriever(search_type="similarity",search_kwargs={"k": 10,"namespace":session_id})
    Multiqueryretriver = MultiQueryRetriever.from_llm(retriever=retriver,llm=model)
    retrive_data = Multiqueryretriver.invoke(query)
    context = full_context(retrive_data)
    prompt = template.invoke({
        'context':context,
        'question':query
    })

    # final_chain =Multiqueryretriver|parallel_chain| template | model
    result= model.invoke(prompt)
    return {'response':result.content}

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# @app.post("/hackrx/run")
# async def run(req: RunRequest):
#     # Download PDF from blob URL
#     response = requests.get(req.documents)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Unable to fetch document from provided URL.")

#     # Save to temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(response.content)
#         tmp_path = tmp.name

#     # Load and split
#     loader = await run_in_threadpool(PyMuPDFLoader, tmp_path)
#     docs = await run_in_threadpool(loader.load)

#     # splitter = RecursiveCharacterTextSplitter(separators='',chunk_size=2000,chunk_overlap=250)
#     splitter = CharacterTextSplitter(chunk_size=2000,chunk_overlap=200,separator='')
#     chunks = await run_in_threadpool(splitter.split_documents, docs)

#     # Create session and store embeddings
#     session_id = str(uuid4())
#     await run_in_threadpool(store.add_documents, chunks, namespace=session_id)

#     # Create retriever
#     retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 10, "namespace": session_id})

#     results = {}
#     for q in req.questions:
#         retrieved_docs = retriever.invoke(q)
#         context = full_context(retrieved_docs)
#         prompt = template.invoke({'context': context, 'question': q})
#         response = model.invoke(prompt)
#         results[q] = response.content

#     return {
#     "answers": list(results.values())
#     }




@app.post("/hackrx/run")
async def run(req: RunRequest):
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
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 10, "namespace": session_id})

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