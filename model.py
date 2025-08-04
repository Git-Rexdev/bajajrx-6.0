import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
import aiohttp
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import UnstructuredFileLoader
from urllib.parse import urlparse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain.vectorstores import FAISS
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
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

model = ChatOpenAI(
    model='gpt-4.1-nano-2025-04-14',
    max_tokens=300,
    temperature=0.2,
    request_timeout=15,
    max_retries=1
)

template = PromptTemplate(
    template="""
You are a helpful assistant for answering questions using provided document context.

The context may include tables (between [TABLE]...[/TABLE]). Use them carefully to answer precisely.

If you don't know the answer, just say so.
Give the response in a short conversational manner.

Context: {context}

Question: {question}
""",
    input_variables=['context', 'question'],
    validate_template=True
)

def full_context(doc):
    context_text = "\n\n".join(i.page_content for i in doc)
    return context_text

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
    parsed_url = urlparse(req.documents)
    file_ext = os.path.splitext(parsed_url.path)[-1].lower()
    if file_ext not in [".pdf", ".docx", ".eml", ".msg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, DOCX, EML, MSG are supported.")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name

    # if file_ext == ".pdf":
    #     loader_cls = PyMuPDFLoader
    # elif file_ext == ".docx":
    #     loader_cls = UnstructuredWordDocumentLoader
    # elif file_ext in [".eml", ".msg"]:
    #     loader_cls = UnstructuredEmailLoader
    # else:
    #     raise HTTPException(status_code=400, detail="Unsupported file type.")


    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        loader = await run_in_threadpool(UnstructuredFileLoader, tmp_path, mode="elements")
        docs = await run_in_threadpool(loader.load)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = await run_in_threadpool(splitter.split_documents, docs)

    # session_id = str(uuid4())


    first_batch = chunks[:100]
    db = await run_in_threadpool(FAISS.from_documents, first_batch, embedding_model)
    
    BATCH_SIZE = 100 

    for i in range(100, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        await run_in_threadpool(db.add_documents, batch)
    

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    async def process_single_question(question: str):
        retrieved_docs = retriever.invoke(question)
        context = full_context(retrieved_docs)
        prompt = template.invoke({'context': context, 'question': question})
        response = model.invoke(prompt)
        return response.content

    tasks = [process_single_question(q) for q in req.questions]
    results = await asyncio.gather(*tasks)
    
    return {"answers": results}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("model:app", host="0.0.0.0", port=port)
