# resume_query_dir/json_resume_query.py
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from collections import defaultdict

# Import centralized prompt
from resume_query_dir.PromptSchema.prompt_generator import prompt

# Import the function to auto-generate JSON from PDFs
from resume_query_dir.pdf_to_json import convert_pdfs_to_json

# Load environment variables
load_dotenv()

# === 1. Auto-generate resume JSON from PDFs if not present ===
resume_json_path = Path("resume_query_dir/document/resume.json")

if not resume_json_path.exists() or resume_json_path.stat().st_size == 0:
    print("Resume JSON not found or empty. Generating from PDFs...")
    convert_pdfs_to_json("resume_query_dir/raw_docs", str(resume_json_path))

# === 2. Load the combined JSON resumes ===
with open(resume_json_path, "r", encoding="utf-8") as f:
    resumes_list = json.load(f)  # list of resume dicts

# === 3. Convert resumes into Documents, leveraging the `keywords` field ===
documents = []
for resume in resumes_list:
    keywords = ", ".join(resume.get("keywords", [])) or "None"
    text = f"Keywords: {keywords}\n\n{json.dumps(resume, indent=2)}"
    metadata = {
        "name": resume.get("name", ""),
        "unique_id": resume.get("unique_id", ""),
        "designation": resume.get("designation", ""),
    }
    documents.append(Document(page_content=text, metadata=metadata))

# === 4. Split into chunks for embeddings ===
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = splitter.split_documents(documents)

# === 5. Build embeddings & FAISS vectorstore ===
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("resume_query_dir/vectorstore")

# === 6. Setup retrievers ===
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=model
)

compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=multiquery_retriever, base_compressor=compressor
)

# === 7. Parser & Chain ===
parser = StrOutputParser()
chain = prompt | model | parser

# === 8. Take user query ===
user_query = input("\nEnter your query about the candidates: ")

# === 9. Retrieve top matching chunks ===
query_docs = compression_retriever.invoke(user_query)

# Merge retrieved chunks by candidate
merged_context = defaultdict(list)
for doc in query_docs:
    uid = doc.metadata.get("unique_id", "unknown")
    merged_context[uid].append(
        f"Candidate Name: {doc.metadata.get('name', 'N/A')}\n"
        f"Designation: {doc.metadata.get('designation', 'N/A')}\n"
        f"Resume ID: {uid}\n"
        f"Resume Content:\n{doc.page_content}"
    )

context_text = "\n\n---\n\n".join(
    "\n\n".join(parts) for parts in merged_context.values()
)

# === 10. Generate final structured JSON output ===
response = chain.invoke({
    "query": user_query,
    "doc": context_text
})

# === 11. Display JSON result ===
try:
    parsed_json = json.loads(response)
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError:
    print("Model returned invalid JSON. Raw output:\n", response)


# python -m resume_query_dir.json_resume_query
