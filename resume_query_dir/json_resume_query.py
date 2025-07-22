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

# Import centralized prompt (schema is now embedded inside it)
from PromptTemplate.prompt_generator import prompt

# Load environment variables
load_dotenv()

# === 1. Load the combined JSON resumes ===
resume_json = Path("resume_query_dir/document/resume.json")
if not resume_json.exists():
    raise FileNotFoundError(f"Resume JSON not found: {resume_json}. "
                            f"Run pdf_to_json.py first to generate it.")

with open(resume_json, "r", encoding="utf-8") as f:
    resumes_list = json.load(f)  # list of resume dicts

# === 2. Convert resumes into Documents, leveraging the `keywords` field ===
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

# === 3. Split into chunks for embeddings ===
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = splitter.split_documents(documents)

# === 4. Build embeddings & FAISS vectorstore ===
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("resume_query_dir/vectorstore")

# === 5. Setup retrievers ===
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=model
)

compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=multiquery_retriever, base_compressor=compressor
)

# === 6. Parser & Chain ===
parser = StrOutputParser()
chain = prompt | model | parser

# === 7. Take user query ===
user_query = input("\nEnter your query about the candidates: ")

# === 8. Retrieve top matching chunks ===
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

# === 9. Generate final structured JSON output ===
response = chain.invoke({
    "query": user_query,
    "doc": context_text
})

# === 10. Display JSON result (only candidates, no sources) ===
try:
    parsed_json = json.loads(response)
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError:
    print("Model returned invalid JSON. Raw output:\n", response)
