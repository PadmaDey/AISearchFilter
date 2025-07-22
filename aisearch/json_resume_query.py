# aisearch/json_resume_query.py
import json
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# Import strict prompt (schema is embedded to avoid dynamic injection issues)
from PromptTemplate.prompt_generator import prompt

# Load API keys and env vars
load_dotenv()

# === 1. Load resumes JSON (combined) ===
resume_json = Path(r"aisearch\document\talentsync.candidates.json")
if not resume_json.exists():
    raise FileNotFoundError(
        f"Resume JSON not found: {resume_json}. Run pdf_to_json.py first to generate it."
    )

with open(resume_json, "r", encoding="utf-8") as f:
    resumes_list = json.load(f)  # list of dicts (each resume)

# === 2. Build Documents with boosted keywords and guaranteed identity fields ===
documents = []
for resume in resumes_list:
    # Build keywords (technical skills + project tech + certs)
    keywords = set(resume.get("technical_skills", []))
    for proj in resume.get("projects", []):
        keywords.update(proj.get("technologies", []))
    for cert in resume.get("certifications", []):
        if isinstance(cert, dict):
            cname = cert.get("name")
            if cname:
                keywords.add(cname)
        elif isinstance(cert, str):
            keywords.add(cert)
    resume["keywords"] = list(keywords)

    # Repeat keywords to emphasize during vector search
    weighted_keywords = " ".join(resume["keywords"] * 3)

    # Ensure name, designation, and ID are prominent for retrieval
    text = (
        f"Candidate Name: {resume.get('name', 'Unknown')}\n"
        f"Designation: {resume.get('designation', 'N/A')}\n"
        f"Resume ID: {resume.get('unique_id', 'N/A')}\n"
        f"KEY SKILLS (weighted): {weighted_keywords}\n\n"
        f"{json.dumps(resume, indent=2)}"
    )

    metadata = {
        "name": resume.get("name", ""),
        "unique_id": resume.get("unique_id", ""),
        "designation": resume.get("designation", ""),
    }
    documents.append(Document(page_content=text, metadata=metadata))

# === 3. Keep full resumes intact (no chunk splitting) ===

# === 4. Build FAISS vectorstore ===
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("resume_query_dir/vectorstore")

# === 5. Setup retrievers ===
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# MultiQuery expands the user query (for better recall across skills phrasing)
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 100}),
    llm=model
)

# LLM-based compressor summarizes chunks for Claude
compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=multiquery_retriever,
    base_compressor=compressor
)

# === 6. Create chain ===
parser = StrOutputParser()
chain = prompt | model | parser

# === 7. Get user query ===
user_query = input("\nEnter your query about the candidates: ")

# === 8. Retrieve relevant resumes ===
query_docs = compression_retriever.invoke(user_query)

# Merge chunks by candidate (for clarity & to avoid fragmented answers)
merged_context = defaultdict(list)
for doc in query_docs:
    uid = doc.metadata.get("unique_id", "unknown")
    merged_context[uid].append(doc.page_content)

context_text = "\n\n---\n\n".join(
    "\n\n".join(parts) for parts in merged_context.values()
)

# === 9. Generate final structured JSON (strict, no hallucinations) ===
response = chain.invoke({
    "query": user_query,
    "doc": context_text
})

# === 10. Print final result ===
try:
    parsed_json = json.loads(response)
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError:
    print("Model returned invalid JSON. Raw output:\n", response)
