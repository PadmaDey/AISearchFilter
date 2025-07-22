# interactive_resume_parser.py
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# Load environment variables (API keys, etc.)
load_dotenv()

# === 1. Load the structured JSON resume ===
resume_json = Path(r"resume_query_single\document\resume.json")
if not resume_json.exists():
    raise FileNotFoundError(f"Resume JSON not found: {resume_json}. "
                            f"Please run the PDF-to-JSON conversion script first.")

with open(resume_json, "r", encoding="utf-8") as f:
    resume_data = json.load(f)

# Convert JSON into a Document object for LangChain
pages = [Document(page_content=json.dumps(resume_data, indent=2))]

# === 2. Split into smaller chunks for processing ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(pages)
full_text = "\n".join([doc.page_content for doc in documents])

# === 3. Build embeddings & vectorstore ===
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("vectorstore")

# === 4. Retriever with LLM-based compression ===
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever, base_compressor=compressor
)

# === 5. JSON Schema for structured query output ===
schema = {
    "answer": {
        "skills": "List of technical or analytical skills (if query is about skills)",
        "tools": "List of BI tools, programming languages, or software (if relevant)",
        "experience_summary": "Short summary of work experience (if query is about experience)",
        "certifications": "List of certifications (if query is about certifications)",
        "projects": "Key projects (if query is about projects)"
    },
    "sources": ["Array of resume text snippets used as evidence"]
}

# Optionally save schema
schema_path = Path(r"resume_query_single\PromptTemplate\schema_json.json")
schema_path.parent.mkdir(parents=True, exist_ok=True)
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

# === 6. Prompt Template (forces JSON output) ===
prompt = PromptTemplate(
    template=(
        "You are a resume analysis assistant. Answer the user's query based only on the provided "
        "document and return the output strictly as valid JSON following this schema:\n{schema}\n\n"
        "Query: {query}\n\n"
        "Document:\n{doc}\n\n"
        "IMPORTANT: Respond ONLY with valid JSON. Do not include any extra explanation."
    ),
    input_variables=["schema", "query", "doc"],
)

parser = StrOutputParser()
chain = prompt | model | parser

# === 7. Ask for user query ===
user_query = input("\nEnter your query about the candidate: ")

# === 8. Fetch relevant chunks ===
query_docs = compression_retriever.invoke(user_query)
context_text = "\n".join([doc.page_content for doc in query_docs])

# === 9. Generate JSON output ===
response = chain.invoke({
    "schema": json.dumps(schema, indent=2),
    "query": user_query,
    "doc": context_text or full_text
})

# === 10. Display JSON result ===
try:
    parsed_json = json.loads(response)
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError:
    print("Model returned invalid JSON. Raw output:\n", response)
