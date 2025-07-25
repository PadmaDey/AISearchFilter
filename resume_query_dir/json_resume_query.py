# ============================================
# resume_query_dir/json_resume_query.py (Deduplicated JSON + Logging + FAISS Guard)
# ============================================
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from PromptSchema.prompt_generator import generate_prompt
from docs_to_json import convert_docs_to_json

load_dotenv()

def normalize_keywords(text: str):
    return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())

def build_ats_scores(resumes, job_desc):
    job_keywords = normalize_keywords(job_desc)
    ats_scores = []
    for resume in resumes:
        resume_text = json.dumps(resume).lower()
        matched_keywords = [kw for kw in job_keywords if kw in resume_text]
        match_score = round(len(matched_keywords) / max(len(job_keywords), 1) * 100, 2)
        ats_scores.append({
            "unique_id": resume.get("unique_id", "unknown"),
            "name": resume.get("name", "N/A"),
            "designation": resume.get("designation", "N/A"),
            "ats_match_score": match_score,
            "matched_keywords": matched_keywords,
            "missing_keywords": list(job_keywords - set(matched_keywords))
        })
    return ats_scores

def build_vectorstore(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", save_dir="resume_query_dir/vectorstore"):
    if not documents:
        print("[WARNING] No resumes to process. Skipping vectorstore build.")
        return None
    embedding_model = HuggingFaceEndpointEmbeddings(repo_id=model_name)
    vectorstore_path = Path(save_dir)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(str(vectorstore_path))
    print(f"[INFO] Vectorstore built and saved at {vectorstore_path}")
    return vectorstore

def create_documents(resumes):
    docs = []
    for resume in resumes:
        keywords = ", ".join(resume.get("keywords", [])) or "None"
        text = f"Keywords: {keywords}\n\n{json.dumps(resume, indent=2)}"
        docs.append(Document(
            page_content=text,
            metadata={
                "name": resume.get("name", ""),
                "unique_id": resume.get("unique_id", ""),
                "designation": resume.get("designation", "")
            }
        ))
    return docs

def chunk_documents(documents, chunk_size=700, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

if __name__ == "__main__":
    job_description = input("\nEnter the job description for ATS evaluation: ").strip()
    resume_dir = input("\nEnter the path to the folder containing resumes (PDF/DOCX): ").strip()

    resume_json_path = Path("resume_query_dir/document/resume.json")
    resume_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume_json_path.exists() or resume_json_path.stat().st_size == 0:
        print(f"[INFO] Resume JSON not found. Converting resumes from '{resume_dir}'...")
        convert_docs_to_json(resume_dir, str(resume_json_path), job_description)
    else:
        print(f"[INFO] Using existing resume JSON: {resume_json_path}")

    # Load the resumes after conversion/deduplication
    with open(resume_json_path, "r", encoding="utf-8") as f:
        resumes_list = json.load(f)

    print(f"[INFO] Loaded {len(resumes_list)} unique resumes for querying (deduplicated).")
    if not resumes_list:
        print("[ERROR] No resumes available after deduplication. Exiting.")
        exit(1)

    # Build ATS scores
    ats_results = build_ats_scores(resumes_list, job_description)

    # Create LangChain Documents and build vectorstore
    documents = create_documents(resumes_list)
    documents = chunk_documents(documents)
    vectorstore = build_vectorstore(documents)
    if not vectorstore:
        print("[ERROR] Vectorstore could not be built (no documents). Exiting.")
        exit(1)

    # Retrieval + Compression
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=model
    )
    compressor = LLMChainExtractor.from_llm(model)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=multiquery_retriever,
        base_compressor=compressor
    )

    # Take user query
    user_query = input("\nEnter your query about the candidates: ").strip()
    query_docs = compression_retriever.invoke(user_query)

    # Organize context by candidate
    merged_context = defaultdict(list)
    for doc in query_docs:
        uid = doc.metadata.get("unique_id", "unknown")
        merged_context[uid].append(
            f"Candidate Name: {doc.metadata.get('name', 'N/A')}\n"
            f"Designation: {doc.metadata.get('designation', 'N/A')}\n"
            f"Resume ID: {uid}\n"
            f"Resume Content:\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join("\n\n".join(parts) for parts in merged_context.values())
    prompt_template = generate_prompt(user_query, context_text)

    # Query LLM
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    response = chain.invoke({"query": user_query, "doc": context_text})

    # Validate JSON output
    try:
        llm_output = json.loads(response)
        candidates = llm_output.get("candidates", [])
    except json.JSONDecodeError:
        candidates = [{"error": "LLM failed to return valid JSON", "raw_output": response}]

    # Enrich with ATS score and total experience
    ats_dict = {a["unique_id"]: a for a in ats_results}
    for c in candidates:
        uid = c.get("unique_id")
        matching_resume = next((r for r in resumes_list if r["unique_id"] == uid), None)
        if matching_resume:
            c["ats_score"] = matching_resume.get("compatibility_analysis", {}).get("overall_score", 0)
            c["total_experience"] = matching_resume.get("total_experience", "Not available")
        else:
            c["ats_score"] = c.get("ats_score", 0)
            c["total_experience"] = c.get("total_experience", "Not available")

    # Output results
    final_output = {"candidates": candidates}
    print("\n[RESULTS]")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


# To run:
# python -m resume_query_dir.json_resume_query
