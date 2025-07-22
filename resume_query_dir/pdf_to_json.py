# resume_query_dir/pdf_to_json.py
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import re

load_dotenv()

# === Model & Prompt Setup ===
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
prompt_template = """
Extract the following details from the resume and return ONLY valid JSON (no explanations):

{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "linkedin": "",
  "github": "",
  "designation": "",
  "education": [
    {{"degree": "", "institution": "", "graduation_year": ""}}
  ],
  "work_experience": [
    {{"company_name": "", "position": "", "duration": "", "responsibilities": []}}
  ],
  "skills": [],
  "certifications": [],
  "projects": [
    {{"name": "", "description": "", "technologies": []}}
  ],
  "languages": []
}}

Resume:
{resume}
"""
parser = StrOutputParser()
chain = PromptTemplate(template=prompt_template, input_variables=["resume"]) | model | parser


# === Helper to Normalize Skills ===
def normalize_skills(skills_list):
    normalized = []
    for skill in skills_list:
        parts = re.split(r"[(),]", skill)  # Split by parentheses and commas
        for p in parts:
            cleaned = p.strip()
            if cleaned:
                normalized.append(cleaned)
    return list(sorted(set(normalized), key=str.lower))  # Deduplicate


# === Core Processing Function ===
def convert_pdfs_to_json(input_dir: str, output_json_path: str) -> list:
    """
    Processes all PDF resumes in the given directory and outputs a combined JSON file.
    Returns the list of structured resumes.
    """
    input_path = Path(input_dir)
    output_path = Path(output_json_path)
    all_resumes = []

    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        resume_text = "\n".join([p.page_content for p in pages])

        raw_output = chain.invoke({"resume": resume_text})

        try:
            parsed_core = json.loads(raw_output)

            # Normalize skills & gather keywords
            skills = normalize_skills(parsed_core.get("skills", []))
            keywords = set(skills)

            # Include technologies from projects
            for proj in parsed_core.get("projects", []):
                for tech in proj.get("technologies", []):
                    keywords.add(tech.strip())

            # Handle certifications (dict or string)
            for cert in parsed_core.get("certifications", []):
                if isinstance(cert, dict):
                    cname = cert.get("name", "").strip()
                    if cname:
                        keywords.add(cname)
                elif isinstance(cert, str):
                    keywords.add(cert.strip())

            # Final schema
            structured_resume = {
                "_id": {"$oid": "auto-generate"},
                "name": parsed_core.get("name", ""),
                "email": parsed_core.get("email", ""),
                "secondary_email": "",
                "primary_phone": parsed_core.get("phone", ""),
                "secondary_phone": "",
                "cv_directory_link": f"CVs/{pdf_file.stem}",
                "unique_id": pdf_file.stem,
                "designation": parsed_core.get("designation", ""),
                "location": parsed_core.get("location", ""),
                "personal_details": {"date_of_birth": "", "age": 0, "gender": ""},
                "social_profiles": {
                    "github": parsed_core.get("github", ""),
                    "linkedin": parsed_core.get("linkedin", ""),
                    "twitter": "",
                    "leetcode": "",
                    "others": []
                },
                "education": parsed_core.get("education", []),
                "work_experience": parsed_core.get("work_experience", []),
                "total_experience": "2+ years",
                "technical_skills": skills,
                "soft_skills": [],
                "languages": parsed_core.get("languages", []),
                "certifications": parsed_core.get("certifications", []),
                "projects": parsed_core.get("projects", []),
                "keywords": list(sorted(keywords, key=str.lower)),
                "batch_id": {"$binary": {"base64": "auto-generate", "subType": "04"}},
                "job_id": {"$oid": "auto-generate"},
                "updated_at": {"$date": datetime.utcnow().isoformat() + "Z"},
                "created_at": {"$date": datetime.utcnow().isoformat() + "Z"},
                "status": "New",
                "compatibility_analysis": {
                    "overall_score": 0,
                    "analysis": {
                        "strengths": [],
                        "gaps": [],
                        "matching_points": [],
                        "improvement_areas": []
                    },
                    "detailed_scoring": {
                        "skills_match": 0,
                        "experience_relevance": 0,
                        "education_fit": 0,
                        "overall_potential": 0
                    }
                }
            }

            all_resumes.append(structured_resume)

        except json.JSONDecodeError:
            print(f"Failed to parse {pdf_file.name}. Raw output:\n{raw_output}")

    # Save combined JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_resumes, f, indent=2)

    print(f"\nProcessed {len(all_resumes)} resumes. Combined JSON saved at {output_path}")
    return all_resumes


# Allow standalone execution
if __name__ == "__main__":
    convert_pdfs_to_json("resume_query_dir/raw_docs", "resume_query_dir/document/resume.json")
