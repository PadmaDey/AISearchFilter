# ================================
# resume_query_dir/docs_to_json.py
# ================================
import json
import uuid
from pathlib import Path
from datetime import datetime
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

load_dotenv()

# === Setup LLM ===
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# === Resume Extraction Prompt (Escaped Braces) ===
extract_prompt = PromptTemplate(
    template="""
Extract the following details from the resume and return ONLY valid JSON (no explanations, no text outside JSON):

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

Resume: {resume}
""",
    input_variables=["resume"],
)
extract_chain = extract_prompt | model | StrOutputParser()

# === ATS Evaluation Prompt (Escaped Braces) ===
ats_prompt = PromptTemplate(
    template="""
You are an ATS evaluator. Compare the following resume details to the job description.
Return ONLY a JSON object with:

{{
  "overall_score": 0,
  "analysis": {{
    "strengths": [],
    "gaps": [],
    "matching_points": [],
    "improvement_areas": []
  }},
  "detailed_scoring": {{
    "skills_match": 0,
    "experience_relevance": 0,
    "education_fit": 0,
    "overall_potential": 0
  }}
}}

Resume: {resume}
Job Description: {job_description}
""",
    input_variables=["resume", "job_description"],
)
ats_chain = ats_prompt | model | StrOutputParser()

# === Helper Functions ===
def normalize_skills(skills_list):
    normalized = []
    for skill in skills_list:
        parts = re.split(r"[(),]", skill)
        for p in parts:
            cleaned = p.strip()
            if cleaned:
                normalized.append(cleaned)
    return list(sorted(set(normalized), key=str.lower))

def parse_duration(duration_str):
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    try:
        # Normalize en-dash/em-dash to a normal hyphen
        duration_str = duration_str.replace("–", "-").replace("—", "-")

        start, end = duration_str.split("-")
        start = start.strip().lower()
        end = end.strip().lower()

        def parse_date(date_str):
            parts = date_str.split()
            if len(parts) == 2:
                month, year = parts
                return int(year), months.get(month[:3], 1)
            elif len(parts) == 1 and parts[0] != "present":
                return int(parts[0]), 1
            else:
                now = datetime.now()
                return now.year, now.month

        sy, sm = parse_date(start)
        ey, em = parse_date(end)
        return (ey - sy) * 12 + (em - sm + 1)
    except Exception:
        return 0


def calculate_total_experience(work_experience):
    total_months = sum(parse_duration(job.get("duration", "")) for job in work_experience)
    return f"{total_months} months"

def load_existing_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# === Main Conversion Function ===
def convert_docs_to_json(input_dir: str, output_json_path: str, job_description: str) -> list:
    input_path = Path(input_dir)
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_resumes = load_existing_json(output_path)

    for file_path in input_path.glob("*"):
        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        print(f"\nProcessing {file_path.name}...")
        try:
            if file_path.suffix.lower() == ".pdf":
                pages = PyPDFLoader(str(file_path)).load()
            else:
                pages = Docx2txtLoader(str(file_path)).load()
            resume_text = "\n".join([p.page_content for p in pages])

            raw_output = extract_chain.invoke({"resume": resume_text})
            parsed = json.loads(raw_output)
            skills = normalize_skills(parsed.get("skills", []))
            total_exp = calculate_total_experience(parsed.get("work_experience", []))

            now = datetime.utcnow().isoformat()
            unique_id = f"{file_path.stem}_{now}"
            existing = next((r for r in all_resumes if r["cv_directory_link"] == f"CVs/{file_path.stem}"), None)

            created_at = existing["created_at"] if existing else now
            status = "Updated" if existing else "New"
            if existing:
                all_resumes.remove(existing)

            ats_result = ats_chain.invoke({"resume": json.dumps(parsed), "job_description": job_description})
            try:
                ats_data = json.loads(ats_result)
            except json.JSONDecodeError:
                ats_data = {
                    "overall_score": 0,
                    "analysis": {"strengths": [], "gaps": [], "matching_points": [], "improvement_areas": []},
                    "detailed_scoring": {"skills_match": 0, "experience_relevance": 0, "education_fit": 0, "overall_potential": 0}
                }

            structured_resume = {
                "_id": str(uuid.uuid4()),
                "name": parsed.get("name", ""),
                "email": parsed.get("email", ""),
                "secondary_email": "",
                "primary_phone": parsed.get("phone", ""),
                "secondary_phone": "",
                "cv_directory_link": f"CVs/{file_path.stem}",
                "unique_id": unique_id,
                "designation": parsed.get("designation", ""),
                "location": parsed.get("location", ""),
                "personal_details": {"date_of_birth": "", "age": "", "gender": "", "marital_status": ""},
                "social_profiles": {
                    "github": parsed.get("github", ""),
                    "linkedin": parsed.get("linkedin", ""),
                    "twitter": "",
                    "leetcode": "",
                    "others": []
                },
                "education": parsed.get("education", []),
                "work_experience": parsed.get("work_experience", []),
                "total_experience": total_exp,
                "technical_skills": skills,
                "soft_skills": [],
                "languages": parsed.get("languages", []),
                "certifications": parsed.get("certifications", []),
                "projects": parsed.get("projects", []),
                "keywords": sorted(set(skills), key=str.lower),
                "job_id": str(uuid.uuid4()),
                "updated_at": now,
                "created_at": created_at,
                "status": status,
                "compatibility_analysis": ats_data
            }
            all_resumes.append(structured_resume)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_resumes, f, indent=2, ensure_ascii=False)
    print(f"\nProcessed {len(all_resumes)} resumes. JSON saved at {output_path}")
    return all_resumes

if __name__ == "__main__":
    print("\nEnter the full path to the resume directory (e.g., resume_query_dir/raw_docs):")
    resume_dir = input().strip()

    output_json = "resume_query_dir/document/resume.json"

    jd_lines = []
    print("\nEnter Job Description (end with empty line):")
    while True:
        line = input()
        if not line.strip():
            break
        jd_lines.append(line)
    job_desc = "\n".join(jd_lines)

    convert_docs_to_json(resume_dir, output_json, job_desc)
