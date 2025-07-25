# pdf_to_json.py
import json
from pathlib import Path
from datetime import datetime, UTC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

def convert_pdf_to_json(pdf_path: str, output_json_path: str) -> dict:
    """Converts a PDF resume into a structured JSON and saves it."""
    
    # === Load PDF ===
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_resume_text = "\n".join([p.page_content for p in pages])

    # === Claude Model ===
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    # === Prompt ===
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
    raw_output = chain.invoke({"resume": full_resume_text})

    # === Validate JSON ===
    try:
        parsed_core = json.loads(raw_output)
    except json.JSONDecodeError:
        print("Claude did not return valid JSON. Raw output:\n", raw_output)
        return {}

    # === Build Final Structured Schema ===
    structured_resume = {
        "_id": {"$oid": "auto-generate"},
        "name": parsed_core.get("name", ""),
        "email": parsed_core.get("email", ""),
        "secondary_email": "",
        "primary_phone": parsed_core.get("phone", ""),
        "secondary_phone": "",
        "cv_directory_link": "CVs/auto",
        "unique_id": "auto",
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
        "technical_skills": parsed_core.get("skills", []),
        "soft_skills": [],
        "languages": parsed_core.get("languages", []),
        "certifications": parsed_core.get("certifications", []),
        "projects": parsed_core.get("projects", []),
        "batch_id": {"$binary": {"base64": "auto-generate", "subType": "04"}},
        "job_id": {"$oid": "auto-generate"},
        "updated_at": {"$date": datetime.now(UTC).isoformat()},
        "created_at": {"$date": datetime.now(UTC).isoformat()},
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

    # Ensure output directory exists
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_resume, f, indent=2)

    print(f"Structured resume saved at {output_path}")
    return structured_resume
