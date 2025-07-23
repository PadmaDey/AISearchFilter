# ============================================
# resume_query_dir/PromptSchema/prompt_generator.py (Schema as variable)
# ============================================

from langchain_core.prompts import PromptTemplate
from pathlib import Path
import json

# Candidate schema (for clarity, now passed as an input variable)
schema = {
    "candidates": [
        {
            "name": "Candidate name",
            "designation": "Current or most recent title",
            "unique_id": "Resume identifier",
            "skills": ["List of relevant skills, tools, or frameworks explicitly mentioned"],
            "experience_summary": "Short summary of work experience (1-2 sentences)",
            "total_experience": "Total professional experience (e.g., '3 years 6 months')",
            "ats_score": "Numerical ATS score (0-100)"
        }
    ]
}

# Prompt template, now referencing {schema} dynamically
prompt = PromptTemplate(
    template=(
        "You are a recruitment assistant. Analyze the resume chunks carefully, focusing on ALL explicit skills, "
        "frameworks, tools, and technologies mentioned in these fields:\n"
        "- technical_skills\n"
        "- keywords (precompiled list of important terms from the resume)\n"
        "- projects (and their technology stacks)\n"
        "- work_experience (responsibilities and achievements)\n"
        "- certifications (if they mention tools or frameworks)\n"
        "- total_experience (overall professional experience)\n"
        "- compatibility_analysis.overall_score (ATS score for the candidate)\n\n"
        "The query may refer to a specific tool, framework, or skill (e.g., Django, Postman). Your job is to:\n"
        "1. Search through ALL these fields and any other text.\n"
        "2. Identify candidates who explicitly worked with or have experience in the queried skill/tool.\n"
        "3. Include each candidate’s ATS score (0-100) from their compatibility analysis.\n"
        "4. Include their total professional experience.\n"
        "5. Ignore candidates who do not explicitly mention the skill/tool.\n\n"
        "Return ONLY valid JSON following this schema (escape curly braces properly, no extra commentary):\n"
        "{schema}\n\n"
        "Query: {query}\n\n"
        "Resume Chunks:\n{doc}\n\n"
        "IMPORTANT: Respond ONLY with valid JSON matching the schema exactly. No extra text."
    ),
    input_variables=["query", "doc", "schema"],
)

def generate_prompt(query: str, context: str) -> PromptTemplate:
    """
    Dynamically generates a prompt with schema included as a variable for readability.
    """
    return PromptTemplate(
        template=prompt.template,
        input_variables=["query", "doc", "schema"]
    ).partial(query=query, doc=context, schema=json.dumps(schema, indent=2))

# Save schema for reference (keeps human-readable JSON without escaping)
schema_path = Path("resume_query_dir/PromptSchema/schema.json")
schema_path.parent.mkdir(parents=True, exist_ok=True)
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)
