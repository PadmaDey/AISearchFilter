from langchain_core.prompts import PromptTemplate
from pathlib import Path
import json

# Candidate-only schema (no sources)
schema = {
    "candidates": [
        {
            "name": "Candidate name",
            "designation": "Current or most recent title",
            "unique_id": "Resume identifier",
            "skills": ["List of relevant skills, tools, or frameworks explicitly mentioned"],
            "experience_summary": "Short summary of work experience (1-2 sentences)"
        }
    ]
}

# Enhanced prompt - considers keywords, skills, and all resume fields
prompt = PromptTemplate(
    template=(
        "You are a recruitment assistant. Analyze the resume chunks carefully, focusing on ALL explicit skills, "
        "frameworks, tools, and technologies mentioned in these fields:\n"
        "- technical_skills\n"
        "- keywords (precompiled list of important terms from the resume)\n"
        "- projects (and their technology stacks)\n"
        "- work_experience (responsibilities and achievements)\n"
        "- certifications (if they mention tools or frameworks)\n\n"
        "The query may refer to a specific tool, framework, or skill (e.g., Django, Postman). Your job is to:\n"
        "1. Search through ALL these fields and any other text.\n"
        "2. Identify candidates who explicitly worked with or have experience in the queried skill/tool.\n"
        "3. Ignore candidates who do not explicitly mention the skill/tool.\n\n"
        "Return ONLY valid JSON with this structure (escape curly braces correctly, no extra fields or commentary):\n"
        "{{\n"
        '  "candidates": [\n'
        "    {{\n"
        '      "name": "Candidate name",\n'
        '      "designation": "Current or most recent title",\n'
        '      "unique_id": "Resume identifier",\n'
        '      "skills": ["List of relevant skills, tools, or frameworks explicitly mentioned"],\n'
        '      "experience_summary": "Short summary of work experience (1-2 sentences)"\n'
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "For each matching candidate, always include:\n"
        "- name\n- designation\n- unique_id\n- a list of skills/tools where the queried skill appears explicitly (include related skills as context)\n"
        "- a concise experience summary (1-2 sentences) highlighting their work with the queried skill.\n\n"
        "Query: {query}\n\nResume Chunks:\n{doc}\n\n"
        "IMPORTANT: Respond ONLY with valid JSON matching the schema above. No explanations, no extra commentary."
    ),
    input_variables=["query", "doc"],  # schema is now built-in, no injection errors
)

# Save schema separately for reference (not used in the prompt directly anymore)
schema_path = Path("resume_query_dir/PromptTemplate/schema.json")
schema_path.parent.mkdir(parents=True, exist_ok=True)
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
