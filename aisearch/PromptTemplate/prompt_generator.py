# aisearch/PromptTemplate/prompt_generator.py
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import json

# Schema for clarity (saved for reference)
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

# Strict, non-hallucinating prompt
# - Ensures names, designations, and skills come only from resume chunks
# - Properly escapes curly braces so LangChain treats them literally
# - Prevents the LLM from fabricating data

prompt = PromptTemplate(
    template=(
        "You are a recruitment assistant. Carefully analyze the resume chunks below. "
        "ONLY include candidates whose names, designations, and skills are explicitly present in the provided text.\n\n"
        "Focus on ALL explicit mentions of skills, frameworks, tools, and technologies in these fields:\n"
        "- technical_skills\n"
        "- keywords (precompiled list of terms)\n"
        "- projects (including technology stacks)\n"
        "- work_experience (responsibilities and achievements)\n"
        "- certifications (if they reference tools or frameworks)\n\n"
        "Include a candidate ONLY if they explicitly mention the queried skill/tool in any of these fields. "
        "NEVER invent names, skills, or summaries not found in the resume chunks.\n\n"
        "Return ONLY valid JSON with this exact structure (escape braces literally):\n"
        "{{{{\n"
        '  "candidates": [\n'
        "    {{{{\n"
        '      "name": "Candidate name",\n'
        '      "designation": "Current or most recent title",\n'
        '      "unique_id": "Resume identifier",\n'
        '      "skills": ["List of relevant skills, tools, or frameworks explicitly mentioned"],\n'
        '      "experience_summary": "Short summary of work experience (1-2 sentences)"\n'
        "    }}}}\n"
        "  ]\n"
        "}}}}\n\n"
        "Query: {query}\n\nResume Chunks:\n{doc}\n\n"
        "IMPORTANT: Respond ONLY with valid JSON matching the schema above. "
        "Do NOT fabricate candidate names or skills. Do NOT add commentary, sources, or extra fields."
    ),
    input_variables=["query", "doc"],  # Only these variables allowed
)

# Save schema for debugging/reference
schema_path = Path("aisearch/PromptTemplate/schema.json")
schema_path.parent.mkdir(parents=True, exist_ok=True)
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
