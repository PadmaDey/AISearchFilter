from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader(r"test_project\raw_docs\Manisha_Ghosh_Data_Analyst_Resume.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15
)

chunks = splitter.split_text(pages)

full_text = "\n".join([page.page_content for page in chunks])

embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.fron_documents(
    documents=pages,
    embedding=embedding_model
)

vectorstore.save_local("vactorstore")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
compressor = LLMChainExtractor.from_llm(model)

compression_retriver = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)
query = "Tell me about the skills, the candidate has."

prompt = PromptTemplate(
    template="Base on the document, answer the question:\n"
            "{query}\n\n"
            "Document:\n{doc}",
    input_variables=['query', 'doc'],
    validate_template=True
)

prompt.save(r'test_project\PromptTemplate\schema.json')

import json
from pathlib import Path
# Load the JSON schema from external file
schema_path = Path(r"test_project\PromptTemplate\schema.json")
with open(schema_path, "r", encoding="utf-8") as f:
    json_schema = json.load(f)

json_output = model.with_structured_output(schema_path)

json_prompt = json_output.invoke(compression_retriver)

parser = StrOutputParser()

chain = compression_retriver | prompt | model | parser

response = chain.invoke({'doc': full_text})

print(response)