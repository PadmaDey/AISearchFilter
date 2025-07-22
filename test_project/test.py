from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load PDF
loader = PyPDFLoader(r"test_project/raw_docs/Manisha_Ghosh_Data_Analyst_Resume.pdf")
pages = loader.load()

full_text = "\n".join([page.page_content for page in pages])

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15
)
chunks = splitter.create_documents([full_text])

# Embedding model
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

vectorstore.save_local("test_project/vectorstore")

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# LLM Model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
compressor = LLMChainExtractor.from_llm(model)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

query = "Tell me about the skills the candidate has."

# Prompt Template
prompt = PromptTemplate(
    template="Based on the document, answer the question:\n"
             "{query}\n\n"
             "Document:\n{doc}",
    input_variables=['query', 'doc']
)

prompt.save(r"test_project/PromptSchema/schema.json")

# Get retrieved docs first (string query only)
retrieved_docs = compression_retriever.invoke(query)
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Run prompt with retrieved text
parser = StrOutputParser()
chain = prompt | model | parser

response = chain.invoke({'query': query, 'doc': retrieved_text})

print(response)
