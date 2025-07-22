# RAG/main.py
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# === 1. Load YouTube Transcript ===
video_id = "Gfr50f6ZBvo"
transcript = ""

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(f"Transcript loaded for video {video_id}.")
except TranscriptsDisabled:
    print(f"No transcript available for video {video_id}.")

# === 2. Split Transcript into Chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# === 3. Create Embeddings & VectorStore ===
embedding_model = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

# === 4. Build Retriever ===
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# === 5. Define LLM and Prompt ===
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say "I don't know."

      Context:
      {context}

      Question: {question}
    """,
    input_variables=["context", "question"]
)

# === 6. Format retrieved docs ===
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === 7. Parallel chain (retriever + context formatting) ===
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# === 8. Combine into the final chain ===
parser = StrOutputParser()
chain = parallel_chain | prompt | llm | parser

# === 9. Take query and run ===
user_query = input("\nEnter your query about the video: ")
response = chain.invoke(user_query)
print(f"\nFinal Answer:\n{response}")
