# indexing_01.py
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
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
    transcript = ""

# === 2. Split Transcript into Chunks ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript])

# === 3. Create Embeddings & VectorStore ===
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embedding_model)

# Export `vector_store` so chain_method.py can import it
__all__ = ["vector_store"]
