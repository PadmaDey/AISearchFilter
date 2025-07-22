from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from indexing_01 import vector_store
from retrieval_02 import retriever, user_query

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"]
)

retrieved_docs = retriever.invoke(user_query)
# print(f"Retrieved {len(retriver_docs)} relevant documents.")

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(f"Context for the question:\n{context_text}")

final_prompt = prompt.invoke(
    {
        "context": context_text,
        "question": user_query
    }
)
# print(final_prompt)

