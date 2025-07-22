
from dotenv import load_dotenv

from indexing_01 import vector_store

load_dotenv()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
# print(retriver)

user_query = input("\nEnter your query about the candidates: ")

retriever.invoke(user_query)
