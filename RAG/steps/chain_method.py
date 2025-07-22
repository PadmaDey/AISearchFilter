
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from retrieval_02 import retriever, user_query
from augmentation_03 import llm, final_prompt

load_dotenv()

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "user_query": RunnablePassthrough(user_query)
    }
)

parser = StrOutputParser()

chain = parallel_chain | final_prompt | llm | parser

response = chain.invoke()
print(f"Response to the query: {response}")