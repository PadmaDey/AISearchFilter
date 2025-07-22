
from dotenv import load_dotenv

from augmentation_03 import llm, final_prompt

load_dotenv()

response = llm.invoke(final_prompt)
# print(f"Response to the query: {response}")
print(f"Response to the query: {response.content}")