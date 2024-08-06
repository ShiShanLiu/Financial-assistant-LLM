from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time
# load api
load_dotenv() # 讀取.env檔案內的APIs
groq_llm = ChatGroq(temperature=0, model="llama3-70b-8192")

# load Duckduckgo search
wrapper = DuckDuckGoSearchAPIWrapper(region="tw-tzh", max_results=5, source="news", time="d")
duck_search = DuckDuckGoSearchRun(api_wrapper=wrapper)

# prompt for chain
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided and please answer in Traditional Chinese.

Context: {context}

Question: {question}"""
)

# Use RunnablePassthrough to pass input from 'question'
# groq_chain = (
#     RunnablePassthrough.assign(context=(lambda x: x["question"]) | duck_search)
#     | prompt
#     | groq_llm
#     | StrOutputParser()
# )

def duckgosearch(question: str):
    groq_chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | duck_search)
        | prompt
        | groq_llm
        | StrOutputParser()
    )
    res = groq_chain.invoke({"question": question})
    return res

# groq_chain.invoke({"question": "美股的最新消息?"})