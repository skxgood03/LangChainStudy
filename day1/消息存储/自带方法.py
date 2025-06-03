# https://python.langchain.com/docs/integrations/memory/redis_chat_message_history/   官方安装教程
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{question}"),
    ]
)
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                   api_key=os.getenv("KEY"),
                   model="qwen/qwq-32b:free", )
# RedisChatMessageHistory 实现自动记录并读取历史history
history = RedisChatMessageHistory(session_id="session_1", redis_url="redis://localhost:6379/0")
parser = StrOutputParser()

chain = prompt_template | model | parser

runnable = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda: history,
)
#清除记录
history.clear()

a = runnable.invoke({"question": "你好，你是谁"})
print(a)
a2 = runnable.invoke({"question": "重复一次"})
print(a2)
a2 = runnable.invoke({"question": "目前我们的对话有多少条？"})
print(a2)