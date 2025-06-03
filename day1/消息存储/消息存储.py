# https://python.langchain.com/docs/integrations/memory/redis_chat_message_history/   官方安装教程
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("KEY"),
                   model="anthropic/claude-3.7-sonnet:beta", )
# 使用RedisChatMessageHistory来存储聊天记录
history = RedisChatMessageHistory(session_id="session_1",redis_url="redis://localhost:6379/0")

# history.add_user_message("你好，你是谁")
# ai1 = model.invoke(history.messages)
# print(ai1.content)
#
# history.add_message(ai1)

history.add_user_message("请重复一次")
ai2 = model.invoke(history.messages)
print(ai2.content)
#存入redis
history.add_message(ai2)