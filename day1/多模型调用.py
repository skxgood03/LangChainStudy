from langchain_openai import ChatOpenAI


#只要符合opai规范的接口都可以使用langchain_openai的ChatOpenAI来调用 ：openrouter...
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key="",
                   model="qwen/qwq-32b:free", )

#当然也可以使用已经写好的包 如：使用Deepseek需要先按照包 pip install -qU "langchain[deepseek]"
# from langchain_deepseek import ChatDeepSeek
# model = ChatDeepSeek(
#                    model="deepseek-chat",api_key="sk-deepseek-v1-76672505e6f23")

#如果在https://python.langchain.com/docs/integrations/providers/没有找到对应的包如通义千问的，则可以去社区版本
#先要安装包 pip install -U langchain-community 再导入通义千问
# from langchain_community.chat_models import ChatTongyi

from langchain_core.messages import HumanMessage, SystemMessage

messages = [SystemMessage(content="请将用户的提问转换为中文，不用要回答问题"),
            HumanMessage(content="who are you") ]
from langchain_core.output_parsers import StrOutputParser

# parser = StrOutputParser()
result = model.invoke(messages)
print(result)
# aaa = parser.invoke(result)