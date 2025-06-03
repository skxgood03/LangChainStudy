import os
from os import environ

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from langchain.tools import tool
from dotenv import load_dotenv
import datetime

load_dotenv()
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                   api_key=os.getenv("KEY"),
                   model="google/gemini-2.0-flash-001", )
#声明工具函数 @tool
#必须要写清楚工具的名称以及作用，否则模型无法理解
@tool
def get_date():
    """获取当前日期"""

    return datetime.datetime.now().strftime("%Y-%m-%d")

@tool
def get_weather(region:str):
    """
    获取指定地区的天气
    :param region: 地区
    :return:
    """

    return f"{region}晴天，25度"



"""第二种注册工具的方式，该方式无需加tool"""
def get_sunset(region:str):
    """
    获取指定地区的日落时间
    :param region: 地区
    :return:
    """

    return f"{region}日落时间为18:30"
getSunsetTool = StructuredTool.from_function(func=get_sunset,description="获取指定地区的日落时间",name="get_sunset")

#绑定工具到模型
llm_with_tools = model.bind_tools([get_date,get_weather,getSunsetTool])
# key必须和函数名相同
all_tools = {"get_date": get_date,"get_weather": get_weather,"get_sunset": getSunsetTool}
# query = "今天是几月几号"
# query = "上海今天天气怎么样"
query = "上海今天天气怎么样,几点有日落"
message=[query]
#询问大模型，大模型会判断是否需要调用工具，并返回一个工具调用请求
ai_msg = llm_with_tools.invoke(message)
print(ai_msg)
message.append(ai_msg)


if ai_msg.additional_kwargs:

    for tool_call in ai_msg.tool_calls:
        #找到需要调用的工具，交给大模型使用
        tool_result = all_tools[tool_call['name'].lower()]
        tool_msg = tool_result.invoke(tool_call)
        message.append(tool_msg)
aa = llm_with_tools.invoke(message).content
#返回结果
print(aa)

