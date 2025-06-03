import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                   api_key=os.getenv("KEY"),
                   model="google/gemini-2.0-flash-001", )

def get_sunset(region:str):
    """
    获取指定地区的日落时间
    :param region: 地区
    :return:
    """

    return f"{region}日落时间为18:30"
getSunsetTool = StructuredTool.from_function(func=get_sunset,description="获取指定地区的日落时间",name="get_sunset")

@tool
def get_weather(region: str):
    """
    获取指定地区的天气
    :param region: 地区
    :return:
    """

    return f"{region}晴天，25度"

# 智能体在回答问题之前会尝试进行推理，是否需要调用工具
agent = initialize_agent(
    tools=[get_weather, getSunsetTool],
    llm=model,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

query = "上海今天天气怎么样什么时候日落"
response = agent.invoke(query)
print(response)
"""
输出：
> Entering new AgentExecutor chain...
I need to find out the weather and sunset time for Shanghai.
Action: get_weather
Action Input: 上海
Observation: 上海晴天，25度
Thought:I have the weather information for Shanghai. Now I need to get the sunset time.
Action: get_sunset
Action Input: 上海
Observation: 上海日落时间为18:30
Thought:I now know the weather and sunset time for Shanghai.
Final Answer: 上海今天晴天，25度，日落时间为18:30

> Finished chain.
{'input': '上海今天天气怎么样什么时候日落', 'output': '上海今天晴天，25度，日落时间为18:30'}
"""