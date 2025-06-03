import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "请将用户的提问转换为{language}，不用要回答问题"),
        ("user", "{question}"),
    ]
)
prompt = prompt_template.invoke({"language": "中文", "question": "who are you"})

#只要符合opai规范的接口都可以使用langchain_openai的ChatOpenAI来调用 ：openrouter...
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("KEY"),
                   model="qwen/qwq-32b:free", )


result = model.invoke(prompt)
print(result)
