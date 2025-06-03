from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "请将用户的提问转换为{language}，不用要回答问题"),
        ("user", "{question}"),
    ]
)

#只要符合opai规范的接口都可以使用langchain_openai的ChatOpenAI来调用 ：openrouter...
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("KEY"),
                   model="qwen/qwq-32b:free", )

#结果解析器，直接获取纯文本回复
parser  = StrOutputParser()
#构建链
chain = prompt_template| model | parser
result = chain.invoke({"language": "中文", "question": "who are you"})
analysis_prompt = ChatPromptTemplate.from_template("我应该怎么回复这句话？{talk}，请给出回复的内容")
chain2 = {'talk':chain}|analysis_prompt | model | parser
print(chain2.invoke({"language": "中文", "question": "who are you"}))