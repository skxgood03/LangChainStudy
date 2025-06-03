from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "请将用户的提问转换为中文，不用要回答问题"),
        ("user", "{question}"),
    ]
)

prompt_template_je = ChatPromptTemplate.from_messages(
    [
        ("system", "请将用户的提问转换为日语，不用要回答问题"),
        ("user", "{question}"),
    ]
)


#只要符合opai规范的接口都可以使用langchain_openai的ChatOpenAI来调用 ：openrouter...
model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("KEY"),
                   model="anthropic/claude-3.7-sonnet:beta", )

#结果解析器，直接获取纯文本回复
parser  = StrOutputParser()
#构建链

#好处就是针对不同的需求可以用不同的大模型，更加合理
chain_ch = prompt_template| model | parser
chain_je = prompt_template_je| model | parser

parallel_chains = RunnableMap(
    {
        "chinese": chain_ch,
        "japanese": chain_je
    }
)
final_chain= parallel_chains | RunnableLambda(lambda x: f"中文：{x['chinese']}\n日语：{x['japanese']}")

print(final_chain.invoke({"question": "who are you"}))


#安装langgraph、grandalf 可将执行过程图形化展示例如果
final_chain.get_graph().print_ascii()