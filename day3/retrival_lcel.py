import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

query = "你能帮助史可轩处理日常事务吗"

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

load_dotenv()
# 阿里云百炼平台的向量模型
embedding = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv("ALY_EMBADING_KEY"))
redis_url = "redis://localhost:6379"  # Redis数据库的连接地址

config = RedisConfig(
    index_name="my_index2",  # 索引名称
    redis_url=redis_url,  # Redis数据库的连接地址
)

# # 创建向量存储实例
vector_store = RedisVectorStore(embedding, config=config)
retriever = vector_store.as_retriever()


# 构建prompt模板，将检索到的段落和用户问题传入
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个问答机器人，你的任务是根据下述给定的已知信息回答用户的问题
        已知信息：{context}
        用户问题：{query}
        如果已知问题不包含用户问题的答案，或者已知信息不足以回答用户的问题，请回答“抱歉，我无法回答这个问题。”
        请不要输出已知信息中不包含的信息或者答案。
        用中文回答用户问题
        """),
    ]
)

from langchain_openai import ChatOpenAI

# 结果解析器，直接获取纯文本回复
parser = StrOutputParser()

model = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_KEY"),
                   model="qwen/qwq-32b:free", )


def collect_documents(segments):
    text = []

    for segment in segments:
        text.append(segment.page_content)
    return text


from operator import itemgetter

chain = ({
             "context": itemgetter("query") | retriever | collect_documents,
             "query": itemgetter("query")
         } | prompt_template | model | parser
         )
response = chain.invoke({"query": query})
print(response
      )
