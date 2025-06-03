from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
# 阿里云百炼平台的向量模型
embedding = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv("ALY_EMBADING_KEY"))
redis_url = "redis://localhost:6379"  # Redis数据库的连接地址

import redis

# 创建Redis客户端
redis_client = redis.from_url(redis_url)

print(redis_client.ping())  # true

from langchain_redis import RedisConfig, RedisVectorStore

config = RedisConfig(
    index_name="my_index",  # 索引名称
    redis_url=redis_url,  # Redis数据库的连接地址
)
# # 创建向量存储实例
vector_store = RedisVectorStore(embedding, config=config)
# # 添加文本到向量存储
vector_store.add_texts(["苹果很甜", "香蕉很长", "西瓜又大又圆"])
# # 查询相似文本
scored_results = vector_store.similarity_search_with_score("什么很甜", k=3)
# for text, score in scored_results:
#     print(f"Text: {text}, Score: {score}")
"""
Text: page_content='苹果很甜', Score: 0.194118976593
Text: page_content='西瓜又大又圆', Score: 0.449708104134
Text: page_content='香蕉很长', Score: 0.496699750423
 """

# 也可以构建检索器
retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# print(retriver.invoke("什么很甜"))
"""
[Document(metadata={}, page_content='苹果很甜'), Document(metadata={}, page_content='苹果很甜'), Document(metadata={}, page_content='西瓜又大又圆')]"""

# -----------------链式使用-----------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('human', '{question}'),
])


def format_prompt_value(prompt):
    return prompt.to_string()


chain = prompt | format_prompt_value | retriver
doc = chain.invoke({"question": "什么水果很甜又很长"})
print(doc)
