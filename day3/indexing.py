import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
#1.加载数据
loader = TextLoader("Q&A.txt", encoding="utf-8")

documents = loader.load()
"""批量加载数据文件
directLoader = DirectoryLoader("day3", glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
documentss = directLoader.load()
"""
# 2.分割
from langchain_text_splitters import CharacterTextSplitter

# 自带的文本分割器(有可能分的不彻底)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n\n", keep_separator=True)
#
# segments = text_splitter.split_documents(documents)

# 手写
text = re.split(r'\n\n', documents[0].page_content)  # 使用正则表达式分割文本
# segments = text_splitter.split_text(documents[0].page_content)  # 使用自定义的文本分割器
segments_documents = text_splitter.create_documents(text)
print(len(segments_documents))  # 打印分割后的段落数量

# 3.保存
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
# # 添加文本到向量存储
vector_store.add_documents(segments_documents)
