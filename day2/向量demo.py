# 安装包 pip install -q dashscope
"""
什么是embedding？
Embedding（嵌入）是将文本、图像或其他数据类型转换为向量表示的一种技术。
计算机本身不理解自然语言，因此需要将文本转换为数字形式，以便进行计算和分析。
看起来就是一个浮点数列表。然后通过一些算法如余弦相似度来计算向量之间的相似性。从而找到文本和文本之间的语义相似度
。最后通过大模型进行仲裁
比如用户问的一些问题，可以通过embedding将问题转换为向量表示，然后与知识库中的文档向量进行比较，找到最相关的文档。

"""
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
import os
import numpy as np
from sklearn.metrics.pairwise  import cosine_similarity
load_dotenv()
#阿里云百炼平台的向量模型
embedding = DashScopeEmbeddings(model="text-embedding-v3",dashscope_api_key=os.getenv("ALY_EMBADING_KEY"))
# text = "Hello, world! This is a test for DashScope embeddings."
#
# query_embadding = embedding.embed_query(text)

# print(query_embadding) #打印向量
# print(len(query_embadding)) # 打印向量的维度


#余弦相似度例子
text = "我喜欢吃苹果"
text2 = "我喜欢吃的水果是苹果"
text3 = "今天天气不错"

vector1 = np.array(embedding.embed_query(text)).reshape(1, -1)
vector2 = np.array(embedding.embed_query(text2)).reshape(1, -1)
vector3 = np.array(embedding.embed_query(text3)).reshape(1, -1)

s12  = cosine_similarity(vector1, vector2)[0][0]  # 计算text和text2的余弦相似度
s13  = cosine_similarity(vector1, vector3)[0][0]  # 计算text和text3的余弦相似度

print(f"'{text}' 和 '{text2}' 的余弦相似度: {s12}")
print(f"'{text}' 和 '{text3}' 的余弦相似度: {s13}")