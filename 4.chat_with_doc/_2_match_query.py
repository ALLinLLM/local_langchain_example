import numpy as np

from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from _1_doc_preprocess import preprocess

local_dir = "/share/public/huggingface_cache/"
model_name = "BAAI/bge-large-zh"
model_kwargs = {"device": "cuda:0"}
encode_kwargs = {"normalize_embeddings": False}
print("loading embedding model...")
hf = HuggingFaceEmbeddings(
    model_name=local_dir + model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
print("embedding model loaded")

user_query = "今年上半年畜牧业发展的怎么样"

query_vector = hf.embed_query(user_query)
query_vector = np.array(query_vector).reshape(1, -1)

docs = preprocess("doc_example.txt")


match_result = []
for doc in docs:
    title = doc.metadata["chap_title"]
    title_vector = hf.embed_query(title)
    title_vector = np.array(title_vector).reshape(1, -1)
    cos_sim = cosine_similarity(query_vector, title_vector)
    cos_sim = round(cos_sim[0][0], 3)

    match_result.append((cos_sim, doc))

# 按照相似度逆序
match_result.sort(key=lambda x: x[0], reverse=True)

# 输出
print(">>> 用户提问：", user_query)
print(">>> 相似度 标题")
for cos_sim, doc in match_result:
    title = doc.metadata["chap_title"]
    chap = doc.metadata["chap"]
    print(cos_sim, chap, title)
