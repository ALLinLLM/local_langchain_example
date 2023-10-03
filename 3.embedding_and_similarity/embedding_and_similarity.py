from langchain.embeddings import HuggingFaceEmbeddings

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
text1 = "我爱吃牛排"
text2 = "我爱吃猪排"
text3 = "我骑自行车"


import numpy as np

vector1 = hf.embed_query(text1)
vector2 = hf.embed_query(text2)
vector3 = hf.embed_query(text3)

print(text1, vector1[:5])
print(text2, vector2[:5])
print(text3, vector3[:5])

vector1 = np.array(vector1).reshape(1, -1)
vector2 = np.array(vector2).reshape(1, -1)
vector3 = np.array(vector3).reshape(1, -1)


from sklearn.metrics.pairwise import cosine_similarity

cos1_2 = cosine_similarity(vector1, vector2)
cos1_3 = cosine_similarity(vector1, vector3)
cos2_3 = cosine_similarity(vector2, vector3)

print(text1, text2, cos1_2)
print(text1, text3, cos1_3)
print(text2, text3, cos2_3)
