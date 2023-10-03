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

# 和大模型聊天
from transformers import AutoTokenizer, AutoModel

llm_name = "THUDM/chatglm-6b"
# 先把llm模型下载到本地某个目录
checkpoint = "/share/public/huggingface_cache/" + llm_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(checkpoint, trust_remote_code=True).half().cuda().eval()
)

most_simliar_doc = match_result[0][1]

prompt_query = f"请根据content标签的内容，回答用户的提问query。\n<content>{most_simliar_doc.page_content}</content>\n<query>{user_query}</query>"
print()
print(">>> 原始输入:", user_query)
answer, history = model.chat(tokenizer, user_query)
print(">>> 大模型的回复：", answer)
print("=" * 20)
print(">>> 拼接后的输入:", prompt_query)
answer, history = model.chat(tokenizer, prompt_query)
print(">>> 大模型的回复：", answer)
