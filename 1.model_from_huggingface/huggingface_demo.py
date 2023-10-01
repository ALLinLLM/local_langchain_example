
import torch
from transformers import AutoTokenizer, AutoModel

llm_name = "THUDM/chatglm-6b"
# 先把llm模型下载到本地某个目录
checkpoint = "/share/public/huggingface_cache/" +llm_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code = True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code = True).half().cuda().eval()

query = "I love this so much!"

outputs = model.chat(tokenizer, query)
print(outputs)