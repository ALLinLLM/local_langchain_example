from langchain.schema import Document
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS, dependable_faiss_import

from _1_doc_preprocess import preprocess

# 和大模型聊天
from transformers import AutoTokenizer, AutoModel

def init_llm():
    llm_name = "THUDM/chatglm-6b"
    # 先把llm模型下载到本地某个目录
    checkpoint = "/share/public/huggingface_cache/" + llm_name
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(checkpoint, trust_remote_code=True).half().cuda().eval()
    )
    return model, tokenizer


def get_embeddings():
    from langchain.embeddings import HuggingFaceEmbeddings

    local_dir = "/share/public/huggingface_cache/"
    model_name = "BAAI/bge-large-zh"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False}
    print("loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=local_dir + model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def get_index_docs(title, text):
    """按照小标题，把文章分割成知识点。
    
    注意：这里的数据结构有所变化，类似“倒排索引”的概念，把文章标题和小标题作为搜索词，而正文放入metadata中，等后续匹配到以后带出正文。
    
    e.g.
    input: 
    一、农业生产形势稳定，畜牧业平稳增长
    上半年，农业（种植业）增加值同比...
    二、工业生产稳步恢复，装备制造业增长较快
    上半年，全国规模以上工业增加值同...

    output:
    {'上半年国民经济恢复向好：农业生产形势稳定，畜牧业平稳增长', metadata: {content:'上半年，农业（种植业）增加值同比...'}}
    {'上半年国民经济恢复向好：工业生产稳步恢复，装备制造业增长较快', metadata: {content:'上半年，农业（种植业）增加值同比...'}}
    """
    raw_docs = preprocess(text)
    index_docs = []
    for raw_doc in raw_docs:
        content_metadata = raw_doc.metadata
        content_metadata['content'] = raw_doc.page_content
        search_index = title + "：" + raw_doc.metadata["chap_title"]
        index_doc = Document(page_content=search_index, metadata=content_metadata)
        index_docs.append(index_doc)
    return index_docs

def create_vector_database(docs, embedding):
    """
    默认用欧式距离，但是nlp的词向量更主流的度量函数是cos相似度
    """
    from langchain.vectorstores.utils import DistanceStrategy
    faiss_vsdb = FAISS.from_documents(docs, embedding, distance_strategy=DistanceStrategy.JACCARD)
    faiss_vsdb.distance_strategy = DistanceStrategy.JACCARD
    return faiss_vsdb


def chat_with_llm(llm, tokenizer, prompt):
    answer, history = llm.chat(tokenizer, prompt)
    return answer

# 程序运行的入口
if __name__ == '__main__':
    import faiss
    # 如果安装的是faiss-cpu，注意观察有没有打开 AVX2指令集优化，对耗时影响较大
    print("FAISS build info:", faiss.get_compile_options(), "gpu support:", faiss.get_num_gpus())
    test_doc = "doc_example.txt"
    title = "上半年国民经济恢复向好"
    test_user_query = "上半年畜牧业增长情况"

    index_docs = get_index_docs(title, test_doc)
    embedding_function = get_embeddings()
    vector_database = create_vector_database(index_docs, embedding_function)

    search_result = vector_database.similarity_search_with_score(test_user_query)
    
    # 显示按相似度排序的结果
    print("用户提问：", test_user_query)
    print("-"*30)
    print("FAISS 知识库向量搜索的结果top k")
    for doc, score in search_result:
        print(f"相似度：{score:.2f} | 搜索词：", doc.page_content)
    print("-"*30)

    # 取回正文部分
    top1_result = search_result[0]
    top1_doc = top1_result[0]
    top1_content = top1_doc.metadata["content"]

    
    print(">>> 通过搜索词，从知识库中匹配到的内容：\n", top1_content)
    print(f">>> 出处：《{title}》 第", top1_doc.metadata["chap"], "章")

    # 拼接到大模型的输入prompt
    prompt = f"请根据content标签的内容，回答用户的提问query。\n<content>{top1_content}</content>\n<query>{test_user_query}</query>"
    llm, tokenizer = init_llm()
    llm_answer = chat_with_llm(llm, tokenizer, prompt)
    print(">>> 大模型回复：" + llm_answer)
    