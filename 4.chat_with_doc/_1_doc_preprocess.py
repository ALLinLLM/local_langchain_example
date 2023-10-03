import re
from langchain.schema import Document


def preprocess(file_path):
    DEFAULT_CHAP = "序言"  # 第一章之前是序言
    DEFAULT_CHAP_TITLE = ""

    # 文章的预处理和文章内容紧耦合，具体文章需要具体处理，这里的例子仅供参考
    with open(file_path, "r") as r:
        lines = r.readlines()
    last_chap = DEFAULT_CHAP
    last_chap_title = DEFAULT_CHAP_TITLE
    cache_content = ""  # 缓存的正文

    docs = []
    for line in lines:
        # 跳过空行
        if line.strip() == "":
            continue
        chap_title_regex_pattern = r"([一二三四五六七八九十])+、(.+)"
        chap, chap_title = DEFAULT_CHAP, DEFAULT_CHAP_TITLE
        # 用正则表达式判断是不是章节标题
        chapter_search_result = re.search(chap_title_regex_pattern, line)
        if chapter_search_result:
            # 当前行是章节标题, 把缓存的内容输出成一个doc
            chap, chap_title = chapter_search_result.groups()
            # 遇到下一个章节，
            # 按照 langchain 框架中的 Document 格式保存文档片段。
            doc = Document(
                page_content=cache_content,
                metadata={"chap": last_chap, "chap_title": last_chap_title},
            )
            docs.append(doc)
            cache_content = ""
            last_chap = chap
            last_chap_title = chap_title
        else:
            # 是正文，追加到缓存
            cache_content += line

    # 遍历结束，处理最后一章
    doc = Document(
        page_content=cache_content,
        metadata={"chap": last_chap, "chap_title": last_chap_title},
    )
    docs.append(doc)
    return docs


if __name__ == "__main__":
    file_path = "doc_example.txt"

    docs = preprocess(file_path)

    # 检查处理后的文章片段内容
    for doc in docs:
        print(
            ">>> 章节: ",
            doc.metadata["chap"],
            "\n>>> 小标题: ",
            doc.metadata["chap_title"],
            "\n>>> 正文: ",
            doc.page_content,
        )
