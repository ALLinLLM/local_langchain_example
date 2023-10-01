# local_langchain_demo
学习如何本地部署langchain大模型


硬件要求：
1. 显卡 ≥ 3090，目前实测 3090 和 4090 可以正常运行ChatGLM-6B


---

## 1. model from huggingface
学习目标：从huggingface上下载清华大学开源的大模型ChatGLM-6B，并使用huggingface的transformers三方库，在本地运行
操作步骤：
1. 自行寻找可以访问huggingface的网络工具，然后使用git clone下载ChatGLM-6B在huggingface网站上的仓库：https://huggingface.co/THUDM/chatglm2-6b。（注意使用ssh协议，http协议容易断）。
2. 根据下载的位置，修改`1.model_from_huggingface/huggingface_demo.py` 文件第7行的具体路径
3. 依次执行下面代码
```python
cd 1.model_from_huggingface
pip install -r requirements.txt
python huggingface_demo.py
```

## 2. langchain demo
学习目标：以api服务的方式运行chatglm-6B，然后使用langchain框架，调用本地的api，实现大模型对话功能
操作步骤：
1. 先开启一个终端，启动api服务
```python
cd 2.langchain_demo
pip install -r requirements.txt
python llm_api_server.py
```
2. 再开启另一个终端，
```python
cd 2.langchain_demo
python langchain_demo.py
```
