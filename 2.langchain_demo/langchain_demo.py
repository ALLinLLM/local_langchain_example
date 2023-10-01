from langchain.llms import ChatGLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_api_host = "127.0.0.1"
llm_api_port = "10000"
# url for a local deployed ChatGLM api server
endpoint_url = f"http://{llm_api_host}:{llm_api_port}"

# direct access endpoint in a proxied environment
# os.environ['NO_PROXY'] = '127.0.0.1'

llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
    top_p=0.9,
    model_kwargs={"sample_model_args": False},
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "北京和上海两座城市有什么不同？"

answer = llm_chain.run(question)
print(answer)
