from langchain_openai import ChatOpenAI
import os

os.environ["DASHSCOPE_API_KEY"] = '替换为你的 API Key' # 替换为你的 API Key

chat = ChatOpenAI(
    model="qwen-plus",   # 可选 qwen-turbo / qwen-max
    api_key=os.environ["DASHSCOPE_API_KEY"], 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    #若为国际版则改为 https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    temperature=0
)

resp = chat.invoke("温度是什么？")
print(resp.content)
