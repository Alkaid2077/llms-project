from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from operator import itemgetter
import os

# 1. 加载 embedding 模型和向量库
try:
    # 加载embedding模型 （用于将query向量化）
    embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')
    #加载 FAISS 向量库 （用于知识召回）
    vector_db = FAISS.load_local('faiss_index/LLM', embeddings, allow_dangerous_deserialization=True) # 替换为你的向量库路径
        # allow_dangerous_deserialization=True 仅在本地加载向量库时使用，请勿在不可信环境下启用
# 添加异常处理
except Exception as e:
    print(f"模型或向量库加载失败: {str(e)}") 
    exit(1)

# 2. 创建检索器 Retriever
# 检索器将 query 转为向量并返回最相似文档片段
retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # 设置返回的相关相似对最高的模块数量为 5

# 3. 配置 API Key，初始化 Chat 模型
# 此处使用阿里云达观智能 DashScope 平台的模型接口
os.environ["DASHSCOPE_API_KEY"] = 'sk-xx' # 替换为你的API Key

chat = ChatOpenAI(
    model="qwen-plus",   # 按照需要更换模型名称
    api_key=os.environ["DASHSCOPE_API_KEY"], 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    #若为国际版则改为 https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    temperature=0
)

# 4. Prompt模板
system_prompt = SystemMessagePromptTemplate.from_template('你是一个有帮助的助手。') # 系统提示
user_prompt = HumanMessagePromptTemplate.from_template(''' 
只基于以下内容回答问题，不要使用你在训练中学到的知识:

{context}

问题: {query}
''') # 用户提示
full_chat_prompt = ChatPromptTemplate.from_messages( 
    [system_prompt, MessagesPlaceholder(variable_name="chat_history"), user_prompt]) # 构建完整的聊天提示模板

# 5. 构建对话链 Chat chain
chat_chain = {
                 "context": itemgetter("query") | retriever,
                 "query": itemgetter("query"),
                 "chat_history": itemgetter("chat_history"),
             } | full_chat_prompt | chat 

# 6. 开始对话
chat_history = [] # 初始化对话历史
while True:
    query = input("请输入问题（输入 exit 退出）：")
    if query.lower() == "exit":
        break 
    response = chat_chain.invoke({'query': query, 'chat_history': chat_history}) # 获取模型响应
    chat_history.extend((HumanMessage(content=query), response)) # 更新对话历史
    print("AI:", response.content)
    chat_history = chat_history[-20:]  # 保留最近 20 条消息