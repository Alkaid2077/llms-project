from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 解析PDF，切成chunk片段
pdf_loader = PyPDFLoader('Google Prompt Guidance.pdf', extract_images=True)   # 调用OCR解析pdf中图片里面的文字

# 切片大小和重叠大小
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10

chunks = pdf_loader.load_and_split(
	text_splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
)   # 切成chunk片段，每 CHUNK_SIZE 个字符切一段，相邻段之间有 CHUNK_OVERLAP 个字符重叠

# 2. 加载embedding模型（直接调用该模型），用于将chunk向量化
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')   #加载embedding模型，此处用的是中文句向量模型

# 3. 将chunk插入到faiss本地向量数据库，faiss为向量索引工具，提供高效的向量索引和搜索功能
vector_db = FAISS.from_documents(chunks,embeddings)
vector_db.save_local('faiss_index/LLM')   #保存faiss本地向量数据库到指定的相对路径，即本代码所在的文件夹中，如不存在则会自动创建

print('faiss saved!')   #帮助确认代码运行到此处