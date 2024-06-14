from embedding import chunk_pdf, get_embeddings
from langchain_community.vectorstores.milvus import Milvus

file_path = 'data/pdf/eink.pdf'
all_splits = chunk_pdf(file_path)
vector_store = Milvus.from_documents(
    documents=all_splits, embedding=get_embeddings())
