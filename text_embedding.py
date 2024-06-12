from embedding import chunk_text, get_embeddings
from langchain_community.vectorstores.milvus import Milvus

file_path = 'data/test.txt'
all_splits = chunk_text(file_path)
vector_store = Milvus.from_documents(
    documents=all_splits, embedding=get_embeddings())
