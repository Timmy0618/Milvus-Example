from langchain.chains import RetrievalQA
from langchain import hub
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_milvus import Milvus
from embedding import chunk_text, get_embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

MILVUS_URI = "http://localhost:19530"

llm = Ollama(
    model="llama2",
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]
    ),
    stop=["<|eot_id|>"],
)

vector_db = Milvus(
    get_embeddings(),
    connection_args={"uri": MILVUS_URI},
    collection_name="LangChainCollection",
)

query = input("\nQuery: ")

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = vector_db.as_retriever()
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

result = retrieval_chain.invoke({"input": query})
print(result)
