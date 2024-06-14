from langchain import hub
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_milvus import Milvus
from embedding import get_embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

prompt = hub.pull("rlm/rag-prompt")
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={'score_threshold': 0.9, 'k': 3}
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = rag_chain.invoke(query)

print(result)
