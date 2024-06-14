from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_pdf(pdf_path, chunk_size=100, chunk_overlap=5):
    loader = PyPDFLoader(pdf_path)
    text_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(text_data)

    return all_splits


def chunk_text(text_path, chunk_size=100, chunk_overlap=5):
    loader = TextLoader(text_path)
    text_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(text_data)

    return all_splits


def get_embeddings(model="llama2"):
    ollama_emb = OllamaEmbeddings(
        base_url="http://127.0.0.1:11434",
        model=model,
    )

    return ollama_emb


if __name__ == "__main__":
    prompt = 'They sky is blue because of rayleigh scattering'
    emb = get_embeddings(prompt)

    print(emb)
