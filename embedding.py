from langchain_community.embeddings import OllamaEmbeddings


def get_embeddings(prompt, model="llama2"):
    ollama_emb = OllamaEmbeddings(
        base_url="http://127.0.0.1:11434",
        model=model,
    )

    r1 = ollama_emb.embed_query(
        prompt
    )

    return r1


if __name__ == "__main__":
    prompt = 'They sky is blue because of rayleigh scattering'
    emb = get_embeddings(prompt)

    print(emb)
