from milvus_db import MilvusHelper
from embedding import get_embeddings


def store_vector(user_id, question_text):
    question_vector = get_embeddings(question_text)

    vector_ids = milvus_helper.insert_vector_to_milvus(
        user_id, question_vector)

    milvus_helper.store_mapping_in_db(user_id, vector_ids, question_text)


def vector_response(user_id, query_text):
    query_vector = get_embeddings(query_text)

    similar_vector_ids = milvus_helper.search_similar_vectors_in_milvus(
        query_vector)

    similar_questions = milvus_helper.find_original_texts_by_vector_ids(
        user_id, similar_vector_ids)

    print("找到的相關問題：")
    for question in similar_questions:
        print(question)


milvus_helper = MilvusHelper()

user_id = 1
question_text = "我每天晚上都睡不好，應該怎麼辦？"

store_vector(user_id, question_text)

user_id = 1
query_text = "睡不好"

vector_response(user_id, query_text)
