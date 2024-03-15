from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import numpy as np
import sqlite3
from embedding import get_embeddings


class MilvusHelper:
    def __init__(self, host="localhost", port="19530", collection_name="user_health"):
        self.collection_name = collection_name
        self.db_path = "./sqlite/sqlite.db"
        self.init(host, port)
        self.create_collection()

    def init(self, host, port):
        connections.connect("default", host=host, port=port)
        print(f"Connected to Milvus at {host}:{port}")

    def create_collection(self):
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
        else:
            fields = [
                FieldSchema(name="user_id", dtype=DataType.INT64,
                            is_primary=True),
                FieldSchema(name="text_vector",
                            dtype=DataType.FLOAT_VECTOR, dim=4096)
            ]
            schema = CollectionSchema(
                fields, description="User Health Questions")

            collection = Collection(name=self.collection_name, schema=schema)
            collection.flush()

            # if no index, collect cant be loaded
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }

            collection.create_index(
                field_name="text_vector",
                index_params=index_params
            )

            utility.index_building_progress(self.collection_name)

            collection.release()

            print(f"Collection '{self.collection_name}' created.")

    def insert_vector_to_milvus(self, user_id, vector, collection_name="user_health"):
        collection = Collection(name=collection_name)
        mr = collection.insert([[user_id], [vector]])
        print(
            f"Inserted vector for user {user_id}, Milvus IDs: {mr.primary_keys}")
        collection.release()

        return mr.primary_keys

    def store_mapping_in_db(self, user_id, vector_ids, text):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for vector_id in vector_ids:
            cursor.execute(
                "SELECT vector_id FROM vector_mapping WHERE vector_id = ?", (vector_id,))
            result = cursor.fetchone()

            if result is None:
                cursor.execute("INSERT INTO vector_mapping (vector_id, user_id, original_text) VALUES (?, ?, ?)",
                               (vector_id, user_id, text))
            else:
                print(
                    f"Vector ID {vector_id} already exists in the database, skipping insert.")

        conn.commit()
        conn.close()
        print(
            f"Stored mapping in DB for user ID {user_id} with vector IDs {vector_ids}.")

    def search_similar_vectors_in_milvus(self, query_vector, top_k=10, collection_name="user_health"):
        collection = Collection(name=collection_name)
        collection.load()
        search_params = {"metric_type": "L2", "params": {"nlist": 1024}}
        results = collection.search(
            [query_vector], "text_vector", search_params, top_k, "user_id > 0")
        ids = [hit.id for result in results for hit in result]
        collection.release()

        print(f"Found similar vector IDs: {ids}")
        return ids

    def find_original_texts_by_vector_ids(self, user_id, vector_ids):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query_placeholders = ','.join('?' for _ in vector_ids)
        query = f"SELECT original_text FROM vector_mapping WHERE user_id = ? AND vector_id IN ({query_placeholders})"
        params = [user_id, *vector_ids]

        cursor.execute(query, params)

        original_texts = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return original_texts


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


if __name__ == "__main__":
    milvus_helper = MilvusHelper()

    user_id = 1
    question_text = "我每天晚上都睡不好，應該怎麼辦？"

    store_vector(user_id, question_text)

    user_id = 1
    query_text = "睡不好不要滑手機"

    vector_response(user_id, query_text)
