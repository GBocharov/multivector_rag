from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pymilvus import MilvusClient
import milvus_db.infrastructure.config as milvus_config
from milvus_db.infrastructure.collection_configs.base_ColQwen_config import default_collection_config

import concurrent.futures

client = MilvusClient(milvus_config.milvus_db_save_dir + '/' +"milvus_demo.db")


def rerank_single_doc(doc_id, data, _client, collection_name, config):
    # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
    doc_colbert_vecs = _client.query(
        collection_name=collection_name,
        filter=f"doc_id in [{doc_id}]",
        output_fields=config.rerank_params.output_fields,
        limit=config.rerank_params.limit
    )
    doc_vecs = np.vstack(
        [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
    )
    score = np.dot(data, doc_vecs.T).max(1).sum()
    return score, doc_id, doc_colbert_vecs[0]['doc']


class Collection(ABC):

    @abstractmethod
    def insert(self, data: Any):
        pass

    @abstractmethod
    def search(self, data: Any) -> bool:
        pass


class ColQwenCollection(Collection):
    _client = client
    def __init__(self, collection_name:str, config = default_collection_config):
        self.collection_name = collection_name
        self.config = config
        if self._client.has_collection(collection_name=self.collection_name):
            self._client.load_collection(collection_name)
        else:
            self._create_collection()

    def insert(self, data: Any):
        colbert_vecs = data["colbert_vecs"]
        seq_length = len(colbert_vecs)

        doc_id = data["doc_id"]
        filepath = data["filepath"]

        vectors_with_metadata = [
            {
                "vector": colbert_vecs[i],
                "seq_id": i,
                "doc_id": doc_id,
                "doc": filepath,
            }
            for i in range(seq_length)
        ]

        return self._client.insert(self.collection_name, vectors_with_metadata)

    def search(self, data, topk=5):
        # Perform a vector search on the collection to find the top-k most similar documents.
        results = self._client.search(
            self.collection_name,
            data,
            search_params=self.config.search_params.search_params,
            output_fields=self.config.search_params.output_fields
        )
        doc_ids = {results[r_id][r]["entity"]["doc_id"] for r_id in range(len(results)) for r in range(len(results[r_id]))}

        scores = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self._client, self.collection_name, self.config
                ): doc_id
                for doc_id in doc_ids
            }

            for future in concurrent.futures.as_completed(futures):
                score, doc_id, doc = future.result()
                scores.append((float(score), doc_id, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        return scores[:topk]

    def _create_collection(self):
        index_params = client.prepare_index_params()

        index_params.add_index(**self.config.vector_index_params.__dict__)
        index_params.add_index(**self.config.scalar_index_params.__dict__)

        self._client.create_collection(
            collection_name=self.collection_name, schema=self.config.schema
        )

    def clear(self):
        self._client.describe_collection(self.collection_name)
        self._create_collection()

