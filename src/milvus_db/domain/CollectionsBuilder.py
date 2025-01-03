import numpy as np
from typing import Any
from abc import ABC, abstractmethod

import concurrent.futures

from milvus_db.infrastructure.collection_configs.base_ColQwen_config import default_collection_config as config


def rerank_single_doc(doc_id, data, _client, collection_name, cnfg):
    # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.

    doc_colbert_vecs = _client.query(
        collection_name=collection_name,
        filter=f"doc_id in [{doc_id}]",
        output_fields=cnfg.rerank_params.output_fields,
        limit=cnfg.rerank_params.limit
    )
    doc_vecs = np.vstack(
        [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
    )
    score = np.dot(data, doc_vecs.T).max(1).sum()
    return score, doc_id, doc_colbert_vecs[0]['doc']


class Collection(ABC):

    @classmethod
    @abstractmethod
    def insert(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def search(cls, *args, **kwargs):
        pass


class ColQwenCollection(Collection):

    @classmethod
    def insert(cls, client, collection_name, data: Any):
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


        response = client.insert(collection_name, vectors_with_metadata)

        return response

    @classmethod
    def search(cls, client, collection_name: str, data, topk=5):
        # Perform a vector search on the collection to find the top-k most similar documents.

        results = client.search(
            collection_name,
            data,
            search_params = config.search_params.search_params,
            output_fields = config.search_params.output_fields
        )
        doc_ids = {results[r_id][r]["entity"]["doc_id"] for r_id in range(len(results)) for r in range(len(results[r_id]))}

        scores = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, client, collection_name, config
                ): doc_id
                for doc_id in doc_ids
            }

            for future in concurrent.futures.as_completed(futures):
                score, doc_id, doc = future.result()
                scores.append((float(score), doc_id, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        return scores[:topk]

    @classmethod
    def _create_collection(cls, client, collection_name: str):

        index_params = client.prepare_index_params()
        index_params.add_index(**config.vector_index_params.__dict__)
        index_params.add_index(**config.scalar_index_params.__dict__)

        client.create_collection(
            collection_name=collection_name, schema=config.schema, index_params=index_params
        )

    @classmethod
    def clear(cls, client, collection_name: str):
        client.drop_collection(collection_name)
        ColQwenCollection._create_collection(client, collection_name)

