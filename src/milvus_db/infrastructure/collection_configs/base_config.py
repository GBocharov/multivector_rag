from dataclasses import dataclass, field
from typing import Dict, List

from pymilvus import DataType, FieldSchema, CollectionSchema


pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True)
vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
seq_id = FieldSchema(name="seq_id", dtype=DataType.INT16)
doc_id = FieldSchema(name="doc_id", dtype=DataType.INT64)
doc = FieldSchema(name="doc", dtype=DataType.VARCHAR,  max_length=65535)

data_schema = CollectionSchema(fields=[pk, vector, seq_id, doc_id, doc], auto_id=True, enable_dynamic_field=True, description="desc of a collection")

@dataclass
class VectorIndexParams:
    index_name: str = "vector_index"
    field_name: str = "vector"
    index_type: str = "HNSW"
    metric_type: str = "IP"
    params: Dict[str, int] = field(default_factory=lambda: {"M": 16, "efConstruction": 500})

@dataclass
class ScalarIndexParams:
    index_name: str = "int32_index"
    field_name: str = "doc_id"
    index_type: str = "INVERTED"

@dataclass
class SearchParams:
    limit: int = 50
    search_params:Dict = field(default_factory=lambda:{"metric_type": "IP", "params": {}})
    output_fields: List[str] = field(default_factory=lambda:["vector", "seq_id", "doc_id", "doc"])


@dataclass
class RerankParams:
    limit: int = 1000
    output_fields: List[str] = field(default_factory=lambda:["seq_id", "vector", "doc"])


@dataclass
class MilvusConfig:
    dim: int = 128
    schema: CollectionSchema = data_schema
    vector_index_params: VectorIndexParams = field(default_factory=VectorIndexParams)
    scalar_index_params: ScalarIndexParams = field(default_factory=ScalarIndexParams)
    search_params: SearchParams = field(default_factory=SearchParams)
    rerank_params: RerankParams = field(default_factory=RerankParams)

default_collection_config = MilvusConfig()