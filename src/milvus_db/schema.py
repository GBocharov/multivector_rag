from pymilvus import DataType, FieldSchema, CollectionSchema


pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True)
vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
seq_id = FieldSchema(name="seq_id", dtype=DataType.INT16)
doc_id = FieldSchema(name="doc_id", dtype=DataType.INT64)
doc = FieldSchema(name="doc", dtype=DataType.VARCHAR,  max_length=65535)

data_schema = CollectionSchema(fields=[pk, vector, seq_id, doc_id, doc], auto_id=True, enable_dynamic_field=True, description="desc of a collection")