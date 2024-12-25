from pymilvus import MilvusClient

client = MilvusClient(uri="./milvus.db")
client.ge
print(client.list_collections())
print(client.describe_collection("colpali"))
print(client.query(collection_name='colpali', limit = 2))
# res = client.query(collection_name="colpali")
# print(res)