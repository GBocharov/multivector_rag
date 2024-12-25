import pickle
from typing import List

import PIL.Image
from fastapi import UploadFile

from src.milvus_db.MilvusColbertCollection import MilvusColbertCollection, client
from src.milvus_db.external.llm_response import image_embeddings, text_embeddings

test_retriever = MilvusColbertCollection(collection_name="test", milvus_client=client)

collections = {
    'test' : test_retriever
}


def get_db_info():
    collections_names = client.list_collections()
    return collections_names

def get_collection_info(collection_name:str):
    if collection_name not in client.list_collections():
        return 'no such collection'
    info = client.get_collection_stats(collection_name)

    info = client.query(collection_name=collection_name, filter="doc_id like \"red%\"", output_fields=['vector', "doc_id"])


    print(info)
    return  info

def drop_db():
    client.delete()

def drop_collection(collection_name:str):
    if collection_name not in client.list_collections():
        return 'no such collection'
    client.drop_collection(collection_name=collection_name)
    collections[collection_name] = MilvusColbertCollection(collection_name="test", milvus_client=client)
    return f'collection {collection_name} successfully deleted'

async def insert_Images(images: List[PIL.Image.Image], names:List[str] = None, collection_name:str = 'test'):
    retriever = collections[collection_name]

    for i in range(len(images)):
        response = await image_embeddings(images[i])
        embedding = pickle.loads(response.content) #embedding = await image_embeddings(images[i])
        print(embedding)
        data = {
            "colbert_vecs": embedding[0],
            "doc_id": i,
            "filepath": names[i] if names else '',
        }
        retriever.insert(data)

# def insert_Texts(texts: List[str], names:List[str] = None, collection_name:str = 'test' ):
#     retriever = collections[collection_name]
#     embeddings = get_text_embeddings(texts)
#
#     for i in range(len(texts)):
#         data = {
#             "colbert_vecs": embeddings[i].float().numpy(),
#             "doc_id": i,
#             "filepath": names[i] if names else '',
#         }
#         retriever.insert(data)
#
# def search_Images(collection_name:str, data: List[PIL.Image.Image]):
#     pass
#
async def search_Texts(query: str, collection_name:str = 'test'):
    retriever = collections[collection_name]
    results = []

    response = await text_embeddings(query)
    query = pickle.loads(response.content)[0]

    result = retriever.search(query, topk=5)
    results.append(result)
    import pprint as pp
    print('------>')
    pp.pprint(results)
    pp.pprint(type(results))

    return results

# def generate(collection_name:str, query: str, image: PIL.Image.Image):
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                 },
#                 {
#                     "type": "text",
#                     "text": f"Перед тобой смешная картинка с текстом: {query}",
#                 },
#             ],
#         }
#     ]
#
#     image = scale_image(image)
#     text_prompt = processor_generation.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs_generation = processor_generation(
#         text=[text_prompt],
#         images=[image],
#         padding=True,
#         return_tensors="pt",
#     ).to(device)
#
#     # Generate the RAG response
#     model.enable_generation()
#     output_ids = model.generate(**inputs_generation, max_new_tokens=128)
#
#     # Ensure that only the newly generated token IDs are retained from output_ids
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
#                      zip(inputs_generation.input_ids, output_ids)]
#
#     # Decode the RAG response
#     output_text = processor_generation.batch_decode(
#         generated_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )
#     model.enable_retrieval()
#     return output_text



# images = [scale_image(Image.open("../pages/" + name)) for name in os.listdir("../pages")]
#
# paths = [name for name in os.listdir("../pages")]
# embeddings = get_image_embeddings(images)
#
# print(paths)
# for i in range(len(images)):
#     data = {
#         "colbert_vecs": embeddings[i].float().numpy(),
#         "doc_id": i,
#         "filepath": paths[i],
#     }
#     retriever.insert(data)
#
# q = [
#     'Задания по математике',
#     'отрицание в математике',
#     'мужик и бритва',
#
# ]
#
# q_e = get_text_embeddings(q)
#
# for query in q_e:
#     query = query.float().numpy()
#     result = retriever.search(query, topk=3)
#     print(result)
