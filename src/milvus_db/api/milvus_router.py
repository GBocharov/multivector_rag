import io
from typing import List

import PIL.Image
#from asyncpg import Connection
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse

import milvus_db.processor as pr
from document_utils.save_to_dir import save_image_to_dir, save_pdf_to_dir_as_images

milvus_router = APIRouter(
    prefix="/milvus_router",
    tags=["milvus"],
)

save_dir = r'/opt/app-root/temp_data/image_data'


@milvus_router.get(
    "/db_info",
)
async def get_db_info(
):
    res =  pr.get_db_info()
    return res


@milvus_router.get(
    "/get_collection_info",
    response_model_exclude_none=True,
)
async def get_collection_info(
collection_name:str = 'test'
):
    res =  pr.get_collection_info(collection_name)

    return res

@milvus_router.get(
    "/clear_collection"
)
async def clear_collection(
collection_name:str = 'test'
):

    return pr.drop_collection(collection_name)




@milvus_router.post(
    "/insert_image",# Set what the media type will be in the autogenerated OpenAPI specification.
    # fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
)
async def insert_image(
files: List[UploadFile]
):
    upload_path = save_dir
    paths = []
    images = []
    for im in files:
        request_object_content = await im.read()
        img = PIL.Image.open(io.BytesIO(request_object_content))

        path = save_image_to_dir(img, upload_path, im.filename)
        paths += path
        images += [img]

    await pr.insert_Images(images, paths)

    return paths

@milvus_router.post(
    "/insert_pdf",
)
async def insert_pdf(
file: UploadFile
):
    upload_path = save_dir
    #request_object_content = await file.read()
    #file = Image.open(io.BytesIO(request_object_content))
    file_path = f"{upload_path}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    imgs, paths = save_pdf_to_dir_as_images(file_path, upload_path, file.filename)
    await pr.insert_Images(imgs , paths)
    return paths

@milvus_router.post(
    "/text_search"
)
async def text_search(
text: str = 'fun pic'
):
    results = await pr.search_Texts(query = text)
    if not results:
        return 'empty collection has been provided'
    print(results)
    return FileResponse(results[0][0][2])