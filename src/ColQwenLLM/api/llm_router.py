import io
import json
import pickle

from PIL import Image
from fastapi import APIRouter, UploadFile
from starlette.responses import StreamingResponse

from src.ColQwenLLM.ColQwen2ForRAG import device
from src.ColQwenLLM.processor import image_query, get_image_embeddings, get_text_embeddings

llm_router = APIRouter(
    prefix="/llm_router",
    tags=["LLM"],
)

@llm_router.post(
    "/llm_info"
)
async def get_device():
    return device


@llm_router.post(
    "/get_image_embeddings"
)
async def image_embeddings(file: UploadFile):

    request_object_content = await file.read()

    img = Image.open(io.BytesIO(request_object_content))
    res = get_image_embeddings([img])

    byte_data = pickle.dumps(res)

    return StreamingResponse(io.BytesIO(byte_data), media_type="application/octet-stream")


@llm_router.post(
    "/get_text_embeddings"
)
async def text_embeddings(text: str = 'test'):

    res = get_text_embeddings([text])

    byte_data = pickle.dumps(res)

    return StreamingResponse(io.BytesIO(byte_data), media_type="application/octet-stream")


@llm_router.post(
    "/chat"
)
async def image_chat(image: UploadFile, query:str = 'test'):
    request_object_content = await image.read()
    img = Image.open(io.BytesIO(request_object_content))

    res = image_query(img, query)
    return res