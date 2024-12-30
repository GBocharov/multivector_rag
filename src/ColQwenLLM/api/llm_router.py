import io
import pickle
import logging.config

from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException
from starlette.responses import StreamingResponse

from ColQwenLLM.domain.ColQwen2ForRAG import device
from ColQwenLLM.domain.processor import image_query, get_image_embeddings, get_text_embeddings
from ColQwenLLM.paths_config import logger_conf_path

logging.config.fileConfig(logger_conf_path)
logger = logging.getLogger('llmLogger')
llm_router = APIRouter()

@llm_router.post("/llm_info")
async def get_device():
    return device

@llm_router.post("/get_image_embeddings")
async def image_embeddings(file: UploadFile):
    try:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))
        res = get_image_embeddings([img])

        byte_data = pickle.dumps(res)
        return StreamingResponse(io.BytesIO(byte_data), media_type="application/octet-stream")
    except Exception as e:
        logger.error("An error occurred while processing image embeddings.", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))

@llm_router.post("/get_text_embeddings")
async def text_embeddings(text: str = 'test'):
    try:
        print(f'text = {text}')
        res = get_text_embeddings([text])

        byte_data = pickle.dumps(res)
        return StreamingResponse(io.BytesIO(byte_data), media_type="application/octet-stream")
    except Exception as e:
        logger.error("An error occurred while processing text embeddings.", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))

@llm_router.post("/chat")
async def image_chat(image: UploadFile, query: str = 'test'):
    try:
        request_object_content = await image.read()
        img = Image.open(io.BytesIO(request_object_content))

        res = image_query(img, query)
        return res
    except Exception as e:
        logger.error("An error occurred during the chat processing.", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))