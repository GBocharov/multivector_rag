import io
import httpx
from PIL.Image import Image


get_device_url = f'http://col-qwen-llm:8001/llm_router/get_device_llm_router_llm_info_post'
image_embeddings_url = f'http://col-qwen-llm:8001/llm_router/get_image_embeddings'
text_embeddings_url = 'http://col-qwen-llm:8001/llm_router/get_text_embeddings?text={text}' # сделать шаблон?

#TODO adapter

async def get_device():
    url = "http://col-qwen-llm:8001/llm_router/get_device_llm_router_llm_info_post"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()  # Проверка на успешный ответ
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP Error: {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Request Error: {str(e)}"}

async def image_embeddings(image: Image, filename: str = 'image'):
    url = "http://col-qwen-llm:8001/llm_router/get_image_embeddings" #"http://localhost:8001/llm_router/get_image_embeddings"
    async with httpx.AsyncClient() as client:
        try:

            buffer = io.BytesIO()
            image.save(buffer, format='PNG')  # Можно указать другой формат, если нужно
            buffer.seek(0)  # Возврат в начало буфера

            files = {
                "file": (filename, buffer)
            }
            response = await client.post(url, files=files)
            response.raise_for_status()  # Проверка на успешный ответ
            return response
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP Error: {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Request Error: {str(e)}"}

async def text_embeddings(text: str = 'test'):
    url = f"http://col-qwen-llm:8001/llm_router/get_text_embeddings?text={text}"
    async with httpx.AsyncClient() as client:
        try:
            #data = {"text": text}
            response = await client.post(url)
            response.raise_for_status()  # Проверка на успешный ответ
            return response
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP Error: {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Request Error: {str(e)}"}

#
#
# async def image_chat(image: UploadFile, query:str = 'test'):
#     request_object_content = await image.read()
#     img = Image.open(io.BytesIO(request_object_content))
#
#     res = image_query(img, query)
#     return res