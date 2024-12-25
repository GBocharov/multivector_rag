from fastapi import FastAPI
from starlette.responses import RedirectResponse

from src.ColQwenLLM.api.llm_router import llm_router

app = FastAPI(
    debug=True,
    title='LLM api',
)

@app.get("/", include_in_schema=False)
async def redirect_from_root() -> RedirectResponse:
    return RedirectResponse(url='/docs')

app.include_router(llm_router)


#python -m uvicorn src.ColQwenLLM.api.main:app --host 0.0.0.0 --port 8001
#python -m uvicorn src.milvus_db.api.main:app --host 0.0.0.0 --port 8000