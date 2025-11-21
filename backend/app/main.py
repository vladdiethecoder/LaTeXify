from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.events import router as events_router

app = FastAPI(title="LaTeXify Streaming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events_router, prefix="/v1/events")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
