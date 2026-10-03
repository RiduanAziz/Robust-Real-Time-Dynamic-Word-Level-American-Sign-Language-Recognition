from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Sign Language Recognition API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, str]:
    return {"name": "sign-language-recognition", "status": "ready"}
