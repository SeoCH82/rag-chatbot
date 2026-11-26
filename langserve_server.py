"""PPT pp. 484, 491: desktop-only /llm service, exposed through ngrok."""
from __future__ import annotations

import hmac
import json
import os
import threading
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langserve import add_routes
from pydantic import BaseModel, ConfigDict, Field, field_validator
from prompt_budget import CONTEXT_TOKENS, OUTPUT_TOKENS, PROMPT_BUDGET, validate_prompt

ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://127.0.0.1:11434"


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=PROMPT_BUDGET)

    @field_validator("prompt")
    @classmethod
    def bounded_prompt(cls, value: str) -> str:
        return validate_prompt(value)


def assert_gpu_loaded(model: str) -> None:
    with urlopen(f"{OLLAMA_URL}/api/ps", timeout=10) as response:
        running = json.load(response).get("models", [])
    names = {model, model if ":" in model else f"{model}:latest"}
    for item in running:
        if item.get("name") in names or item.get("model") in names:
            if item.get("size_vram", 0) > 0:
                return
            break
    raise RuntimeError("The local model is not using GPU memory.")


def read_api_key() -> str:
    key = os.environ.get("RAG_BACKEND_API_KEY", "")
    if not key:
        path = ROOT / ".server_secrets" / "backend.json"
        if path.is_file():
            key = json.loads(path.read_text(encoding="utf-8"))["api_key"]
    if len(key) < 32:
        raise RuntimeError("Run the backend setup to create its private access key.")
    return key


def create_app(*, api_key: str, model: str = "Llama3_unslot_Q8", llm=None,
               gpu_check=None) -> FastAPI:
    if len(api_key) < 32:
        raise ValueError("A private backend access key of at least 32 characters is required.")
    # This backend never selects a cloud model or a remote Ollama endpoint.
    local_llm = llm if llm is not None else ChatOllama(
        model=model, base_url=OLLAMA_URL, temperature=0.1, num_ctx=CONTEXT_TOKENS,
        num_predict=OUTPUT_TOKENS, repeat_penalty=1.1, keep_alive="5m",
    )
    verify_gpu = gpu_check if gpu_check is not None else lambda: assert_gpu_loaded(model)
    slot = threading.Lock()
    server = FastAPI(title="Local GPU LangServe", docs_url=None,
                     redoc_url=None, openapi_url=None)

    @server.middleware("http")
    async def authenticate(request, call_next):
        if request.url.path != "/health":
            expected = f"Bearer {api_key}".encode("utf-8")
            provided = request.headers.get("authorization", "").encode("utf-8")
            if not hmac.compare_digest(provided, expected):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @server.get("/health")
    def health():
        return {"status": "ok", "service": "local-gpu-langserve"}

    def generate(data: dict) -> str:
        # No client-configurable model, file paths, or Ollama URLs are accepted.
        prompt = PromptInput.model_validate(data).prompt
        if not slot.acquire(blocking=False):
            raise HTTPException(503, "The GPU is busy. Try again shortly.")
        try:
            result = local_llm.invoke(prompt)
            # If the runner falls back to CPU, do not report a GPU success.
            verify_gpu()
            return str(result.content)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, "Local GPU inference is unavailable.") from None
        finally:
            slot.release()

    runnable = RunnableLambda(generate).with_types(input_type=PromptInput, output_type=str)
    add_routes(server, runnable, path="/llm", enabled_endpoints=["invoke"], config_keys=[])
    return server


if __name__ == "__main__":
    import uvicorn
    config = json.loads((ROOT / "deployment_config.json").read_text(encoding="utf-8"))
    # ngrok forwards to loopback; only this backend, never Ollama, is tunneled.
    uvicorn.run(create_app(api_key=read_api_key(), model=config["model"]),
                host="127.0.0.1", port=int(config["backend_port"]), access_log=False)
