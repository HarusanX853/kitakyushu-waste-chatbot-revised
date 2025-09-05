# backend/services/llm_service.py
import os
import time
from typing import Generator, List, Dict, Any
import ollama

DEFAULT_LLM = os.getenv("LLM_MODEL", "llama3.1-swallow")   # 你在 ollama 本地的模型名
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 可按需：os.environ["OLLAMA_HOST"] = OLLAMA_HOST

SYSTEM_JA = (
    "あなたは自治体のごみ分別案内ボットです。"
    "常に日本語で、提供された情報の範囲内で簡潔かつ正確に回答してください。"
    "不明な場合は『申し訳ございませんが、該当する情報が見つかりません。』と答えてください。"
)

class LLMService:
    def __init__(self, model: str = None):
        self.model = model or DEFAULT_LLM

    def generate_blocking(self, prompt: str, temperature: float = 0.1, num_ctx: int = 4096) -> Dict[str, Any]:
        start = time.time()
        res = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_JA},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature, "num_ctx": num_ctx},
            stream=False,
        )
        latency_ms = int((time.time() - start) * 1000)
        content = res["message"]["content"]
        usage = {k: res.get(k) for k in ("prompt_eval_count", "eval_count", "eval_duration", "prompt_eval_duration")}
        return {"reply": content, "latency_ms": latency_ms, "usage": usage}

    def generate_stream(self, prompt: str, temperature: float = 0.1, num_ctx: int = 4096) -> Generator[str, None, None]:
        yield f'data: {{"type":"start"}}\n\n'
        start = time.time()
        first_token_ms = None
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_JA},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature, "num_ctx": num_ctx},
            stream=True,
        )
        try:
            for chunk in stream:
                if first_token_ms is None:
                    first_token_ms = int((time.time() - start) * 1000)
                    yield f'data: {{"type":"first_token","latency_ms":{first_token_ms}}}\n\n'
                delta = chunk["message"]["content"]
                if delta:
                    yield f'data: {{"type":"chunk","content":{delta!r}}}\n\n'
        finally:
            total_ms = int((time.time() - start) * 1000)
            yield f'data: {{"type":"end","latency_ms":{total_ms}}}\n\n'
            yield "data: [DONE]\n\n"
