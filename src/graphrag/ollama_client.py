"""Thin wrapper around the Ollama HTTP API."""
import json
import logging
import time
from typing import Any

import httpx

from graphrag.config import OLLAMA_BASE_URL, LLM_TEMPERATURE, LLM_MAX_RETRIES

logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous client for the Ollama REST API."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 300.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text completion."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        resp = self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = LLM_TEMPERATURE,
    ) -> dict[str, Any]:
        """Generate and parse JSON output. Retries on malformed JSON."""
        for attempt in range(LLM_MAX_RETRIES + 1):
            text = self.generate(
                model, prompt, system=system, temperature=temperature
            )
            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s...",
                    attempt + 1,
                    LLM_MAX_RETRIES + 1,
                    text[:200],
                )
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(1)
        raise ValueError(f"Failed to parse JSON after {LLM_MAX_RETRIES + 1} attempts")

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts."""
        resp = self._client.post(
            "/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def list_models(self) -> list[dict[str, Any]]:
        """List locally available models."""
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json()["models"]

    def pull_model(self, model: str) -> None:
        """Pull a model from the Ollama registry."""
        logger.info("Pulling model: %s", model)
        resp = self._client.post(
            "/api/pull",
            json={"name": model, "stream": False},
            timeout=1800.0,  # 30 min for large models
        )
        resp.raise_for_status()
        logger.info("Model pulled: %s", model)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
