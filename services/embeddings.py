import time
from typing import Protocol, Sequence

import httpx

from models.retrieval import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        max_batch_size: int = 2048,
        max_retries: int = 3,
        retry_sleep_seconds: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_sleep_seconds < 0:
            raise ValueError("retry_sleep_seconds must not be negative")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.client = client or httpx.Client(timeout=timeout)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("embedding input texts must not be blank")
        if not cleaned:
            return []

        vectors: list[list[float]] = []
        for batch in _batches(cleaned, self.max_batch_size):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._post_embeddings_with_retry(texts)
        payload = response.json()
        data = sorted(payload.get("data", []), key=lambda item: item["index"])
        vectors = [list(item["embedding"]) for item in data]
        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding response returned {len(vectors)} vectors for "
                f"{len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"embedding response vector has {len(vector)} dimensions, "
                    f"expected {self.dimensions}"
                )
        return vectors

    def _post_embeddings_with_retry(self, texts: list[str]) -> httpx.Response:
        attempts = self.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._post_embeddings(texts)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if not _is_retryable_status(exc.response.status_code):
                    raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc

            if attempt < attempts - 1:
                time.sleep(self.retry_sleep_seconds * (2 ** attempt))

        assert last_exc is not None
        raise last_exc

    def _post_embeddings(self, texts: list[str]) -> httpx.Response:
        return self.client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
                "dimensions": self.dimensions,
            },
        )


def _batches(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}
