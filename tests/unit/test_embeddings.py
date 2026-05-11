import httpx
import pytest

from services.embeddings import OpenAIEmbeddingProvider


def test_openai_embedding_provider_posts_to_embeddings_endpoint():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = request.read().decode("utf-8")
        assert '"model":"text-embedding-3-small"' in payload
        assert '"encoding_format":"float"' in payload
        assert '"dimensions":3' in payload
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        base_url="https://api.openai.test/v1",
        retry_sleep_seconds=0,
        client=client,
    )

    vectors = provider.embed_documents(["first", "second"])

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert requests[0].url == "https://api.openai.test/v1/embeddings"
    assert requests[0].headers["Authorization"] == "Bearer test-key"


def test_openai_embedding_provider_rejects_blank_inputs():
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    with pytest.raises(ValueError):
        provider.embed_documents(["valid", " "])


def test_openai_embedding_provider_batches_large_document_sets():
    batch_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode("utf-8")
        batch_size = payload.count('"doc-')
        batch_sizes.append(batch_size)
        offset = sum(batch_sizes[:-1])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(offset + index)]}
                    for index in range(batch_size)
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        dimensions=1,
        max_batch_size=2,
        retry_sleep_seconds=0,
        client=client,
    )

    vectors = provider.embed_documents(["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"])

    assert batch_sizes == [2, 2, 1]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_openai_embedding_provider_retries_retryable_statuses():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        dimensions=1,
        max_retries=2,
        retry_sleep_seconds=0,
        client=client,
    )

    assert provider.embed_query("hello") == [1.0]
    assert calls == 2


def test_openai_embedding_provider_does_not_retry_non_retryable_statuses():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        dimensions=1,
        max_retries=2,
        retry_sleep_seconds=0,
        client=client,
    )

    with pytest.raises(httpx.HTTPStatusError):
        provider.embed_query("hello")
    assert calls == 1
