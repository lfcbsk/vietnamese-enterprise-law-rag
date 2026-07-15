from src.indexing import embedder as embedder_module
from src.indexing.embedder import EmbedderConfig, LawEmbedder


class _ModelStub:
    pass


def test_embedder_prefers_local_model_cache(monkeypatch) -> None:
    calls: list[dict] = []

    def create_model(model_name: str, **kwargs):
        calls.append({"model_name": model_name, **kwargs})
        return _ModelStub()

    monkeypatch.setattr(embedder_module, "SentenceTransformer", create_model)

    LawEmbedder(EmbedderConfig(model_name="test/model", device="cpu"))

    assert calls == [
        {
            "model_name": "test/model",
            "device": "cpu",
            "local_files_only": True,
        }
    ]


def test_embedder_downloads_only_when_local_cache_is_missing(monkeypatch) -> None:
    calls: list[dict] = []

    def create_model(model_name: str, **kwargs):
        calls.append({"model_name": model_name, **kwargs})
        if kwargs.get("local_files_only"):
            raise OSError("model is not cached")
        return _ModelStub()

    monkeypatch.setattr(embedder_module, "SentenceTransformer", create_model)

    LawEmbedder(EmbedderConfig(model_name="test/model", device="cpu"))

    assert calls == [
        {
            "model_name": "test/model",
            "device": "cpu",
            "local_files_only": True,
        },
        {
            "model_name": "test/model",
            "device": "cpu",
        },
    ]
