import os
import pytest
from unittest.mock import patch

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - fallback when python-dotenv is missing
    def load_dotenv(path: str = ".env"):
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())



def check_environment():
    """Validate environment configuration and external services."""
    load_dotenv()
    required = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [name for name in required if not os.environ.get(name)]
    assert not missing, f"Missing environment variables: {', '.join(missing)}"

    from sentence_transformers import SentenceTransformer  # imported lazily
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(["Test sentence"])
    assert embedding is not None

    from pinecone import Pinecone  # imported lazily
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    stats = index.describe_index_stats()
    assert "total_vector_count" in stats


def test_check_environment_success(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "index")
    import sys
    from types import ModuleType
    sentence_transformers_mock = ModuleType("sentence_transformers")
    sentence_transformers_mock.SentenceTransformer = object
    pinecone_mock = ModuleType("pinecone")
    pinecone_mock.Pinecone = object
    with patch.dict(sys.modules, {
        "sentence_transformers": sentence_transformers_mock,
        "pinecone": pinecone_mock
    }):
        with patch("sentence_transformers.SentenceTransformer") as MockModel, \
             patch("pinecone.Pinecone") as MockPinecone:
            model_inst = MockModel.return_value
            model_inst.encode.return_value = [0.1]
            pc_inst = MockPinecone.return_value
            index_inst = pc_inst.Index.return_value
            index_inst.describe_index_stats.return_value = {"total_vector_count": 1}

            check_environment()

            model_inst.encode.assert_called_once_with(["Test sentence"])
            pc_inst.Index.assert_called_once_with("index")
            index_inst.describe_index_stats.assert_called_once()


def test_missing_env_vars(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    with pytest.raises(AssertionError, match="Missing environment variables"):
        check_environment()


def test_model_exception(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "index")
    import types, sys
    with patch.dict(sys.modules, {
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "pinecone": types.SimpleNamespace(Pinecone=object)
    }):
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("model")):
            with pytest.raises(RuntimeError):
                check_environment()


def test_pinecone_exception(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "index")
    import types, sys
    with patch.dict(sys.modules, {
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "pinecone": types.SimpleNamespace(Pinecone=object)
    }):
        with patch("sentence_transformers.SentenceTransformer") as MockModel:
            MockModel.return_value.encode.return_value = [0.1]
            with patch("pinecone.Pinecone", side_effect=RuntimeError("pc")):
                with pytest.raises(RuntimeError):
                    check_environment()
