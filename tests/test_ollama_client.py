"""Tests for Ollama client wrapper."""
import json
import pytest
from unittest.mock import patch, MagicMock
from graphrag.ollama_client import OllamaClient


@pytest.fixture
def client():
    return OllamaClient(base_url="http://localhost:11434")


class TestGenerate:
    def test_generate_returns_text(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello world"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.generate("test-model", "Say hello")
            assert result == "Hello world"

    def test_generate_json_parses_response(self, client):
        json_str = '{"legal_concepts": ["fairness"], "legal_tests": []}'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": json_str}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.generate_json("test-model", "Extract entities")
            assert result == {"legal_concepts": ["fairness"], "legal_tests": []}


class TestEmbed:
    def test_embed_returns_vectors(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed("test-model", ["text1", "text2"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]


class TestListModels:
    def test_list_models(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:32b", "size": 20_000_000_000},
                {"name": "deepseek-r1:8b", "size": 5_200_000_000},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "get", return_value=mock_response):
            result = client.list_models()
            assert len(result) == 2
            assert result[0]["name"] == "qwen3:32b"
