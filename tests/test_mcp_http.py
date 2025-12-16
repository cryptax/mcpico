import pytest
import asyncio
import sys, os

# Ensure project root is on sys.path so top-level modules are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp_http import start_http_server


class DummyResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


class DummyAsyncClient:
    def __init__(self, responses):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None):
        # pop first response
        if not self._responses:
            return DummyResponse(500, {}, "no responses")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_start_http_server(monkeypatch):
    # Simulate two POST responses: init and tools list
    init_resp = DummyResponse(200, {"result": {}})
    initialized_ack = DummyResponse(200, {})
    tools_resp = DummyResponse(200, {"result": {"tools": [{"name": "t1", "description": "d"}]}})

    def fake_client_factory(timeout=30.0):
        # Order: initialize response, initialized ack, tools list
        return DummyAsyncClient([init_resp, initialized_ack, tools_resp])

    # Inject a fake httpx module so the dynamic import inside start_http_server uses our fake
    import types, sys
    fake_httpx = types.SimpleNamespace(AsyncClient=fake_client_factory)
    sys.modules["httpx"] = fake_httpx

    class Client:
        def __init__(self):
            self.mcp_connections = {}

        def debug_log(self, *args, **kwargs):
            pass

    client = Client()
    await start_http_server(client, "test_http", {"url": "http://example.local"})

    assert "test_http" in client.mcp_connections
    conn = client.mcp_connections["test_http"]
    assert conn["type"] == "http"
    assert conn["tools"] and len(conn["tools"]) == 1
