import asyncio
import json
import pytest
import sys, os

# Ensure project root is on sys.path so top-level modules are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp_stdio import start_stdio_server


class DummyStream:
    def __init__(self, responses=None):
        self.writes = []
        self._responses = responses or []

    def write(self, data: bytes):
        self.writes.append(data)

    async def drain(self):
        return None

    async def readline(self):
        if not self._responses:
            await asyncio.sleep(0)
            return b"\n"
        return self._responses.pop(0)


class DummyProcess:
    def __init__(self, responses=None):
        self.stdin = DummyStream()
        self.stdout = DummyStream(responses=responses or [])
        self.stderr = DummyStream()

    async def wait(self):
        return None


@pytest.mark.asyncio
async def test_start_stdio_server(monkeypatch):
    # Prepare dummy process that replies to initialize and tools/list
    init_resp = json.dumps({}) + "\n"
    tools_resp = json.dumps({"result": {"tools": []}}) + "\n"
    proc = DummyProcess(responses=[init_resp.encode(), tools_resp.encode()])

    async def fake_create(*args, **kwargs):
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create)

    class Client:
        def __init__(self):
            self.mcp_connections = {}

        def debug_log(self, *args, **kwargs):
            pass

    client = Client()
    await start_stdio_server(client, "test_stdio", {"command": "cmd", "args": []})

    assert "test_stdio" in client.mcp_connections
    conn = client.mcp_connections["test_stdio"]
    assert conn["type"] == "stdio"
    assert isinstance(conn["process"], DummyProcess)
    assert conn["tools"] == []
