import json
from rich.console import Console

console = Console()


async def start_http_server(client, server_name: str, server_config: dict):
    """Connect to an HTTP/SSE MCP server (extracted from mcpico.MCPClient)."""
    try:
        import httpx
        client.debug_log(f"Connecting to HTTP/SSE MCP server: {server_name}", server_config)

        url = server_config["url"]

        # Try SSE-style GET first (not required for test/mocking)
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}

        client.debug_log(f"Attempting SSE connection to {url}")

        async with httpx.AsyncClient(timeout=30.0) as client_http:
            # Attempt to discover working endpoint by posting initialize
            possible_endpoints = [url]
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mcpico", "version": "1.0.0"},
                },
            }

            connected = False
            working_endpoint = None

            for endpoint in possible_endpoints:
                try:
                    client.debug_log(f"Trying endpoint: {endpoint}", init_request)
                    response = await client_http.post(endpoint, json=init_request)
                    if response.status_code == 200:
                        working_endpoint = endpoint
                        init_response = response.json()
                        client.debug_log(f"HTTP initialize response from {server_name}", init_response)
                        connected = True
                        break
                    else:
                        client.debug_log(f"Endpoint {endpoint} returned {response.status_code}: {response.text}")
                except Exception as e:
                    client.debug_log(f"Endpoint {endpoint} failed", str(e))
                    continue

            if not connected:
                console.print(f"[red]Could not find working endpoint for {server_name}[/red]")
                return

            # Send initialized notification
            initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            await client_http.post(working_endpoint, json=initialized)

            # Request tools list
            tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            tools_response = await client_http.post(working_endpoint, json=tools_request)

            if tools_response.status_code != 200:
                console.print(f"[red]Failed to get tools from {server_name}: status {tools_response.status_code}[/red]")
                client.debug_log(f"Tools request error from {server_name}", tools_response.text)
                return

            tools_data = tools_response.json()
            client.debug_log(f"Tools response from {server_name}", tools_data)

            tools = tools_data.get("result", {}).get("tools", [])

            client.mcp_connections[server_name] = {"type": "http", "url": working_endpoint, "tools": tools}

            console.print(f"[green]Connected to HTTP MCP server: {server_name} ({len(tools)} tools)[/green]")

    except Exception as e:
        console.print(f"[red]Failed to connect to HTTP MCP server {server_name}: {e}[/red]")
        client.debug_log(f"Exception connecting to {server_name}", str(e))


async def call_http_tool(client, server_name: str, tool_name: str, arguments: dict) -> dict:
    """Call a tool on an HTTP MCP server."""
    try:
        import httpx
        conn = client.mcp_connections[server_name]
        url = conn["url"]

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        client.debug_log(f"Calling HTTP MCP tool: {server_name}::{tool_name}", request)

        async with httpx.AsyncClient(timeout=120.0) as client_http:
            response = await client_http.post(url, json=request)

            if response.status_code != 200:
                error_msg = f"HTTP error {response.status_code}: {response.text}"
                client.debug_log(f"HTTP tool call error", error_msg)
                return {"error": error_msg}

            response_data = response.json()
            client.debug_log(f"HTTP MCP tool response", response_data)

            return response_data.get("result", {})
    except Exception as e:
        error_msg = f"Exception calling HTTP tool: {e}"
        client.debug_log(f"HTTP tool call exception", error_msg)
        return {"error": error_msg}
