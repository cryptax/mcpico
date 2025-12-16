import asyncio
import json
from rich.console import Console

console = Console()


async def start_stdio_server(client, server_name: str, server_config: dict):
    """Start a stdio MCP server (extracted from mcpico.MCPClient)."""
    try:
        client.debug_log(f"Starting stdio MCP server: {server_name}", server_config)
        process = await asyncio.create_subprocess_exec(
            server_config["command"],
            *server_config.get("args", []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        client.mcp_connections[server_name] = {
            "type": "stdio",
            "process": process,
            "tools": [],
        }
        await stdio_initialize(client, server_name)
        console.print(f"[green]Started MCP server: {server_name}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to start {server_name}: {e}[/red]")
        client.debug_log(f"Error starting {server_name}", str(e))


async def stdio_initialize(client, server_name: str):
    """Initialize stdio MCP server and fetch tools."""
    conn = client.mcp_connections[server_name]
    process = conn["process"]

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

    client.debug_log("Sending MCP initialize", init_request)

    process.stdin.write((json.dumps(init_request) + "\n").encode())
    await process.stdin.drain()

    response_line = await process.stdout.readline()
    client.debug_log("MCP initialize response", response_line.decode())

    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    process.stdin.write((json.dumps(initialized) + "\n").encode())
    await process.stdin.drain()

    tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    process.stdin.write((json.dumps(tools_request) + "\n").encode())
    await process.stdin.drain()

    tools_response = await process.stdout.readline()
    if tools_response:
        try:
            tools_data = json.loads(tools_response)
            conn["tools"] = tools_data.get("result", {}).get("tools", [])
            client.debug_log(f"MCP tools loaded for {server_name}", conn["tools"])
        except Exception:
            client.debug_log("Failed to parse tools response", tools_response)


async def call_stdio_tool(client, server_name: str, tool_name: str, arguments: dict) -> dict:
    """Call a tool on a stdio MCP server."""
    conn = client.mcp_connections[server_name]
    process = conn["process"]

    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    client.debug_log(f"Calling MCP tool: {server_name}::{tool_name}", request)

    process.stdin.write((json.dumps(request) + "\n").encode())
    await process.stdin.drain()

    response_line = await process.stdout.readline()
    response = json.loads(response_line)

    client.debug_log(f"MCP tool response", response)

    return response.get("result", {})
