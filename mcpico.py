#!/usr/bin/env python3
"""
MCPico - A command-line MCP client for interacting with MCP servers and LLMs
"""

import json
import os
import sys
import re
import asyncio
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from greetings import banner, pico_greetings

# Configuration file path
CONFIG_FILE = Path.home() / ".config" / "mcpico" / "config.json"
HISTORY_FILE = Path.home() / ".config" / "mcpico" / "history"
DEBUG_DIR = Path("/tmp/mcpico_debug")

console = Console()

# Default configuration
DEFAULT_CONFIG = {
    "current_provider": "anthropic",
    "debug": False,
    "providers": {
        "anthropic": {
            "api_key": "YOUR_API_KEY_HERE",
            "model": "claude-sonnet-4-20250514",
            "api_url": "https://api.anthropic.com/v1/messages",
            "type": "anthropic"
        },
        "groq": {
            "api_key": "",
            "model": "llama-3.3-70b-versatile",
            "api_url": "https://api.groq.com/openai/v1/chat/completions",
            "type": "openai"
        },
        "lmstudio": {
            "api_key": "",
            "model": "local-model",
            "api_url": "http://localhost:1234/v1/chat/completions",
            "type": "openai"
        }
    },
    "mcp_servers": {
        "example_stdio": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
            "enabled": False
        },
        "example_http": {
            "type": "http",
            "url": "http://localhost:8080/mcp",
            "enabled": False
        }
    }
}


class MCPClient:
    def __init__(self):
        self.config = self.load_config()
        self.mcp_connections = {}
        self.conversation_history = []
        self.should_exit = False
        self.tool_mapping = {}  # Store tool name mappings
        self.current_dir = Path.cwd()  # Track current directory
    
    def normalize_path(self, path_str: str) -> Path:
        """Normalize path by expanding ~ and making relative paths absolute"""
        path = Path(path_str)
        # Expand ~
        path = path.expanduser()
        # Make relative paths absolute based on current directory
        if not path.is_absolute():
            path = self.current_dir / path
        return path.resolve()
        
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            console.print(f"[yellow]Created default config at {CONFIG_FILE}[/yellow]")
            console.print("[yellow]Please edit it with your API keys and MCP server details[/yellow]")
            return DEFAULT_CONFIG
    
    def save_config(self):
        """Save current configuration to file"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_current_provider(self) -> Dict:
        """Get current provider configuration"""
        provider_name = self.config.get("current_provider", "anthropic")
        return self.config["providers"].get(provider_name, {})
    
    def debug_log(self, message: str, data: Any = None):
        """Log debug information if debug mode is enabled"""
        if self.config.get("debug", False):
            console.print(f"[dim cyan]DEBUG: {message}[/dim cyan]")
            if data:
                # Make a deep copy to avoid modifying original
                import copy
                data_copy = copy.deepcopy(data) if isinstance(data, (dict, list)) else data
                
                # Censor API keys
                def censor_keys(obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if key in ["x-api-key", "Authorization"]:
                                if isinstance(value, str):
                                    obj[key] = "***CENSORED***"
                            elif isinstance(value, (dict, list)):
                                censor_keys(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            censor_keys(item)
                
                censor_keys(data_copy)
                
                # Shorten tools list if present
                if isinstance(data_copy, dict):
                    if "body" in data_copy and isinstance(data_copy["body"], dict):
                        if "tools" in data_copy["body"] and isinstance(data_copy["body"]["tools"], list):
                            tools_list = data_copy["body"]["tools"]
                            if len(tools_list) > 3:
                                data_copy["body"]["tools"] = tools_list[:3] + [
                                    {"_truncated": f"... and {len(tools_list) - 3} more tools"}
                                ]
                    elif "tools" in data_copy and isinstance(data_copy["tools"], list):
                        tools_list = data_copy["tools"]
                        if len(tools_list) > 3:
                            data_copy["tools"] = tools_list[:3] + [
                                {"_truncated": f"... and {len(tools_list) - 3} more tools"}
                            ]
                
                console.print(Panel(
                    json.dumps(data_copy, indent=2),
                    title="Debug Data",
                    border_style="dim cyan"
                ))
    
    def save_debug_to_file(self, request_data: Dict, response_data: Any, response_status: int):
        """Save debug information to file"""
        if self.config.get("debug", False):
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = asyncio.get_event_loop().time()
            debug_file = DEBUG_DIR / f"request_response_{int(timestamp)}.json"
            
            debug_info = {
                "timestamp": timestamp,
                "request": request_data,
                "response_status": response_status,
                "response": response_data
            }
            
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
            
            console.print(f"[dim cyan]Debug saved to: {debug_file}[/dim cyan]")
    
    async def list_models(self, provider_name: str = None):
        """List available models for a provider"""
        if provider_name is None:
            provider_name = self.config.get("current_provider")
        
        if provider_name not in self.config["providers"]:
            console.print(f"[red]Unknown provider: {provider_name}[/red]")
            return
        
        provider = self.config["providers"][provider_name]
        
        try:
            if provider.get("type") == "anthropic":
                # Anthropic doesn't have a models endpoint, show known models
                models = [
                    "claude-opus-4-20250514",
                    "claude-sonnet-4-20250514",
                    "claude-sonnet-4-5-20250929",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022"
                ]
                table = Table(title=f"{provider_name} Models")
                table.add_column("Model", style="cyan")
                table.add_column("Current", style="green")
                
                for model in models:
                    current = "✓" if model == provider["model"] else ""
                    table.add_row(model, current)
                
                console.print(table)
                console.print("[dim]Note: Anthropic API doesn't provide a models list endpoint[/dim]")
            
            elif provider.get("type") == "openai":
                # Try OpenAI-compatible models endpoint
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {}
                    if provider.get("api_key"):
                        headers["Authorization"] = f"Bearer {provider['api_key']}"
                    
                    # Try to get models list
                    models_url = provider["api_url"].replace("/chat/completions", "/models")
                    
                    self.debug_log(f"Fetching models from: {models_url}")
                    
                    response = await client.get(models_url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("data", [])
                        
                        table = Table(title=f"{provider_name} Models")
                        table.add_column("Model ID", style="cyan")
                        table.add_column("Current", style="green")
                        
                        for model in models:
                            model_id = model.get("id", "unknown")
                            current = "✓" if model_id == provider["model"] else ""
                            table.add_row(model_id, current)
                        
                        console.print(table)
                    else:
                        console.print(f"[yellow]Could not retrieve models (status {response.status_code})[/yellow]")
                        console.print(f"[dim]Current model: {provider['model']}[/dim]")
        
        except Exception as e:
            console.print(f"[red]Error listing models: {e}[/red]")
            console.print(f"[dim]Current model: {provider['model']}[/dim]")
    
    async def start_stdio_server(self, server_name: str, server_config: Dict):
        """Start a stdio MCP server"""
        try:
            process = await asyncio.create_subprocess_exec(
                server_config["command"],
                *server_config["args"],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.mcp_connections[server_name] = {
                "type": "stdio",
                "process": process,
                "tools": []
            }
            # Initialize and get tools
            await self.stdio_initialize(server_name)
            console.print(f"[green]Started MCP server: {server_name}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start {server_name}: {e}[/red]")
    
    async def stdio_initialize(self, server_name: str):
        """Initialize stdio MCP server and fetch tools"""
        conn = self.mcp_connections[server_name]
        process = conn["process"]
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-terminal-client", "version": "1.0.0"}
            }
        }
        
        self.debug_log("Sending MCP initialize", init_request)
        
        process.stdin.write((json.dumps(init_request) + "\n").encode())
        await process.stdin.drain()
        
        # Read response
        response_line = await process.stdout.readline()
        self.debug_log("MCP initialize response", response_line.decode())
        
        # Send initialized notification
        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        process.stdin.write((json.dumps(initialized) + "\n").encode())
        await process.stdin.drain()
        
        # Get tools list
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        process.stdin.write((json.dumps(tools_request) + "\n").encode())
        await process.stdin.drain()
        
        tools_response = await process.stdout.readline()
        if tools_response:
            tools_data = json.loads(tools_response)
            conn["tools"] = tools_data.get("result", {}).get("tools", [])
            self.debug_log(f"MCP tools loaded for {server_name}", conn["tools"])
    
    async def call_stdio_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Any:
        """Call a tool on a stdio MCP server"""
        conn = self.mcp_connections[server_name]
        process = conn["process"]
        
        # Normalize any file paths in arguments
        normalized_args = {}
        for key, value in arguments.items():
            if isinstance(value, str) and ("path" in key.lower() or "file" in key.lower()):
                # Try to normalize as a path
                try:
                    normalized_value = str(self.normalize_path(value))
                    normalized_args[key] = normalized_value
                    self.debug_log(f"Normalized {key}: {value} -> {normalized_value}")
                except:
                    normalized_args[key] = value
            else:
                normalized_args[key] = value
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": normalized_args
            }
        }
        
        self.debug_log(f"Calling MCP tool: {server_name}::{tool_name}", request)
        
        process.stdin.write((json.dumps(request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await process.stdout.readline()
        response = json.loads(response_line)
        
        self.debug_log(f"MCP tool response", response)
        
        return response.get("result", {})
    
    async def initialize_mcp_servers(self):
        """Initialize all configured MCP servers"""
        for server_name, server_config in self.config.get("mcp_servers", {}).items():
            if not server_config.get("enabled", True):
                continue
                
            if server_config["type"] == "stdio":
                await self.start_stdio_server(server_name, server_config)
            # HTTP servers would be initialized here
    
    def get_available_tools(self) -> List[Dict]:
        """Get all available tools from all MCP servers"""
        tools = []
        tool_mapping = {}  # Store mapping for later use
        
        for server_name, conn in self.mcp_connections.items():
            for tool in conn.get("tools", []):
                # Sanitize tool name to match Anthropic requirements: ^[a-zA-Z0-9_-]{1,128}$
                sanitized_name = f"{server_name}_{tool['name']}"
                sanitized_name = sanitized_name.replace("::", "_").replace(" ", "_")
                # Remove any invalid characters
                sanitized_name = "".join(c for c in sanitized_name if c.isalnum() or c in "_-")
                sanitized_name = sanitized_name[:128]  # Limit to 128 chars
                
                # Store only API-compliant fields
                tool_def = {
                    "name": sanitized_name,
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {})
                }
                
                tools.append(tool_def)
                
                # Keep separate mapping for internal use
                tool_mapping[sanitized_name] = {
                    "original_name": tool['name'],
                    "server": server_name
                }
        
        # Store mapping as instance variable for use in send_to_llm
        self.tool_mapping = tool_mapping
        return tools
    
    async def send_to_llm(self, user_message: str, files: List[Path] = None) -> str:
        """Send message to LLM with optional file attachments"""
        provider = self.get_current_provider()
        
        # Build message content
        content = []
        
        # Handle file attachments - don't send to Claude, use MCP tools instead
        file_context = ""
        if files:
            for file_path in files:
                if file_path.exists():
                    # Add file information to the message context
                    file_context += f"\n[File attached: {file_path}]"
                    console.print(f"[cyan]File attached (for MCP tools): {file_path}[/cyan]")
        
        # Combine message with file context
        full_message = user_message
        if file_context:
            full_message += file_context
        
        content.append({"type": "text", "text": full_message})
        
        # Build messages with history
        messages = self.conversation_history + [{"role": "user", "content": content}]
        
        # Get available MCP tools
        tools = self.get_available_tools()
        
        # Prepare request based on provider type
        provider_type = provider.get("type", "anthropic")
        
        if provider_type == "anthropic":
            # Anthropic format
            request_data = {
                "model": provider["model"],
                "max_tokens": 4096,
                "messages": messages
            }
            if tools:
                request_data["tools"] = tools
            
            headers = {
                "x-api-key": provider["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        else:
            # OpenAI-compatible format (LM Studio, Groq, etc.)
            # Convert Anthropic message format to OpenAI format
            openai_messages = []
            for msg in messages:
                if isinstance(msg["content"], list):
                    # Extract text from content blocks
                    text_parts = [block["text"] for block in msg["content"] if block["type"] == "text"]
                    openai_messages.append({
                        "role": msg["role"],
                        "content": " ".join(text_parts)
                    })
                else:
                    openai_messages.append(msg)
            
            request_data = {
                "model": provider["model"],
                "messages": openai_messages
            }
            
            # Add tools in OpenAI format if available
            if tools:
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["input_schema"]
                        }
                    })
                request_data["tools"] = openai_tools
            
            headers = {
                "content-type": "application/json"
            }
            if provider.get("api_key"):
                headers["Authorization"] = f"Bearer {provider['api_key']}"
        
        # Debug output
        self.debug_log(f"Request to {provider['api_url']}", {
            "headers": {k: v[:20] + "..." if k.lower() in ["authorization", "x-api-key"] else v 
                       for k, v in headers.items()},
            "body": request_data
        })
        
        # Send request
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    provider["api_url"],
                    json=request_data,
                    headers=headers
                )
                
                # Debug output
                self.debug_log(f"Response status: {response.status_code}")
                
                # Save to file if debug enabled
                response_data = None
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                self.save_debug_to_file(
                    {"url": provider["api_url"], "headers": headers, "body": request_data},
                    response_data,
                    response.status_code
                )
                
                # Debug output on error
                if response.status_code != 200:
                    console.print(f"[red]API Error {response.status_code}:[/red]")
                    console.print(response.text)
                    return f"Error: {response.status_code} - {response.text}"
                
                result = response_data if isinstance(response_data, dict) else response.json()
                self.debug_log("Response body", result)
                
        except httpx.HTTPStatusError as e:
            console.print(f"[red]HTTP Error: {e}[/red]")
            console.print(f"[red]Response: {e.response.text}[/red]")
            raise
        except KeyError as e:
            console.print(f"[red]KeyError accessing response: {e}[/red]")
            self.debug_log("Full result that caused error", result)
            raise
        
        # Extract response based on provider type
        if provider_type == "anthropic":
            assistant_message = result
            response_text = ""
            
            # Process tool calls in a loop until no more tool_use blocks
            while True:
                has_tool_use = False
                
                # Handle tool use
                for block in assistant_message.get("content", []):
                    if block["type"] == "text":
                        response_text += block["text"]
                    elif block["type"] == "tool_use":
                        has_tool_use = True
                        
                        # Find original tool info from mapping
                        tool_name = block["name"]
                        if tool_name not in self.tool_mapping:
                            console.print(f"[red]Tool not found in mapping: {tool_name}[/red]")
                            continue
                        
                        tool_info = self.tool_mapping[tool_name]
                        server_name = tool_info["server"]
                        original_tool_name = tool_info["original_name"]
                        
                        # Ask user for approval
                        console.print(Panel(
                            f"[yellow]Tool:[/yellow] {server_name}::{original_tool_name}\n"
                            f"[yellow]Arguments:[/yellow]\n{json.dumps(block['input'], indent=2)}",
                            title="🔧 Tool Call Request",
                            border_style="yellow"
                        ))
                        
                        # Get user input for approval
                        approval = input("Approve? [Y/n/edit]: ").strip().lower()
                        
                        if approval == 'n':
                            console.print("[red]Tool call rejected[/red]")
                            # Send rejection back to Claude
                            tool_result = {"error": "User rejected tool call"}
                        elif approval == 'edit':
                            console.print("[cyan]Enter new arguments (JSON format):[/cyan]")
                            try:
                                new_args_str = input("> ")
                                new_args = json.loads(new_args_str)
                                console.print(f"[cyan]Calling tool with modified arguments...[/cyan]")
                                tool_result = await self.call_stdio_tool(
                                    server_name, 
                                    original_tool_name, 
                                    new_args
                                )
                            except json.JSONDecodeError:
                                console.print("[red]Invalid JSON, using original arguments[/red]")
                                tool_result = await self.call_stdio_tool(
                                    server_name, 
                                    original_tool_name, 
                                    block["input"]
                                )
                        else:  # Default to yes
                            console.print(f"[green]✓ Executing tool: {server_name}::{original_tool_name}[/green]")
                            tool_result = await self.call_stdio_tool(
                                server_name, 
                                original_tool_name, 
                                block["input"]
                            )
                        
                        # Display tool result to user
                        console.print("\n[bold cyan]Tool Result:[/bold cyan]")
                        if isinstance(tool_result, dict):
                            # Pretty print structured data
                            if "content" in tool_result:
                                result_content = tool_result["content"]
                                if isinstance(result_content, list):
                                    for item in result_content:
                                        if isinstance(item, dict) and "text" in item:
                                            console.print(Panel(item["text"], border_style="dim cyan"))
                                        else:
                                            console.print(Panel(json.dumps(item, indent=2), border_style="dim cyan"))
                                else:
                                    console.print(Panel(str(result_content), border_style="dim cyan"))
                            else:
                                console.print(Panel(json.dumps(tool_result, indent=2), border_style="dim cyan"))
                        else:
                            console.print(Panel(str(tool_result), border_style="dim cyan"))
                        console.print()
                        
                        # Add to conversation history if not already added
                        if not self.conversation_history or self.conversation_history[-1]["role"] != "assistant":
                            self.conversation_history.append({"role": "user", "content": content})
                            self.conversation_history.append({"role": "assistant", "content": assistant_message["content"]})
                        
                        # Send tool result back
                        tool_result_content = [{
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": json.dumps(tool_result)
                        }]
                        
                        # Continue the conversation with tool result
                        continue_request = {
                            "model": provider["model"],
                            "max_tokens": 4096,
                            "messages": self.conversation_history + [{"role": "user", "content": tool_result_content}]
                        }
                        if tools:
                            continue_request["tools"] = tools
                        
                        self.debug_log("Continuing conversation with tool result", continue_request)
                        
                        async with httpx.AsyncClient(timeout=120.0) as client:
                            continue_response = await client.post(
                                provider["api_url"],
                                json=continue_request,
                                headers=headers
                            )
                            
                            # Save debug info
                            continue_result_data = continue_response.json()
                            self.save_debug_to_file(
                                {"url": provider["api_url"], "headers": headers, "body": continue_request},
                                continue_result_data,
                                continue_response.status_code
                            )
                            
                            continue_result = continue_result_data
                        
                        self.debug_log("Continue response", continue_result)
                        
                        # Update history with tool result and response
                        self.conversation_history.append({"role": "user", "content": tool_result_content})
                        self.conversation_history.append({"role": "assistant", "content": continue_result["content"]})
                        
                        # Set the assistant_message to the new response to check for more tool calls
                        assistant_message = continue_result
                        
                        # Break out of the for loop to restart checking from the beginning
                        break
                
                # If no tool_use was found, we're done
                if not has_tool_use:
                    break
            
            # If we haven't added the final message to history yet, do it now
            if not self.conversation_history or self.conversation_history[-1]["content"] != assistant_message["content"]:
                if not self.conversation_history or self.conversation_history[-2:] != [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": assistant_message["content"]}
                ]:
                    self.conversation_history.append({"role": "user", "content": content})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message["content"]})
            
            return response_text
        else:
            # OpenAI format response handling
            message_data = result.get("choices", [{}])[0].get("message", {})
            response_text = message_data.get("content") or ""  # Handle None or missing content
            
            # Check for tool calls in OpenAI format
            if "tool_calls" in message_data and message_data["tool_calls"]:
                # Handle OpenAI-style tool calls
                for tool_call in message_data["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    
                    if tool_name not in self.tool_mapping:
                        console.print(f"[red]Tool not found in mapping: {tool_name}[/red]")
                        continue
                    
                    tool_info = self.tool_mapping[tool_name]
                    server_name = tool_info["server"]
                    original_tool_name = tool_info["original_name"]
                    
                    # Ask user for approval
                    console.print(Panel(
                        f"[yellow]Tool:[/yellow] {server_name}::{original_tool_name}\n"
                        f"[yellow]Arguments:[/yellow]\n{json.dumps(tool_args, indent=2)}",
                        title="🔧 Tool Call Request",
                        border_style="yellow"
                    ))
                    
                    approval = input("Approve? [Y/n/edit]: ").strip().lower()
                    
                    if approval == 'n':
                        console.print("[red]Tool call rejected[/red]")
                        tool_result = {"error": "User rejected tool call"}
                    elif approval == 'edit':
                        console.print("[cyan]Enter new arguments (JSON format):[/cyan]")
                        try:
                            new_args_str = input("> ")
                            new_args = json.loads(new_args_str)
                            console.print(f"[cyan]Calling tool with modified arguments...[/cyan]")
                            tool_result = await self.call_stdio_tool(server_name, original_tool_name, new_args)
                        except json.JSONDecodeError:
                            console.print("[red]Invalid JSON, using original arguments[/red]")
                            tool_result = await self.call_stdio_tool(server_name, original_tool_name, tool_args)
                    else:
                        console.print(f"[green]✓ Executing tool: {server_name}::{original_tool_name}[/green]")
                        tool_result = await self.call_stdio_tool(server_name, original_tool_name, tool_args)
                    
                    # Display tool result to user
                    console.print("\n[bold cyan]Tool Result:[/bold cyan]")
                    if isinstance(tool_result, dict):
                        if "content" in tool_result:
                            result_content = tool_result["content"]
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and "text" in item:
                                        console.print(Panel(item["text"], border_style="dim cyan"))
                                    else:
                                        console.print(Panel(json.dumps(item, indent=2), border_style="dim cyan"))
                            else:
                                console.print(Panel(str(result_content), border_style="dim cyan"))
                        else:
                            console.print(Panel(json.dumps(tool_result, indent=2), border_style="dim cyan"))
                    else:
                        console.print(Panel(str(tool_result), border_style="dim cyan"))
                    console.print()
                    
                    # Continue conversation with tool result
                    self.conversation_history.append({"role": "user", "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": response_text if response_text else "", "tool_calls": message_data["tool_calls"]})
                    
                    # Add tool result message
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result)
                    }
                    self.conversation_history.append(tool_message)
                    
                    # Make follow-up request
                    follow_up_request = {
                        "model": provider["model"],
                        "messages": self.conversation_history
                    }
                    if tools:
                        openai_tools = []
                        for tool in tools:
                            openai_tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "description": tool["description"],
                                    "parameters": tool["input_schema"]
                                }
                            })
                        follow_up_request["tools"] = openai_tools
                    
                    self.debug_log("Follow-up request with tool result", follow_up_request)
                    
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        follow_up_response = await client.post(
                            provider["api_url"],
                            json=follow_up_request,
                            headers=headers
                        )
                        
                        if follow_up_response.status_code != 200:
                            console.print(f"[red]Follow-up API Error {follow_up_response.status_code}:[/red]")
                            console.print(follow_up_response.text)
                            return response_text
                        
                        follow_up_result = follow_up_response.json()
                    
                    self.debug_log("Follow-up response", follow_up_result)
                    
                    # Extract response, handling case where content might be null
                    follow_up_message = follow_up_result["choices"][0]["message"]
                    response_text = follow_up_message.get("content", "")
                    
                    # Update history
                    self.conversation_history.append({"role": "assistant", "content": response_text if response_text else ""})
                
                return response_text if response_text else "Tool executed successfully."
            
            # Store in simplified format for OpenAI (no tool calls)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if should continue, False if should exit"""
        if command == "/quit" or command == "/exit":
            self.should_exit = True
            sys.exit(0)  # Immediate exit
        
        elif command == "/clear":
            self.conversation_history = []
            console.print("[green]Conversation history cleared[/green]")
        
        elif command == "/reset":
            self.conversation_history = []
            console.print("[green]Context reset - conversation history cleared[/green]")
        
        elif command == "/debug":
            self.config["debug"] = not self.config.get("debug", False)
            self.save_config()
            status = "enabled" if self.config["debug"] else "disabled"
            console.print(f"[green]Debug mode {status}[/green]")
        
        elif command == "/history":
            console.print(Panel(json.dumps(self.conversation_history, indent=2), title="Conversation History"))
        
        elif command == "/providers":
            table = Table(title="Providers")
            table.add_column("Name", style="cyan")
            table.add_column("Model", style="yellow")
            table.add_column("Type", style="magenta")
            table.add_column("Current", style="green")
            
            for name, config in self.config["providers"].items():
                current = "✓" if name == self.config["current_provider"] else ""
                table.add_row(
                    name,
                    config.get('model', 'N/A'),
                    config.get('type', 'unknown'),
                    current
                )
            
            console.print(table)
        
        elif command.startswith("/use "):
            args = command.split(" ", 1)[1]
            
            # Check if it's provider:model format
            if ":" in args:
                provider_name, model_name = args.split(":", 1)
                if provider_name in self.config["providers"]:
                    self.config["current_provider"] = provider_name
                    self.config["providers"][provider_name]["model"] = model_name
                    self.save_config()
                    console.print(f"[green]Switched to provider: {provider_name} with model: {model_name}[/green]")
                else:
                    console.print(f"[red]Unknown provider: {provider_name}[/red]")
            else:
                # Just provider name
                provider_name = args
                if provider_name in self.config["providers"]:
                    self.config["current_provider"] = provider_name
                    self.save_config()
                    provider = self.config["providers"][provider_name]
                    console.print(f"[green]Switched to provider: {provider_name} ({provider['model']})[/green]")
                else:
                    console.print(f"[red]Unknown provider: {provider_name}[/red]")
        
        elif command.startswith("/model "):
            model_name = command.split(" ", 1)[1]
            provider_name = self.config.get("current_provider")
            if provider_name in self.config["providers"]:
                self.config["providers"][provider_name]["model"] = model_name
                self.save_config()
                console.print(f"[green]Switched model to: {model_name} (provider: {provider_name})[/green]")
            else:
                console.print(f"[red]No current provider set[/red]")
        
        elif command == "/tools":
            tools = self.get_available_tools()
            if tools:
                table = Table(title="Available MCP Tools")
                table.add_column("Tool", style="cyan")
                table.add_column("Description", style="white")
                
                for tool in tools:
                    tool_name = tool['name']
                    # Get original name from mapping if available
                    if tool_name in self.tool_mapping:
                        display_name = f"{self.tool_mapping[tool_name]['server']}::{self.tool_mapping[tool_name]['original_name']}"
                    else:
                        display_name = tool_name
                    table.add_row(display_name, tool['description'])
                
                console.print(table)
            else:
                console.print("[yellow]No MCP tools available[/yellow]")
        
        elif command == "/help":
            help_text = """
**Available Commands:**
- `/quit` or `/exit` - Exit the client
- `/clear` or `/reset` - Clear conversation history and reset context
- `/history` - Show conversation history
- `/providers` - List available providers
- `/use <provider>` - Switch to a different provider
- `/use <provider>:<model>` - Switch to a provider and model
- `/model <model>` - Change model for current provider
- `/models [provider]` - List available models for current or specified provider
- `/tools` - List available MCP tools
- `/debug` - Toggle debug mode (shows HTTP requests/responses)
- `/help` - Show this help message

**Attaching Files:**
Type `@` followed by the file path to attach files to your message.
Example: `@/path/to/file.pdf Summarize this document`
Example: `Locate the main using @/path/to/binary`

**Examples:**
- `/use anthropic` - Switch to Anthropic provider
- `/use groq:llama-3.3-70b-versatile` - Switch to Groq with specific model
- `/model claude-opus-4-20250514` - Change to Opus model on current provider
            """
            console.print(Panel(Markdown(help_text), title="Help"))
        
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
        
        return True
    
    async def run(self):
        """Main interactive loop"""
        console.print(Panel.fit(
            "[bold cyan]MCPico - MCP Terminal Client[/bold cyan]\n"
            f"Config: {CONFIG_FILE}\n"
            "Type /help for commands",
            border_style="cyan"
        ))
        
        # Initialize MCP servers
        await self.initialize_mcp_servers()
        
        # Show current provider
        provider = self.get_current_provider()
        console.print(f"[green]Using: {self.config['current_provider']} ({provider['model']})[/green]")
        
        # Show available tools
        tools = self.get_available_tools()
        if tools:
            console.print(f"[cyan]Loaded {len(tools)} MCP tools[/cyan]")
        
        # Create prompt session
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        session = PromptSession(history=FileHistory(str(HISTORY_FILE)))
        
        while not self.should_exit:
            try:
                # Get user input
                user_input = await session.prompt_async("You: ")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    # For async commands like /models, we need to await them
                    if user_input.startswith("/models"):
                        parts = user_input.split()
                        provider_name = parts[1] if len(parts) > 1 else None
                        await self.list_models(provider_name)
                    else:
                        should_continue = self.handle_command(user_input)
                        if not should_continue:
                            # Exit flag was set, break the loop
                            break
                    continue
                
                # Parse file attachments - look for @/path/to/file pattern
                files = []
                message = user_input
                
                # Use regex to find @/path patterns (handles spaces in paths if quoted)
                import re
                file_pattern = r'@([^\s]+)'
                matches = re.finditer(file_pattern, user_input)
                
                for match in matches:
                    file_path_str = match.group(1)
                    file_path = self.normalize_path(file_path_str)
                    if file_path.exists():
                        files.append(file_path)
                        console.print(f"[cyan]Attached: {file_path}[/cyan]")
                        # Remove the @/path from the message
                        message = message.replace(match.group(0), "", 1).strip()
                    else:
                        console.print(f"[yellow]File not found: {file_path}[/yellow]")
                
                # Clean up extra spaces
                message = " ".join(message.split())
                
                # Send to LLM
                console.print("[dim]Thinking...[/dim]")
                response = await self.send_to_llm(message, files)
                
                # Display response
                console.print("\n[bold cyan]Assistant:[/bold cyan]")
                console.print(Markdown(response))
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if self.config.get("debug", False):
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                continue
        
        # Cleanup
        console.print("[yellow]Shutting down...[/yellow]")
        for server_name, conn in self.mcp_connections.items():
            if conn["type"] == "stdio":
                conn["process"].terminate()
                await conn["process"].wait()
        
        console.print("[green]Goodbye![/green]")

async def main():
    client = MCPClient()
    pico_greetings()
    banner()
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
