#!/usr/bin/env python3
"""
MCPico - A command-line MCP client for interacting with MCP servers and LLMs
"""

import json
import os
import sys
import re
import asyncio
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Import provider handlers
from providers import AnthropicProvider, OpenAIProvider

# Import greetings
try:
    from greetings import pico_greetings, banner
except ImportError:
    def pico_greetings():
        pass
    def banner():
        pass

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
            "models_url": "http://localhost:1234/v1/models",
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
        self.tool_mapping = {}
        self.current_dir = Path.cwd()
    
    def normalize_path(self, path_str: str) -> Path:
        """Normalize path by expanding ~ and making relative paths absolute"""
        path = Path(path_str)
        path = path.expanduser()
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
                data_copy = copy.deepcopy(data) if isinstance(data, (dict, list)) else data
                
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
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {}
                    if provider.get("api_key"):
                        headers["Authorization"] = f"Bearer {provider['api_key']}"
                    
                    # Check if custom models_url is specified in config
                    if "models_url" in provider:
                        models_url = provider["models_url"]
                    else:
                        # Try to construct models URL from api_url
                        # Handle different endpoint patterns
                        api_url = provider["api_url"]
                        if "/chat/completions" in api_url:
                            models_url = api_url.replace("/chat/completions", "/models")
                        elif "/completions" in api_url:
                            models_url = api_url.replace("/completions", "/models")
                        else:
                            # Assume /models is at the same base
                            from urllib.parse import urljoin
                            models_url = urljoin(api_url, "/v1/models")
                    
                    self.debug_log(f"Fetching models from: {models_url}")
                    
                    try:
                        response = await client.get(models_url, headers=headers)
                        
                        if response.status_code == 200:
                            data = response.json()
                            models = data.get("data", [])
                            
                            if not models:
                                console.print(f"[yellow]No models returned from {models_url}[/yellow]")
                                console.print(f"[dim]Current model: {provider['model']}[/dim]")
                                return
                            
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
                            console.print(f"[yellow]URL tried: {models_url}[/yellow]")
                            console.print(f"[dim]Current model: {provider['model']}[/dim]")
                            console.print(f"[dim]Tip: Add 'models_url' to provider config if endpoint differs[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Could not connect to models endpoint: {e}[/yellow]")
                        console.print(f"[dim]Current model: {provider['model']}[/dim]")
                        console.print(f"[dim]Tip: Add 'models_url' to provider config if endpoint differs[/dim]")
        
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
            await self.stdio_initialize(server_name)
            console.print(f"[green]Started MCP server: {server_name}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start {server_name}: {e}[/red]")
    
    async def stdio_initialize(self, server_name: str):
        """Initialize stdio MCP server and fetch tools"""
        conn = self.mcp_connections[server_name]
        process = conn["process"]
        
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcpico", "version": "1.0.0"}
            }
        }
        
        self.debug_log("Sending MCP initialize", init_request)
        
        process.stdin.write((json.dumps(init_request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await process.stdout.readline()
        self.debug_log("MCP initialize response", response_line.decode())
        
        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        process.stdin.write((json.dumps(initialized) + "\n").encode())
        await process.stdin.drain()
        
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
        """Call a tool on a stdio MCP server (arguments already normalized)"""
        conn = self.mcp_connections[server_name]
        process = conn["process"]
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        self.debug_log(f"Calling MCP tool: {server_name}::{tool_name}", request)
        
        process.stdin.write((json.dumps(request) + "\n").encode())
        await process.stdin.drain()
        
        response_line = await process.stdout.readline()
        response = json.loads(response_line)
        
        self.debug_log(f"MCP tool response", response)
        
        return response.get("result", {})
    
    def _normalize_tool_arguments(self, arguments: Dict) -> Dict:
        """Normalize file paths in tool arguments"""
        normalized_args = {}
        for key, value in arguments.items():
            if isinstance(value, str) and ("path" in key.lower() or "file" in key.lower()):
                try:
                    normalized_value = str(self.normalize_path(value))
                    normalized_args[key] = normalized_value
                    self.debug_log(f"Normalized {key}: {value} -> {normalized_value}")
                except:
                    normalized_args[key] = value
            else:
                normalized_args[key] = value
        return normalized_args
    
    async def initialize_mcp_servers(self):
        """Initialize all configured MCP servers"""
        for server_name, server_config in self.config.get("mcp_servers", {}).items():
            if not server_config.get("enabled", True):
                continue
                
            if server_config["type"] == "stdio":
                await self.start_stdio_server(server_name, server_config)
    
    def get_available_tools(self) -> List[Dict]:
        """Get all available tools from all MCP servers"""
        tools = []
        tool_mapping = {}
        
        for server_name, conn in self.mcp_connections.items():
            for tool in conn.get("tools", []):
                sanitized_name = f"{server_name}_{tool['name']}"
                sanitized_name = sanitized_name.replace("::", "_").replace(" ", "_")
                sanitized_name = "".join(c for c in sanitized_name if c.isalnum() or c in "_-")
                sanitized_name = sanitized_name[:128]
                
                tool_def = {
                    "name": sanitized_name,
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {})
                }
                
                tools.append(tool_def)
                tool_mapping[sanitized_name] = {
                    "original_name": tool['name'],
                    "server": server_name
                }
        
        self.tool_mapping = tool_mapping
        return tools
    
    async def send_to_llm(self, user_message: str, files: List[Path] = None) -> str:
        """Send message to LLM with optional file attachments"""
        provider = self.get_current_provider()
        
        # Build message content
        content = []
        file_context = ""
        if files:
            for file_path in files:
                if file_path.exists():
                    file_context += f"\n[File attached: {file_path}]"
                    console.print(f"[cyan]File attached (for MCP tools): {file_path}[/cyan]")
        
        full_message = user_message
        if file_context:
            full_message += file_context
        
        content.append({"type": "text", "text": full_message})
        
        # Build messages with history
        messages = self.conversation_history + [{"role": "user", "content": content}]
        tools = self.get_available_tools()
        
        provider_type = provider.get("type", "anthropic")
        
        try:
            if provider_type == "anthropic":
                result = await AnthropicProvider.send_message(
                    provider, messages, tools,
                    debug_callback=self.debug_log,
                    save_debug_callback=self.save_debug_to_file
                )
                
                if "error" in result:
                    return f"Error: {result['error']}"
                
                # Handle tool calls
                response_text, updated_history = await AnthropicProvider.handle_tool_calls(
                    result, self.tool_mapping, self.call_stdio_tool,
                    self.conversation_history, provider, tools,
                    path_normalizer=self._normalize_tool_arguments,
                    debug_callback=self.debug_log,
                    save_debug_callback=self.save_debug_to_file
                )
                
                # Update conversation history
                if not updated_history:
                    self.conversation_history.append({"role": "user", "content": content})
                    self.conversation_history.append({"role": "assistant", "content": result["content"]})
                else:
                    self.conversation_history = updated_history
                
                return response_text
            
            else:  # OpenAI-compatible
                result = await OpenAIProvider.send_message(
                    provider, messages, tools,
                    debug_callback=self.debug_log,
                    save_debug_callback=self.save_debug_to_file
                )
                
                if "error" in result:
                    return f"Error: {result['error']}"
                
                # Handle tool calls
                response_text, updated_history = await OpenAIProvider.handle_tool_calls(
                    result, self.tool_mapping, self.call_stdio_tool,
                    self.conversation_history, provider, tools, user_message,
                    path_normalizer=self._normalize_tool_arguments,
                    debug_callback=self.debug_log,
                    save_debug_callback=self.save_debug_to_file
                )
                
                # Update conversation history
                if not updated_history:
                    message_data = result.get("choices", [{}])[0].get("message", {})
                    response_text = message_data.get("content") or ""
                    self.conversation_history.append({"role": "user", "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                else:
                    self.conversation_history = updated_history
                
                return response_text
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if self.config.get("debug", False):
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands"""
        if command == "/quit" or command == "/exit":
            self.should_exit = True
            sys.exit(0)
        
        elif command == "/clear" or command == "/reset":
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
                table.add_row(name, config.get('model', 'N/A'), config.get('type', 'unknown'), current)
            
            console.print(table)
        
        elif command.startswith("/use "):
            args = command.split(" ", 1)[1]
            
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
Example: `Locate the main using @~/binary`

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
        
        await self.initialize_mcp_servers()
        
        provider = self.get_current_provider()
        console.print(f"[green]Using: {self.config['current_provider']} ({provider['model']})[/green]")
        
        tools = self.get_available_tools()
        if tools:
            console.print(f"[cyan]Loaded {len(tools)} MCP tools[/cyan]")
        
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        session = PromptSession(history=FileHistory(str(HISTORY_FILE)))
        
        while not self.should_exit:
            try:
                user_input = await session.prompt_async("You: ")
                
                if not user_input.strip():
                    continue
                
                if user_input.startswith("/"):
                    if user_input.startswith("/models"):
                        parts = user_input.split()
                        provider_name = parts[1] if len(parts) > 1 else None
                        await self.list_models(provider_name)
                    else:
                        should_continue = self.handle_command(user_input)
                        if not should_continue:
                            break
                    continue
                
                # Parse file attachments
                files = []
                message = user_input
                file_pattern = r'@([^\s]+)'
                matches = re.finditer(file_pattern, user_input)
                
                for match in matches:
                    file_path_str = match.group(1)
                    file_path = self.normalize_path(file_path_str)
                    if file_path.exists():
                        files.append(file_path)
                        console.print(f"[cyan]Attached: {file_path}[/cyan]")
                        message = message.replace(match.group(0), "", 1).strip()
                    else:
                        console.print(f"[yellow]File not found: {file_path}[/yellow]")
                
                message = " ".join(message.split())
                
                console.print("[dim]Thinking...[/dim]")
                response = await self.send_to_llm(message, files)
                
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
        
        console.print("[yellow]Shutting down...[/yellow]")
        for server_name, conn in self.mcp_connections.items():
            if conn["type"] == "stdio":
                conn["process"].terminate()
                await conn["process"].wait()
        
        console.print("[green]Goodbye![/green]")


async def main():
    client = MCPClient()
    await client.run()


if __name__ == "__main__":
    try:
        pico_greetings()
        banner()
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
