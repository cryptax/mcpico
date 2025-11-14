"""
LLM Provider handlers for MCPico
Handles Anthropic and OpenAI-compatible APIs
"""

import json
import httpx
from typing import Dict, List, Any, Optional, Callable
from rich.console import Console
from rich.panel import Panel

console = Console()


class ToolApproval:
    """Handle tool approval UI"""
    
    @staticmethod
    def display_and_approve(server_name: str, tool_name: str, arguments: Dict) -> tuple[bool, Dict]:
        """
        Display tool call and get user approval
        Returns: (approved, modified_arguments)
        """
        console.print(Panel(
            f"[yellow]Tool:[/yellow] {server_name}::{tool_name}\n"
            f"[yellow]Arguments:[/yellow]\n{json.dumps(arguments, indent=2)}",
            title="🔧 Tool Call Request",
            border_style="yellow"
        ))
        
        approval = input("Approve? [Y/n/edit]: ").strip().lower()
        
        if approval == 'n':
            console.print("[red]Tool call rejected[/red]")
            return False, arguments
        elif approval == 'edit':
            console.print("[cyan]Enter new arguments (JSON format):[/cyan]")
            try:
                new_args_str = input("> ")
                new_args = json.loads(new_args_str)
                return True, new_args
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON, using original arguments[/red]")
                return True, arguments
        else:
            return True, arguments
    
    @staticmethod
    def display_result(tool_result: Any):
        """Display tool result nicely"""
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


async def execute_tool_with_approval(
    tool_name: str,
    tool_args: Dict,
    tool_mapping: Dict,
    tool_executor: Callable,
    path_normalizer: Callable = None
) -> Any:
    """
    Common function to execute a tool with user approval
    Returns: tool_result
    """
    if tool_name not in tool_mapping:
        console.print(f"[red]Tool not found in mapping: {tool_name}[/red]")
        return {"error": f"Tool {tool_name} not found"}
    
    tool_info = tool_mapping[tool_name]
    server_name = tool_info["server"]
    original_tool_name = tool_info["original_name"]
    
    # Normalize paths if normalizer provided
    if path_normalizer:
        tool_args = path_normalizer(tool_args)
    
    # Get approval
    approved, modified_args = ToolApproval.display_and_approve(
        server_name, original_tool_name, tool_args
    )
    
    if not approved:
        return {"error": "User rejected tool call"}
    
    console.print(f"[green]✓ Executing tool: {server_name}::{original_tool_name}[/green]")
    tool_result = await tool_executor(server_name, original_tool_name, modified_args)
    
    # Display result
    ToolApproval.display_result(tool_result)
    
    return tool_result


async def send_api_request(
    url: str,
    headers: Dict,
    request_data: Dict,
    debug_callback: Optional[Callable] = None,
    save_debug_callback: Optional[Callable] = None
) -> Dict:
    """Common function to send API requests"""
    if debug_callback:
        censored_headers = {k: "***CENSORED***" if k.lower() in ["x-api-key", "authorization"] else v 
                           for k, v in headers.items()}
        debug_callback("API Request", {"headers": censored_headers, "body": request_data})
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=request_data, headers=headers)
        
        if response.status_code != 200:
            console.print(f"[red]API Error {response.status_code}:[/red]")
            console.print(response.text)
            error_result = {"error": f"{response.status_code}: {response.text}"}
            
            if save_debug_callback:
                save_debug_callback(
                    {"url": url, "headers": headers, "body": request_data},
                    error_result,
                    response.status_code
                )
            
            return error_result
        
        response_data = response.json()
        
        if save_debug_callback:
            save_debug_callback(
                {"url": url, "headers": headers, "body": request_data},
                response_data,
                response.status_code
            )
        
        if debug_callback:
            debug_callback("API Response", response_data)
        
        return response_data


class AnthropicProvider:
    """Handle Anthropic API interactions"""
    
    @staticmethod
    def format_tools(tools: List[Dict]) -> List[Dict]:
        """Format tools for Anthropic API"""
        return tools
    
    @staticmethod
    async def send_message(
        provider_config: Dict,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        debug_callback=None,
        save_debug_callback=None
    ) -> Dict:
        """Send message to Anthropic API"""
        request_data = {
            "model": provider_config["model"],
            "max_tokens": 4096,
            "messages": messages
        }
        if tools:
            request_data["tools"] = tools
        
        headers = {
            "x-api-key": provider_config["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        return await send_api_request(
            provider_config["api_url"],
            headers,
            request_data,
            debug_callback,
            save_debug_callback
        )
    
    @staticmethod
    async def handle_tool_calls(
        assistant_message: Dict,
        tool_mapping: Dict,
        tool_executor: Callable,
        conversation_history: List[Dict],
        provider_config: Dict,
        tools: List[Dict],
        path_normalizer: Callable = None,
        debug_callback=None,
        save_debug_callback=None
    ) -> tuple[str, List[Dict]]:
        """
        Handle Anthropic tool calls in a loop
        Returns: (response_text, updated_history)
        """
        response_text = ""
        current_message = assistant_message
        updated_history = conversation_history.copy()
        
        headers = {
            "x-api-key": provider_config["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        while True:
            has_tool_use = False
            
            for block in current_message.get("content", []):
                if block["type"] == "text":
                    response_text += block["text"]
                elif block["type"] == "tool_use":
                    has_tool_use = True
                    
                    # Execute tool with approval
                    tool_result = await execute_tool_with_approval(
                        block["name"],
                        block["input"],
                        tool_mapping,
                        tool_executor,
                        path_normalizer
                    )
                    
                    # Update history
                    if not updated_history or updated_history[-1]["role"] != "assistant":
                        updated_history.append({"role": "assistant", "content": current_message["content"]})
                    
                    # Send tool result back
                    tool_result_content = [{
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(tool_result)
                    }]
                    
                    continue_request = {
                        "model": provider_config["model"],
                        "max_tokens": 4096,
                        "messages": updated_history + [{"role": "user", "content": tool_result_content}]
                    }
                    if tools:
                        continue_request["tools"] = tools
                    
                    if debug_callback:
                        debug_callback("Continuing with tool result", continue_request)
                    
                    continue_result = await send_api_request(
                        provider_config["api_url"],
                        headers,
                        continue_request,
                        debug_callback,
                        save_debug_callback
                    )
                    
                    if "error" in continue_result:
                        return response_text, updated_history
                    
                    updated_history.append({"role": "user", "content": tool_result_content})
                    updated_history.append({"role": "assistant", "content": continue_result["content"]})
                    
                    current_message = continue_result
                    break
            
            if not has_tool_use:
                break
        
        return response_text, updated_history


class OpenAIProvider:
    """Handle OpenAI-compatible API interactions (Groq, LM Studio, etc.)"""
    
    @staticmethod
    def format_tools(tools: List[Dict]) -> List[Dict]:
        """Format tools for OpenAI API"""
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
        return openai_tools
    
    @staticmethod
    def convert_messages(messages: List[Dict]) -> List[Dict]:
        """Convert Anthropic message format to OpenAI format"""
        openai_messages = []
        for msg in messages:
            if isinstance(msg["content"], list):
                text_parts = [block["text"] for block in msg["content"] if block.get("type") == "text"]
                openai_messages.append({
                    "role": msg["role"],
                    "content": " ".join(text_parts)
                })
            else:
                openai_messages.append(msg)
        return openai_messages
    
    @staticmethod
    async def send_message(
        provider_config: Dict,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        debug_callback=None,
        save_debug_callback=None
    ) -> Dict:
        """Send message to OpenAI-compatible API"""
        openai_messages = OpenAIProvider.convert_messages(messages)
        
        request_data = {
            "model": provider_config["model"],
            "messages": openai_messages
        }
        
        if tools:
            request_data["tools"] = OpenAIProvider.format_tools(tools)
        
        headers = {"content-type": "application/json"}
        if provider_config.get("api_key"):
            headers["Authorization"] = f"Bearer {provider_config['api_key']}"
        
        return await send_api_request(
            provider_config["api_url"],
            headers,
            request_data,
            debug_callback,
            save_debug_callback
        )
    
    @staticmethod
    async def handle_tool_calls(
        result: Dict,
        tool_mapping: Dict,
        tool_executor: Callable,
        conversation_history: List[Dict],
        provider_config: Dict,
        tools: List[Dict],
        user_message: str,
        path_normalizer: Callable = None,
        debug_callback=None,
        save_debug_callback=None
    ) -> tuple[str, List[Dict]]:
        """
        Handle OpenAI-style tool calls in a loop (like Anthropic)
        Returns: (response_text, updated_history)
        """
        updated_history = conversation_history.copy()
        current_result = result
        response_text = ""
        
        headers = {"content-type": "application/json"}
        if provider_config.get("api_key"):
            headers["Authorization"] = f"Bearer {provider_config['api_key']}"
        
        # Loop until no more tool calls
        while True:
            message_data = current_result.get("choices", [{}])[0].get("message", {})
            
            # Collect any text response
            if message_data.get("content"):
                response_text += message_data["content"]
            
            # Check if there are tool calls
            if "tool_calls" not in message_data or not message_data["tool_calls"]:
                # No more tool calls, we're done
                break
            
            # Process each tool call in this response
            tool_results_to_send = []
            
            for tool_call in message_data["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # Execute tool with approval (common function)
                tool_result = await execute_tool_with_approval(
                    tool_name,
                    tool_args,
                    tool_mapping,
                    tool_executor,
                    path_normalizer
                )
                
                # Collect tool result for this call
                tool_results_to_send.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result)
                })
            
            # Add assistant message with tool calls to history
            updated_history.append({
                "role": "assistant",
                "content": message_data.get("content"),
                "tool_calls": message_data["tool_calls"]
            })
            
            # Add all tool results to history
            updated_history.extend(tool_results_to_send)
            
            # Make follow-up request with tool results
            follow_up_request = {
                "model": provider_config["model"],
                "messages": updated_history
            }
            if tools:
                follow_up_request["tools"] = OpenAIProvider.format_tools(tools)
            
            if debug_callback:
                debug_callback("Follow-up request with tool results", follow_up_request)
            
            follow_up_result = await send_api_request(
                provider_config["api_url"],
                headers,
                follow_up_request,
                debug_callback,
                save_debug_callback
            )
            
            if "error" in follow_up_result:
                # Error occurred, return what we have
                return response_text if response_text else "Error in follow-up request", updated_history
            
            # Update current_result for next iteration
            current_result = follow_up_result
        
        # Add final assistant message (without tool calls)
        if message_data.get("content"):
            updated_history.append({
                "role": "assistant",
                "content": message_data.get("content")
            })
        
        return response_text if response_text else "Tool executed successfully.", updated_history
