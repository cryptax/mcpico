# McPico - MCP client for the terminal

**M**c**P**ico, or **MCP**ico if you prefer, is a lightweight MCP client for the terminal.

It supports:

- File attachment :white_check_mark:
- MCP servers http based or stdio-based :white_check_mark:
- Multi-MCP *to test*
- Anthropic :white_check_mark:
- Groq :white_check_mark:
- LM Studio *to test*

Most of the code has been written by AI... and/but it works pretty well. It has been tested on Linux.

- Nov 8, 2025. This is **alpha** stage. I'll work on it, and improve it in the following days.

## Installation

```
git clone https://github.com/cryptax/mcpico
pip install -r requirements.txt
```

## Configuration

In `~/.config/mcpico/config.json`, setup your API keys and URLs:

```json
{
  "current_provider": "anthropic",
  "providers": {
    "anthropic": {
      "api_key": "CENSORED",
      "model": "claude-sonnet-4-20250514",
      "api_url": "https://api.anthropic.com/v1/messages",
      "type": "anthropic"
    },
  ...
}
```

Get your inspiration from the default configuration in `mcpico.py`

## Use
  



