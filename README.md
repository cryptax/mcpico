# McPico - MCP client for the terminal

**M**c**P**ico, or **MCP**ico if you prefer, is a lightweight MCP client for the terminal.

It supports:

- File attachment :white_check_mark:
- MCP servers http based or stdio-based :white_check_mark:
- Multi-MCP *to test*
- Anthropic + r2mcp :white_check_mark:
- Groq + r2mcp: *to test and fix*
- LM Studio + r2mcp: *to test and fix*

Most of the code has been written by AI... and/but it works pretty well. It has been tested on Linux.

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
  
View the [asciinema](https://asciinema.org/) videos in the `./demo` directory:

- `mcpico-use.cast`: how to use McPico and select a given provider/model
- `mcpico-file.cast`: how to attach a file
- `mcpico-mcp.cast`: example of MCP use with McPico. McPico is configured to use a [Radare2 MCP](https://github.com/radareorg/radare-mcp), and it locates the main inside a binary.
