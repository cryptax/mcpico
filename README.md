# McPico - MCP client for the terminal

**M**c**P**ico, or **MCP**ico if you prefer, is a lightweight MCP client for the terminal.

It supports:

- File attachment :white_check_mark:
- MCP servers stdio-based :white_check_mark:
- MCP servers http-based: *to test and fix*
- Anthropic + r2mcp :white_check_mark:
- Groq + r2mcp: :white_check_mark:
- LM Studio + r2mcp: :white_check_mark:
- Multi-MCP *to test*

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

Get your inspiration from the default configuration in `mcpico.py`.
More info [here](./doc/config.md).

## Run it

`python3 mcpico.py`

Then use `/help` for various commands.

## Demo
  
View the [asciinema](https://asciinema.org/) videos in the `./demo` directory, or online.

[![asciicast](https://asciinema.org/a/QXMTovZacPPSAJsObPZN32Zj8.svg)](https://asciinema.org/a/QXMTovZacPPSAJsObPZN32Zj8)
[![asciicast](https://asciinema.org/a/755841.svg)](https://asciinema.org/a/755841)

