Configuration file: `~/.config/mcpico/config.json`

| providers | List of providers |
| mcp_servers | List of MCP servers |
| debug | Log requests and responses to `/tmp/mcpico` |
| current_provider | Last used provider |

# Configuring providers

In the configuration json file, there is a `providers` section where you can list all providers you wish to use.

Each provider has an API key, a default model name, a URL, and a type.

Currently, we only support types `openai`, `anthropic`.


| api_key | Paste your API Key here |
| model   | Default model name to use. Leave blank if you don't know |
| api_url | URL to contact for chat completions |
| models_url | If you have a special URL to list models, specify it here. Otherwise, the program will try and guess the correct URL (e.g `api_url/v1/models`, or replace `chat/completions` with `models`) |
| type    | We only support `anthropic` or `openai`. Note it's a *type*. The type `openai` works for several providers e.g groq and lm studio |


## Example for Groq

```json

{
  "providers": {
    "groq": {
      "api_key": "PASTE-YOUR-GROQ-KEY-HERE",
      "model": "qwen/qwen3-32b",
      "api_url": "https://api.groq.com/openai/v1/chat/completions",
      "type": "openai"
    }

}
```

## Example for Anthropic

```json
{
  "providers": {
    "anthropic": {
      "api_key": "PASTE-YOUR-KEY-HERE",
      "model": "claude-sonnet-4-20250514",
      "api_url": "https://api.anthropic.com/v1/messages",
      "type": "anthropic"
    },
}
```

## Example for LM Studio

```json
    "lmstudio": {
      "api_key": "",
      "model": "local-model",
       "api_url": "http://HOST:PORT/v1/chat/completions",
       "models_url": "http://HOST:PORT/v1/models",
      "type": "openai"
    }
```    


# Configuring MCP servers


They are listed in `mcp_servers` of the configuration file. We support only 2 types of MCP servers: stdio and http.

## Example for r2mcp

```json
  "mcp_servers": {
    "r2mcp": {
      "type": "stdio",
      "command": "r2pm",
      "args": [
        "-r",
        "r2mcp"
      ]
    },
```

# Other configuration options

- `debug`: true/false. Logs requests and responses in `/tmp/mcpico`
- `current_provider`: name.

## Example

```json
{
  "current_provider": "groq",
  "providers": {
    "anthropic": {
      "api_key": "CENSORED",
      "model": "claude-sonnet-4-20250514",
      "api_url": "https://api.anthropic.com/v1/messages",
      "type": "anthropic"
    },
    "groq": {
      "api_key": "CENSORED",
      "model": "qwen/qwen3-32b",
      "api_url": "https://api.groq.com/openai/v1/chat/completions",
      "type": "openai"
    }
  },
  "mcp_servers": {
    "r2mcp": {
      "type": "stdio",
      "command": "r2pm",
      "args": [
        "-r",
        "r2mcp"
      ]
    }
  },
  "debug": true
}
```