# 🏀 Hooplytics MCP — Setup Guide

## What you can ask Claude

- *"Project LeBron James' next game — which stat has the biggest edge?"*
- *"Is there value on Jokic rebounds at 12.5 tonight?"*
- *"Give me tonight's slate brief and rank the loudest mispricings on my roster."*

---

## Setup (5 minutes)

### Step 1 — Clone the repo and install dependencies

```bash
git clone https://github.com/texasbe2trill/hooplytics.git
cd hooplytics
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[web]"
pip install "mcp[cli]"
```

### Step 2 — Add your API keys

Create a `.env` file in the repo root:

```
ODDS_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

> Get your Odds API key at [the-odds-api.com](https://the-odds-api.com/) and your Anthropic key at [console.anthropic.com](https://console.anthropic.com/).

### Step 3 — Connect Claude Desktop

Open (or create) the Claude Desktop config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the `hooplytics` block inside `"mcpServers"` (replace the paths):

```json
{
  "mcpServers": {
    "hooplytics": {
      "command": "/absolute/path/to/hooplytics/.venv/bin/python",
      "args": ["/absolute/path/to/hooplytics/hooplytics_mcp_server.py"],
      "env": {
        "ODDS_API_KEY": "your_odds_api_key",
        "ANTHROPIC_API_KEY": "your_anthropic_key"
      }
    }
  }
}
```

> **Find your absolute path:** Run `pwd` in the repo directory (macOS/Linux) or `cd` (Windows).
> **Windows path:** use `.venv\Scripts\python.exe` and backslashes.

### Step 4 — Relaunch Claude Desktop

Fully quit Claude Desktop (macOS: **Cmd+Q**) and reopen it. A simple window close is not enough.

That's it — Hooplytics tools will appear in Claude automatically.

---

## Available Tools

| Tool | What it does |
|------|-------------|
| `project_player` | Next-game projections across all 8 RACE models |
| `analyze_prop` | MORE/LESS call vs a sportsbook line |
| `player_decisions` | Full 8-stat decision table with live line comparison |
| `score_scenario` | Score a hypothetical stat scenario |
| `live_line_board` | Full roster edge board sorted by projection gap |
| `generate_scout_report` | AI-written analytical prose (Anthropic or OpenAI) |
| `player_analytics` | Game log, trends, and volatility breakdown |
| `roster_manage` | Add, remove, list, or reset your tracked roster |
| `slate_brief` | AI daily slate summary with mispricing rankings |

---

## Example Prompts

- *"Project Anthony Edwards' next game."*
- *"The line on Jokic rebounds is 12.5 — is there an edge?"*
- *"Generate a scout report on Jayson Tatum's points for tonight."*
- *"Give me tonight's slate brief. Which players on my roster have the biggest mispricings?"*
- *"Pull the last 10 games for De'Aaron Fox and show me his assist trend."*

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: hooplytics` | Run `pip install -e .` with the venv active |
| `ModuleNotFoundError: mcp` | Run `pip install "mcp[cli]"` |
| `ODDS_API_KEY not set` | Add the key to `.env` or the `env` block in the config |
| `Odds API 401` | Key is invalid — refresh it at [the-odds-api.com](https://the-odds-api.com/) |
| Claude shows "No tools available" | Fully quit Claude Desktop (Cmd+Q) and relaunch |
| Player not found | Use the full player name; Hooplytics does fuzzy matching |
