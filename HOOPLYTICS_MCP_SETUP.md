# 🏀 Hooplytics MCP — Setup & Config Guide

## What this gives you in Claude

Once connected, Claude can answer questions like:

> *"Project LeBron James' next game and tell me which stat has the biggest edge."*
> *"What's the MORE/LESS call on Shai Gilgeous-Alexander points tonight? The line is 31.5."*
> *"Project Cade Cunningham as if he's playing 38 minutes a night — what does Hooplytics project for his fantasy score?"*
> *"Generate a Scout report on Victor Wembanyama rebounds using Claude."*
> *"What's tonight's slate brief? Rank the loudest mispricings on my roster."*
> *"Add Anthony Edwards to my roster and show me his analytics for the last 10 games."*

---

## Prerequisites

```bash
# 1. Clone the Hooplytics repo (if not already done)
git clone https://github.com/texasbe2trill/hooplytics.git
cd hooplytics

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Hooplytics + MCP SDK
pip install -e ".[web]"            # installs anthropic + openai SDKs too
pip install "mcp[cli]"             # MCP Python SDK

# 4. Set your API keys in .env (same file Hooplytics already uses)
echo "ODDS_API_KEY=your_key_here"        >> .env
echo "ANTHROPIC_API_KEY=your_key_here"   >> .env   # for Scout + Slate Brief
echo "OPENAI_API_KEY=your_key_here"      >> .env   # optional second provider
```

> `hooplytics_mcp_server.py` already ships in the repo root — no copy step needed.

---

## Claude Desktop Configuration

**Config file locations:**

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

> **Before editing:** Run `pwd` (macOS/Linux) or `cd` (Windows) inside the repo to get the absolute path you'll need below.

### If your config file is new or empty

Create it with this content (replace the paths and keys):

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

**Windows:** use `.venv\Scripts\python.exe` and backslashes in the path.

### If your config file already has content

Add only the `"hooplytics"` entry inside the existing `"mcpServers"` object — do **not** paste a second top-level `{` block, as that produces a JSON parse error:

```json
{
  "existing-key": { ... },
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

> **OpenAI key:** Only needed if you use `provider="openai"` in Scout reports or Slate Brief. Omit the `OPENAI_API_KEY` line if you only use Anthropic.

> **Roster persistence:** The MCP roster is shared with the Hooplytics CLI at `~/.hooplytics/roster.json` — adding a player via Claude shows up in `hooplytics roster list` and vice versa. Set `HOOPLYTICS_ROSTER_PATH` in the `env` block above to override the location.

---

## Remote / SSE Mode (optional)

If you want to expose Hooplytics MCP to a web client or Claude.ai (via custom connector):

```bash
# Start SSE server on port 8765
python hooplytics_mcp_server.py --transport sse --port 8765
```

The server runs FastMCP's built-in SSE transport directly — no separate ASGI host needed.

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `project_player` | Next-game projections across all 8 RACE models |
| `analyze_prop` | Single-stat MORE/LESS call vs a sportsbook line |
| `player_decisions` | Full 8-stat decision table with live line comparison |
| `score_scenario` | Score a hypothetical box-score row |
| `live_line_board` | Full roster edge board, sorted by projection gap |
| `generate_scout_report` | AI-generated analytical prose (Anthropic or OpenAI) |
| `player_analytics` | Game log, trends, volatility, vs-baseline comparison |
| `roster_manage` | Add, remove, list, or reset the tracked player roster |
| `slate_brief` | One-paragraph AI daily slate read on tonight's mispricings |

---

## Example Claude Prompts

### Projections
```
Project Anthony Edwards' next game. Which of the 8 models is most
bullish and which is most bearish?
```

### Prop Analysis
```
The line on Nikola Jokic rebounds tonight is 12.5.
Run the Hooplytics prop analysis and tell me if there's an edge.
```

### Scenario What-If
```
Score this hypothetical scenario for Cade Cunningham:
What does Hooplytics project if he averages 38 minutes,
shoots 48% from the field, and takes 8 threes per game?
```

> **Note:** `score_scenario` scores a feature row through the RACE models. It works best with rolling-window or per-game-rate inputs (`min_l5`, `fg_pct`, `fg3a_l10`, etc.) rather than a single raw box-score line. For a raw "what if tonight" question, `project_player` with `last_n` is the better tool.

### Scout Report
```
Generate a Hooplytics Scout report on Jayson Tatum's points
for tonight's game. Use Anthropic and hybrid grounding mode.
```

### Slate Brief
```
Give me tonight's slate brief from Hooplytics.
Which players on my roster have the biggest mispricings?
```

### Analytics Deep Dive
```
Pull the last 10 games for De'Aaron Fox. Is he trending
above or below his season baseline on assists?
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: hooplytics` | Run `pip install -e .` inside the repo with the venv active |
| `ModuleNotFoundError: mcp` | Run `pip install "mcp[cli]"` |
| `ODDS_API_KEY not set` | Add it to `.env` or the `env` block in `claude_desktop_config.json` |
| `Odds API 401 Unauthorized` | Your `ODDS_API_KEY` is invalid or expired. Refresh it at [the-odds-api.com](https://the-odds-api.com/) |
| `[Errno 30] Read-only file system: 'data'` | The server is not running from the repo root. Ensure you're using the latest `hooplytics_mcp_server.py` — it auto-corrects its working directory on startup |
| `Unexpected non-whitespace character after JSON` | You pasted the `mcpServers` block as a second top-level object. Merge it into your existing config instead — see the "already has content" example above |
| Claude shows "No tools available" | Fully quit Claude Desktop (macOS: Cmd+Q) and relaunch — a simple window close does not reload MCP servers |
| Slow first response | First call loads `bundles/race_fast.joblib` from disk — usually fast. If that file is missing, models train from scratch on first call and may take several minutes |
| Player not found | Try a more complete name; Hooplytics uses RapidFuzz for fuzzy matching |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Desktop                       │
│           (or any MCP-compatible client)                │
└───────────────────────┬─────────────────────────────────┘
                        │  MCP Protocol (stdio / SSE)
                        ▼
┌─────────────────────────────────────────────────────────┐
│           hooplytics_mcp_server.py                      │
│                                                         │
│  project_player ──────────► hooplytics.predict          │
│  analyze_prop ─────────────► hooplytics.predict         │
│  player_decisions ─────────► hooplytics.predict         │
│  score_scenario ───────────► hooplytics.predict         │
│  live_line_board ──────────► hooplytics.odds            │
│  generate_scout_report ────► hooplytics.ai_agent        │
│  player_analytics ─────────► hooplytics.data            │
│  roster_manage ────────────► local JSON roster          │
│  slate_brief ──────────────► odds + predict + ai_agent  │
└───────────────┬───────────────────────────────┬─────────┘
                │                               │
                ▼                               ▼
┌──────────────────────┐          ┌─────────────────────────┐
│   Hooplytics Core    │          │     External APIs        │
│                      │          │                          │
│  RACE ML Models      │          │  nba_api (game logs)     │
│  Calibration layer   │          │  The Odds API (lines)    │
│  Feature pipelines   │          │  Anthropic API (prose)   │
│  PlayerStore cache   │          │  OpenAI API (optional)   │
└──────────────────────┘          └─────────────────────────┘
```
