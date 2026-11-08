# grind

A small terminal agent harness that works with **any chat model** and has a **goal mode**: give it a goal and it keeps
working (shell, files, search, web) until the goal is done and verified, or until you interrupt and redirect it.

It also ships a one-command **RunPod vLLM server** so you can run open models such as
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a rented GPU, with a watchdog that deletes the pod when
you stop using it.

```
grind "make the test suite pass"          # autonomous goal, then an interactive prompt
grind                                     # interactive chat with tools
grind -m claude -p "summarize README.md"  # one-shot
grind pod up                              # start Qwen3.8-27B on RunPod (~15 min to ready)
```

## Why not omnigent?

[omnigent](https://github.com/omnigent-ai/omnigent) is a meta-harness: a server, web UI and orchestration layer
(~600k lines) that wraps other agents (Claude Code, Codex, OpenCode, Qwen Code, ...). It's great when you want
multi-agent supervision from any device. For "one terminal, any model, keep going until done" it's a lot of
machinery, and its Claude Code wrapper is the one piece that touches `~/.claude`. grind is ~1,000 lines you can read
in one sitting, owns its own loop, and never touches your Claude Code setup. The pod server speaks the OpenAI API,
so omnigent (or OpenCode, Qwen Code, Codex, ...) can use it too if you want to try them later.

## Install

```bash
uv tool install --editable ~/Projects/grind     # puts `grind` on PATH (~/.local/bin)
```

Config is `~/.config/grind/config.toml`; sessions are in `~/.local/share/grind/sessions`.

## Goal mode

`grind "<goal>"` or `/goal <goal>` inside the prompt. The agent:

1. gets the goal plus instructions to keep going until it's achieved and verified;
2. loops: model call → tool calls → results → model call ...;
3. if the model ends a turn without calling `goal_done`, grind nudges it to continue (and pauses for you after 3
   stalls in a row);
4. finishes only when the model calls `goal_done` with a summary of what it did and how it verified it.

**Ctrl-C** at any time interrupts the model or the running command and pauses. Type new instructions (they go in
as an interjection and the goal continues), press Enter to resume, or `/stop` to end the goal. Sessions are saved
after every step: `grind -r` resumes the latest session in the current directory and `/continue` resumes its goal.

Long goals don't run out of context: at 70% of the model's window grind asks the model for a handoff summary
and continues from that (the goal and the `update_plan` checklist are carried over).

## Approvals

| mode | file edits in cwd | edits outside cwd | shell / fetch |
|------|------|------|------|
| `ask` | ask | ask | ask |
| `edits` (default) | run | ask | ask |
| `auto` (`-y`) | run | run | run |

At an approval prompt, `a` switches the session to `auto`, and typing anything else denies the call and sends your
text to the model as feedback. For unattended goals, use `-y` in a directory you don't mind the agent changing.

## Models

Every profile in `config.toml` is one endpoint. `provider = "openai"` covers any OpenAI-compatible server;
`provider = "anthropic"` uses the Anthropic API. Built-in profiles:

- `qwen`: the RunPod vLLM server (filled in by `grind pod up`)
- `claude`: Claude Opus 5.5 (needs `ANTHROPIC_API_KEY`)

Add your own:

```toml
default_model = "qwen"
env_files = ["~/Desktop/.env"]     # KEY=VALUE files loaded at startup (never overrides real env vars)

[models.gpt]
provider = "openai"
base_url = "https://api.openai.com/v1"
model = "gpt-5.5"
api_key_env = "OPENAI_API_KEY"
effort_style = "openai"            # sends reasoning_effort
effort = "medium"
context_window = 400000

[models.openrouter]
provider = "openai"
base_url = "https://openrouter.ai/api/v1"
model = "qwen/qwen3.8-27b"
api_key_env = "OPENROUTER_API_KEY"

[models.local]
provider = "openai"
base_url = "http://localhost:11434/v1"   # Ollama, LM Studio, llama.cpp server, ...
model = "qwen3:8b"
context_window = 32768
```

Profile fields: `provider`, `base_url`, `model`, `api_key` or `api_key_env`, `context_window`, `max_tokens`,
`temperature`, `top_p`, `extra_body` (merged into the request), `effort` + `effort_style` (`qwen` → chat-template
kwargs, `openai` → `reasoning_effort`; Anthropic always uses `output_config.effort`), and `preserve_reasoning`
(send reasoning back in history, which Qwen3.8 uses for multi-turn agent work).

Switch mid-session with `/model <name>`; the history carries over.

## RunPod: open models on a rented GPU

```bash
grind pod presets               # models `up` can serve
grind pod up                    # default preset (Qwen3.8-27B), waits until vLLM is ready
grind pod up -m heretic-9b      # another preset
grind                           # use it (`up` makes the pod's profile the default model)
grind pod status                # balance, cost so far, idle timer
grind pod logs                  # vLLM log tail
grind pod down                  # delete it now
```

| preset | model | profile | notes |
|---|---|---|---|
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `qwen` | strongest coder; 31 GB |
| `heretic-9b` | [xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic](https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic) | `heretic` | DeepSeek-V4-Pro reasoning distilled into Qwen3.5-9B, refusals removed; 19 GB bf16, faster, weaker at code |

One pod runs at a time. Add a preset for any vLLM-servable Hugging Face model in `config.toml`; anything not
set falls back to the shared `[pod]` settings (image, parsers, context, limits):

```toml
[pod.presets.my-model]
hf_model = "org/model"
served_name = "my-model"
gpus = ["NVIDIA A40", "NVIDIA L40S"]
profile = "my-model"          # the grind profile `up` points at the pod
tool_parser = "hermes"        # vLLM --tool-call-parser for the model's tool format
reasoning_parser = ""         # "" for non-reasoning models
mtp = false

[models.my-model]
provider = "openai"
model = "my-model"
context_window = 32768
```

Startup is ~15 min: ~7 min image pull, ~5 min weight download, ~3 min load/compile. On an L40 it gives 64K
context (KV room for ~120K tokens). A small coding goal (write a CLI + 12 pytest tests, fix a failure, verify)
took ~1 minute end to end.

What `up` does:

- Serves **Qwen/Qwen3.8-27B-FP8** (31 GB) with the stock `vllm/vllm-openai` image on the cheapest available 48 GB
  GPU (A40 → A6000 → L40S → RTX 6000 Ada → L40, ~$0.35–0.80/hr), 64K context, `qwen3_coder` tool parsing, the
  `qwen3` reasoning parser, and MTP speculative decoding.
- Generates a random API key for vLLM, so the public RunPod proxy URL isn't an open endpoint.
- Starts a watchdog inside the pod that **deletes the pod** after 30 min idle (`--idle`), a 3 h hard cap
  (`--max-hours`), 10 min after a vLLM crash, or 40 min without getting ready. It uses the pod-scoped key RunPod
  injects into every pod, so your account API key never leaves your machine.
- Uses no network volume, so a deleted pod costs nothing. The trade-off is re-downloading the weights on each `up`
  (a few minutes on RunPod's network). The `[pod]` table in `config.toml` overrides the model, image, GPU list,
  context length and limits.

Your RunPod key is read from `RUNPOD_API_KEY` (or `RUN_POD_API_KEY`), from the environment or an `env_files` entry.

Community-cloud pods run on third-party hosts. Don't send secrets through a model hosted there.

## Layout

```
src/grind/
  agent.py      loop, goal mode, interrupts, approvals, compaction, sessions
  providers.py  OpenAI-compatible + Anthropic streaming adapters
  tools.py      bash, read/write/edit, glob, grep, fetch_url, update_plan, ask_user, goal_done
  ui.py         terminal rendering and input
  app.py        CLI + REPL
  pod.py        RunPod lifecycle
  watchdog.py   runs inside the pod: status/logs endpoint + self-delete
```

`uv run pytest` runs the offline tests (a scripted fake model; no network).
