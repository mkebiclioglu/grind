import argparse
import os
import sys

from rich.markup import escape

from . import __version__, config
from .agent import Agent, list_sessions
from .ui import UI

HELP = """[bold]Commands[/]
  /goal <text>     work autonomously until the goal is done (Ctrl-C to interrupt and redirect)
  /continue        resume a paused goal
  /stop            drop the current goal
  /model [name]    show or switch the model profile (history is kept)
  /models          list model profiles
  /approval MODE   ask | edits | auto   (edits = file edits in cwd run freely, shell asks)
  /effort LEVEL    reasoning effort for the current model (e.g. low | medium | high | xhigh | off)
  /thinking        toggle showing the model's reasoning
  /plan            show the current plan
  /compact         summarize the conversation to free context
  /clear           start a new session
  /usage           token usage for this session
  /exit            quit (Ctrl-D also works)
Anything else is sent as a chat message. Esc+Enter inserts a newline."""


def build_parser():
    ap = argparse.ArgumentParser(
        prog="grind", description="Terminal agent for any chat model. `grind \"<goal>\"` works until the goal is done.",
        epilog="Subcommands: `grind pod up|down|status|logs|wait` (RunPod vLLM server), `grind models`, `grind sessions`.")
    ap.add_argument("goal", nargs="*", help="a goal to work on autonomously")
    ap.add_argument("-m", "--model", help="model profile name (see `grind models`)")
    ap.add_argument("-p", "--print", dest="prompt", help="one-shot: answer this prompt (with tools) and exit")
    ap.add_argument("-y", "--yolo", action="store_true", help="approval mode auto: never ask before running tools")
    ap.add_argument("--approval", choices=["ask", "edits", "auto"])
    ap.add_argument("-r", "--resume", nargs="?", const="latest", help="resume the latest session here, or a session id")
    ap.add_argument("-C", "--cwd", default=".", help="working directory")
    ap.add_argument("--max-steps", type=int, default=0, help="pause a goal after this many model calls (0 = no limit)")
    ap.add_argument("-x", "--exit", action="store_true", help="exit when the goal finishes instead of opening the prompt")
    ap.add_argument("--hide-thinking", action="store_true")
    ap.add_argument("-V", "--version", action="version", version=f"grind {__version__}")
    return ap


def show_models(cfg, current=None):
    for name, p in cfg["models"].items():
        mark = "*" if name == (current or cfg.get("default_model")) else " "
        where = p.get("base_url") or ("api.anthropic.com" if p.get("provider") == "anthropic" else "(base_url not set)")
        key = "key ok" if config.api_key(p) else "no key"
        print(f"{mark} {name:<12} {p.get('provider', 'openai'):<10} {p.get('model', ''):<28} {where}  [{key}]")


def run(argv):
    cfg = config.load()
    if argv[:1] == ["models"]:
        return show_models(cfg)
    if argv[:1] == ["sessions"]:
        for s in list_sessions()[:20]:
            print(f"{s['id']}  {s.get('profile', ''):<10} {s['cwd']}  {(s.get('goal') or '')[:60]}")
        return

    a = build_parser().parse_args(argv)
    cwd = os.path.abspath(os.path.expanduser(a.cwd))
    approval = "auto" if a.yolo else a.approval
    ui = UI(interactive=a.prompt is None, show_reasoning=not a.hide_thinking)

    session = None
    if a.resume:
        found = list_sessions(None if a.resume != "latest" else cwd)
        if a.resume != "latest":
            found = [s for s in found if s["id"].startswith(a.resume)]
        if not found:
            sys.exit("No session to resume.")
        session = found[0]
        cwd = session["cwd"]

    profile = a.model or (session or {}).get("profile") or cfg.get("default_model")
    try:
        agent = Agent(cfg, profile, cwd, ui, approval=approval, session=session, max_steps=a.max_steps)
    except KeyError as e:
        sys.exit(str(e))
    _check_profile(agent, ui)

    if a.prompt is not None:
        agent.chat(a.prompt)
        return

    ui.console.print(f"[bold]grind[/] {__version__} · model [bold]{agent.profile_name}[/] "
                     f"({escape(agent.profile.get('model', ''))}) · approval {agent.approval} · {escape(cwd)}")
    if session:
        ui.console.print(f"[dim]Resumed session {session['id']} ({len(agent.messages)} messages).[/]")
        if agent.goal:
            ui.console.print(f"[dim]Unfinished goal: {escape(agent.goal[:200])}. /continue to resume it.[/]")
    else:
        ui.console.print("[dim]/help for commands. Ctrl-C interrupts the agent so you can redirect it.[/]")

    if a.goal:
        _guard(lambda: agent.start_goal(" ".join(a.goal)), ui)
        if a.exit or not sys.stdin.isatty():
            return
    repl(agent, cfg, ui)


def _check_profile(agent, ui):
    p = agent.profile
    if p.get("provider", "openai") == "openai" and not p.get("base_url"):
        ui.warn(f"Profile {agent.profile_name!r} has no base_url. "
                + ("Run `grind pod up` to start the RunPod server." if agent.profile_name == cfg_pod_profile(agent.cfg) else ""))
    elif not config.api_key(p) and p.get("api_key_env"):
        ui.warn(f"Profile {agent.profile_name!r}: ${p['api_key_env']} is not set.")


def cfg_pod_profile(cfg):
    return cfg.get("pod", {}).get("profile", "qwen")


def _guard(fn, ui):
    try:
        fn()
    except KeyboardInterrupt:
        ui.end_stream()
        ui.info("Interrupted.")
    except Exception as e:  # noqa: BLE001 - keep the REPL alive on API errors
        ui.error(f"{type(e).__name__}: {e}")


def repl(agent, cfg, ui):
    while True:
        try:
            line = ui.prompt(f"{agent.profile_name}{' [goal]' if agent.goal else ''} › ")
        except KeyboardInterrupt:
            continue
        if line is None or line.strip() in ("/exit", "/quit"):
            return
        line = line.strip()
        if not line:
            continue
        if not line.startswith("/"):
            _guard(lambda: agent.chat(line), ui)
            continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/help":
            ui.console.print(HELP)
        elif cmd == "/goal":
            if not arg:
                ui.console.print(f"Current goal: {escape(agent.goal or '(none)')}")
            else:
                _guard(lambda: agent.start_goal(arg), ui)
        elif cmd == "/continue":
            if agent.goal:
                _guard(agent.resume_goal, ui)
            else:
                ui.info("No goal to continue.")
        elif cmd == "/stop":
            agent.goal = None
            agent.save()
            ui.info("Goal cleared.")
        elif cmd == "/models":
            show_models(cfg, agent.profile_name)
        elif cmd == "/model":
            if not arg:
                ui.info(f"{agent.profile_name}: {agent.profile.get('model')}")
            else:
                try:
                    agent.set_profile(arg)
                    _check_profile(agent, ui)
                    ui.info(f"Switched to {arg} ({agent.profile.get('model')}).")
                except KeyError as e:
                    ui.error(str(e))
        elif cmd == "/approval":
            if arg in ("ask", "edits", "auto"):
                agent.approval = arg
            ui.info(f"Approval mode: {agent.approval}")
        elif cmd == "/effort":
            if arg:
                agent.profile["effort"] = arg
                agent.set_profile(agent.profile_name)
            ui.info(f"Effort: {agent.profile.get('effort', '(default)')}")
        elif cmd == "/thinking":
            ui.show_reasoning = not ui.show_reasoning
            ui.info(f"Show reasoning: {ui.show_reasoning}")
        elif cmd == "/plan":
            ui.plan(agent.tools.plan) if agent.tools.plan else ui.info("No plan.")
        elif cmd == "/compact":
            _guard(lambda: agent.maybe_compact(bool(agent.goal), force=True), ui)
        elif cmd == "/clear":
            agent = Agent(cfg, agent.profile_name, agent.tools.cwd, ui, approval=agent.approval, max_steps=agent.max_steps)
            ui.info("New session.")
        elif cmd == "/usage":
            u = agent.usage
            ui.info(f"input {u['input_tokens']:,} · output {u['output_tokens']:,} · last context {agent.last_context:,} "
                    f"of {agent.profile.get('context_window', '?'):,}")
        else:
            ui.warn(f"Unknown command {cmd}. /help lists commands.")
