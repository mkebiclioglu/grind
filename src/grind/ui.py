import json
import sys

from rich.console import Console
from rich.markup import escape

from . import config

DIM, RESET, ITAL = "\x1b[2m", "\x1b[0m", "\x1b[3m"


class UI:
    def __init__(self, interactive=True, show_reasoning=True):
        self.console = Console(highlight=False)
        self.interactive = interactive and sys.stdin.isatty()
        self.show_reasoning = show_reasoning
        self._mode = None  # None | "text" | "reasoning"
        self._session = None

    # -- streaming ------------------------------------------------------------

    def on_event(self, ev):
        out = sys.stdout
        if ev.kind == "reasoning":
            if not self.show_reasoning:
                if self._mode != "reasoning":
                    self._switch("reasoning")
                    out.write(f"{DIM}thinking…{RESET}")
                return
            if self._mode != "reasoning":
                self._switch("reasoning")
                out.write(DIM + ITAL)
            out.write(ev.text)
        elif ev.kind == "text":
            if self._mode != "text":
                self._switch("text")
            out.write(ev.text)
        elif ev.kind == "retry":
            self._switch(None)
            self.console.print(f"[yellow]⟳ {escape(ev.text)}[/]")
        out.flush()

    def _switch(self, mode):
        if self._mode is not None:
            sys.stdout.write(RESET + "\n")
            if self._mode == "reasoning":
                sys.stdout.write("\n")
        self._mode = mode

    def end_stream(self):
        if self._mode is not None:
            sys.stdout.write(RESET + "\n")
            sys.stdout.flush()
        self._mode = None

    # -- tools ------------------------------------------------------------------

    def tool_call(self, name, args):
        self.end_stream()
        if name == "bash":
            detail = args.get("command", "")
        elif name in ("read_file", "write_file", "edit_file"):
            detail = args.get("path", "")
            if name == "read_file" and args.get("offset"):
                detail += f" @{args['offset']}"
        elif name in ("glob", "grep"):
            detail = args.get("pattern", "") + (f"  in {args['path']}" if args.get("path") else "")
        elif name == "fetch_url":
            detail = args.get("url", "")
        elif name in ("update_plan", "goal_done", "ask_user"):
            return
        else:
            detail = json.dumps(args)[:200]
        first, *rest = (detail or "").splitlines() or [""]
        more = f" [dim](+{len(rest)} lines)[/]" if rest else ""
        self.console.print(f"[bold cyan]●[/] [bold]{name}[/] {escape(first[:220])}{more}")

    def tool_result(self, text, is_error):
        lines = (text or "").splitlines()
        shown = lines[:6]
        style = "red" if is_error else "dim"
        for l in shown:
            self.console.print(f"  [{style}]│ {escape(l[:200])}[/]")
        if len(lines) > len(shown):
            self.console.print(f"  [dim]│ … {len(lines) - len(shown)} more lines[/]")

    def plan(self, items):
        mark = {"done": "[green]✔[/]", "in_progress": "[yellow]▸[/]", "pending": "[dim]○[/]"}
        self.console.print("[bold]Plan[/]")
        for i in items:
            self.console.print(f"  {mark.get(i.get('status'), '○')} {escape(str(i.get('step')))}")

    # -- messages -------------------------------------------------------------

    def info(self, msg):
        self.end_stream()
        self.console.print(f"[dim]{escape(msg)}[/]")

    def warn(self, msg):
        self.end_stream()
        self.console.print(f"[yellow]{escape(msg)}[/]")

    def error(self, msg):
        self.end_stream()
        self.console.print(f"[red]{escape(msg)}[/]")

    def goal_finished(self, goal):
        self.end_stream()
        self.console.print("[bold green]✔ Goal complete.[/]")

    # -- input ------------------------------------------------------------------

    def _input(self, prompt):
        if not self.interactive:
            return None
        try:
            return input(prompt)
        except EOFError:
            return None

    def approve(self, name, args):
        if not self.interactive:
            return "n: no user available to approve; approval mode is not auto"
        if name == "bash":
            self.console.print(f"[dim]{escape(args.get('command', ''))}[/]")
        try:
            a = self._input(f"Allow {name}? [y]es / [n]o / [a]lways / or type feedback: ")
        except KeyboardInterrupt:
            return "stop"
        a = (a or "").strip()
        return {"y": "y", "yes": "y", "": "y", "a": "a", "always": "a", "n": "n", "no": "n"}.get(a.lower(), a)

    def ask(self, question):
        self.console.print(f"[bold magenta]?[/] {escape(question)}")
        try:
            return (self._input("> ") or "").strip()
        except KeyboardInterrupt:
            return ""

    def paused(self, goal_mode, reason):
        """None = stop, "" = resume, text = new instructions."""
        if not self.interactive:
            return None
        if goal_mode:
            self.console.print(f"[bold yellow]⏸ Paused ({reason}).[/] Type new instructions, Enter to resume, "
                               "or /stop to end the goal.")
        else:
            self.console.print(f"[yellow]⏸ {reason}.[/] Type instructions to continue, or Enter to drop back to the prompt.")
        try:
            t = self._input("» ")
        except KeyboardInterrupt:
            return None
        if t is None or t.strip() in ("/stop", "/exit"):
            return None
        return t.strip()

    def prompt(self, label):
        if not sys.stdin.isatty():
            line = sys.stdin.readline()
            return line.rstrip("\n") if line else None
        if self._session is None:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings

            kb = KeyBindings()

            @kb.add("escape", "enter")
            def _(event):
                event.current_buffer.insert_text("\n")

            config.DATA.mkdir(parents=True, exist_ok=True)
            self._session = PromptSession(history=FileHistory(str(config.DATA / "history")), key_bindings=kb)
        try:
            return self._session.prompt(label)
        except EOFError:
            return None
