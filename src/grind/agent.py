import json
import platform
import time
import uuid
from datetime import date
from pathlib import Path

from . import config, providers
from .tools import GOAL_DONE, MUTATING, SCHEMAS, Tools

SYSTEM = """You are grind, an agent working in the user's terminal on {os}.
Working directory: {cwd}
Today's date: {date}

You act through tools: run shell commands, read/write/edit files, search, and fetch URLs.

How to work:
- Investigate before changing things. Work in small steps and check each one (run the code, the tests, the command).
- For multi-step work, keep a checklist with update_plan and update it as steps finish.
- Keep messages short. The user sees your tool calls; don't narrate them.
- Read a file before editing it. Use edit_file for changes to existing files.
- Do not do destructive or irreversible things (deleting data you didn't create, force-pushing, spending money,
  messaging other people) unless the user asked for exactly that.
- Stay inside the working directory unless the task requires otherwise.
- If a tool fails, read the error and adapt; don't repeat the same failing call.
{project}"""

GOAL_KICKOFF = """GOAL: {goal}

You are in autonomous goal mode. Keep working, step after step, until the goal is fully achieved and verified.
Then call goal_done with a summary of what you did and how you verified it.
- Make reasonable decisions yourself. Use ask_user only if you are blocked on something only the user can provide.
- The user may interrupt with new instructions; the latest instructions win.
- Don't stop to report progress or ask for permission to continue; just continue."""

NUDGE = ("You ended your turn without calling goal_done. If the goal is fully achieved and verified, call goal_done "
         "now. Otherwise continue with the next concrete step (use a tool).")

COMPACT = """The conversation is about to be compacted to save context. Write a handoff summary that lets you continue
seamlessly without the earlier messages. Include: the goal and any user instructions/preferences given along the way;
what has been done so far; the current state (files created/changed with paths, commands that work, test status);
open problems and what was tried; the exact next steps. Be specific and complete but concise. Do not call tools."""

PROJECT_FILES = ("AGENTS.md", "CLAUDE.md", "GRIND.md")


def _project_instructions(cwd):
    for name in PROJECT_FILES:
        p = Path(cwd) / name
        if p.is_file():
            return f"\nProject instructions from {name}:\n{p.read_text()[:20000]}\n"
    return ""


class Stop(Exception):
    """User asked to stop the current goal."""


class Agent:
    def __init__(self, cfg, profile_name, cwd, ui, approval=None, session=None, max_steps=0):
        self.cfg = cfg
        self.ui = ui
        self.tools = Tools(cwd)
        self.approval = approval or cfg.get("approval", "edits")
        self.max_steps = max_steps
        self.set_profile(profile_name)
        self.messages = []
        self.goal = None
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_context = 0
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        if session:
            self._load(session)
        self.system = SYSTEM.format(os=f"{platform.system()} {platform.machine()}", cwd=self.tools.cwd,
                                    date=date.today().isoformat(), project=_project_instructions(self.tools.cwd))

    # -- setup / persistence ----------------------------------------------

    def set_profile(self, name):
        models = self.cfg["models"]
        if name not in models:
            raise KeyError(f"no model profile {name!r}; have: {', '.join(models)}")
        self.profile_name = name
        self.profile = models[name]
        self.provider = providers.make(self.profile)

    def _load(self, data):
        self.id = data["id"]
        self.messages = data["messages"]
        self.goal = data.get("goal")
        self.tools.plan = data.get("plan", [])
        self.usage = data.get("usage", self.usage)

    def save(self):
        data = {"id": self.id, "cwd": str(self.tools.cwd), "profile": self.profile_name, "goal": self.goal,
                "plan": self.tools.plan, "usage": self.usage, "updated": time.time(), "messages": self.messages}
        # transcripts hold command output and file contents: keep them user-only
        config.write_private(config.SESSIONS / f"{self.id}.json", json.dumps(data))

    def add(self, msg):
        self.messages.append(msg)
        self.save()

    # -- public entry points ----------------------------------------------

    def chat(self, text):
        self.add({"role": "user", "content": text})
        self._run(goal_mode=False)

    def start_goal(self, goal):
        self.goal = goal
        self.add({"role": "user", "content": GOAL_KICKOFF.format(goal=goal)})
        self._run(goal_mode=True)

    def resume_goal(self):
        self.add({"role": "user", "content": "Resuming after a break. Continue working on the goal."})
        self._run(goal_mode=True)

    # -- the loop ------------------------------------------------------------

    def _run(self, goal_mode):
        steps = nudges = 0
        while True:
            if goal_mode and self.max_steps and steps >= self.max_steps:
                self.ui.warn(f"Reached --max-steps {self.max_steps}.")
                if not self._pause(goal_mode, reason="step limit"):
                    return
                steps = 0
            try:
                outcome = self._step(goal_mode)
            except KeyboardInterrupt:
                self.tools.interrupt()
                self._repair()
                self.ui.end_stream()
                if not self._pause(goal_mode):
                    return
                continue
            except Stop:
                return
            steps += 1
            if outcome == "done":
                self.ui.goal_finished(self.goal)
                self.goal = None
                self.save()
                return
            if outcome == "tools":
                nudges = 0
                continue
            if not goal_mode:
                return
            nudges += 1
            if nudges > 3:
                self.ui.warn("The model keeps stopping without finishing.")
                if not self._pause(goal_mode, reason="stalled"):
                    return
                nudges = 0
                continue
            self.add({"role": "user", "content": NUDGE})

    def _pause(self, goal_mode, reason="interrupted"):
        """Ask the user what to do. Returns False to stop the loop."""
        text = self.ui.paused(goal_mode, reason)
        if text is None:
            if goal_mode:
                self.ui.info("Goal paused. /continue resumes it, /goal sets a new one.")
                self.save()
            return False
        if text == "":
            if not goal_mode:
                return False
            self.add({"role": "user", "content": "Continue."})
            return True
        self.add({"role": "user", "content": f"[The user interrupted with new instructions]\n{text}"})
        return True

    def _repair(self):
        """After an interrupt, make sure every tool call has a result."""
        if not self.messages:
            return
        last_asst = max((i for i, m in enumerate(self.messages) if m["role"] == "assistant"), default=None)
        if last_asst is None:
            return
        calls = self.messages[last_asst].get("tool_calls") or []
        answered = {m.get("tool_call_id") for m in self.messages[last_asst + 1:] if m["role"] == "tool"}
        for c in calls:
            if c["id"] not in answered:
                self.add({"role": "tool", "tool_call_id": c["id"], "content": "Interrupted by the user before this finished.",
                          "_is_error": True})

    def tool_schemas(self, goal_mode):
        return SCHEMAS + ([GOAL_DONE] if goal_mode or self.goal else [])

    def _call_model(self, messages, goal_mode, quiet=False):
        on_event = (lambda e: None) if quiet else self.ui.on_event
        res = self.provider.complete(self.system, messages, self.tool_schemas(goal_mode), on_event)
        if not quiet:
            self.ui.end_stream()
        u = res.usage or {}
        self.usage["input_tokens"] += u.get("input_tokens") or 0
        self.usage["output_tokens"] += u.get("output_tokens") or 0
        self.last_context = (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
        return res

    def _step(self, goal_mode):
        self.maybe_compact(goal_mode)
        try:
            res = self._call_model(self.messages, goal_mode)
        except providers.ContextOverflow:
            self.ui.warn("Context overflow; compacting and retrying.")
            self.maybe_compact(goal_mode, force=True)
            res = self._call_model(self.messages, goal_mode)
        msg = res.message
        if res.stop == "length" or res.stop == "max_tokens":
            self.ui.warn("Response hit max_tokens.")
            if msg.get("tool_calls"):  # args are probably cut off
                msg.pop("tool_calls")
                msg.pop("_anthropic", None)
                msg["content"] = (msg.get("content") or "") + "\n[output truncated at max_tokens]"
        self.add(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return "text"
        done = False
        for c in calls:
            result, is_error = self._exec(c, goal_mode)
            m = {"role": "tool", "tool_call_id": c["id"], "content": result}
            if is_error:
                m["_is_error"] = True
            self.add(m)
            if c["function"]["name"] == "goal_done" and not is_error:
                done = True
        return "done" if done else "tools"

    def _needs_approval(self, name, args):
        if name not in MUTATING or self.approval == "auto":
            return False
        if self.approval == "ask":
            return True
        if name in ("write_file", "edit_file"):
            return not self.tools.inside_cwd(args.get("path", ""))
        return True  # bash, fetch_url in "edits" mode

    def _exec(self, call, goal_mode):
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            self.ui.tool_call(name, {})
            msg = f"Invalid JSON arguments ({e}). Call the tool again with valid JSON."
            self.ui.tool_result(msg, True)
            return msg, True
        self.ui.tool_call(name, args)

        if name == "goal_done":
            return "Goal marked complete.", False
        if name == "ask_user":
            answer = self.ui.ask(args.get("question", ""))
            return (answer or "(no answer; use your best judgment)"), False
        if name == "update_plan":
            out = self.tools.update_plan(args.get("items") or [])
            self.ui.plan(self.tools.plan)
            self.save()
            return out, False

        if self._needs_approval(name, args):
            verdict = self.ui.approve(name, args)
            if verdict == "a":
                self.approval = "auto"
                self.ui.info("Approval mode is now auto for this session.")
            elif verdict == "stop":
                raise KeyboardInterrupt
            elif verdict != "y":
                note = f": {verdict}" if verdict not in ("n", "") else ""
                msg = f"The user denied this {name} call{note}"
                self.ui.tool_result(msg, True)
                return msg, True

        fn = getattr(self.tools, name, None)
        if fn is None or name.startswith("_"):
            msg = f"Unknown tool {name!r}."
            self.ui.tool_result(msg, True)
            return msg, True
        try:
            out = fn(**args)
            self.ui.tool_result(out, False)
            return out, False
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - tool errors go back to the model
            msg = f"{type(e).__name__}: {e}"
            self.ui.tool_result(msg, True)
            return msg, True

    # -- context management -------------------------------------------------

    def maybe_compact(self, goal_mode, force=False):
        limit = int(self.profile.get("context_window", 65536))
        if not force and self.last_context < limit * 0.7:
            return
        if len(self.messages) < 3:
            return
        self.ui.info(f"Compacting context ({self.last_context:,} of {limit:,} tokens)...")
        history = self.messages
        summary = None
        for _ in range(3):
            try:
                res = self._call_model(history + [{"role": "user", "content": COMPACT}], goal_mode, quiet=True)
                summary = res.message.get("content") or ""
                break
            except providers.ContextOverflow:
                # Drop the oldest third, starting the remainder at a user message.
                cut = len(history) // 3
                while cut < len(history) and history[cut]["role"] != "user":
                    cut += 1
                history = history[cut:] or history[-2:]
        if not summary:
            self.ui.warn("Compaction failed; keeping only the latest messages.")
            summary = "(Earlier context was lost.)"
        parts = []
        if self.goal:
            parts.append(GOAL_KICKOFF.format(goal=self.goal))
        parts.append("## Summary of earlier work (context was compacted)\n" + summary)
        if self.tools.plan:
            parts.append("## Current plan\n" + self.tools.plan_text())
        parts.append("Continue from here.")
        self.messages = [{"role": "user", "content": "\n\n".join(parts)}]
        self.last_context = 0
        self.save()
        self.ui.info("Compacted.")


def list_sessions(cwd=None):
    if not config.SESSIONS.exists():
        return []
    out = []
    for p in config.SESSIONS.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if cwd is None or d.get("cwd") == str(cwd):
            out.append(d)
    return sorted(out, key=lambda d: d.get("updated", 0), reverse=True)
