"""Tools the model can call. Schemas are OpenAI function format."""

import fnmatch
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import httpx

MAX_OUT = 30_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build", ".next"}


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


SCHEMAS = [
    _fn("bash", "Run a shell command (bash) in the working directory. The directory persists between calls "
        "(cd works); environment variables do not. Use for git, tests, builds, installs, and anything else. "
        "Output is truncated to ~30k chars. Avoid interactive commands.",
        {"command": {"type": "string"},
         "timeout": {"type": "integer", "description": "seconds, default 120, max 1800"}},
        ["command"]),
    _fn("read_file", "Read a text file. Returns numbered lines.",
        {"path": {"type": "string"},
         "offset": {"type": "integer", "description": "1-based first line, default 1"},
         "limit": {"type": "integer", "description": "max lines, default 2000"}},
        ["path"]),
    _fn("write_file", "Create or overwrite a file with the given content. Creates parent directories.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit_file", "Replace an exact string in a file. old_string must match exactly once "
        "(include surrounding lines to make it unique) unless replace_all is true.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
         "replace_all": {"type": "boolean"}},
        ["path", "old_string", "new_string"]),
    _fn("glob", "Find files by glob pattern, e.g. '**/*.py'. Skips .git, node_modules, venvs.",
        {"pattern": {"type": "string"}, "path": {"type": "string", "description": "directory, default cwd"}},
        ["pattern"]),
    _fn("grep", "Search file contents with a regular expression. Returns path:line:text matches.",
        {"pattern": {"type": "string"}, "path": {"type": "string"},
         "include": {"type": "string", "description": "file glob filter, e.g. '*.ts'"}},
        ["pattern"]),
    _fn("fetch_url", "HTTP GET a URL and return its text (HTML is stripped to text). Max ~30k chars.",
        {"url": {"type": "string"}}, ["url"]),
    _fn("update_plan", "Record your current plan as a checklist. Call it at the start of multi-step work and "
        "whenever a step changes status. The plan survives context compaction.",
        {"items": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done"]}},
            "required": ["step", "status"]}}},
        ["items"]),
    _fn("ask_user", "Ask the user a question and wait for the answer. Only for decisions you genuinely "
        "cannot make yourself (missing credentials, ambiguous requirements with real consequences).",
        {"question": {"type": "string"}}, ["question"]),
]

GOAL_DONE = _fn("goal_done", "Call this ONLY when the goal is fully achieved and verified. This ends autonomous mode.",
                {"summary": {"type": "string", "description": "what was done and how it was verified"}},
                ["summary"])

MUTATING = {"bash", "write_file", "edit_file", "fetch_url"}


def _trunc(s, limit=MAX_OUT):
    if len(s) <= limit:
        return s
    half = limit // 2
    return s[:half] + f"\n\n[... {len(s) - limit} chars truncated ...]\n\n" + s[-half:]


class Tools:
    def __init__(self, cwd):
        self.cwd = Path(cwd).resolve()
        self.proc = None
        self.plan = []

    def path(self, p):
        p = Path(os.path.expanduser(p))
        return (p if p.is_absolute() else self.cwd / p).resolve()

    def inside_cwd(self, p):
        try:
            self.path(p).relative_to(self.cwd)
            return True
        except ValueError:
            return False

    def interrupt(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # -- tools -------------------------------------------------------------

    def bash(self, command, timeout=120):
        timeout = max(1, min(int(timeout or 120), 1800))
        marker = "__GRIND_CWD__"
        script = f"{command}\n__rc=$?\nprintf '\\n{marker}%s' \"$PWD\"\nexit $__rc"
        self.proc = subprocess.Popen(["bash", "-c", script], cwd=self.cwd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            out, _ = self.proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            self.interrupt()
            out, _ = self.proc.communicate()
            timed_out = True
        rc = self.proc.returncode
        self.proc = None
        text = out.decode("utf-8", "replace")
        if marker in text:
            text, new_cwd = text.rsplit(marker, 1)
            text = text.removesuffix("\n")
            if new_cwd.strip() and Path(new_cwd.strip()).is_dir():
                self.cwd = Path(new_cwd.strip())
        text = _trunc(text)
        if timed_out:
            text += f"\n[timed out after {timeout}s; process killed]"
        elif rc:
            text += f"\n[exit code {rc}]"
        return text or "(no output)"

    def read_file(self, path, offset=1, limit=2000):
        p = self.path(path)
        if not p.is_file():
            raise FileNotFoundError(f"{p} is not a file")
        lines = p.read_text(errors="replace").splitlines()
        offset = max(1, int(offset or 1))
        chunk = lines[offset - 1: offset - 1 + int(limit or 2000)]
        body = "\n".join(f"{i:>6}\t{l[:2000]}" for i, l in enumerate(chunk, offset))
        more = len(lines) - (offset - 1 + len(chunk))
        if more > 0:
            body += f"\n[{more} more lines; use offset={offset + len(chunk)}]"
        return _trunc(body) or "(empty file)"

    def write_file(self, path, content):
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content)
        return f"{'Overwrote' if existed else 'Created'} {p} ({len(content.splitlines())} lines)"

    def edit_file(self, path, old_string, new_string, replace_all=False):
        p = self.path(path)
        text = p.read_text()
        n = text.count(old_string)
        if not old_string or n == 0:
            raise ValueError("old_string not found; read the file and copy the text exactly")
        if n > 1 and not replace_all:
            raise ValueError(f"old_string occurs {n} times; add surrounding context or set replace_all")
        p.write_text(text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1))
        return f"Edited {p} ({n if replace_all else 1} replacement{'s' if replace_all and n > 1 else ''})"

    def _walk(self, root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                yield Path(dirpath) / f

    def glob(self, pattern, path=None):
        root = self.path(path or ".")
        hits = []
        for f in self._walk(root):
            rel = f.relative_to(root).as_posix()
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f.name, pattern) or Path(rel).match(pattern):
                hits.append(rel)
                if len(hits) >= 500:
                    break
        return "\n".join(sorted(hits)) or "no matches"

    def grep(self, pattern, path=None, include=None):
        root = self.path(path or ".")
        if shutil.which("rg"):
            cmd = ["rg", "-n", "--no-heading", "--max-columns", "300", "-m", "50", pattern, str(root)]
            if include:
                cmd[1:1] = ["-g", include]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
            return _trunc(out.replace(str(root) + "/", "")) or "no matches"
        rx = re.compile(pattern)
        hits = []
        files = [root] if root.is_file() else self._walk(root)
        for f in files:
            if include and not fnmatch.fnmatch(f.name, include):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(root) if f != root else f.name}:{i}:{line[:300]}")
            except (OSError, UnicodeDecodeError):
                continue
            if len(hits) > 1000:
                break
        return _trunc("\n".join(hits)) or "no matches"

    def fetch_url(self, url):
        r = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "grind-agent/0.1"})
        text = r.text
        if "html" in r.headers.get("content-type", ""):
            text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text)
        return f"HTTP {r.status_code}\n" + _trunc(text.strip())

    def update_plan(self, items):
        self.plan = items
        return "Plan updated:\n" + self.plan_text()

    def plan_text(self):
        mark = {"done": "[x]", "in_progress": "[~]", "pending": "[ ]"}
        return "\n".join(f"{mark.get(i.get('status'), '[ ]')} {i.get('step')}" for i in self.plan) or "(no plan)"
