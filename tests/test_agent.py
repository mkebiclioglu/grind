import json

import pytest

from grind import agent as agent_mod
from grind import config
from grind.providers import Result
from grind.tools import Tools


class FakeUI:
    def __init__(self):
        self.log = []
        self.show_reasoning = True

    def __getattr__(self, name):
        def rec(*a, **k):
            self.log.append((name, a))
            if name == "paused":
                return None
            if name == "approve":
                return "y"
            if name == "ask":
                return "blue"
        return rec


def call(name, **args):
    return {"id": f"c_{name}_{len(json.dumps(args))}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete(self, system, messages, tools, on_event):
        self.seen.append([m["role"] for m in messages])
        content, calls = self.turns.pop(0)
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        return Result(msg, {"input_tokens": 10, "output_tokens": 5}, "stop")


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")

    def _make(turns, approval="auto"):
        cfg = config._merge(config.DEFAULTS, {})
        a = agent_mod.Agent(cfg, "qwen", tmp_path, FakeUI(), approval=approval)
        a.provider = Scripted(turns)
        return a
    return _make


def test_goal_mode_nudges_until_done(make_agent, tmp_path):
    a = make_agent([
        ("", [call("write_file", path="hello.txt", content="hi\n")]),
        ("I wrote the file.", None),  # stops without goal_done -> nudge
        ("", [call("bash", command="cat hello.txt")]),
        ("", [call("goal_done", summary="wrote and checked hello.txt")]),
    ])
    a.start_goal("create hello.txt")
    assert (tmp_path / "hello.txt").read_text() == "hi\n"
    assert a.goal is None
    assert any(m["role"] == "user" and m["content"] == agent_mod.NUDGE for m in a.messages)
    tool_msgs = [m for m in a.messages if m["role"] == "tool"]
    assert "hi" in tool_msgs[1]["content"]


def test_chat_mode_stops_on_text(make_agent):
    a = make_agent([("", [call("bash", command="echo 42")]), ("It's 42.", None)])
    a.chat("what is 6*7")
    assert a.messages[-1]["content"] == "It's 42."
    assert not any(t["function"]["name"] == "goal_done" for t in a.tool_schemas(False))


def test_stalled_goal_pauses(make_agent):
    a = make_agent([("thinking about it", None)] * 5)
    a.start_goal("x")
    assert any(e[0] == "paused" for e in a.ui.log)
    assert a.goal == "x"  # still pending


def test_edit_and_errors(make_agent, tmp_path):
    (tmp_path / "f.py").write_text("a = 1\na = 1\n")
    a = make_agent([
        ("", [call("edit_file", path="f.py", old_string="a = 1", new_string="a = 2")]),
        ("", [call("edit_file", path="f.py", old_string="a = 1\na", new_string="a = 2\na")]),
        ("", [call("nope")]),
        ("done", None),
    ])
    a.chat("edit")
    tools = [m for m in a.messages if m["role"] == "tool"]
    assert "occurs 2 times" in tools[0]["content"] and tools[0]["_is_error"]
    assert (tmp_path / "f.py").read_text() == "a = 2\na = 1\n"
    assert "Unknown tool" in tools[2]["content"]


def test_approval_edits_mode(make_agent, tmp_path):
    a = make_agent([], approval="edits")
    assert not a._needs_approval("write_file", {"path": "x.txt"})
    assert a._needs_approval("write_file", {"path": "/etc/hosts"})
    assert a._needs_approval("bash", {"command": "ls"})
    assert not a._needs_approval("read_file", {"path": "/etc/hosts"})


def test_bash_cwd_persists(tmp_path):
    (tmp_path / "sub").mkdir()
    t = Tools(tmp_path)
    t.bash("cd sub")
    assert t.cwd == (tmp_path / "sub").resolve()
    assert "exit code 3" in t.bash("exit 3")


def test_compaction_replaces_history(make_agent):
    a = make_agent([("Summary: did stuff.", None)])
    a.goal = "g"
    a.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                  {"role": "user", "content": "more"}]
    a.last_context = 10**9
    a.maybe_compact(True)
    assert len(a.messages) == 1 and "Summary: did stuff." in a.messages[0]["content"] and "GOAL: g" in a.messages[0]["content"]


def test_anthropic_message_conversion():
    from grind.providers import AnthropicProvider
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [call("bash", command="ls"), call("bash", command="pwd")]},
        {"role": "tool", "tool_call_id": call("bash", command="ls")["id"], "content": "a"},
        {"role": "tool", "tool_call_id": call("bash", command="pwd")["id"], "content": "b", "_is_error": True},
        {"role": "user", "content": "next"},
    ]
    out = AnthropicProvider._messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert out[2]["content"][1]["is_error"] is True
