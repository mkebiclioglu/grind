"""Config lives in ~/.config/grind/config.toml (override with GRIND_HOME).

A *profile* is one model endpoint. Any OpenAI-compatible server (vLLM,
OpenAI, OpenRouter, Ollama, Gemini, Groq, ...) uses provider = "openai";
Anthropic's API uses provider = "anthropic".
"""

import copy
import os
import tomllib
from pathlib import Path

import tomli_w

HOME = Path(os.environ.get("GRIND_HOME", "~/.config/grind")).expanduser()
CONFIG_PATH = HOME / "config.toml"
POD_STATE = HOME / "pod.json"
DATA = Path(os.environ.get("GRIND_DATA", "~/.local/share/grind")).expanduser()
SESSIONS = DATA / "sessions"

DEFAULTS = {
    "default_model": "qwen",
    "approval": "edits",  # ask | edits | auto
    "env_files": ["~/.config/grind/.env"],
    "models": {
        "qwen": {
            "provider": "openai",
            "base_url": "",  # filled in by `grind pod up`
            "model": "qwen3.8-27b",
            "api_key_env": "GRIND_POD_KEY",
            "context_window": 65536,
            "max_tokens": 16384,
            "temperature": 1.0,
            "top_p": 0.95,
            "extra_body": {"top_k": 20},
            "effort_style": "qwen",
            "effort": "medium",
            "preserve_reasoning": True,
        },
        "claude": {
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "effort": "high",
            "api_key_env": "ANTHROPIC_API_KEY",
            "context_window": 200000,
            "max_tokens": 64000,
        },
    },
    "pod": {
        "hf_model": "Qwen/Qwen3.8-27B-FP8",
        "served_name": "qwen3.8-27b",
        "image": "vllm/vllm-openai:v0.32.0-cu129",
        "gpus": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40"],
        "cuda_versions": ["12.9", "13.0"],
        "cloud": ["COMMUNITY", "SECURE"],
        "disk_gb": 70,
        "max_model_len": 65536,
        "idle_minutes": 30,
        "max_hours": 3,
        "mtp": True,
        "extra_args": ["--language-model-only", "--gpu-memory-utilization", "0.92"],
        "profile": "qwen",
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "models":
            out[k] = _merge(out[k], v)
        elif k == "models" and isinstance(v, dict):
            models = out.setdefault("models", {})
            for name, prof in v.items():
                models[name] = _merge(models.get(name, {}), prof) if isinstance(prof, dict) else prof
        else:
            out[k] = v
    return out


def load():
    user = {}
    if CONFIG_PATH.exists():
        user = tomllib.loads(CONFIG_PATH.read_text())
    cfg = _merge(DEFAULTS, user)
    for f in cfg.get("env_files", []):
        load_env_file(Path(f).expanduser())
    return cfg


def load_user():
    return tomllib.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


def save_user(user):
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(tomli_w.dumps(user))
    CONFIG_PATH.chmod(0o600)


def update_profile(name, **fields):
    user = load_user()
    user.setdefault("models", {}).setdefault(name, {}).update(fields)
    save_user(user)


def load_env_file(path):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def api_key(profile):
    if profile.get("api_key"):
        return profile["api_key"]
    env = profile.get("api_key_env")
    return os.environ.get(env, "") if env else ""
