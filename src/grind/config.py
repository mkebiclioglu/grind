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
        "heretic": {
            "provider": "openai",
            "base_url": "",  # filled in by `grind pod up -m heretic-9b`
            "model": "heretic-9b",
            "context_window": 65536,
            "max_tokens": 16384,
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body": {"top_k": 20},
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
    # Settings shared by every pod; each preset below can override any of them.
    "pod": {
        "default_preset": "qwen3.8-27b",
        "image": "vllm/vllm-openai:v0.32.0-cu129",
        "cuda_versions": ["12.9", "13.0"],
        "cloud": ["COMMUNITY", "SECURE"],
        "disk_gb": 70,
        "max_model_len": 65536,
        "idle_minutes": 30,
        "max_hours": 3,
        "mtp": True,
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "extra_args": ["--language-model-only", "--gpu-memory-utilization", "0.92", "--max-num-seqs", "16"],
        # Shell lines run before `vllm serve`. The cu129 image ships a torchcodec built for CUDA 13
        # that crashes vLLM at import; it's only needed for video input, so drop it.
        "pre_serve": ["pip uninstall -y -q torchcodec >/dev/null 2>&1 || true"],
        "presets": {
            "qwen3.8-27b": {
                "hf_model": "Qwen/Qwen3.8-27B-FP8",
                "served_name": "qwen3.8-27b",
                "gpus": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40"],
                "profile": "qwen",
            },
            "heretic-9b": {
                "hf_model": "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic",
                "served_name": "heretic-9b",
                "gpus": ["NVIDIA A40", "NVIDIA RTX PRO 4500 Blackwell", "NVIDIA GeForce RTX 5090",
                         "NVIDIA RTX A6000", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S", "NVIDIA L40"],
                "disk_gb": 50,
                "profile": "heretic",
            },
        },
    },
}


def pod_settings(cfg, preset=None):
    """Shared pod settings merged with one preset. Returns (name, settings)."""
    pod = cfg["pod"]
    name = preset or pod.get("default_preset")
    presets = pod.get("presets", {})
    if name not in presets:
        raise KeyError(f"no pod preset {name!r}; have: {', '.join(presets)}")
    shared = {k: v for k, v in pod.items() if k not in ("presets", "default_preset")}
    return name, {**shared, **presets[name]}


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


def write_private(path, text):
    """Write a file only the user can read (keys, transcripts); never world-readable, even briefly."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    tmp.chmod(0o600)
    tmp.replace(path)


def save_user(user):
    write_private(CONFIG_PATH, tomli_w.dumps(user))


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
