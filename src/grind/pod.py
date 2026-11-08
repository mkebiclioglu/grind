"""`grind pod up|down|status|logs`: a vLLM server on a RunPod GPU.

The pod runs the stock vLLM image with a tiny watchdog (watchdog.py) that
deletes the pod when idle, crashed, or past a hard runtime cap. No network
volume is used, so a deleted pod costs nothing; the trade-off is that each
`up` re-downloads the weights (a few minutes on RunPod's network).
"""

import base64
import json
import os
import secrets
import shlex
import sys
import time
from importlib import resources

import httpx
from rich.console import Console

from . import config

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
console = Console()


def _key():
    for name in ("RUNPOD_API_KEY", "RUN_POD_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    sys.exit("No RunPod key: set RUNPOD_API_KEY (or list an env file under env_files in config.toml).")


def _client():
    return httpx.Client(headers={"Authorization": f"Bearer {_key()}"}, timeout=30)


def _state():
    return json.loads(config.POD_STATE.read_text()) if config.POD_STATE.exists() else None


def _save_state(s):
    config.HOME.mkdir(parents=True, exist_ok=True)
    if s is None:
        config.POD_STATE.unlink(missing_ok=True)
        return
    config.POD_STATE.write_text(json.dumps(s, indent=2))
    config.POD_STATE.chmod(0o600)


def balance():
    with _client() as c:
        r = c.post(GQL, json={"query": "query { myself { clientBalance currentSpendPerHr } }"})
        d = r.json()["data"]["myself"]
        return d["clientBalance"], d["currentSpendPerHr"]


def _start_script(p, key_env="GRIND_VLLM_KEY"):
    args = [
        "vllm", "serve", p["hf_model"],
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", p["served_name"],
        "--max-model-len", str(p["max_model_len"]),
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        *p.get("extra_args", []),
    ]
    if p.get("mtp"):
        args += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})]
    serve = " ".join(shlex.quote(a) for a in args) + f' --api-key "${key_env}"'
    return (
        'echo "$GRIND_WATCHDOG_B64" | base64 -d > /tmp/grind_watchdog.py\n'
        "python3 /tmp/grind_watchdog.py &\n"
        + "".join(f"{line}\n" for line in p.get("pre_serve", []))
        + f"{serve} >> /tmp/vllm.log 2>&1\n"
        + 'echo "[grind] vllm exited with code $?" >> /tmp/vllm.log\n'
        "sleep infinity\n"
    )


def up(idle=None, max_hours=None, gpu=None, mtp=None, wait=True):
    cfg = config.load()
    p = dict(cfg["pod"])
    if idle is not None:
        p["idle_minutes"] = idle
    if max_hours is not None:
        p["max_hours"] = max_hours
    if mtp is not None:
        p["mtp"] = mtp
    gpus = [gpu] if gpu else p["gpus"]

    st = _state()
    if st:
        with _client() as c:
            r = c.get(f"{REST}/pods/{st['id']}")
        if r.status_code == 200 and r.json().get("desiredStatus") == "RUNNING":
            console.print(f"Pod {st['id']} is already running. Use `grind pod status` or `grind pod down`.")
            return
        _save_state(None)

    bal, _ = balance()
    console.print(f"RunPod balance: [bold]${bal:.2f}[/]")
    vkey = secrets.token_urlsafe(32)
    watchdog = resources.files("grind").joinpath("watchdog.py").read_bytes()
    body = {
        "name": "grind-vllm",
        "imageName": p["image"],
        "gpuTypeIds": gpus,
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "allowedCudaVersions": p["cuda_versions"],
        "containerDiskInGb": p["disk_gb"],
        "volumeInGb": 0,
        "ports": ["8000/http", "8001/http"],
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [_start_script(p)],
        "env": {
            "GRIND_VLLM_KEY": vkey,
            "GRIND_WATCHDOG_B64": base64.b64encode(watchdog).decode(),
            "GRIND_IDLE_MIN": str(p["idle_minutes"]),
            "GRIND_MAX_HOURS": str(p["max_hours"]),
        },
    }
    pod = None
    with _client() as c:
        for cloud in p["cloud"]:
            body["cloudType"] = cloud
            r = c.post(f"{REST}/pods", json=body)
            if r.status_code < 300:
                pod = r.json()
                break
            console.print(f"[dim]{cloud} cloud: {r.status_code} {r.text[:200]}[/]")
    if not pod:
        sys.exit("No matching GPU available right now. Try again later or pass --gpu.")

    gpu_name = (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("displayName") or "?"
    cost = pod.get("costPerHr") or pod.get("adjustedCostPerHr")
    st = {
        "id": pod["id"],
        "key": vkey,
        "created": time.time(),
        "gpu": gpu_name,
        "cost_per_hr": cost,
        "url": f"https://{pod['id']}-8000.proxy.runpod.net/v1",
        "watchdog": f"https://{pod['id']}-8001.proxy.runpod.net",
    }
    _save_state(st)
    config.update_profile(p["profile"], base_url=st["url"], api_key=vkey, model=p["served_name"],
                          context_window=p["max_model_len"])
    console.print(f"Pod [bold]{pod['id']}[/] on {gpu_name} at ${cost}/hr "
                  f"(auto-deletes after {p['idle_minutes']} min idle or {p['max_hours']}h).")
    console.print(f"Profile [bold]{p['profile']}[/] now points at {st['url']}")
    if wait:
        wait_ready(st)


def _watchdog_status(st, timeout=10):
    try:
        r = httpx.get(st["watchdog"] + "/status", headers={"Authorization": f"Bearer {st['key']}"}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except httpx.HTTPError:
        pass
    return None


def wait_ready(st=None, timeout_min=45):
    st = st or _state()
    if not st:
        sys.exit("No pod. Run `grind pod up`.")
    console.print("Waiting for vLLM (image pull, weight download, load, graph capture). Ctrl-C stops waiting; the pod keeps going.")
    t0 = time.time()
    last = None
    try:
        while time.time() - t0 < timeout_min * 60:
            s = _watchdog_status(st)
            el = int(time.time() - t0)
            if s is None:
                line = "container starting (pulling image)..."
            elif s["phase"] == "ready":
                console.print(f"[green]Ready[/] after {el//60}m{el%60:02d}s. Run `grind` (profile qwen).")
                return True
            elif s["phase"] == "crashed":
                console.print("[red]vLLM crashed.[/] Last log lines:")
                for l in s.get("last_log", []):
                    console.print("  " + l, markup=False)
                console.print("See `grind pod logs`. The watchdog deletes the pod in 10 min; `grind pod down` deletes it now.")
                return False
            else:
                line = (s.get("last_log") or ["starting..."])[-1][:150]
            if line != last:
                console.print(f"[dim]{el//60:>2}m{el%60:02d}s[/] {line}", markup=True, highlight=False)
                last = line
            time.sleep(10)
    except KeyboardInterrupt:
        console.print("Stopped waiting. `grind pod status` to check again.")
        return False
    console.print("[yellow]Still not ready.[/] Check `grind pod logs`.")
    return False


def status():
    st = _state()
    bal, spend = balance()
    console.print(f"Balance ${bal:.2f}, spending ${spend:.2f}/hr")
    if not st:
        console.print("No grind pod.")
        return
    with _client() as c:
        r = c.get(f"{REST}/pods/{st['id']}")
    if r.status_code != 200:
        console.print(f"Pod {st['id']} is gone (watchdog or manual delete).")
        _save_state(None)
        return
    pod = r.json()
    up_min = (time.time() - st["created"]) / 60
    console.print(f"Pod {st['id']} {pod.get('desiredStatus')} on {st['gpu']} at ${pod.get('costPerHr')}/hr, "
                  f"up {up_min:.0f} min (~${up_min/60*float(pod.get('costPerHr') or 0):.2f})")
    s = _watchdog_status(st)
    if s:
        idle = f", idle {s['idle_s']//60} of {int(s['idle_limit_s']//60)} min" if s["phase"] == "ready" else ""
        console.print(f"vLLM: {s['phase']}{idle}, self-delete {'armed' if s['can_self_delete'] else '[red]NOT available[/]'}")
        if s["phase"] != "ready":
            for l in s.get("last_log", []):
                console.print("  " + l, markup=False, highlight=False)
    else:
        console.print("Watchdog not reachable yet (container still starting).")


def logs():
    st = _state()
    if not st:
        sys.exit("No pod.")
    r = httpx.get(st["watchdog"] + "/logs", headers={"Authorization": f"Bearer {st['key']}"}, timeout=15)
    print(r.text if r.status_code == 200 else f"HTTP {r.status_code}")


def down():
    st = _state()
    if not st:
        console.print("No grind pod recorded.")
        return
    with _client() as c:
        r = c.delete(f"{REST}/pods/{st['id']}")
    if r.status_code in (200, 204, 404):
        console.print(f"Pod {st['id']} deleted.")
        _save_state(None)
    else:
        console.print(f"[red]Delete failed[/] {r.status_code} {r.text[:300]}")
