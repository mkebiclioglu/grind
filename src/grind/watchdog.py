"""Runs *inside* the RunPod pod, next to vLLM. Python stdlib only.

Jobs:
  * serve /status and /logs on port 8001 (bearer-auth with the vLLM key) so the
    local CLI can show startup progress;
  * delete the pod when it has been idle, crashed, failed to start, or hit the
    hard runtime cap, so a forgotten pod can't drain the account.

It deletes the pod with the pod-scoped RUNPOD_API_KEY that RunPod injects into
every pod; the user's own API key never leaves the laptop.
"""

import http.server
import json
import os
import threading
import time
import urllib.request

KEY = os.environ.get("GRIND_VLLM_KEY", "")
IDLE_S = float(os.environ.get("GRIND_IDLE_MIN", "30")) * 60
MAX_S = float(os.environ.get("GRIND_MAX_HOURS", "3")) * 3600
STARTUP_S = float(os.environ.get("GRIND_STARTUP_MIN", "40")) * 60
CRASH_GRACE_S = 600
LOG = "/tmp/vllm.log"
MARK = "/root/.grind_started"

try:
    START = float(open(MARK).read().strip())
except Exception:
    START = time.time()
    with open(MARK, "w") as f:
        f.write(str(START))

state = {
    "phase": "starting",
    "ready_at": None,
    "last_activity": time.time(),
    "crashed_at": None,
    "terminating": None,
    "tokens": 0,
}


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[grind-watchdog] {msg}\n")


def tail(n=60):
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return lines[-n:]
    except FileNotFoundError:
        return []


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def healthy():
    try:
        return get("http://127.0.0.1:8000/health")[0] == 200
    except Exception:
        return False


def metrics():
    """Return (total tokens processed, requests in flight)."""
    try:
        _, body = get("http://127.0.0.1:8000/metrics")
    except Exception:
        return None, 0
    tokens = inflight = 0.0
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except Exception:
            continue
        if name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
            tokens += val
        elif name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            inflight += val
    return tokens, inflight


def vllm_alive():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ")
        except OSError:
            continue
        if b"vllm serve" in cmd or b"vllm.entrypoints" in cmd:
            return True
    return False


def terminate(reason):
    if state["terminating"]:
        return
    state["terminating"] = reason
    log(f"deleting pod: {reason}")
    pod = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    if not (pod and key):
        log("no RUNPOD_POD_ID/RUNPOD_API_KEY in env; cannot self-delete")
        return
    attempts = [
        ("DELETE", f"https://rest.runpod.io/v1/pods/{pod}", None),
        ("POST", "https://api.runpod.io/graphql",
         {"query": f'mutation {{ podTerminate(input: {{podId: "{pod}"}}) }}'}),
        ("POST", f"https://rest.runpod.io/v1/pods/{pod}/stop", None),
    ]
    for method, url, body in attempts:
        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                text = r.read().decode()[:300]
            if "error" not in text.lower():
                log(f"{method} {url.split('/v1')[0]} ok")
                return
            log(f"{method} {url} -> {text}")
        except Exception as e:
            log(f"{method} {url} failed: {e}")
    state["terminating"] = None  # let the loop retry


def loop():
    last_tokens = None
    while True:
        now = time.time()
        if now - START > MAX_S:
            terminate(f"hard cap of {MAX_S/3600:.1f}h reached")
        if healthy():
            if state["phase"] != "ready":
                state.update(phase="ready", ready_at=now, last_activity=now, crashed_at=None)
                log("vLLM is ready")
            tokens, inflight = metrics()
            if tokens is not None:
                state["tokens"] = tokens
                if tokens != last_tokens or inflight > 0:
                    state["last_activity"] = now
                last_tokens = tokens
            if now - state["last_activity"] > IDLE_S:
                terminate(f"idle for {IDLE_S/60:.0f} min")
        elif not vllm_alive():
            if state["phase"] != "crashed":
                state.update(phase="crashed", crashed_at=now)
                log(f"vLLM process is gone; deleting pod in {CRASH_GRACE_S//60} min")
            if now - state["crashed_at"] > CRASH_GRACE_S:
                terminate("vLLM crashed")
        elif state["phase"] == "starting" and now - START > STARTUP_S:
            terminate(f"not ready after {STARTUP_S/60:.0f} min")
        time.sleep(20)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            self.send_response(401)
            self.end_headers()
            return
        now = time.time()
        if self.path.startswith("/logs"):
            body = "\n".join(tail(400)).encode()
            ctype = "text/plain"
        else:
            payload = dict(state)
            payload.update(
                uptime_s=round(now - START),
                idle_s=round(now - state["last_activity"]) if state["phase"] == "ready" else 0,
                idle_limit_s=IDLE_S,
                max_s=MAX_S,
                can_self_delete=bool(os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_POD_ID")),
                last_log=[l for l in tail(15) if l.strip()][-3:],
            )
            body = json.dumps(payload).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    log(f"watchdog up: idle={IDLE_S/60:.0f}m cap={MAX_S/3600:.1f}h "
        f"self_delete={'yes' if os.environ.get('RUNPOD_API_KEY') else 'NO'}")
    threading.Thread(target=loop, daemon=True).start()
    http.server.ThreadingHTTPServer(("0.0.0.0", 8001), Handler).serve_forever()
