import argparse
import sys


def pod_main(argv):
    from . import pod

    ap = argparse.ArgumentParser(prog="grind pod", description="Manage the RunPod vLLM server.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="start a pod and wait until the model is served")
    u.add_argument("-m", "--model", help="preset name (see `grind pod presets`)")
    u.add_argument("--idle", type=int, help="minutes idle before the pod deletes itself")
    u.add_argument("--max-hours", type=float, help="hard runtime cap in hours")
    u.add_argument("--gpu", help='exact RunPod GPU id, e.g. "NVIDIA L40S"')
    u.add_argument("--no-mtp", action="store_true", help="disable MTP speculative decoding")
    u.add_argument("--no-wait", action="store_true")
    sub.add_parser("presets", help="list the models `up` can serve")
    sub.add_parser("wait", help="wait for the current pod to be ready")
    sub.add_parser("status", help="pod, vLLM and balance status")
    sub.add_parser("logs", help="last vLLM log lines")
    sub.add_parser("down", help="delete the pod now")
    a = ap.parse_args(argv)
    if a.cmd == "up":
        pod.up(preset=a.model, idle=a.idle, max_hours=a.max_hours, gpu=a.gpu, mtp=False if a.no_mtp else None, wait=not a.no_wait)
    elif a.cmd == "presets":
        pod.presets()
    elif a.cmd == "wait":
        pod.wait_ready()
    elif a.cmd == "status":
        pod.status()
    elif a.cmd == "logs":
        pod.logs()
    elif a.cmd == "down":
        pod.down()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["pod"]:
        from . import config
        config.load()
        return pod_main(argv[1:])
    from .app import run
    return run(argv)
