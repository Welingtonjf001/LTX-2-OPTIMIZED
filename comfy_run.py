"""Submit an API-format workflow to ComfyUI, wait for it, report outputs/errors.

Usage: python comfy_run.py --workflow api.json [--server http://127.0.0.1:8188]
                           [--timeout 1800]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    server = args.server.rstrip("/")

    wf = json.load(open(args.workflow, encoding="utf-8"))
    client_id = str(uuid.uuid4())

    try:
        res = post_json(f"{server}/prompt", {"prompt": wf, "client_id": client_id})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"[run] REJEITADO pelo servidor (HTTP {e.code}):", file=sys.stderr)
        try:
            err = json.loads(body)
            print(json.dumps(err, indent=2, ensure_ascii=False)[:4000], file=sys.stderr)
        except Exception:
            print(body[:4000], file=sys.stderr)
        return 2

    pid = res.get("prompt_id")
    print(f"[run] prompt_id={pid} enfileirado", flush=True)

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        hist = get_json(f"{server}/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            done = status.get("completed")
            smsg = status.get("status_str")
            elapsed = time.time() - t0
            if done or smsg == "success":
                print(f"[run] CONCLUIDO em {elapsed:.1f}s", flush=True)
                for nid, out in (entry.get("outputs") or {}).items():
                    for key, items in out.items():
                        for it in items if isinstance(items, list) else []:
                            if isinstance(it, dict) and it.get("filename"):
                                print(f"       saida: {key} -> {it.get('subfolder','')}/{it['filename']}")
                return 0
            if smsg == "error" or status.get("messages"):
                for m in status.get("messages", []):
                    if isinstance(m, list) and m and "error" in str(m[0]).lower():
                        print(f"[run] ERRO: {json.dumps(m, ensure_ascii=False)[:2000]}", file=sys.stderr)
                if smsg == "error":
                    print(f"[run] FALHOU em {elapsed:.1f}s", file=sys.stderr)
                    return 3
        time.sleep(5)

    print(f"[run] TIMEOUT apos {args.timeout}s", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
