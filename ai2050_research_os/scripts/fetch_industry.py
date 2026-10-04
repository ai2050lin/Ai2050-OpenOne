#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_industry.py - public-release industry downloader.

Reads config/industry_sources.json, fetches each source URL on demand, and
caches the raw artifact + metadata into cache/industry/ (gitignored).

Design contract (README "公开平台发布形态" section):
  * The repo ships only this config + script; industry content is NOT committed.
  * Every fetched artifact is hashed (sha256) into cache/industry/manifest.json;
    promotion of an industry.json record to status=verified requires the archived
    sha256 to be present and its primary text re-checked.
  * Offline-safe: unreachable sources are skipped and reported, never fatal.

Usage:
  python fetch_industry.py                # fetch all sources
  python fetch_industry.py --id SRC-CTRACER
  python fetch_industry.py --list         # show configured sources
  python fetch_industry.py --manifest     # print cache manifest summary
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
OS_ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(OS_ROOT, "config", "industry_sources.json")


def load_config():
    with io.open(CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_dir(cfg):
    d = os.path.join(OS_ROOT, cfg["fetch_policy"]["cache_dir"].split("ai2050_research_os" + os.sep)[-1]
                     if os.sep in cfg["fetch_policy"]["cache_dir"]
                     else cfg["fetch_policy"]["cache_dir"])
    # normalise: always <OS_ROOT>/cache/industry
    d = os.path.join(OS_ROOT, "cache", "industry")
    os.makedirs(d, exist_ok=True)
    return d


def load_manifest(d):
    p = os.path.join(d, "manifest.json")
    if os.path.exists(p):
        with io.open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"entries": {}}


def save_manifest(d, manifest):
    p = os.path.join(d, "manifest.json")
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def fetch_one(src, policy, d, manifest):
    sid = src["id"]
    urls = [u for u in (src.get("api_url"), src.get("url")) if u]
    entry = manifest["entries"].get(sid, {})
    for url in urls:
        req = urllib.request.Request(
            url, headers={"User-Agent": policy.get("user_agent", "AI2050-fetcher/1.0")})
        for attempt in range(1 + int(policy.get("retry", 2))):
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=policy.get("timeout_seconds", 30)) as r:
                    raw = r.read()
                sha = hashlib.sha256(raw).hexdigest()
                fname = "%s_%s.bin" % (sid, sha[:12])
                with io.open(os.path.join(d, fname), "wb") as f:
                    f.write(raw)
                entry = {
                    "source_id": sid,
                    "name": src.get("name"),
                    "url": url,
                    "status": "ok",
                    "http_bytes": len(raw),
                    "sha256": sha,
                    "file": fname,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "elapsed_s": round(time.time() - t0, 2),
                    "content_type": r.headers.get("Content-Type", ""),
                }
                manifest["entries"][sid] = entry
                return entry
            except Exception as e:  # noqa: BLE001 - offline-safe by design
                last = "%s: %s" % (type(e).__name__, e)
                time.sleep(1.0)
    entry = {"source_id": sid, "name": src.get("name"), "status": "failed",
             "error": last if urls else "no url configured",
             "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest["entries"][sid] = entry
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", help="fetch only this source id")
    ap.add_argument("--list", action="store_true", help="list configured sources")
    ap.add_argument("--manifest", action="store_true", help="print cache manifest summary")
    args = ap.parse_args()

    cfg = load_config()
    policy = cfg["fetch_policy"]
    d = cache_dir(cfg)

    if args.list:
        for s in cfg["sources"]:
            print("%-16s %-14s %s" % (s["id"], s["type"], s["name"]))
        return 0

    if args.manifest:
        m = load_manifest(d)
        for sid, e in sorted(m["entries"].items()):
            print("%-16s %-6s %8sB  sha=%s  %s" %
                  (sid, e.get("status"), e.get("http_bytes", "-"),
                   (e.get("sha256") or "-")[:12], e.get("url", "")))
        return 0

    sources = cfg["sources"]
    if args.id:
        sources = [s for s in sources if s["id"] == args.id]
        if not sources:
            print("unknown source id: %s" % args.id)
            return 2

    manifest = load_manifest(d)
    ok = 0
    for src in sources:
        e = fetch_one(src, policy, d, manifest)
        flag = "OK " if e.get("status") == "ok" else "ERR"
        print("[%s] %-16s %s" % (flag, src["id"], e.get("url", "")))
        ok += 1 if e.get("status") == "ok" else 0
        save_manifest(d, manifest)

    print("fetched %d/%d -> cache: %s" % (ok, len(sources), d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
