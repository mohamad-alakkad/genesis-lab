#!/usr/bin/env python3
"""Verify a GENESIS checkpoint manifest (schema genesis.manifest/1).

Usage: python verify_manifest.py <manifest.json> [--dir <files_dir>] [--quiet]

Checks required fields, then the size and SHA-256 of every listed file (relative to --dir,
default: the manifest's directory). Prints findings only, at most 20 lines.
Exit codes: 0 valid, 1 invalid, 2 usage error or unreadable manifest. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import sys

REQUIRED = ["schema", "run_id", "checkpoint_id", "global_step", "created_at",
            "files", "world_spec_version", "code_commit", "verified"]
MAX_LINES = 20


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(manifest, base_dir):
    findings = []
    for key in REQUIRED:
        if key not in manifest:
            findings.append(f"missing field: {key}")
    if not str(manifest.get("schema", "")).startswith("genesis.manifest/"):
        findings.append(f"unexpected schema: {manifest.get('schema')!r}")
    if manifest.get("verified") is not True:
        findings.append("verified flag is not true")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        findings.append("files list is empty or invalid")
        return findings
    for entry in files:
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not rel:
            findings.append("file entry without path")
            continue
        path = os.path.normpath(os.path.join(base_dir, rel))
        if not os.path.abspath(path).startswith(os.path.abspath(base_dir)):
            findings.append(f"{rel}: path escapes the checkpoint directory")
            continue
        if not os.path.isfile(path):
            findings.append(f"{rel}: missing")
            continue
        size = os.path.getsize(path)
        if entry.get("size") is not None and size != entry["size"]:
            findings.append(f"{rel}: size {size} != manifest {entry['size']}")
        expected = str(entry.get("sha256", "")).lower()
        if not expected:
            findings.append(f"{rel}: no sha256 in manifest")
        elif sha256_of(path) != expected:
            findings.append(f"{rel}: sha256 mismatch")
    return findings


def main():
    ap = argparse.ArgumentParser(description="Verify a GENESIS checkpoint manifest.")
    ap.add_argument("manifest")
    ap.add_argument("--dir", default=None, help="directory holding the listed files")
    ap.add_argument("--quiet", action="store_true", help="no output on success")
    args = ap.parse_args()
    try:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read manifest: {exc.__class__.__name__}")
        return 2
    if not isinstance(manifest, dict):
        print("manifest is not a JSON object")
        return 2
    base_dir = args.dir or os.path.dirname(os.path.abspath(args.manifest))
    findings = verify(manifest, base_dir)
    if findings:
        for line in findings[:MAX_LINES]:
            print(f"FAIL {line}")
        if len(findings) > MAX_LINES:
            print(f"... {len(findings) - MAX_LINES} more")
        return 1
    if not args.quiet:
        print(f"OK {manifest.get('checkpoint_id')} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
