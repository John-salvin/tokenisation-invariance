"""Provenance records for computed numbers.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "paper_numbers.yaml"

MARKER = "# --- robustness package -- appended, never hand-edited ---"


def git_sha() -> str:
    for cwd in (REPO_ROOT,):
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd,
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return out.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return os.environ.get("TOKINV_GIT_SHA", "unavailable-no-git-on-cluster")


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def code_sha256(script: Path, modules: list[str]) -> str:
    files = [Path(script)]
    files += [REPO_ROOT / "src" / f"{m}.py" for m in modules]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"cannot stamp provenance, missing: {missing}")
    return _hash_files(files)


def stamp(script: Path, modules: list[str], seed: int | None = None,
          **extra) -> dict:
    return {
        "script": str(Path(script).relative_to(REPO_ROOT)),
        "modules": sorted(modules),
        "git_sha": git_sha(),
        "code_sha256": code_sha256(Path(script), modules),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "host": platform.node(),
        **extra,
    }


def register(namespace: str, values: dict, provenance: dict,
             registry: Path = REGISTRY) -> None:
    text = registry.read_text(encoding="utf-8")
    before = yaml.safe_load(text)
    if not isinstance(before, dict) or "claims" not in before or "meta" not in before:
        raise RuntimeError(f"{registry} does not look like the anchor registry; refusing to write")

    head = text.split(MARKER, 1)[0].rstrip("\n")
    existing = (before.get("robustness") or {})
    existing[namespace] = {
        "provenance": provenance,
        "values": json.loads(json.dumps(values, default=float)),
    }

    block = yaml.safe_dump({"robustness": existing}, allow_unicode=True,
                           sort_keys=False, default_flow_style=False)
    registry.write_text(f"{head}\n\n{MARKER}\n{block}", encoding="utf-8")

    after = yaml.safe_load(registry.read_text(encoding="utf-8"))
    if after["claims"] != before["claims"] or after["meta"] != before["meta"]:
        registry.write_text(text, encoding="utf-8")
        raise RuntimeError("claims/meta changed during a robustness write -- rolled back")
    if after["robustness"][namespace]["values"].keys() != existing[namespace]["values"].keys():
        registry.write_text(text, encoding="utf-8")
        raise RuntimeError("robustness block did not round-trip -- rolled back")

    print(f"[provenance] registered robustness.{namespace} "
          f"({len(values)} values, code {provenance['code_sha256']})")
