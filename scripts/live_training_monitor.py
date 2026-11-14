"""Standalone zero-dependency live training monitor for MudraForge Dual-Stream runs.

Serves a real-time local web dashboard over HTTP with JSON telemetry endpoints,
live file tailing, multi-panel HTML5 Canvas charts, generalization gap tracking,
ETA estimation, and discrepancy inspection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.parse import parse_qs, urlparse


_DEVICE_RE = re.compile(
    r"Device:\s*(?P<device>.+?)\s*\|\s*Batch:\s*(?P<batch>\d+)\s*\|\s*AMP:\s*(?P<amp>True|False)"
)
_ARCH_RE = re.compile(
    r"Architecture:\s*(?P<arch>.+?)\s*\|\s*Raw Vocab:\s*(?P<raw_vocab>\d+)\s*\|\s*Logic Vocab:\s*(?P<logic_vocab>\d+)"
)
_SPLIT_RE = re.compile(
    r"Train samples:\s*(?P<train>\d+)\s*\|\s*Val samples:\s*(?P<val>\d+)\s*\|\s*Eff Batch:\s*(?P<eff_batch>\d+)"
)
_PARAMS_RE = re.compile(
    r"DomainSensor \(Model B\):\s*(?P<domain>[\d,]+)\s*\|\s*"
    r"LogicReasoner \(Model A\):\s*(?P<logic>[\d,]+)\s*\|\s*Total:\s*(?P<total>[\d,]+)"
)
_DISC_CHECK_RE = re.compile(
    r"\[DISCREPANCY-CHECK\]\s*(?P<msg>.+?:\s*(?P<clean>\d+)/(?P<total>\d+)\s*clean\s*\(errors=(?P<errors>\d+)\))"
)
_EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s*\((?P<duration>[\d.]+)s\)\s*\|\s*"
    r"NLL:\s*(?P<nll>[\d.]+)\s*\|\s*"
    r"DomTok:\s*(?P<dom_tok>[\d.]+)%\s*\|\s*"
    r"(?:TrainBIO:\s*(?P<train_bio>[\d.]+)%\s*\|\s*)?"
    r"(?:TrainRelF1:\s*(?P<train_rel_f1>[\d.]+)%\s*\|\s*)?"
    r"BIO:\s*(?P<val_bio>[\d.]+)%\s*\|\s*"
    r"Mod:\s*(?P<mod>[\d.]+)%\s*\|\s*"
    r"F1:\s*(?P<f1>[\d.]+)%\s*\|\s*"
    r"RelF1:\s*(?P<val_rel_f1>[\d.]+)%\s*\|\s*"
    r"Unit:\s*(?P<unit>[\d.]+)%\s*\|\s*"
    r"EM:\s*(?P<em>[\d.]+)%\s*\|\s*"
    r"SemPass:\s*(?P<sem_pass>[\d.]+)%\s*\|\s*"
    r"Disc:\s*(?P<disc>\d+)"
    r"(?P<best>\s*\[BEST SAVED\])?"
)
_RL_EPOCH_RE = re.compile(
    r"RL Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s*"
    r"(?:\((?P<duration>[\d.]+)s\)\s*)?\|\s*"
    r"Mean USVM Reward:\s*(?P<mean_reward>[-+]?[\d.]+)\s*\|\s*"
    r"Vector Pass:\s*(?P<vector_pass>[\d.]+)%\s*\|\s*"
    r"BIO Agree:\s*(?P<bio_agree>[\d.]+)%"
    r"(?P<best>\s*\[BEST SAVED\])?"
)
_DATASET_GEN_RE = re.compile(
    r"Generated\s+(?P<samples>\d+)\s+samples\s+->\s+(?P<path>.+)"
)


def _format_iso_utc(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _default_search_dirs(extra_dirs: list[str] | None = None) -> list[Path]:
    dirs: list[Path] = []
    if extra_dirs is not None:
        for raw in extra_dirs:
            p = Path(raw).expanduser()
            if p.exists():
                dirs.append(p)
        return [d.resolve() for d in dirs]

    brain_root = Path.home() / ".gemini" / "antigravity" / "brain"
    if brain_root.exists():
        conv_dirs: list[tuple[float, Path]] = []
        for conv_dir in brain_root.iterdir():
            tasks_dir = conv_dir / ".system_generated" / "tasks"
            if tasks_dir.is_dir():
                try:
                    conv_dirs.append((tasks_dir.stat().st_mtime, tasks_dir))
                except OSError:
                    continue
        conv_dirs.sort(key=lambda pair: pair[0], reverse=True)
        for _, tasks_dir in conv_dirs[:6]:
            dirs.append(tasks_dir)

    cwd_logs = Path.cwd() / "logs"
    if cwd_logs.is_dir():
        dirs.append(cwd_logs)
    seen: set[str] = set()
    unique: list[Path] = []
    for d in dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _looks_like_training_log(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            head = f.read(16384)
    except OSError:
        return False
    return (
        "train_dual_stream.py" in head
        or "DomainSensor (Model B):" in head
        or "[DISCREPANCY-CHECK]" in head
        or "Epoch " in head
        or "RL Epoch " in head
    )


def discover_training_logs(
    search_dirs: list[str] | None = None,
    pinned_log: Path | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return candidate training log files ordered newest first."""
    candidates: dict[str, tuple[float, int, Path]] = {}
    if pinned_log is not None and pinned_log.exists():
        resolved = pinned_log.resolve()
        try:
            st = resolved.stat()
            candidates[str(resolved)] = (st.st_mtime, st.st_size, resolved)
        except OSError:
            pass

    for directory in _default_search_dirs(search_dirs):
        try:
            for item in directory.glob("*.log"):
                if not item.is_file():
                    continue
                resolved = item.resolve()
                key = str(resolved)
                if key not in candidates:
                    st = resolved.stat()
                    candidates[key] = (st.st_mtime, st.st_size, resolved)
        except OSError:
            continue

    items_with_stat = sorted(candidates.values(), key=lambda triple: triple[0], reverse=True)
    results: list[dict[str, Any]] = []
    pinned_resolved = str(pinned_log.resolve()) if (pinned_log is not None and pinned_log.exists()) else None
    for mtime, size, path in items_with_stat[: max(limit * 4, 40)]:
        if str(path) != pinned_resolved and not _looks_like_training_log(path):
            continue
        results.append(
            {
                "name": path.name,
                "path": str(path),
                "mtime": mtime,
                "mtime_utc": _format_iso_utc(mtime),
                "size_bytes": size,
            }
        )
        if len(results) >= limit:
            break
    return results


def _read_discrepancies(checkpoint_path: Path | None, max_items: int = 25) -> list[dict[str, Any]]:
    if checkpoint_path is None:
        return []
    disc_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_discrepancies.jsonl")
    if not disc_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = disc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines[-max_items:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            records.append({"raw": stripped})
    records.reverse()
    return records


def _inspect_checkpoint(checkpoint_path: Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    exists = checkpoint_path.is_file()
    disc_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_discrepancies.jsonl")
    info: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": exists,
        "size_bytes": None,
        "mtime": None,
        "mtime_utc": None,
        "discrepancy_log_exists": disc_path.is_file(),
    }
    if exists:
        try:
            st = checkpoint_path.stat()
            info["size_bytes"] = st.st_size
            info["mtime"] = st.st_mtime
            info["mtime_utc"] = _format_iso_utc(st.st_mtime)
        except OSError:
            pass
    return info


def parse_training_log(
    raw_text: str,
    *,
    log_path: str | None = None,
    checkpoint_path: Path | None = None,
    file_mtime: float | None = None,
    now: float | None = None,
    stale_after_sec: float = 240.0,
    tail_limit: int = 120,
) -> dict[str, Any]:
    """Parse training log lines into structured telemetry for live monitoring."""
    now_ts = time.time() if now is None else now
    lines = [line.rstrip("\r") for line in raw_text.splitlines()]
    non_empty_lines = [line for line in lines if line.strip()]

    metadata: dict[str, Any] = {
        "device": None,
        "batch_size": None,
        "amp": None,
        "architecture": None,
        "raw_vocab": None,
        "logic_vocab": None,
        "train_samples": None,
        "val_samples": None,
        "eff_batch": None,
        "params": None,
        "discrepancy_check": None,
        "dataset_generated": None,
    }
    epochs: list[dict[str, Any]] = []
    rl_epochs: list[dict[str, Any]] = []
    completion_markers: list[str] = []

    for line in non_empty_lines:
        if m := _DEVICE_RE.search(line):
            metadata["device"] = m.group("device").strip()
            metadata["batch_size"] = int(m.group("batch"))
            metadata["amp"] = m.group("amp") == "True"
            continue
        if m := _ARCH_RE.search(line):
            metadata["architecture"] = m.group("arch").strip()
            metadata["raw_vocab"] = int(m.group("raw_vocab"))
            metadata["logic_vocab"] = int(m.group("logic_vocab"))
            continue
        if m := _SPLIT_RE.search(line):
            metadata["train_samples"] = int(m.group("train"))
            metadata["val_samples"] = int(m.group("val"))
            metadata["eff_batch"] = int(m.group("eff_batch"))
            continue
        if m := _PARAMS_RE.search(line):
            metadata["params"] = {
                "domain_sensor": int(m.group("domain").replace(",", "")),
                "logic_reasoner": int(m.group("logic").replace(",", "")),
                "total": int(m.group("total").replace(",", "")),
            }
            continue
        if m := _DISC_CHECK_RE.search(line):
            metadata["discrepancy_check"] = {
                "message": m.group("msg").strip(),
                "clean": int(m.group("clean")),
                "total": int(m.group("total")),
                "errors": int(m.group("errors")),
                "passed": int(m.group("errors")) == 0,
            }
            continue
        if m := _DATASET_GEN_RE.search(line):
            metadata["dataset_generated"] = {
                "samples": int(m.group("samples")),
                "path": m.group("path").strip(),
            }
            continue
        if m := _EPOCH_RE.search(line):
            train_bio = float(m.group("train_bio")) if m.group("train_bio") is not None else None
            train_rel = float(m.group("train_rel_f1")) if m.group("train_rel_f1") is not None else None
            val_bio = float(m.group("val_bio"))
            val_rel = float(m.group("val_rel_f1"))
            epochs.append(
                {
                    "epoch": int(m.group("epoch")),
                    "total_epochs": int(m.group("total_epochs")),
                    "duration_sec": float(m.group("duration")),
                    "nll": float(m.group("nll")),
                    "dom_tok": float(m.group("dom_tok")),
                    "train_bio": train_bio,
                    "train_rel_f1": train_rel,
                    "val_bio": val_bio,
                    "mod": float(m.group("mod")),
                    "f1": float(m.group("f1")),
                    "val_rel_f1": val_rel,
                    "unit": float(m.group("unit")),
                    "em": float(m.group("em")),
                    "sem_pass": float(m.group("sem_pass")),
                    "disc": int(m.group("disc")),
                    "best_saved": bool(m.group("best")),
                    "bio_gap": round(train_bio - val_bio, 2) if train_bio is not None else None,
                    "rel_gap": round(train_rel - val_rel, 2) if train_rel is not None else None,
                }
            )
            continue
        if m := _RL_EPOCH_RE.search(line):
            rl_epochs.append(
                {
                    "epoch": int(m.group("epoch")),
                    "total_epochs": int(m.group("total_epochs")),
                    "duration_sec": float(m.group("duration")) if m.group("duration") else None,
                    "mean_reward": float(m.group("mean_reward")),
                    "vector_pass": float(m.group("vector_pass")),
                    "bio_agree": float(m.group("bio_agree")),
                    "best_saved": bool(m.group("best")),
                }
            )
            continue
        if "Stage 1 Supervised complete" in line or "Stage 2 RL complete" in line or "Training complete" in line:
            completion_markers.append(line.strip())

    file_age_sec = max(0.0, now_ts - file_mtime) if file_mtime is not None else None

    stage = "SUPERVISED"
    if rl_epochs:
        stage = "RL"

    current_epoch = epochs[-1]["epoch"] if epochs else 0
    total_epochs = epochs[-1]["total_epochs"] if epochs else 0
    if rl_epochs:
        current_epoch = rl_epochs[-1]["epoch"]
        total_epochs = rl_epochs[-1]["total_epochs"]

    elapsed_sec = round(sum(e["duration_sec"] for e in epochs), 1)
    # Exclude epoch 1 from avg step time once epoch >= 2 because epoch 1 includes dataset/CUDA warmup.
    steady_durations = [e["duration_sec"] for e in epochs[1:]] if len(epochs) > 1 else [e["duration_sec"] for e in epochs]
    avg_epoch_sec = round(sum(steady_durations) / len(steady_durations), 2) if steady_durations else None

    remaining_epochs = max(0, total_epochs - current_epoch) if total_epochs > 0 else 0
    eta_sec = round(avg_epoch_sec * remaining_epochs, 1) if (avg_epoch_sec is not None and not rl_epochs) else None

    best_epoch = None
    best_saved_rows = [e for e in epochs if e["best_saved"]]
    if best_saved_rows:
        best_epoch = best_saved_rows[-1]
    elif epochs:
        best_epoch = max(epochs, key=lambda e: (e["val_rel_f1"], e["val_bio"], -e["nll"]))

    rl_done_markers = [m for m in completion_markers if "Stage 2 RL complete" in m or "Training complete" in m]
    sup_done_markers = [m for m in completion_markers if "Stage 1 Supervised complete" in m]

    if not non_empty_lines:
        status = "IDLE"
    elif rl_epochs:
        if rl_epochs[-1]["epoch"] >= rl_epochs[-1]["total_epochs"] or rl_done_markers:
            status = "COMPLETED"
        elif file_age_sec is not None and file_age_sec > stale_after_sec:
            status = "STALE"
        else:
            status = "RUNNING"
    elif (epochs and epochs[-1]["epoch"] >= epochs[-1]["total_epochs"]) or sup_done_markers or rl_done_markers:
        if epochs and epochs[-1]["epoch"] >= epochs[-1]["total_epochs"] and not (sup_done_markers or rl_done_markers) and (
            file_age_sec is not None and file_age_sec <= 10.0
        ):
            status = "RUNNING"
        else:
            status = "COMPLETED"
    elif file_age_sec is not None and file_age_sec > stale_after_sec:
        status = "STALE"
    else:
        status = "RUNNING"

    progress_pct = round((current_epoch / total_epochs) * 100.0, 1) if total_epochs > 0 else 0.0

    deltas: dict[str, float | None] = {}
    if len(epochs) >= 2:
        prev = epochs[-2]
        curr = epochs[-1]
        for key in ("nll", "val_bio", "val_rel_f1", "f1", "unit", "mod", "dom_tok", "em", "sem_pass", "disc"):
            deltas[key] = round(curr[key] - prev[key], 4)
    elif len(epochs) == 1:
        for key in ("nll", "val_bio", "val_rel_f1", "f1", "unit", "mod", "dom_tok", "em", "sem_pass", "disc"):
            deltas[key] = None

    return {
        "generated_at_utc": _format_iso_utc(now_ts),
        "log_path": log_path,
        "log_name": Path(log_path).name if log_path else None,
        "file_mtime": file_mtime,
        "file_mtime_utc": _format_iso_utc(file_mtime),
        "file_age_sec": round(file_age_sec, 1) if file_age_sec is not None else None,
        "status": status,
        "stage": stage,
        "metadata": metadata,
        "progress": {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "progress_pct": progress_pct,
            "elapsed_sec": elapsed_sec,
            "avg_epoch_sec": avg_epoch_sec,
            "eta_sec": eta_sec,
        },
        "latest": epochs[-1] if epochs else None,
        "deltas_from_prev": deltas,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "rl_epochs": rl_epochs,
        "checkpoint": _inspect_checkpoint(checkpoint_path),
        "discrepancies": _read_discrepancies(checkpoint_path),
        "tail_lines": non_empty_lines[-tail_limit:],
        "total_log_lines": len(non_empty_lines),
    }


def load_telemetry(
    *,
    requested_log: str | None = None,
    default_log: Path | None = None,
    checkpoint_path: Path | None = None,
    search_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve target log (explicit or auto-discovered latest) and return telemetry payload."""
    available_logs = discover_training_logs(search_dirs=search_dirs, pinned_log=default_log)
    target_path: Path | None = None

    if requested_log and requested_log not in ("auto", "default"):
        candidate = Path(requested_log).expanduser()
        if candidate.is_file():
            target_path = candidate.resolve()
    elif (not requested_log or requested_log == "default") and default_log is not None and default_log.is_file():
        target_path = default_log.resolve()
    elif available_logs:
        target_path = Path(available_logs[0]["path"])

    if target_path is None or not target_path.is_file():
        payload = parse_training_log(
            "",
            log_path=str(target_path) if target_path else None,
            checkpoint_path=checkpoint_path,
        )
        payload["available_logs"] = available_logs
        return payload

    try:
        st = target_path.stat()
        raw_text = target_path.read_text(encoding="utf-8", errors="replace")
        file_mtime = st.st_mtime
    except OSError as exc:
        payload = parse_training_log(
            f"[MONITOR ERROR] Failed to read {target_path}: {exc}",
            log_path=str(target_path),
            checkpoint_path=checkpoint_path,
        )
        payload["available_logs"] = available_logs
        return payload

    payload = parse_training_log(
        raw_text,
        log_path=str(target_path),
        checkpoint_path=checkpoint_path,
        file_mtime=file_mtime,
    )
    payload["available_logs"] = available_logs
    return payload


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MudraForge Live Training Monitor</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #11182c;
    --panel-alt: #162038;
    --border: #253352;
    --text: #e6edf7;
    --muted: #8ea0c0;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #a78bfa;
    --teal: #2dd4bf;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    --sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.45;
  }
  .shell {
    max-width: 1520px;
    margin: 0 auto;
    padding: 16px 20px 28px;
  }
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 14px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .pulse {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
  }
  .pulse.running {
    background: var(--green);
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
    animation: pulse-ring 1.8s infinite;
  }
  .pulse.completed { background: var(--accent); }
  .pulse.stale { background: var(--amber); }
  .pulse.idle { background: var(--red); }
  @keyframes pulse-ring {
    0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
    70% { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
  }
  h1 {
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.2px;
  }
  .subtitle {
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    margin-top: 2px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    border: 1px solid var(--border);
    background: var(--panel-alt);
  }
  .badge.running { color: var(--green); border-color: rgba(34, 197, 94, 0.4); }
  .badge.completed { color: var(--accent); border-color: rgba(56, 189, 248, 0.4); }
  .badge.stale { color: var(--amber); border-color: rgba(245, 158, 11, 0.4); }
  .badge.idle { color: var(--red); border-color: rgba(239, 68, 68, 0.4); }
  select, button, input {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-family: var(--sans);
  }
  button { cursor: pointer; font-weight: 600; }
  button:hover { border-color: var(--accent); }
  .progress-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 14px;
  }
  .progress-head {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
    font-family: var(--mono);
    font-size: 12px;
  }
  .bar-track {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: #090e1c;
    border: 1px solid var(--border);
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--teal));
    transition: width 0.35s ease;
  }
  .meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 10px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
  }
  .meta-strip span strong { color: var(--text); font-weight: 600; }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
  }
  .kpi {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 13px;
  }
  .kpi-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
  }
  .kpi-value {
    font-family: var(--mono);
    font-size: 21px;
    font-weight: 700;
  }
  .kpi-sub {
    margin-top: 4px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    gap: 6px;
  }
  .delta-up { color: var(--green); }
  .delta-down { color: var(--red); }
  .delta-flat { color: var(--muted); }
  .charts-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 14px;
  }
  @media (max-width: 1024px) {
    .charts-grid { grid-template-columns: 1fr; }
  }
  .chart-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    position: relative;
  }
  .chart-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }
  .chart-title {
    font-size: 13px;
    font-weight: 700;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .swatch {
    width: 10px;
    height: 3px;
    border-radius: 2px;
    display: inline-block;
  }
  .canvas-wrap {
    position: relative;
    width: 100%;
    height: 235px;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  .tooltip {
    position: fixed;
    pointer-events: none;
    background: rgba(9, 14, 28, 0.96);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 7px 9px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
    z-index: 50;
    display: none;
    white-space: nowrap;
  }
  .panel-tabs {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .tab-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    background: var(--panel-alt);
    border-bottom: 1px solid var(--border);
    padding: 8px 12px;
  }
  .tab-buttons { display: flex; gap: 6px; }
  .tab-btn.active {
    background: var(--accent);
    color: #06101f;
    border-color: var(--accent);
  }
  .tab-pane { display: none; padding: 12px; }
  .tab-pane.active { display: block; }
  .table-wrap { overflow-x: auto; max-height: 360px; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 11.5px;
  }
  th, td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    white-space: nowrap;
  }
  th {
    position: sticky;
    top: 0;
    background: var(--panel-alt);
    color: var(--muted);
    font-weight: 700;
    z-index: 2;
  }
  th:first-child, td:first-child { text-align: left; }
  tr.best-row { background: rgba(34, 197, 94, 0.08); }
  .tag-best {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(34, 197, 94, 0.18);
    color: var(--green);
    font-size: 10px;
    font-weight: 700;
  }
  .log-box {
    background: #070b16;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 11.5px;
    max-height: 360px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .hidden { display: none !important; }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div class="brand">
      <div id="statusPulse" class="pulse idle"></div>
      <div>
        <h1>MudraForge Dual-Stream Live Training Monitor</h1>
        <div id="logSubtitle" class="subtitle">Resolving active training log...</div>
      </div>
    </div>
    <div class="controls">
      <span id="statusBadge" class="badge idle">IDLE</span>
      <select id="logSelector" title="Select training log"></select>
      <button id="pauseBtn" type="button">Pause Live</button>
      <button id="refreshBtn" type="button">Refresh Now</button>
      <span id="lastPoll" class="subtitle">Poll: 1.5s</span>
    </div>
  </div>

  <div class="progress-card">
    <div class="progress-head">
      <span id="epochProgressText">Epoch 0 / 0 (0.0%)</span>
      <span id="timingText">Elapsed: -- | Avg/Epoch: -- | ETA: --</span>
      <span id="bestCheckpointText">Best Saved: --</span>
    </div>
    <div class="bar-track"><div id="progressBar" class="bar-fill"></div></div>
    <div id="metaStrip" class="meta-strip"></div>
  </div>

  <div id="kpiGrid" class="kpi-grid"></div>

  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">NLL Loss &amp; Epoch Duration</span>
        <div class="legend">
          <span class="legend-item"><i class="swatch" style="background:#38bdf8"></i>NLL</span>
          <span class="legend-item"><i class="swatch" style="background:#f59e0b"></i>Sec/Epoch (right)</span>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="chartLoss"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">BIO &amp; Relation F1 (Train vs Val)</span>
        <div class="legend">
          <span class="legend-item"><i class="swatch" style="background:#22c55e"></i>Val BIO</span>
          <span class="legend-item"><i class="swatch" style="background:#86efac"></i>TrainBIO</span>
          <span class="legend-item"><i class="swatch" style="background:#a78bfa"></i>Val RelF1</span>
          <span class="legend-item"><i class="swatch" style="background:#ddd6fe"></i>TrainRelF1</span>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="chartBioRel"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">Fine-Grained Structural Accuracies (%)</span>
        <div class="legend">
          <span class="legend-item"><i class="swatch" style="background:#38bdf8"></i>Role F1</span>
          <span class="legend-item"><i class="swatch" style="background:#2dd4bf"></i>Span Unit</span>
          <span class="legend-item"><i class="swatch" style="background:#f59e0b"></i>Modality</span>
          <span class="legend-item"><i class="swatch" style="background:#f472b6"></i>DomTok</span>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="chartAcc"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">Logic Exact Match, SemPass &amp; Discrepancies</span>
        <div class="legend">
          <span class="legend-item"><i class="swatch" style="background:#22c55e"></i>EM (%)</span>
          <span class="legend-item"><i class="swatch" style="background:#38bdf8"></i>SemPass (%)</span>
          <span class="legend-item"><i class="swatch" style="background:#ef4444"></i>Disc (right)</span>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="chartExec"></canvas></div>
    </div>
  </div>

  <div id="rlSection" class="chart-card hidden" style="margin-bottom:14px;">
    <div class="chart-head">
      <span class="chart-title">Stage 2 USVM Reinforcement Learning</span>
      <div class="legend">
        <span class="legend-item"><i class="swatch" style="background:#22c55e"></i>Vector Pass (%)</span>
        <span class="legend-item"><i class="swatch" style="background:#38bdf8"></i>BIO Agree (%)</span>
        <span class="legend-item"><i class="swatch" style="background:#f59e0b"></i>Mean Reward (right)</span>
      </div>
    </div>
    <div class="canvas-wrap"><canvas id="chartRl"></canvas></div>
  </div>

  <div class="panel-tabs">
    <div class="tab-bar">
      <div class="tab-buttons">
        <button type="button" class="tab-btn active" data-tab="epochsTab">Epoch History</button>
        <button type="button" class="tab-btn" data-tab="logTab">Live Log Tail</button>
        <button type="button" class="tab-btn" data-tab="discTab">Discrepancies (<span id="discTabCount">0</span>)</button>
      </div>
      <div class="controls">
        <label style="color:var(--muted);font-size:12px;">
          <input type="checkbox" id="newestFirst" checked /> Newest First
        </label>
        <input id="logFilter" type="text" placeholder="Filter log lines..." style="width:190px;" />
      </div>
    </div>
    <div id="epochsTab" class="tab-pane active">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Epoch</th>
              <th>Sec</th>
              <th>NLL</th>
              <th>DomTok</th>
              <th>TrainBIO</th>
              <th>Val BIO</th>
              <th>BIO Gap</th>
              <th>Mod</th>
              <th>Role F1</th>
              <th>TrainRelF1</th>
              <th>Val RelF1</th>
              <th>Rel Gap</th>
              <th>Unit</th>
              <th>EM</th>
              <th>SemPass</th>
              <th>Disc</th>
            </tr>
          </thead>
          <tbody id="epochTableBody"></tbody>
        </table>
      </div>
    </div>
    <div id="logTab" class="tab-pane">
      <div id="logBox" class="log-box">Waiting for log lines...</div>
    </div>
    <div id="discTab" class="tab-pane">
      <div id="discBox" class="log-box">No discrepancy JSONL records found yet.</div>
    </div>
  </div>
</div>
<div id="tooltip" class="tooltip"></div>

<script>
(function() {
  const state = {
    paused: false,
    selectedLog: "default",
    data: null,
    timerId: null
  };

  const els = {
    pulse: document.getElementById("statusPulse"),
    subtitle: document.getElementById("logSubtitle"),
    badge: document.getElementById("statusBadge"),
    logSelector: document.getElementById("logSelector"),
    pauseBtn: document.getElementById("pauseBtn"),
    refreshBtn: document.getElementById("refreshBtn"),
    lastPoll: document.getElementById("lastPoll"),
    epochProgressText: document.getElementById("epochProgressText"),
    timingText: document.getElementById("timingText"),
    bestCheckpointText: document.getElementById("bestCheckpointText"),
    progressBar: document.getElementById("progressBar"),
    metaStrip: document.getElementById("metaStrip"),
    kpiGrid: document.getElementById("kpiGrid"),
    epochTableBody: document.getElementById("epochTableBody"),
    logBox: document.getElementById("logBox"),
    discBox: document.getElementById("discBox"),
    discTabCount: document.getElementById("discTabCount"),
    newestFirst: document.getElementById("newestFirst"),
    logFilter: document.getElementById("logFilter"),
    rlSection: document.getElementById("rlSection"),
    tooltip: document.getElementById("tooltip")
  };

  function fmtDur(sec) {
    if (sec === null || sec === undefined || Number.isNaN(sec)) return "--";
    const s = Math.max(0, Math.round(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rem = s % 60;
    if (h > 0) return `${h}h ${m}m ${rem}s`;
    if (m > 0) return `${m}m ${rem}s`;
    return `${rem}s`;
  }

  function fmtNum(v, decimals = 2, suffix = "") {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    return Number(v).toFixed(decimals) + suffix;
  }

  function fmtDelta(v, decimals = 2, suffix = "", lowerBetter = false) {
    if (v === null || v === undefined || Number.isNaN(v)) {
      return { text: "first epoch", cls: "delta-flat" };
    }
    const sign = v > 0 ? "+" : "";
    const text = `${sign}${Number(v).toFixed(decimals)}${suffix} vs prev`;
    if (Math.abs(v) < 1e-6) return { text, cls: "delta-flat" };
    const good = lowerBetter ? v < 0 : v > 0;
    return { text, cls: good ? "delta-up" : "delta-down" };
  }

  function updateLogSelector(logs, activePath) {
    const prev = state.selectedLog;
    const options = [{ value: "default", label: "Pinned / Active Run" }, { value: "auto", label: "Auto-Latest Log" }];
    for (const item of logs || []) {
      options.push({ value: item.path, label: `${item.name} (${Math.round((item.size_bytes || 0)/1024)} KB)` });
    }
    const serialized = JSON.stringify(options);
    if (els.logSelector.dataset.opts !== serialized) {
      els.logSelector.dataset.opts = serialized;
      els.logSelector.innerHTML = "";
      for (const opt of options) {
        const el = document.createElement("option");
        el.value = opt.value;
        el.textContent = opt.label;
        els.logSelector.appendChild(el);
      }
      if ([...els.logSelector.options].some(o => o.value === prev)) {
        els.logSelector.value = prev;
      }
    }
    els.subtitle.textContent = activePath || "No active training log found";
  }

  function renderKpis(data) {
    const l = data.latest;
    const d = data.deltas_from_prev || {};
    const best = data.best_epoch;
    if (!l) {
      els.kpiGrid.innerHTML = `<div class="kpi"><div class="kpi-label">Status</div><div class="kpi-value">Waiting</div><div class="kpi-sub"><span>No completed epochs parsed yet</span></div></div>`;
      return;
    }

    const cards = [
      {
        label: "NLL Loss",
        value: fmtNum(l.nll, 4),
        delta: fmtDelta(d.nll, 4, "", true),
        sub: best ? `Best Ep ${best.epoch}: ${fmtNum(best.nll, 4)}` : ""
      },
      {
        label: "Val BIO / TrainBIO",
        value: fmtNum(l.val_bio, 2, "%"),
        delta: fmtDelta(d.val_bio, 2, "%"),
        sub: `Train: ${fmtNum(l.train_bio, 2, "%")} | Gap: ${fmtNum(l.bio_gap, 2, "%")}`
      },
      {
        label: "Val RelF1 / TrainRelF1",
        value: fmtNum(l.val_rel_f1, 2, "%"),
        delta: fmtDelta(d.val_rel_f1, 2, "%"),
        sub: `Train: ${fmtNum(l.train_rel_f1, 2, "%")} | Gap: ${fmtNum(l.rel_gap, 2, "%")}`
      },
      {
        label: "Role F1",
        value: fmtNum(l.f1, 2, "%"),
        delta: fmtDelta(d.f1, 2, "%"),
        sub: best ? `Best Ep ${best.epoch}: ${fmtNum(best.f1, 2, "%")}` : ""
      },
      {
        label: "Span Unit Acc",
        value: fmtNum(l.unit, 2, "%"),
        delta: fmtDelta(d.unit, 2, "%"),
        sub: `Modality: ${fmtNum(l.mod, 2, "%")}`
      },
      {
        label: "DomTok Acc",
        value: fmtNum(l.dom_tok, 2, "%"),
        delta: fmtDelta(d.dom_tok, 2, "%"),
        sub: `Step Time: ${fmtNum(l.duration_sec, 1, "s")}`
      },
      {
        label: "Exact Match (EM)",
        value: fmtNum(l.em, 2, "%"),
        delta: fmtDelta(d.em, 2, "%"),
        sub: `SemPass: ${fmtNum(l.sem_pass, 1, "%")}`
      },
      {
        label: "Val Discrepancies",
        value: String(l.disc),
        delta: fmtDelta(d.disc, 0, "", true),
        sub: best ? `Best Ep ${best.epoch} Disc: ${best.disc}` : ""
      }
    ];

    els.kpiGrid.innerHTML = cards.map(c => `
      <div class="kpi">
        <div class="kpi-label">${c.label}</div>
        <div class="kpi-value">${c.value}</div>
        <div class="kpi-sub">
          <span class="${c.delta.cls}">${c.delta.text}</span>
          <span>${c.sub}</span>
        </div>
      </div>
    `).join("");
  }

  function renderMeta(data) {
    const m = data.metadata || {};
    const parts = [];
    if (m.device) parts.push(`<span>Device: <strong>${m.device}</strong></span>`);
    if (m.architecture) parts.push(`<span>Arch: <strong>${m.architecture}</strong></span>`);
    if (m.params) parts.push(`<span>Params: <strong>${m.params.total.toLocaleString()}</strong> (DS ${m.params.domain_sensor.toLocaleString()} / LR ${m.params.logic_reasoner.toLocaleString()})</span>`);
    if (m.train_samples !== null) parts.push(`<span>Split: <strong>${m.train_samples} train / ${m.val_samples} val</strong> (batch ${m.eff_batch || m.batch_size})</span>`);
    if (m.discrepancy_check) {
      const color = m.discrepancy_check.passed ? "var(--green)" : "var(--red)";
      parts.push(`<span>Contract Check: <strong style="color:${color}">${m.discrepancy_check.message}</strong></span>`);
    }
    if (data.checkpoint && data.checkpoint.exists) {
      const mb = (data.checkpoint.size_bytes / (1024 * 1024)).toFixed(2);
      parts.push(`<span>Checkpoint: <strong>${mb} MB</strong> (${data.checkpoint.mtime_utc || ""})</span>`);
    }
    els.metaStrip.innerHTML = parts.join("");
  }

  function drawChart(canvasId, rows, leftSeries, rightSeries = [], opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.max(240, Math.floor(rect.width));
    const h = Math.max(180, Math.floor(rect.height));
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = { left: 46, right: rightSeries.length ? 46 : 16, top: 12, bottom: 26 };
    const pw = Math.max(40, w - pad.left - pad.right);
    const ph = Math.max(40, h - pad.top - pad.bottom);

    ctx.strokeStyle = "#1e2942";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ph * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    if (!rows || rows.length === 0) {
      ctx.fillStyle = "#8ea0c0";
      ctx.font = "12px monospace";
      ctx.fillText("Waiting for epoch metrics...", pad.left + 12, pad.top + ph / 2);
      canvas._chartMeta = null;
      return;
    }

    function computeRange(seriesList, fixedMin, fixedMax) {
      let minV = Infinity;
      let maxV = -Infinity;
      for (const s of seriesList) {
        for (const r of rows) {
          const val = r[s.key];
          if (val !== null && val !== undefined && !Number.isNaN(val)) {
            minV = Math.min(minV, val);
            maxV = Math.max(maxV, val);
          }
        }
      }
      if (!Number.isFinite(minV)) return [0, 1];
      if (fixedMin !== undefined) minV = fixedMin;
      if (fixedMax !== undefined) maxV = fixedMax;
      if (Math.abs(maxV - minV) < 1e-6) {
        minV = Math.max(0, minV - 1);
        maxV = maxV + 1;
      } else if (fixedMin === undefined || fixedMax === undefined) {
        const span = maxV - minV;
        if (fixedMin === undefined) minV = Math.max(0, minV - span * 0.08);
        if (fixedMax === undefined) maxV = maxV + span * 0.08;
      }
      return [minV, maxV];
    }

    const [lMin, lMax] = computeRange(leftSeries, opts.leftMin, opts.leftMax);
    const [rMin, rMax] = rightSeries.length ? computeRange(rightSeries, opts.rightMin, opts.rightMax) : [0, 1];

    ctx.fillStyle = "#8ea0c0";
    ctx.font = "10.5px monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const val = lMax - ((lMax - lMin) * i) / 4;
      const y = pad.top + (ph * i) / 4 + 3;
      ctx.fillText(val.toFixed(opts.leftDecimals ?? 1), pad.left - 6, y);
    }
    if (rightSeries.length) {
      ctx.textAlign = "left";
      for (let i = 0; i <= 4; i++) {
        const val = rMax - ((rMax - rMin) * i) / 4;
        const y = pad.top + (ph * i) / 4 + 3;
        ctx.fillText(val.toFixed(opts.rightDecimals ?? 0), w - pad.right + 6, y);
      }
    }

    const xForIndex = idx => rows.length === 1 ? pad.left + pw / 2 : pad.left + (pw * idx) / (rows.length - 1);
    const yLeft = v => pad.top + ph - ((v - lMin) / (lMax - lMin)) * ph;
    const yRight = v => pad.top + ph - ((v - rMin) / (rMax - rMin)) * ph;

    ctx.textAlign = "center";
    ctx.fillStyle = "#8ea0c0";
    const step = Math.max(1, Math.ceil(rows.length / 8));
    for (let i = 0; i < rows.length; i += step) {
      ctx.fillText("E" + rows[i].epoch, xForIndex(i), h - 8);
    }
    if (rows.length > 1 && (rows.length - 1) % step !== 0) {
      ctx.fillText("E" + rows[rows.length - 1].epoch, xForIndex(rows.length - 1), h - 8);
    }

    function drawSeries(series, yMapper) {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = series.dashed ? 1.5 : 2.1;
      ctx.setLineDash(series.dashed ? [4, 3] : []);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < rows.length; i++) {
        const val = rows[i][series.key];
        if (val === null || val === undefined || Number.isNaN(val)) continue;
        const x = xForIndex(i);
        const y = yMapper(val);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = series.color;
      for (let i = 0; i < rows.length; i++) {
        const val = rows[i][series.key];
        if (val === null || val === undefined || Number.isNaN(val)) continue;
        ctx.beginPath();
        ctx.arc(xForIndex(i), yMapper(val), rows[i].best_saved && series.highlightBest ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (const s of leftSeries) drawSeries(s, yLeft);
    for (const s of rightSeries) drawSeries(s, yRight);

    canvas._chartMeta = { rows, leftSeries, rightSeries, pad, pw, ph, xForIndex };
  }

  function attachChartHover(canvasId) {
    const canvas = document.getElementById(canvasId);
    canvas.addEventListener("mousemove", ev => {
      const meta = canvas._chartMeta;
      if (!meta || !meta.rows.length) {
        els.tooltip.style.display = "none";
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      let nearestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < meta.rows.length; i++) {
        const d = Math.abs(meta.xForIndex(i) - mx);
        if (d < bestDist) {
          bestDist = d;
          nearestIdx = i;
        }
      }
      const row = meta.rows[nearestIdx];
      const lines = [`Epoch ${row.epoch}${row.best_saved ? " [BEST]" : ""}`];
      for (const s of [...meta.leftSeries, ...meta.rightSeries]) {
        const v = row[s.key];
        if (v !== null && v !== undefined) {
          lines.push(`${s.label}: ${Number(v).toFixed(s.decimals ?? 2)}${s.suffix || ""}`);
        }
      }
      els.tooltip.textContent = lines.join("\n");
      els.tooltip.style.whiteSpace = "pre";
      els.tooltip.style.display = "block";
      els.tooltip.style.left = Math.min(window.innerWidth - 220, ev.clientX + 14) + "px";
      els.tooltip.style.top = Math.max(8, ev.clientY - 24) + "px";
    });
    canvas.addEventListener("mouseleave", () => {
      els.tooltip.style.display = "none";
    });
  }

  function renderCharts(data) {
    const epochs = data.epochs || [];
    drawChart(
      "chartLoss",
      epochs,
      [{ key: "nll", label: "NLL", color: "#38bdf8", decimals: 4, highlightBest: true }],
      [{ key: "duration_sec", label: "Sec", color: "#f59e0b", decimals: 1, suffix: "s" }],
      { leftDecimals: 2, rightDecimals: 0 }
    );
    drawChart(
      "chartBioRel",
      epochs,
      [
        { key: "val_bio", label: "Val BIO", color: "#22c55e", suffix: "%", highlightBest: true },
        { key: "train_bio", label: "TrainBIO", color: "#86efac", suffix: "%", dashed: true },
        { key: "val_rel_f1", label: "Val RelF1", color: "#a78bfa", suffix: "%", highlightBest: true },
        { key: "train_rel_f1", label: "TrainRelF1", color: "#ddd6fe", suffix: "%", dashed: true }
      ],
      [],
      { leftMin: 0, leftMax: 100, leftDecimals: 0 }
    );
    drawChart(
      "chartAcc",
      epochs,
      [
        { key: "f1", label: "Role F1", color: "#38bdf8", suffix: "%" },
        { key: "unit", label: "Unit", color: "#2dd4bf", suffix: "%" },
        { key: "mod", label: "Modality", color: "#f59e0b", suffix: "%" },
        { key: "dom_tok", label: "DomTok", color: "#f472b6", suffix: "%" }
      ],
      [],
      { leftMin: 0, leftMax: 100, leftDecimals: 0 }
    );
    drawChart(
      "chartExec",
      epochs,
      [
        { key: "em", label: "EM", color: "#22c55e", suffix: "%" },
        { key: "sem_pass", label: "SemPass", color: "#38bdf8", suffix: "%" }
      ],
      [{ key: "disc", label: "Disc", color: "#ef4444", decimals: 0 }],
      { leftMin: 0, leftMax: 100, leftDecimals: 0, rightDecimals: 0 }
    );

    const rl = data.rl_epochs || [];
    if (rl.length > 0) {
      els.rlSection.classList.remove("hidden");
      drawChart(
        "chartRl",
        rl,
        [
          { key: "vector_pass", label: "Vector Pass", color: "#22c55e", suffix: "%", highlightBest: true },
          { key: "bio_agree", label: "BIO Agree", color: "#38bdf8", suffix: "%" }
        ],
        [{ key: "mean_reward", label: "Reward", color: "#f59e0b", decimals: 3 }],
        { leftMin: 0, leftMax: 100, leftDecimals: 0, rightDecimals: 2 }
      );
    } else {
      els.rlSection.classList.add("hidden");
    }
  }

  function renderTablesAndLogs(data) {
    const rows = [...(data.epochs || [])];
    if (els.newestFirst.checked) rows.reverse();
    els.epochTableBody.innerHTML = rows.map(r => `
      <tr class="${r.best_saved ? "best-row" : ""}">
        <td>Ep ${r.epoch}/${r.total_epochs} ${r.best_saved ? '<span class="tag-best">BEST</span>' : ""}</td>
        <td>${fmtNum(r.duration_sec, 1, "s")}</td>
        <td>${fmtNum(r.nll, 4)}</td>
        <td>${fmtNum(r.dom_tok, 2, "%")}</td>
        <td>${fmtNum(r.train_bio, 2, "%")}</td>
        <td>${fmtNum(r.val_bio, 2, "%")}</td>
        <td>${fmtNum(r.bio_gap, 2, "%")}</td>
        <td>${fmtNum(r.mod, 2, "%")}</td>
        <td>${fmtNum(r.f1, 2, "%")}</td>
        <td>${fmtNum(r.train_rel_f1, 2, "%")}</td>
        <td>${fmtNum(r.val_rel_f1, 2, "%")}</td>
        <td>${fmtNum(r.rel_gap, 2, "%")}</td>
        <td>${fmtNum(r.unit, 2, "%")}</td>
        <td>${fmtNum(r.em, 2, "%")}</td>
        <td>${fmtNum(r.sem_pass, 1, "%")}</td>
        <td>${r.disc}</td>
      </tr>
    `).join("") || `<tr><td colspan="16" style="text-align:center;color:var(--muted);">No epochs completed yet</td></tr>`;

    const q = (els.logFilter.value || "").trim().toLowerCase();
    const tail = (data.tail_lines || []).filter(line => !q || line.toLowerCase().includes(q));
    const wasBottom = els.logBox.scrollHeight - els.logBox.scrollTop - els.logBox.clientHeight < 28;
    els.logBox.textContent = tail.join("\n") || "No matching log lines.";
    if (wasBottom) els.logBox.scrollTop = els.logBox.scrollHeight;

    const discs = data.discrepancies || [];
    els.discTabCount.textContent = String(discs.length);
    els.discBox.textContent = discs.length
      ? discs.map(d => JSON.stringify(d, null, 2)).join("\n---\n")
      : "No discrepancy JSONL records found alongside checkpoint.";
  }

  function render(data) {
    state.data = data;
    updateLogSelector(data.available_logs, data.log_path);

    const statusLower = (data.status || "IDLE").toLowerCase();
    els.pulse.className = `pulse ${statusLower}`;
    els.badge.className = `badge ${statusLower}`;
    const ageNote = data.file_age_sec !== null && data.file_age_sec !== undefined ? ` | log age ${fmtDur(data.file_age_sec)}` : "";
    els.badge.textContent = `${data.status || "IDLE"} (${data.stage || "SUPERVISED"})`;

    const p = data.progress || {};
    els.epochProgressText.textContent = `Epoch ${p.current_epoch || 0} / ${p.total_epochs || 0} (${fmtNum(p.progress_pct, 1, "%")})`;
    els.timingText.textContent = `Elapsed: ${fmtDur(p.elapsed_sec)} | Avg/Epoch: ${p.avg_epoch_sec ? p.avg_epoch_sec + "s" : "--"} | ETA: ${fmtDur(p.eta_sec)}${ageNote}`;
    els.bestCheckpointText.textContent = data.best_epoch
      ? `Best Saved: Ep ${data.best_epoch.epoch} (RelF1 ${fmtNum(data.best_epoch.val_rel_f1, 2, "%")}, BIO ${fmtNum(data.best_epoch.val_bio, 2, "%")})`
      : "Best Saved: --";
    els.progressBar.style.width = `${Math.min(100, Math.max(0, p.progress_pct || 0))}%`;
    els.lastPoll.textContent = `Updated: ${data.generated_at_utc || "--"}`;

    renderMeta(data);
    renderKpis(data);
    renderCharts(data);
    renderTablesAndLogs(data);
  }

  async function fetchMetrics() {
    try {
      const url = `/api/metrics?log=${encodeURIComponent(state.selectedLog)}`;
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      render(data);
    } catch (err) {
      els.lastPoll.textContent = `Poll error: ${err.message}`;
    }
  }

  function schedulePoll() {
    if (state.timerId) clearInterval(state.timerId);
    state.timerId = setInterval(() => {
      if (!state.paused) fetchMetrics();
    }, 1500);
  }

  for (const id of ["chartLoss", "chartBioRel", "chartAcc", "chartExec", "chartRl"]) {
    attachChartHover(id);
  }

  els.logSelector.addEventListener("change", () => {
    state.selectedLog = els.logSelector.value;
    fetchMetrics();
  });
  els.pauseBtn.addEventListener("click", () => {
    state.paused = !state.paused;
    els.pauseBtn.textContent = state.paused ? "Resume Live" : "Pause Live";
  });
  els.refreshBtn.addEventListener("click", fetchMetrics);
  els.newestFirst.addEventListener("change", () => {
    if (state.data) renderTablesAndLogs(state.data);
  });
  els.logFilter.addEventListener("input", () => {
    if (state.data) renderTablesAndLogs(state.data);
  });
  window.addEventListener("resize", () => {
    if (state.data) renderCharts(state.data);
  });

  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.addEventListener("click", () => {
      for (const b of document.querySelectorAll(".tab-btn")) b.classList.remove("active");
      for (const pane of document.querySelectorAll(".tab-pane")) pane.classList.remove("active");
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  }

  fetchMetrics();
  schedulePoll();
})();
</script>
</body>
</html>
"""


def make_handler_class(
    *,
    default_log: Path | None,
    checkpoint_path: Path | None,
    search_dirs: list[str] | None,
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to the configured monitor paths."""

    class LiveMonitorHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Suppress per-request console noise unless explicitly needed.
            return

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    DASHBOARD_HTML.encode("utf-8"),
                )
                return

            if parsed.path == "/api/metrics":
                qs = parse_qs(parsed.query)
                req_log = qs.get("log", ["default"])[0]
                payload = load_telemetry(
                    requested_log=req_log,
                    default_log=default_log,
                    checkpoint_path=checkpoint_path,
                    search_dirs=search_dirs,
                )
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)
                return

            if parsed.path == "/api/logs":
                logs = discover_training_logs(search_dirs=search_dirs, pinned_log=default_log)
                body = json.dumps({"logs": logs}, ensure_ascii=True).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)
                return

            if parsed.path == "/api/health":
                body = json.dumps({"ok": True, "pid": os.getpid()}, ensure_ascii=True).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)
                return

            self._send_bytes(
                HTTPStatus.NOT_FOUND,
                "application/json; charset=utf-8",
                json.dumps({"error": "Not found"}).encode("utf-8"),
            )

    return LiveMonitorHandler


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    log_path: Path | None = None,
    checkpoint_path: Path | None = None,
    search_dirs: list[str] | None = None,
) -> ThreadingHTTPServer:
    """Instantiate the ThreadingHTTPServer for live monitor serving."""
    handler_cls = make_handler_class(
        default_log=log_path,
        checkpoint_path=checkpoint_path,
        search_dirs=search_dirs,
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live training monitor for MudraForge Dual-Stream")
    parser.add_argument("--log", default=None, help="Path to training log file (omit for auto-discovery)")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v2_dual_stream_sup.pt",
        help="Optional checkpoint path for mtime/size and sibling *_discrepancies.jsonl inspection",
    )
    parser.add_argument("--tasks-dir", action="append", default=None, help="Extra directory to scan for *.log runs")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--once-json", action="store_true", help="Print parsed telemetry JSON once and exit")
    args = parser.parse_args(argv)

    log_path = Path(args.log).expanduser() if args.log else None
    ckpt_path = Path(args.checkpoint).expanduser() if args.checkpoint else None

    if args.once_json:
        telemetry = load_telemetry(
            requested_log="default" if log_path else "auto",
            default_log=log_path,
            checkpoint_path=ckpt_path,
            search_dirs=args.tasks_dir,
        )
        print(json.dumps(telemetry, indent=2))
        return 0

    server = create_server(
        host=args.host,
        port=args.port,
        log_path=log_path,
        checkpoint_path=ckpt_path,
        search_dirs=args.tasks_dir,
    )
    actual_port = server.server_address[1]
    print(f"Live Training Monitor listening on http://{args.host}:{actual_port}/", flush=True)
    if log_path:
        print(f"Pinned log: {log_path}", flush=True)
    if ckpt_path:
        print(f"Checkpoint watch: {ckpt_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
