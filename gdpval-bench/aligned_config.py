#!/usr/bin/env python3
"""Load AA-aligned run config from YAML.

Convention (one model => one yaml, isolated by model name):
  aligned_<model>.yaml
  e.g. aligned_GLM-5.2.yaml, aligned_agnes-2.5-flash-fp8-v3.1.yaml

Select config:
  --model GLM-5.2              # => aligned_GLM-5.2.yaml
  --config aligned_GLM-5.2.yaml
  env ALIGNED_CONFIG=...
  default: aligned.yaml
"""
from __future__ import annotations

import collections
import collections.abc
import os
import re
import sys
from typing import Any

# PyYAML<5.4 on Python 3.10+ needs this shim.
if not hasattr(collections, "Hashable"):
    collections.Hashable = collections.abc.Hashable  # type: ignore[attr-defined]

import yaml

ROOT = os.path.expanduser("~/gdpval-bench")
DEFAULT_CONFIG = os.path.join(ROOT, "aligned.yaml")


def sanitize_model_name(model: str) -> str:
    """Filename-safe model id; keep common chars, map the rest to '_'."""
    name = model.strip()
    name = re.sub(r"[^\w.\-+]", "_", name)
    return name


def config_path_for_model(model: str) -> str:
    """Resolve aligned_<model>.yaml under ROOT."""
    fname = f"aligned_{sanitize_model_name(model)}.yaml"
    return os.path.join(ROOT, fname)


def list_model_configs() -> list[str]:
    """Return model-name stems for aligned_*.yaml (excluding TEMPLATE/smoke helpers)."""
    out = []
    for fn in sorted(os.listdir(ROOT)):
        if not (fn.startswith("aligned_") and fn.endswith(".yaml")):
            continue
        stem = fn[len("aligned_"):-len(".yaml")]
        if stem in {"TEMPLATE", "smoke20", "brave2", "topk050"} or stem.startswith("smoke"):
            continue
        out.append(stem)
    return out


def resolve_config_path(argv: list[str] | None = None) -> str:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" in argv and "--model" in argv:
        raise SystemExit("use either --config or --model, not both")
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 >= len(argv):
            raise SystemExit("usage: --config <path.yaml>")
        path = argv[i + 1]
    elif "--model" in argv:
        i = argv.index("--model")
        if i + 1 >= len(argv):
            raise SystemExit("usage: --model <model-name>  (loads aligned_<model>.yaml)")
        path = config_path_for_model(argv[i + 1])
        if not os.path.exists(path):
            known = ", ".join(list_model_configs()) or "(none)"
            raise SystemExit(
                f"config not found for model {argv[i+1]!r}: {path}\n"
                f"known: {known}\n"
                f"tip: cp aligned_TEMPLATE.yaml {os.path.basename(path)}"
            )
    else:
        path = os.environ.get("ALIGNED_CONFIG", DEFAULT_CONFIG)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return path


def load(path: str | None = None) -> dict[str, Any]:
    path = path or resolve_config_path()
    if not os.path.exists(path):
        raise SystemExit(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config must be a mapping: {path}")

    cand = cfg.setdefault("candidate", {})
    gen = cfg.setdefault("gen", {})
    judge = cfg.setdefault("judge", {})
    round1 = cfg.setdefault("round1", {})

    model = cand.get("model")
    if not model:
        raise SystemExit(f"candidate.model required in {path}")
    if not cand.get("base_url"):
        raise SystemExit(f"candidate.base_url required in {path}")

    # Warn if filename model stem mismatches candidate.model (common copy-paste footgun).
    base = os.path.basename(path)
    if base.startswith("aligned_") and base.endswith(".yaml"):
        stem = base[len("aligned_"):-len(".yaml")]
        if stem not in {"TEMPLATE", "smoke20", "brave2", "topk050", "agnes"} and stem != sanitize_model_name(model):
            print(
                f"WARN: config file stem {stem!r} != candidate.model {model!r} "
                f"(isolation uses candidate.model for output dirs)",
                flush=True,
            )

    cand.setdefault("api_key", "x")
    # Prefer yaml brave_api_key; fall back to BRAVE_API_KEY env.
    brave = (cfg.get("brave_api_key") or os.environ.get("BRAVE_API_KEY") or "").strip()
    cfg["brave_api_key"] = brave or None
    if cfg["brave_api_key"]:
        os.environ["BRAVE_API_KEY"] = cfg["brave_api_key"]

    gen.setdefault("n_samples", 3)
    gen.setdefault("workers", 12)
    gen.setdefault("max_turns", 22)
    gen.setdefault("max_tokens", 48000)
    # Optional sampling; omitted keys are left to the server default.
    sampling = gen.setdefault("sampling", {})
    for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if k in gen and k not in sampling:
            sampling[k] = gen[k]
    gen["sampling"] = sampling

    judge.setdefault("model", "gpt-5.5")
    judge.setdefault("base_url", "https://agrouter-ng-test.kiwiar.com/v1")
    judge.setdefault("n_samples", gen["n_samples"])
    judge.setdefault("workers", 4)
    judge.setdefault("max_pages", 12)
    if not judge.get("api_key"):
        judge["api_key"] = os.environ.get("JUDGE_API_KEY", "")
    if not judge["api_key"]:
        raise SystemExit("judge.api_key missing (set in yaml or JUDGE_API_KEY)")

    judge.setdefault("results_file", "judge_aligned_{model}_results.json")
    round1.setdefault("results_file", "round1_{model}_results.json")

    cfg["_path"] = path
    cfg["_root"] = ROOT
    return cfg


def results_path(cfg: dict[str, Any], key: str) -> str:
    """Resolve results filename; {model} -> candidate.model. Relative paths under ROOT."""
    section = "round1" if key == "round1" else "judge"
    tmpl = cfg[section]["results_file"]
    name = tmpl.format(model=cfg["candidate"]["model"])
    if os.path.isabs(name):
        return name
    return os.path.join(cfg["_root"], name)


def tasks_list(cfg: dict[str, Any]) -> list:
    tasks_file = cfg.get("tasks_file", "config.json")
    path = tasks_file if os.path.isabs(tasks_file) else os.path.join(cfg["_root"], tasks_file)
    import json
    return json.load(open(path))["tasks"]


def pop_config_args(argv: list[str] | None = None) -> tuple[list[str], str]:
    """Return (remaining_argv, config_path). Leaves other flags intact."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path = resolve_config_path(argv)
    out = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--config", "--model"):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    os.environ["ALIGNED_CONFIG"] = path
    return out, path
