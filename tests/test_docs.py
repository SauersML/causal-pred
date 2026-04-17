"""Tests for the documentation trio: ARCHITECTURE.md, CLAUDE.md, RUNBOOK.md.

These are smoke-level contract tests.  They do NOT check prose quality -- they
check that the files exist, cover the non-negotiable topics documented in the
team task brief, and that every shell command the RUNBOOK shows is at least
parseable by ``shlex``.
"""

from __future__ import annotations

import os
import re
import shlex

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_architecture_exists():
    path = os.path.join(DOCS_DIR, "ARCHITECTURE.md")
    assert os.path.exists(path), "docs/ARCHITECTURE.md is missing"
    text = _read(path)
    assert len(text) >= 2000, (
        f"docs/ARCHITECTURE.md is only {len(text)} chars; expected >= 2000"
    )


def test_claude_md_mentions_uv_and_gam():
    path = os.path.join(REPO_ROOT, "CLAUDE.md")
    assert os.path.exists(path), "CLAUDE.md is missing at the repo root"
    text = _read(path)
    assert "uv run" in text, "CLAUDE.md must instruct teammates to use `uv run`"
    assert "gam" in text, "CLAUDE.md must mention the `gam` library"
    assert "gnomon" in text, "CLAUDE.md must mention the `gnomon` CLI"
    assert "pygam" not in text.lower(), (
        "CLAUDE.md must NOT reintroduce `pygam` -- it is a tripwire"
    )


_CODE_FENCE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)


def _runbook_command_lines():
    path = os.path.join(DOCS_DIR, "RUNBOOK.md")
    text = _read(path)
    for match in _CODE_FENCE.finditer(text):
        lang = (match.group(1) or "").lower()
        if lang and lang not in {"sh", "bash", "shell", "console", "zsh"}:
            continue
        block = match.group(2)
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("uv run") or line.startswith("gnomon"):
                yield line


def test_runbook_is_executable():
    path = os.path.join(DOCS_DIR, "RUNBOOK.md")
    assert os.path.exists(path), "docs/RUNBOOK.md is missing"

    lines = list(_runbook_command_lines())
    assert lines, "RUNBOOK.md has no `uv run`/`gnomon` code lines"
    for line in lines:
        try:
            parts = shlex.split(line)
        except ValueError as exc:  # pragma: no cover - diagnostic path
            pytest.fail(f"shlex failed to parse RUNBOOK command: {line!r} ({exc})")
        assert parts, f"RUNBOOK command parsed to empty tokens: {line!r}"


def test_readme_points_to_docs():
    path = os.path.join(REPO_ROOT, "README.md")
    text = _read(path)
    for doc in ("docs/ARCHITECTURE.md", "CLAUDE.md", "docs/RUNBOOK.md"):
        assert doc in text, f"README.md must link to {doc}"
