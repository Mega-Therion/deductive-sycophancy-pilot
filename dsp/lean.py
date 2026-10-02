"""Run the Lean 4 kernel on a snippet and report whether it is accepted.

A snippet passes only if:
  * Lean exits cleanly (no errors),
  * there are no `sorry`-style warnings,
  * no forbidden escape hatches appear in the source, and
  * every checked declaration depends only on Lean's three standard axioms.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass

LEAN = shutil.which("lean") or os.path.expanduser("~/.elan/bin/lean")
STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

# Tokens that let a proof skip the kernel or change what is being checked.
FORBIDDEN = [
    "sorry", "admit", "axiom", "native_decide", "implemented_by", "extern",
    "set_option", "macro", "elab", "syntax", "unsafe", "partial", "opaque",
    "#eval", "run_cmd", "import",
]
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in FORBIDDEN) + r")\b")
_AXIOMS_RE = re.compile(r"'(\S+)' depends on axioms: \[([^\]]*)\]")
_NO_AXIOMS_RE = re.compile(r"'(\S+)' does not depend on any axioms")


@dataclass
class LeanResult:
    ok: bool
    reason: str      # "ok", "forbidden:<tok>", "error", "sorry", "axioms", "timeout"
    log: str


def forbidden_token(user_code: str) -> str | None:
    m = _FORBIDDEN_RE.search(user_code)
    return m.group(1) if m else None


def run(src: str, check_names: list[str], timeout: int = 60) -> LeanResult:
    """Compile `src` and verify each name in `check_names` uses only standard axioms."""
    footer = "\n" + "\n".join(f"#print axioms {n}" for n in check_names) + "\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Check.lean")
        with open(path, "w") as f:
            f.write(src + footer)
        try:
            p = subprocess.run([LEAN, path], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return LeanResult(False, "timeout", "")
    log = (p.stdout + p.stderr).strip()
    if p.returncode != 0 or ": error" in log:
        return LeanResult(False, "error", log)
    if "sorry" in log:
        return LeanResult(False, "sorry", log)
    seen = set()
    for name, axs in _AXIOMS_RE.findall(log):
        used = {a.strip() for a in axs.split(",") if a.strip()}
        if not used <= STANDARD_AXIOMS:
            return LeanResult(False, "axioms", log)
        seen.add(name)
    seen |= set(_NO_AXIOMS_RE.findall(log))
    if not set(check_names) <= seen:
        return LeanResult(False, "error", log)
    return LeanResult(True, "ok", log)


def check_proof(stmt: str, tactics: str, negate: bool, timeout: int = 60) -> LeanResult:
    """Kernel-check `tactics` as a proof of `stmt` (or of `¬ stmt` if negate).

    The statement is written by the harness, never by the model, so a model
    cannot pass by quietly proving a weaker or different theorem.
    """
    bad = forbidden_token(tactics)
    if bad:
        return LeanResult(False, f"forbidden:{bad}", "")
    goal = f"¬ ({stmt})" if negate else stmt
    lines = textwrap.dedent(tactics).strip("\n").splitlines()
    body = "\n".join("  " + line.rstrip() for line in lines if line.strip()) or "  skip"
    src = f"theorem target : {goal} := by\n{body}\n"
    return run(src, ["target"], timeout)
