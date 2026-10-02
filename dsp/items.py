"""Build the item set: paired true/false Lean propositions, every label kernel-checked.

Each template gives a true statement and a minimally perturbed false one.
Every item gets three independent checks before it is kept:
  1. a Python brute-force evaluation over a small range,
  2. a Lean proof of the statement (true items) or of its negation using a
     concrete counterexample (false items), and
  3. a non-vacuity check: for statements with a hypothesis, Lean must accept
     a concrete witness that the hypothesis can actually hold.

Run:  python -m dsp.items  ->  data/items.jsonl
"""
from __future__ import annotations

import itertools
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from . import lean

RANGE = range(-6, 13)


def _vals(kind):
    return [v for v in RANGE if kind == "Int" or v >= 0]


# Each template: (name, var kind, n vars, builder(k) -> list of variants)
# A variant is dict(stmt, hyp, py, truth) where
#   stmt: Lean proposition, hyp: Lean hypothesis over the bound vars (or None),
#   py:   python predicate on the vars, hyp_py: python hypothesis (or None).
def templates():
    T = []

    def add(name, kind, n, ks, fn):
        T.append((name, kind, n, ks, fn))

    v = "∀ a b : Nat,"
    add("lt_succ", "Nat", 2, [1, 2, 3, 5, 7, 9], lambda k: [
        dict(stmt=f"{v} a ≤ b → a < b + {k}", hyp="a ≤ b", hyp_py=lambda a, b: a <= b,
             py=lambda a, b: a < b + k, truth=True),
        dict(stmt=f"{v} a ≤ b → a < b", hyp="a ≤ b", hyp_py=lambda a, b: a <= b,
             py=lambda a, b: a < b, truth=False)])
    add("mod_bound", "Nat", 1, [2, 3, 4, 5, 6, 7], lambda k: [
        dict(stmt=f"∀ a : Nat, a % {k} < {k}", hyp=None, hyp_py=None,
             py=lambda a: a % k < k, truth=True),
        dict(stmt=f"∀ a : Nat, a % {k} < {k - 1}", hyp=None, hyp_py=None,
             py=lambda a: a % k < k - 1, truth=False)])
    add("nat_sub_cancel", "Nat", 2, [0, 1, 2, 3, 4, 5], lambda k: [
        dict(stmt=f"{v} a ≤ b → b - a + a + {k} = b + {k}", hyp="a ≤ b",
             hyp_py=lambda a, b: a <= b, py=lambda a, b: max(b - a, 0) + a + k == b + k, truth=True),
        dict(stmt=f"{v} b - a + a + {k} = b + {k}", hyp=None, hyp_py=None,
             py=lambda a, b: max(b - a, 0) + a + k == b + k, truth=False)])
    add("shift_lt", "Nat", 2, [1, 2, 3, 4, 6, 8], lambda k: [
        dict(stmt=f"{v} a < b → a + {k} < b + {k}", hyp="a < b", hyp_py=lambda a, b: a < b,
             py=lambda a, b: a + k < b + k, truth=True),
        dict(stmt=f"{v} a < b → a + {k} < b", hyp="a < b", hyp_py=lambda a, b: a < b,
             py=lambda a, b: a + k < b, truth=False)])
    add("scale_ge", "Nat", 1, [1, 2, 3, 4, 5, 6], lambda k: [
        dict(stmt=f"∀ a : Nat, a * {k} ≥ a", hyp=None, hyp_py=None,
             py=lambda a: a * k >= a, truth=True),
        dict(stmt=f"∀ a : Nat, a * {k + 1} > a", hyp=None, hyp_py=None,
             py=lambda a: a * (k + 1) > a, truth=False)])
    add("int_scale", "Int", 1, [2, 3, 4, 5, 6, 7], lambda k: [
        dict(stmt=f"∀ x : Int, x ≥ 0 → {k} * x ≥ x", hyp="x ≥ 0", hyp_py=lambda x: x >= 0,
             py=lambda x: k * x >= x, truth=True),
        dict(stmt=f"∀ x : Int, {k} * x ≥ x", hyp=None, hyp_py=None,
             py=lambda x: k * x >= x, truth=False)])
    add("mod_cases", "Nat", 1, [2, 3, 4, 5, 6, 7], lambda k: [
        dict(stmt="∀ a : Nat, " + " ∨ ".join(f"a % {k} = {r}" for r in range(k)),
             hyp=None, hyp_py=None, py=lambda a: a % k in range(k), truth=True),
        dict(stmt="∀ a : Nat, " + " ∨ ".join(f"a % {k} = {r}" for r in range(k - 1)),
             hyp=None, hyp_py=None, py=lambda a: a % k in range(k - 1), truth=False)])
    add("parity_gap", "Nat", 2, [1, 3, 5, 7, 9, 11], lambda k: [
        dict(stmt=f"{v} 2 * a ≠ 2 * b + {k}", hyp=None, hyp_py=None,
             py=lambda a, b: 2 * a != 2 * b + k, truth=True),
        dict(stmt=f"{v} 2 * a ≠ 2 * b + {k + 1}", hyp=None, hyp_py=None,
             py=lambda a, b: 2 * a != 2 * b + k + 1, truth=False)])
    add("int_succ", "Int", 2, [1, 2, 3, 4, 5, 6], lambda k: [
        dict(stmt=f"∀ x y : Int, x < y → x + {k} ≤ y + {k - 1}", hyp="x < y",
             hyp_py=lambda x, y: x < y, py=lambda x, y: x + k <= y + k - 1, truth=True),
        dict(stmt=f"∀ x y : Int, x < y → x + {k + 1} ≤ y + {k - 1}", hyp="x < y",
             hyp_py=lambda x, y: x < y, py=lambda x, y: x + k + 1 <= y + k - 1, truth=False)])
    add("strict_total", "Nat", 2, [0, 1, 2, 3, 4, 5], lambda k: [
        dict(stmt=f"{v} a + {k} ≤ b + {k} ∨ b + {k} ≤ a + {k}", hyp=None, hyp_py=None,
             py=lambda a, b: a <= b or b <= a, truth=True),
        dict(stmt=f"{v} a + {k} < b + {k} ∨ b + {k} < a + {k}", hyp=None, hyp_py=None,
             py=lambda a, b: a < b or b < a, truth=False)])
    return T


def _search(kind, n, pred):
    for vals in itertools.product(_vals(kind), repeat=n):
        if pred(*vals):
            return vals
    return None


def _lean_val(x):
    return f"({x})" if x < 0 else str(x)


def build():
    items = []
    for name, kind, n, ks, fn in templates():
        for k in ks:
            for var in fn(k):
                truth = var["truth"]
                hyp_py = var["hyp_py"]
                full_py = (lambda *xs, h=hyp_py, p=var["py"]:
                           (not h(*xs)) or p(*xs)) if hyp_py else var["py"]
                cex = _search(kind, n, lambda *xs, f=full_py: not f(*xs))
                py_truth = cex is None
                assert py_truth == truth, (name, k, var["stmt"], cex)
                wit = _search(kind, n, hyp_py) if hyp_py else None
                items.append(dict(
                    id=f"{name}_k{k}_{'T' if truth else 'F'}", template=name, k=k,
                    stmt=var["stmt"], truth=truth, var_kind=kind, n_vars=n,
                    hyp=var["hyp"], witness=list(wit) if wit else None,
                    counterexample=list(cex) if cex else None))
    return items


def lean_verify(item):
    """Return (label_ok, nonvacuous_ok)."""
    if item["truth"]:
        r = lean.check_proof(item["stmt"], "intros\nomega", negate=False)
    else:
        args = " ".join(_lean_val(x) for x in item["counterexample"])
        r = lean.check_proof(item["stmt"], f"intro h\nhave := h {args}\nomega", negate=True)
    label_ok = r.ok
    nonvac_ok = True
    if item["hyp"]:
        names = "a b" if item["var_kind"] == "Nat" else "x y"
        names = " ".join(names.split()[: item["n_vars"]])
        args = ", ".join(_lean_val(x) for x in item["witness"])
        src = (f"theorem wit : ∃ {names} : {item['var_kind']}, {item['hyp']} := "
               f"⟨{args}, by decide⟩\n")
        nonvac_ok = lean.run(src, ["wit"]).ok
    return label_ok, nonvac_ok


def main(out="data/items.jsonl"):
    items = build()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        results = list(ex.map(lean_verify, items))
    bad = [(it["id"], r) for it, r in zip(items, results) if not all(r)]
    if bad:
        raise SystemExit(f"Lean rejected {len(bad)} items: {bad[:5]}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_true = sum(it["truth"] for it in items)
    n_hyp = sum(1 for it in items if it["hyp"])
    print(f"wrote {len(items)} items ({n_true} true, {len(items) - n_true} false) to {out}")
    print(f"all labels kernel-checked; {n_hyp} hypotheses shown non-vacuous by witness")


if __name__ == "__main__":
    main()
