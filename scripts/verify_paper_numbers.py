#!/usr/bin/env python3
"""Recompute every number registered in paper_numbers.yaml from the table it
names, and fail on any mismatch.

Run the notebooks (or scripts/run_all.py) first: they rebuild the tables
under results/tables/ from the committed data, so a pass here means every
number in the paper follows from the data.

    python scripts/verify_paper_numbers.py [-v] [--only PREFIX]

Exit status 0 iff every claim passes.

Registry grammar (paper_numbers.yaml), one mapping per claim:

    id        unique identifier
    desc      what the number is
    value     the value exactly as printed in the tex
    tol       absolute tolerance
    tex       "<line>: <verbatim snippet>"    (traceability only, not checked)
    source    path to the source file, relative to the repo root
    where     {column: value} equality filters; a list value means "in"
    column    the column the value is read from
    agg       first | mean | median | min | max | sum | count | wmean |
              spearman | argmax_of
    weight    column of weights, for agg: wmean
    x, y      columns, for agg: spearman
    of        column whose argmax is taken, for agg: argmax_of
    sort_by   column to sort rows by before a positional agg (first/last)
    scale     multiply the recomputed value by this before comparing
    abs       true -> compare |recomputed|
    expr      instead of source/column: a Python arithmetic expression over
              other claim ids written as {other_id}
"""
from __future__ import annotations

import argparse
import ast
import math
import operator
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]

_ALLOWED_BINOP = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}
_ALLOWED_FUNC = {"abs": abs, "min": min, "max": max}


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression over numbers, abs/min/max only."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOP:
            return _ALLOWED_BINOP[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _ALLOWED_FUNC and not node.keywords):
            return float(_ALLOWED_FUNC[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f"disallowed expression element: {ast.dump(node)}")
    return ev(ast.parse(expr, mode="eval"))


_cache: dict[Path, pd.DataFrame] = {}


def load(path: Path) -> pd.DataFrame:
    if path not in _cache:
        if not path.exists():
            raise FileNotFoundError(path)
        _cache[path] = pd.read_csv(path)
    return _cache[path]


def select(df: pd.DataFrame, where: dict | None) -> pd.DataFrame:
    if not where:
        return df
    m = pd.Series(True, index=df.index)
    for col, val in where.items():
        if col not in df.columns:
            raise KeyError(f"no column {col!r}; have {list(df.columns)}")
        series = df[col]
        if isinstance(val, list):
            m &= series.isin(val)
        elif isinstance(val, bool):
            # tolerate True/"True"/"true" as written by different writers
            m &= series.astype(str).str.lower() == str(val).lower()
        else:
            m &= series.astype(str) == str(val)
    return df[m]


def recompute(claim: dict, repo: Path, resolved: dict[str, float]) -> float:
    if "expr" in claim:
        expr = re.sub(r"\{([A-Za-z0-9_]+)\}",
                      lambda m: repr(resolved[m.group(1)]), claim["expr"])
        return safe_eval(expr)

    df = select(load(repo / claim["source"]), claim.get("where"))
    agg = claim.get("agg", "first")
    if df.empty:
        # "no rows match" is the answer for a count claim, and an error for
        # everything else: a silently empty selection must never read as a pass.
        if agg in ("count", "count_positive"):
            return 0.0
        raise ValueError(f"filter {claim.get('where')} selected zero rows")
    if claim.get("sort_by"):
        df = df.sort_values(claim["sort_by"])

    if agg == "spearman":
        # Pearson correlation of the ranks; identical to scipy.stats.spearmanr
        # for these data and keeps the checker dependency-free.
        val = float(df[claim["x"]].rank().corr(df[claim["y"]].rank()))
    elif agg == "count":
        val = float(len(df))
    elif agg == "nunique":
        val = float(df[claim["column"]].nunique())
    elif agg == "count_positive":
        # "improves in five of the six" is a count over a signed column, and
        # must be recomputed rather than restated as a hand-picked row filter,
        # or the claim would assert its own answer.
        val = float((df[claim["column"]].astype(float) > 0).sum())
    elif agg == "mean_of_differences":
        # row-wise mean of (a_i - b_i) over paired column lists; used where the
        # paper averages a per-language difference rather than differencing the
        # pooled values, which is not the same number
        a = [df[c].astype(float) for c in claim["a_columns"]]
        b = [df[c].astype(float) for c in claim["b_columns"]]
        if len(a) != len(b):
            raise ValueError("a_columns and b_columns must be the same length")
        diffs = [float((x - y).iloc[0]) for x, y in zip(a, b)]
        val = sum(diffs) / len(diffs)
    elif agg == "argmax_of":
        val = float(df.loc[df[claim["of"]].idxmax(), claim["column"]])
    elif agg == "argmin_of":
        val = float(df.loc[df[claim["of"]].idxmin(), claim["column"]])
    elif agg == "wmean":
        w = df[claim["weight"]].astype(float)
        val = float((df[claim["column"]].astype(float) * w).sum() / w.sum())
    else:
        if claim.get("group_by"):
            # collapse to one value per group first, then aggregate across
            # groups: "the range of per-tokeniser means", not "the range of cells"
            df = (df.groupby(claim["group_by"], as_index=False)[claim["column"]]
                    .mean())
        col = df[claim["column"]].astype(float)
        val = {
            "first": lambda c: float(c.iloc[0]),
            "last": lambda c: float(c.iloc[-1]),
            "mean": lambda c: float(c.mean()),
            "median": lambda c: float(c.median()),
            "min": lambda c: float(c.min()),
            "max": lambda c: float(c.max()),
            "sum": lambda c: float(c.sum()),
            "nunique": lambda c: float(c.nunique()),
        }[agg](col)

    if claim.get("abs"):
        val = abs(val)
    if claim.get("scale") is not None:
        val *= float(claim["scale"])
    return val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--only", default=None, help="only ids starting with this prefix")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--log", default=None, help="also write a markdown report here")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else REPO
    reg_path = Path(args.registry) if args.registry else repo / "paper_numbers.yaml"
    doc = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    claims = doc["claims"]

    ids = [c["id"] for c in claims]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        print(f"FATAL: duplicate claim ids: {sorted(dupes)}")
        return 2

    resolved: dict[str, float] = {}
    passed = failed = errored = 0
    failures: list[str] = []

    # expr claims read other claims, so evaluate inputs first
    deps = lambda c: set(re.findall(r"\{([A-Za-z0-9_]+)\}", c.get("expr", "")))
    ordered, done, pending = [], set(), list(claims)
    while pending:
        ready = [c for c in pending if deps(c) <= done or not deps(c) & {x["id"] for x in pending}]
        if not ready:
            ready = pending
        for c in ready:
            ordered.append(c)
            done.add(c["id"])
        pending = [c for c in pending if c["id"] not in done]

    for c in ordered:
        cid = c["id"]
        if args.only and not cid.startswith(args.only):
            continue
        try:
            got = recompute(c, repo, resolved)
        except Exception as e:  # a claim that cannot be recomputed is a failure
            errored += 1
            failures.append(f"  ERROR  {cid}: {type(e).__name__}: {e}")
            print(f"ERROR  {cid}: {type(e).__name__}: {e}")
            continue
        resolved[cid] = got
        want, tol = float(c["value"]), float(c["tol"])
        diff = abs(got - want)
        # 1e-9 absorbs float noise at an exact half-unit rounding boundary
        ok = diff <= tol + 1e-9 or (math.isnan(got) and math.isnan(want))
        if ok:
            passed += 1
            if args.verbose:
                print(f"ok     {cid:44s} printed={want:<12g} recomputed={got:<14.6g} |d|={diff:.2e}")
        else:
            failed += 1
            msg = (f"  FAIL   {cid}\n"
                   f"           printed    = {want}\n"
                   f"           recomputed = {got:.6g}\n"
                   f"           |diff|     = {diff:.6g}  (tol {tol})\n"
                   f"           source     = {c.get('source', 'expr: ' + str(c.get('expr')))}\n"
                   f"           tex        = {c.get('tex', '-')}\n"
                   f"           desc       = {c.get('desc', '-')}")
            failures.append(msg)
            print(f"FAIL   {cid:44s} printed={want:<12g} recomputed={got:<14.6g} |d|={diff:.2e}")

    total = passed + failed + errored
    print("\n" + "=" * 72)
    if failures:
        print("FAILURES AND ERRORS IN DETAIL\n")
        for f in failures:
            print(f)
        print()
    print(f"SUMMARY  {passed} passed, {failed} failed, {errored} errored "
          f"({total} claims checked, {len(claims)} registered)")
    print("=" * 72)
    if args.log:
        lines = ["# Verification report", "",
                 f"{passed} passed, {failed} failed, {errored} errored "
                 f"({total} claims checked, {len(claims)} registered).", ""]
        if failures:
            lines += ["## Failures", "", "```"] + failures + ["```"]
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text("\n".join(lines) + "\n")
    return 0 if (failed == 0 and errored == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
