"""Evaluation workflow.

1. Runs the agent on dataset/sample_requests.csv (public solved examples) and
   scores every output column against the published values.
2. Validates a predictions file (default: output.csv) against the output
   contract: schema, column order, one row per request, allowed values,
   bounds, plan/option matching, spending-change validity.

Usage (from the repository root):
    python code/evaluation/main.py                 # samples score + output.csv validation
    python code/evaluation/main.py --verbose       # also print every mismatch
    python code/evaluation/main.py --output other.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agent.validate import COLUMNS, validate_row  # noqa: E402
from main import Agent  # noqa: E402


def _num(x: str):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _plan_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        pa = [(p.split(":")[0], float(p.split(":")[1])) for p in a.split("|")]
        pb = [(p.split(":")[0], float(p.split(":")[1])) for p in b.split("|")]
    except (IndexError, ValueError):
        return False
    return len(pa) == len(pb) and all(x[0] == y[0] and abs(x[1] - y[1]) < 0.011 for x, y in zip(pa, pb))


def _changes_equal(a: str, b: str) -> bool:
    norm = lambda s: sorted(p.rsplit(":", 1)[0] + (f":{float(p.rsplit(':', 1)[1]):.2f}" if p.startswith("reduce_to") else "")
                            for p in s.split("|")) if s != "none" else []
    try:
        return norm(a) == norm(b)
    except ValueError:
        return a == b


def score_samples(agent: Agent, verbose: bool) -> dict:
    fields = Counter()
    n = 0
    safe_close = 0
    for req in agent.ds.samples:
        row, errors, *_ = agent.decide(req)
        exp = req.expected
        n += 1
        got_safe, exp_safe = _num(row["amount_safe_to_pay"]), _num(exp["amount_safe_to_pay"])
        checks = {
            "amount_safe_to_pay(exact)": exp_safe is not None and abs(got_safe - exp_safe) <= 0.011,
            "affordability_status": row["affordability_status"] == exp["affordability_status"],
            "recommended_payment_method": row["recommended_payment_method"] == exp["recommended_payment_method"],
            "payment_plan": _plan_equal(row["payment_plan"], exp["payment_plan"]),
            "earliest_date_for_full_payment": row["earliest_date_for_full_payment"] == exp["earliest_date_for_full_payment"],
            "spending_changes_needed": _changes_equal(row["spending_changes_needed"], exp["spending_changes_needed"]),
            "contract_valid": not errors,
        }
        if exp_safe and got_safe is not None and abs(got_safe - exp_safe) <= 0.05 * max(exp_safe, 1e-9):
            safe_close += 1
        for k, ok in checks.items():
            fields[k] += int(ok)
        if verbose and not all(checks.values()):
            print(f"\n--- {req.request_id} ({req.user_id})")
            for k, ok in checks.items():
                if not ok:
                    col = k.split("(")[0]
                    print(f"  {k}: got={row.get(col)!r} expected={exp.get(col)!r}")
            for e in errors:
                print("  !", e)
    print(f"\nSample evaluation on {n} solved requests")
    for k, v in fields.items():
        print(f"  {k:34s} {v:3d}/{n}  ({100 * v / n:.0f}%)")
    print(f"  {'amount_safe_to_pay(within 5%)':34s} {safe_close:3d}/{n}  ({100 * safe_close / n:.0f}%)")
    return {"n": n, **fields}


def validate_output(agent: Agent, path: Path) -> int:
    if not path.exists():
        print(f"\n{path} not found - run `python code/main.py` first")
        return 1
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)
    problems = []
    if header != COLUMNS:
        problems.append(f"header mismatch: {header}")
    want = [r.request_id for r in agent.ds.requests]
    got = [r["request_id"] for r in rows]
    if sorted(want) != sorted(got):
        problems.append(f"request ids differ: missing={set(want) - set(got)} extra={set(got) - set(want)}")
    if len(got) != len(set(got)):
        problems.append("duplicate request ids")
    by_id = {r.request_id: r for r in agent.ds.requests}
    for row in rows:
        req = by_id.get(row["request_id"])
        if req is None:
            continue
        problems += validate_row(row, req, agent.ds.options_by_request.get(req.request_id, []),
                                 agent.ds.events_by_id, agent.ds.profiles[req.user_id])
    dist = Counter(r["affordability_status"] for r in rows)
    meth = Counter(r["recommended_payment_method"] for r in rows)
    print(f"\nOutput validation: {path} ({len(rows)} rows)")
    print("  status distribution:", dict(dist))
    print("  method distribution:", dict(meth))
    print(f"  contract problems: {len(problems)}")
    for p in problems[:40]:
        print("   !", p)
    return len(problems)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE.parent.parent / "dataset"))
    ap.add_argument("--output", default=str(HERE.parent.parent / "output.csv"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    agent = Agent(Path(args.dataset))
    score_samples(agent, args.verbose)
    return 1 if validate_output(agent, Path(args.output)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
