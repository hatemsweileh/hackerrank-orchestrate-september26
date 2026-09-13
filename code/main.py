"""Buy or Wait? - affordability decision agent.

Usage (from the repository root):
    python code/main.py                      # predict dataset/requests.csv -> output.csv
    python code/main.py --samples            # run on dataset/sample_requests.csv instead
    python code/main.py --only request_31    # restrict to specific request ids
    python code/main.py --trace request_31   # print the reconstructed forecast for a request

Environment (all optional): OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY,
LLM_PROVIDER, LLM_MODEL, LLM_DISABLE=1. Without a key the agent runs fully
offline (template message parser + local OCR for images).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent.data import load_dataset  # noqa: E402
from agent.explain import explain  # noqa: E402
from agent.forecast import Forecast  # noqa: E402
from agent.images import ImageExtractor  # noqa: E402
from agent.llm import LLMClient, UsageTracker  # noqa: E402
from agent.planner import decide, fmt_amount_2  # noqa: E402
from agent.state import Config, StateBuilder  # noqa: E402
from agent.validate import COLUMNS, validate_row  # noqa: E402


def fmt_safe(x: float) -> str:
    s = f"{round(x + 1e-9, 2):.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


class Agent:
    def __init__(self, dataset_dir: Path, config: Config = None, cache_dir: Path = None):
        self.ds = load_dataset(dataset_dir)
        self.tracker = UsageTracker()
        self.llm = LLMClient(self.tracker, cache_path=(cache_dir or HERE / "cache") / "llm_cache.json")
        self.images = ImageExtractor(self.llm, self.tracker)
        self.config = config or Config()
        self.builder = StateBuilder(self.ds, self.llm, self.images, self.config)

    def decide(self, request):
        state = self.builder.build(request)
        fc = Forecast(state, debits_before_credits=self.config.debits_before_credits,
                      day_order=self.config.day_order)
        profile = self.ds.profiles[request.user_id]
        options = self.ds.options_by_request.get(request.request_id, [])
        decision = decide(request, profile, options, fc, wait_with_changes=self.config.wait_with_changes)
        row = {
            "request_id": request.request_id,
            "amount_safe_to_pay": fmt_safe(decision.amount_safe_to_pay),
            "affordability_status": decision.status,
            "recommended_payment_method": decision.plan.method if decision.plan else "not_recommended",
            "payment_plan": "|".join(f"{d.isoformat()}:{fmt_amount_2(a)}" for d, a in decision.plan.payments)
            if decision.plan else "none",
            "earliest_date_for_full_payment": decision.earliest_full_date.isoformat()
            if decision.earliest_full_date else "",
            "spending_changes_needed": "|".join(c.render() for c in decision.plan.changes)
            if decision.plan and decision.plan.changes else "none",
            "decision_explanation": explain(decision, request, profile, self.ds.events_by_id),
        }
        if row["affordability_status"] == "affordable_now":
            row["earliest_date_for_full_payment"] = request.request_date.isoformat()
        errors = validate_row(row, request, options, self.ds.events_by_id, profile)
        return row, errors, state, fc, decision


def write_usage_report(tracker: UsageTracker, n_requests: int, path: Path, runtime_s: float, llm: LLMClient) -> None:
    per = tracker.summary()
    lines = [
        "# Token Usage and Cost Report",
        "",
        "Generated automatically by `code/main.py` at the end of the full-dataset run that produced `output.csv`.",
        "",
        f"- Requests processed: **{n_requests}**",
        f"- Runtime: **{runtime_s:.1f} s**",
        f"- LLM provider configured: **{llm.provider or 'none (offline mode)'}**"
        + (f", model `{llm.model}`" if llm.provider else ""),
        "",
        "## Per model",
        "",
        "| provider/model | calls | live calls | cache hits | failures | input tokens | output tokens | "
        "total tokens | est. cost (USD) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tot_in = tot_out = tot_calls = 0
    tot_cost = 0.0
    for key, s in per.items():
        total = s["input_tokens"] + s["output_tokens"]
        lines.append(f"| {key} | {s['calls']} | {s['live_calls']} | {s['cache_hits']} | {s['failures']} | "
                     f"{s['input_tokens']} | {s['output_tokens']} | {total} | {s['cost_usd']:.4f} |")
        tot_in += s["input_tokens"]
        tot_out += s["output_tokens"]
        tot_calls += s["calls"]
        tot_cost += s["cost_usd"]
    if not per:
        lines.append("| (no model calls) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0000 |")
    n = max(n_requests, 1)
    lines += [
        "",
        "## Overall",
        "",
        f"- Model calls: **{tot_calls}**",
        f"- Input tokens: **{tot_in}**, output tokens: **{tot_out}**, total: **{tot_in + tot_out}**",
        f"- Average tokens per request: **{(tot_in + tot_out) / n:.1f}**",
        f"- Estimated total cost: **${tot_cost:.4f}**; per request: **${tot_cost / n:.6f}**",
        "",
        "## Offline (non-LLM) AI components",
        "",
        "| component | invocations | tokens | cost |",
        "|---|---|---|---|",
    ]
    for k, v in tracker.offline_extractions.items():
        lines.append(f"| {k} (local rapidocr ONNX model) | {v} | 0 | $0 |")
    if not tracker.offline_extractions:
        lines.append("| (none) | 0 | 0 | $0 |")
    lines += [
        "",
        "Token counts come from the provider usage fields returned by each API response; cache hits re-use the",
        "recorded counts of the original call and are not billed again. Costs use the list prices in",
        "`code/agent/llm.py::PRICING` and are estimates. Deterministic components (financial forecast, plan",
        "ranking, validation, message template parser) use no tokens.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Buy or Wait? affordability agent")
    ap.add_argument("--dataset", default=str(HERE.parent / "dataset"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--samples", action="store_true", help="run on sample_requests.csv")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--trace", nargs="*", default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    agent = Agent(Path(args.dataset))
    requests = agent.ds.samples if args.samples else agent.ds.requests
    if args.only:
        requests = [r for r in requests if r.request_id in set(args.only)]
    default_out = HERE.parent / ("sample_output.csv" if args.samples else "output.csv")
    out_path = Path(args.output) if args.output else default_out

    rows, all_errors = [], []
    for req in requests:
        try:
            row, errors, state, fc, decision = agent.decide(req)
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the whole run
            print(f"  ! {req.request_id}: could not be processed ({type(exc).__name__}: {exc}); "
                  f"writing a conservative not_recommended row")
            rows.append({
                "request_id": req.request_id, "amount_safe_to_pay": "0", "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended", "payment_plan": "none",
                "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                "decision_explanation": "Do not proceed yet: this request's financial records could not be "
                                        "verified, so no payment can be confirmed as safe.",
            })
            all_errors.append(f"{req.request_id}: processing error {exc}")
            continue
        rows.append(row)
        all_errors.extend(errors)
        if args.trace is not None and (not args.trace or req.request_id in args.trace):
            print(f"\n=== {req.request_id} {req.user_id} {req.request_date} req={req.requested_amount} "
                  f"bal={state.balance} min={state.minimum}")
            for line in state.log:
                print("  log:", line)
            for line in fc.trace():
                print("  ", line)
            print("  decision:", json.dumps(row, ensure_ascii=False))
            print("  rejected:", decision.rejected)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    agent.llm.save_cache()
    runtime = time.time() - t0
    if not args.samples and not args.only and not args.no_report:
        write_usage_report(agent.tracker, len(rows), HERE / "evaluation" / "usage_report.md", runtime, agent.llm)
    print(f"wrote {len(rows)} rows to {out_path} in {runtime:.1f}s; contract violations: {len(all_errors)}")
    for e in all_errors[:50]:
        print("  !", e)
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
