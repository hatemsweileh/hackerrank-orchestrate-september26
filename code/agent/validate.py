"""Deterministic output validation against the published contract."""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List

COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation",
]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
_PLAN_ITEM = re.compile(r"^\d{4}-\d{2}-\d{2}:\d+(\.\d{1,2})?$")
_CHANGE = re.compile(r"^(stop:[\w-]+|reduce_to:[\w-]+:\d+(\.\d{1,2})?)$")


def validate_row(row: Dict[str, str], request, options, events_by_id, profile) -> List[str]:
    """Return a list of human-readable contract violations for one output row."""
    errs: List[str] = []
    rid = row["request_id"]
    try:
        safe = float(row["amount_safe_to_pay"])
        if not (-1e-9 <= safe <= request.requested_amount + 1e-9):
            errs.append(f"{rid}: amount_safe_to_pay {safe} outside [0, requested]")
    except ValueError:
        errs.append(f"{rid}: amount_safe_to_pay not numeric")
        safe = None
    status, method = row["affordability_status"], row["recommended_payment_method"]
    if status not in STATUSES:
        errs.append(f"{rid}: bad status {status}")
    if method not in METHODS:
        errs.append(f"{rid}: bad method {method}")

    plan = row["payment_plan"]
    items = []
    if plan != "none":
        for part in plan.split("|"):
            if not _PLAN_ITEM.match(part):
                errs.append(f"{rid}: malformed plan item {part!r}")
                continue
            d, a = part.split(":")
            items.append((date.fromisoformat(d), float(a)))
        if [i[0] for i in items] != sorted(i[0] for i in items):
            errs.append(f"{rid}: plan not chronological")
    if method in {"not_recommended"} and plan != "none":
        errs.append(f"{rid}: not_recommended must have plan none")
    if method != "not_recommended" and plan == "none":
        errs.append(f"{rid}: {method} requires a plan")

    earliest = row["earliest_date_for_full_payment"]
    if status == "affordable_now" and earliest != request.request_date.isoformat():
        errs.append(f"{rid}: affordable_now requires earliest == request_date")

    if method == "installments":
        match = any(
            len(o.schedule()) == len(items)
            and all(d == od and abs(a - oa) < 0.005 for (d, a), (od, oa) in zip(items, o.schedule()))
            for o in options if o.payment_method == "installments"
        )
        if not match:
            errs.append(f"{rid}: installment plan does not match a supplied option")
    if method == "partial_payment":
        if len(items) != 2:
            errs.append(f"{rid}: partial_payment needs exactly two payments")
        elif abs(sum(a for _, a in items) - request.requested_amount) > 0.011:
            errs.append(f"{rid}: partial payments do not sum to requested_amount")
        if status != "affordable_with_plan":
            errs.append(f"{rid}: partial_payment requires affordable_with_plan")
        if not request.allows_partial_payment:
            errs.append(f"{rid}: request does not allow partial payment")
    if method in {"full_payment", "partial_payment", "installments"} and method not in profile.payment_methods:
        errs.append(f"{rid}: user does not consider {method}")

    changes = row["spending_changes_needed"]
    if changes != "none":
        parts = changes.split("|")
        if len(parts) > 3:
            errs.append(f"{rid}: more than three spending changes")
        seen = set()
        for p in parts:
            if not _CHANGE.match(p):
                errs.append(f"{rid}: malformed spending change {p!r}")
                continue
            eid = p.split(":")[1]
            if eid in seen:
                errs.append(f"{rid}: event {eid} changed twice")
            seen.add(eid)
            ev = events_by_id.get(eid)
            if ev is None or ev.user_id != request.user_id:
                errs.append(f"{rid}: spending change references unknown event {eid}")
            elif p.startswith("stop:") and "stoppable" not in ev.flexibility:
                errs.append(f"{rid}: {eid} is not stoppable")
            elif p.startswith("reduce_to:") and "reducible" not in ev.flexibility:
                errs.append(f"{rid}: {eid} is not reducible")
    if not row["decision_explanation"].strip():
        errs.append(f"{rid}: empty explanation")
    return errs
