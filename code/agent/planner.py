"""Candidate plan generation, deterministic safety validation and ranking.

The planner never trusts a plan it has not simulated: every candidate is
checked against the day-by-day forecast (`Forecast.plan_is_safe`). Ranking
follows the published order exactly:

1. complete the full request by desired_completion_date
2. require no spending changes
3. minimise the total amount paid
4. start payment earlier
5. use fewer payments
6. lowest payment_option_id
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

from .data import PaymentOption, Profile, Request
from .money import EPS, r2


@dataclass
class SpendingChange:
    kind: str            # "stop" | "reduce_to"
    event_id: str        # representative (latest) event id of the flexible series
    series_key: str
    new_amount: Optional[float] = None
    label: str = ""

    def render(self) -> str:
        from .money import fmt_amount

        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{fmt_amount_2(self.new_amount)}"


def fmt_amount_2(x: float) -> str:
    v = r2(x)
    return str(int(round(v))) if abs(v - round(v)) < EPS else f"{v:.2f}"


@dataclass
class Plan:
    method: str
    payments: List[Tuple[date, float]]
    changes: List[SpendingChange] = field(default_factory=list)
    option_id: str = ""

    @property
    def total(self) -> float:
        return sum(a for _, a in self.payments)

    @property
    def start(self) -> date:
        return self.payments[0][0]

    @property
    def end(self) -> date:
        return self.payments[-1][0]


def rank_key(plan: Plan, request: Request):
    return (
        0 if plan.end <= request.desired_completion_date else 1,
        0 if not plan.changes else 1,
        round(plan.total, 2),
        plan.start,
        len(plan.payments),
        _option_order(plan.option_id),
    )


def _option_order(option_id: str):
    digits = "".join(ch for ch in option_id if ch.isdigit())
    return (int(digits) if digits else 10**9, option_id)


def installment_allowed(option: PaymentOption, profile: Profile) -> bool:
    if "installments" not in profile.payment_methods or profile.max_installment_months is None:
        return False
    return option.number_of_payments <= profile.max_installment_months


@dataclass
class Decision:
    amount_safe_to_pay: float
    status: str
    plan: Optional[Plan]
    earliest_full_date: Optional[date]
    candidates_considered: int
    rejected: List[str]


def decide(request: Request, profile: Profile, options: List[PaymentOption], forecast,
           wait_with_changes: bool = False) -> Decision:
    """Build every eligible plan, keep only simulated-safe ones, rank them."""
    req_amt = request.requested_amount
    rd = request.request_date
    deadline = request.desired_completion_date
    safe_today = max(0.0, min(req_amt, forecast.max_payment_on(rd)))
    safe_today = r2(safe_today)
    earliest = forecast.earliest_full_payment_date(req_amt)

    accepted = set(profile.payment_methods)
    safe_plans: List[Plan] = []
    rejected: List[str] = []
    considered = 0

    def consider(plan: Plan, allow_changes: bool) -> None:
        nonlocal considered
        considered += 1
        if plan.end > deadline:
            rejected.append(f"{plan.method}{'/' + plan.option_id if plan.option_id else ''}: ends after deadline")
            return
        if forecast.plan_is_safe(plan.payments):
            safe_plans.append(plan)
            return
        if allow_changes:
            changes = forecast.find_spending_changes(plan.payments, profile)
            if changes is not None:
                plan.changes = changes
                safe_plans.append(plan)
                return
        rejected.append(f"{plan.method}{'/' + plan.option_id if plan.option_id else ''}: breaks minimum balance")

    if "full_payment" in accepted:
        consider(Plan("full_payment", [(rd, req_amt)]), allow_changes=True)

    if ("partial_payment" in accepted and request.allows_partial_payment
            and 0 < safe_today < req_amt and earliest is not None and earliest <= deadline):
        consider(Plan("partial_payment", [(rd, safe_today), (earliest, r2(req_amt - safe_today))]),
                 allow_changes=False)

    for opt in options:
        if opt.payment_method != "installments":
            continue
        if not installment_allowed(opt, profile):
            rejected.append(f"installments/{opt.payment_option_id}: outside user installment preference")
            continue
        consider(Plan("installments", opt.schedule(), option_id=opt.payment_option_id), allow_changes=True)

    if "full_payment" in accepted and earliest is not None and earliest > rd:
        consider(Plan("wait", [(earliest, req_amt)]), allow_changes=False)

    if wait_with_changes and "full_payment" in accepted and (earliest is None or earliest > deadline):
        # the full amount never becomes safe unaided before the deadline: find the first day on which
        # permitted flexible spending changes make a single full payment safe
        d, last_day = rd + timedelta(days=1), min(deadline, forecast.state.end)
        while d <= last_day:
            considered += 1
            changes = forecast.find_spending_changes([(d, req_amt)], profile)
            if changes:
                safe_plans.append(Plan("wait", [(d, req_amt)], changes))
                break
            d += timedelta(days=1)

    if not safe_plans:
        return Decision(safe_today, "not_affordable", None, earliest, considered, rejected)

    best = sorted(safe_plans, key=lambda p: rank_key(p, request))[0]
    if best.changes:
        status = "affordable_with_plan"
    elif best.method == "wait":
        status = "affordable_later"
    elif best.method == "full_payment" and not best.changes and best.start == rd:
        status = "affordable_now"
    else:
        status = "affordable_with_plan"
    return Decision(safe_today, status, best, earliest, considered, rejected)
