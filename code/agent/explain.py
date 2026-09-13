"""Grounded decision explanations.

Explanations are rendered from the final, validated decision so every
number quoted is a number the engine actually used. The wording follows the
style of the published examples (one action sentence + one safety fact).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from .money import fmt_money_text


def long_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _change_phrase(changes, events_by_id, ccy: str) -> str:
    parts = []
    for i, ch in enumerate(changes):
        ev = events_by_id.get(ch.event_id)
        name = (ev.description if ev else ch.event_id).strip()
        name = name[0].lower() + name[1:] if name else name
        verb = "stop" if ch.kind == "stop" else "reduce"
        if ch.kind == "stop":
            phrase = f"{verb} the {name}"
        else:
            phrase = f"{verb} the {name} to {fmt_money_text(ch.new_amount, ccy)}"
        parts.append(phrase)
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return text[0].upper() + text[1:]


def explain(decision, request, profile, events_by_id, lowest_balance: Optional[float] = None) -> str:
    ccy = profile.home_currency
    req = fmt_money_text(request.requested_amount, ccy)
    minimum = fmt_money_text(profile.minimum_balance, ccy)
    plan = decision.plan

    if plan is None:
        if decision.amount_safe_to_pay > 0 and "partial_payment" in profile.payment_methods \
                and "installments" not in profile.payment_methods and request.allows_partial_payment:
            return (f"Do not proceed with the {req} request. Although "
                    f"{fmt_money_text(decision.amount_safe_to_pay, ccy)} is available today, the full amount "
                    f"cannot be completed safely within 90 days.")
        return (f"Do not make this payment by {long_date(request.desired_completion_date)}. None of the "
                f"available options keeps the {minimum} minimum protected.")

    if plan.method == "wait":
        if plan.changes:
            return (f"{_change_phrase(plan.changes, events_by_id, ccy)}, then pay {req} in full on "
                    f"{long_date(plan.start)}. This leaves at least {minimum} available.")
        if plan.start == request.desired_completion_date:
            return (f"Pay {req} in full on {long_date(plan.start)}. Paying earlier would take the balance "
                    f"below the {minimum} minimum.")
        return (f"Wait until {long_date(plan.start)}, then pay {req} in full. Paying sooner would put the "
                f"{minimum} minimum at risk.")
    if plan.method == "partial_payment":
        first, second = plan.payments
        return (f"Pay {fmt_money_text(first[1], ccy)} today and the remaining {fmt_money_text(second[1], ccy)} "
                f"on {long_date(second[0])}. This completes the full request and keeps the {minimum} "
                f"minimum protected.")
    if plan.method == "installments":
        n = len(plan.payments)
        lead = (f"Use {n} installments of {fmt_money_text(plan.payments[0][1], ccy)}, starting "
                f"{long_date(plan.start)}.")
        if plan.changes:
            lead = f"{_change_phrase(plan.changes, events_by_id, ccy)}, then use {n} installments of " \
                   f"{fmt_money_text(plan.payments[0][1], ccy)}, starting {long_date(plan.start)}."
        return f"{lead} This leaves at least {minimum} available."
    # full payment today
    if plan.changes:
        return (f"{_change_phrase(plan.changes, events_by_id, ccy)}, then pay {req} today. This leaves at "
                f"least {minimum} available.")
    return f"Pay {req} today. This leaves at least {minimum} available over the next 90 days."
