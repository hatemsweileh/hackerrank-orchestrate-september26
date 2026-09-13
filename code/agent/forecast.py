"""Deterministic day-by-day cash simulator.

For every day t of the window two points are tracked (both minus the minimum balance):
    low[t]  lowest intraday point (after debits that clear before same-day income)
    end[t]  end-of-day position
headroom[t] = min(low[t], end[t]).

A recommended payment on day d settles after that day's income, so it lowers
end[d] and every later day. The safety invariant is enforced on *every* day:

    min(low[t] - paid_before(t), end[t] - paid_up_to(t)) >= 0   for all t

Because a payment lowers every later point by the same amount, the largest
safe single payment on day d is min(end[d], headroom[d+1:]) when the days
before d are safe. That identity gives an exact `amount_safe_to_pay` and
`earliest_date_for_full_payment` without searching.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .planner import SpendingChange
from .state import FinancialState

EPS = 0.005


class Forecast:
    def __init__(self, state: FinancialState, debits_before_credits: bool = False, day_order: str = "net"):
        self.state = state
        self.start = state.start
        self.n = (state.end - state.start).days + 1
        self.debits_before_credits = debits_before_credits
        # day_order: "net" (end-of-day netting), "streams_first" (day-to-day spending and one-off debits
        # clear before same-day income, monthly bills after it), "streams_only_first"
        self.day_order = day_order
        self.low, self.end = self._paths({})
        self.headroom = [min(a, b) for a, b in zip(self.low, self.end)]

    # ----------------------------------------------------------------- engine
    def _idx(self, d: date) -> int:
        return (d - self.start).days

    def _paths(self, changes: Dict[str, SpendingChange]) -> Tuple[List[float], List[float]]:
        early = [0.0] * self.n       # debits that clear before same-day income
        late = [0.0] * self.n        # debits that clear after same-day income
        credit = [0.0] * self.n
        for s in self.state.series:
            ch = changes.get(s.key)
            for d in s.dates:
                i = self._idx(d)
                if not 0 <= i < self.n:
                    continue
                amt = s.amount_on(d)
                if ch is not None:
                    if ch.kind == "stop":
                        continue
                    amt = min(amt, ch.new_amount)
                if s.direction == "credit":
                    credit[i] += amt
                elif self.debits_before_credits or (self.day_order != "net" and s.key.startswith("stream:")):
                    early[i] += amt
                else:
                    late[i] += amt
        for c in self.state.one_offs:
            i = self._idx(c.on)
            if 0 <= i < self.n:
                if c.amount >= 0:
                    credit[i] += c.amount
                elif self.debits_before_credits or self.day_order == "streams_first":
                    early[i] -= c.amount
                else:
                    late[i] -= c.amount
        low, end, bal = [], [], self.state.balance
        mn = self.state.minimum
        for i in range(self.n):
            intraday = bal - early[i]
            bal = intraday + credit[i] - late[i]
            low.append(min(intraday, bal) - mn)
            end.append(bal - mn)
        return low, end

    def _check(self, low: List[float], end: List[float], payments: List[Tuple[date, float]]) -> List[float]:
        per_day = [0.0] * self.n
        for d, amt in payments:
            i = max(0, self._idx(d))
            if i < self.n:
                per_day[i] += amt
        out, before = [], 0.0
        for t in range(self.n):
            out.append(min(low[t] - before, end[t] - before - per_day[t]))
            before += per_day[t]
        return out

    # ------------------------------------------------------------- public API
    def max_payment_on(self, d: date) -> float:
        i = max(0, self._idx(d))
        if i >= self.n:
            return 0.0
        if (i > 0 and min(self.headroom[:i]) < -EPS) or self.low[i] < -EPS:
            return 0.0
        return min([self.end[i]] + self.headroom[i + 1:])

    def earliest_full_payment_date(self, amount: float) -> Optional[date]:
        after = [float("inf")] * self.n
        run = float("inf")
        for i in range(self.n - 1, -1, -1):
            after[i] = run
            run = min(run, self.headroom[i])
        prefix_ok = True
        for i in range(self.n):
            if (prefix_ok and self.low[i] >= -EPS and self.end[i] >= amount - EPS
                    and after[i] >= amount - EPS):
                return self.start + timedelta(days=i)
            if self.headroom[i] < -EPS:
                prefix_ok = False
        return None

    def plan_is_safe(self, payments: List[Tuple[date, float]]) -> bool:
        return min(self._check(self.low, self.end, payments)) >= -EPS

    def lowest_headroom(self, payments, changes: Optional[List[SpendingChange]] = None) -> float:
        low, end = self._paths({c.series_key: c for c in changes}) if changes else (self.low, self.end)
        return min(self._check(low, end, payments))

    def change_candidates(self, profile) -> List[SpendingChange]:
        out: List[SpendingChange] = []
        protected = set(profile.protected_categories)
        for s in self.state.series:
            if s.direction != "debit" or not s.dates or s.category in protected:
                continue
            if "stoppable" in s.flexibility and s.category in profile.stoppable_categories:
                out.append(SpendingChange("stop", s.last_event_id, s.key, None, s.description))
            if ("reducible" in s.flexibility and s.category in profile.reducible_categories
                    and s.minimum_allowed is not None and s.minimum_allowed < s.amount - EPS):
                out.append(SpendingChange("reduce_to", s.last_event_id, s.key, s.minimum_allowed, s.description))
        return out

    def find_spending_changes(self, payments, profile, max_changes: int = 3) -> Optional[List[SpendingChange]]:
        """Greedy least-disruptive search: repeatedly add the flexible change with
        the smallest per-occurrence saving that still improves the plan's lowest
        point, until the plan is safe (max three changes, one per event)."""
        series_by_key = {s.key: s for s in self.state.series}
        chosen: List[SpendingChange] = []
        current = self.lowest_headroom(payments)
        candidates = self.change_candidates(profile)
        while len(chosen) < max_changes:
            if current >= -EPS:
                return chosen if chosen else None
            options = []
            for c in candidates:
                if any(x.series_key == c.series_key for x in chosen):
                    continue
                trial = self.lowest_headroom(payments, chosen + [c])
                if trial > current + EPS:
                    s = series_by_key[c.series_key]
                    saving = s.amount if c.kind == "stop" else s.amount - c.new_amount
                    options.append((round(saving, 2), 0 if c.kind == "stop" else 1, c.event_id, c, trial))
            if not options:
                return None
            options.sort(key=lambda o: o[:3])
            _, _, _, pick, trial = options[0]
            chosen.append(pick)
            current = trial
        return chosen if current >= -EPS else None

    def trace(self, payments=None, changes=None, limit: int = 400) -> List[str]:
        low, end = self._paths({c.series_key: c for c in changes}) if changes else (self.low, self.end)
        h = self._check(low, end, payments or [])
        flows: Dict[date, List[str]] = {}
        for s in self.state.series:
            for d in s.dates:
                flows.setdefault(d, []).append(f"{'+' if s.direction == 'credit' else '-'}{s.amount_on(d):.2f} {s.key}")
        for c in self.state.one_offs:
            flows.setdefault(c.on, []).append(f"{c.amount:+.2f} {c.label} [{c.source}]")
        lines = []
        for i in range(self.n):
            d = self.start + timedelta(days=i)
            if d in flows or i == 0:
                lines.append(f"{d} headroom={h[i]:.2f} :: " + "; ".join(flows.get(d, [])))
        return lines[:limit]
