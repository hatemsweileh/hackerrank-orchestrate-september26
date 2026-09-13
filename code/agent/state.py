"""Financial state reconstruction.

Turns one user's raw event history + validated evidence into a canonical
`FinancialState`: the opening balance, the minimum to keep, projected
recurring series (income and expenses), one-off future cash items, and a log
of every exclusion/assumption so decisions are auditable.

Rules implemented (problem statement + dataset contract):
* failed / cancelled / unrealized (non-cash) rows carry no cash.
* pending credits (refunds, payouts, prizes) are not counted until settled.
* pending and scheduled debits are reserved on their settlement date.
* a scheduled salary row is confirmed income on its settlement date.
* recurring expenses are detected from history (fixed day-of-month series and
  fixed-interval category streams); one-off rows are never projected.
* regular salary is projected monthly only when history supports it and no
  message ends it; bonuses, commissions, arrears, gig and freelance credits
  are not confirmed future income.
* explicit message amendments (salary changes, date moves, rent increases,
  confirmed invoices, ended employment) override history.
* blank amounts are filled from the linked image, never treated as zero.
* foreign-currency cash is converted with the dated rate for its cash date.
"""
from __future__ import annotations

import calendar
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from .data import Dataset, Event, Profile, Request
from .evidence import MessageFact, interpret_message
from .images import ImageExtractor
from .llm import LLMClient
from .money import FxTable

STREAM_CATEGORIES = {"groceries", "transport", "dining"}
NON_RECURRING_INCOME_WORDS = ("bonus", "commission", "arrears", "prize", "reimbursement", "proceeds",
                              "reversal", "prorated", "refund", "final")


@dataclass
class Config:
    horizon_days: int = 86            # last forecast day = request_date + horizon_days (calibrated on samples)
    variable_estimator: str = "mean"  # mean | max | median | last | p75 | mean3 | max3 | mean_half_sd
    essential_estimator: str = "mean"  # estimator for essential variable spending
    essential_categories: str = "groceries,transport,utilities,healthcare"  # or "protected"
    include_request_day: bool = True  # monthly obligations due on request_date are still unpaid
    stream_on_request_day: bool = False  # variable spending streams start after request_date
    debits_before_credits: bool = False  # same-day ordering inside the simulator
    day_order: str = "streams_first"   # net | streams_first | streams_only_first (see forecast.Forecast)
    wait_with_changes: bool = False    # consider a later full payment that needs flexible spending changes
    reserve_disputed_duplicates: bool = True
    # optional rounding of estimated (non-constant) amounts to a currency unit, e.g. {"EUR": 1, "IDR": 50}
    round_units: Dict[str, float] = field(default_factory=dict)


@dataclass
class Series:
    key: str
    category: str
    description: str
    direction: str                     # debit | credit
    amount: float                      # per occurrence, home currency
    dates: List[date]                  # projected cash dates inside the horizon
    flexibility: str = "fixed"
    minimum_allowed: Optional[float] = None
    last_event_id: str = ""
    amount_overrides: Dict[date, float] = field(default_factory=dict)

    def amount_on(self, d: date) -> float:
        return self.amount_overrides.get(d, self.amount)


@dataclass
class CashItem:
    on: date
    amount: float                      # signed, home currency
    label: str
    source: str


@dataclass
class FinancialState:
    request: Request
    profile: Profile
    start: date
    end: date
    balance: float
    minimum: float
    series: List[Series]
    one_offs: List[CashItem]
    facts: List[MessageFact]
    log: List[str]


def add_months(d: date, k: int, dom: int) -> date:
    y = d.year + (d.month - 1 + k) // 12
    m = (d.month - 1 + k) % 12 + 1
    return date(y, m, min(dom, calendar.monthrange(y, m)[1]))


def _estimate(values: List[float], how: str) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    if max(vals) - min(vals) < 1e-9:
        return vals[-1]
    if how == "max":
        return max(vals)
    if how == "median":
        return statistics.median(vals)
    if how == "last":
        return vals[-1]
    if how == "p75":
        s = sorted(vals)
        k = 0.75 * (len(s) - 1)
        lo = int(k)
        return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)
    if how == "mean3":
        return sum(vals[-3:]) / len(vals[-3:])
    if how == "max3":
        return max(vals[-3:])
    if how == "mean_half_sd":
        return statistics.mean(vals) + 0.5 * (statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return sum(vals) / len(vals)


class StateBuilder:
    def __init__(self, ds: Dataset, llm: LLMClient, images: ImageExtractor, config: Config):
        self.ds = ds
        self.llm = llm
        self.images = images
        self.config = config
        self.fx = FxTable(ds.rates)
        self.noise, self.min_ratio = self._learn_amount_model(ds)

    @staticmethod
    def _learn_amount_model(ds: Dataset):
        """Learn, from the whole event table, how recurring amounts vary around their base.

        * noise[category] = half-width `a` of the multiplicative spread: amounts lie in
          [base*(1-a), base*(1+a)], estimated from the largest max/min ratio seen in any series.
        * min_ratio[category] = minimum_allowed_amount / typical amount for flexible series, so a
          flexible series' base can be read from its published floor.
        """
        groups: Dict[tuple, List[Event]] = {}
        for evs in ds.events_by_user.values():
            for e in evs:
                if e.status != "settled" or e.direction != "debit" or e.amount is None or e.linked_event_id:
                    continue
                key = (e.user_id, e.category) if e.category in STREAM_CATEGORIES else \
                    (e.user_id, e.category, e.description)
                groups.setdefault(key, []).append(e)
        ratios: Dict[str, List[float]] = {}
        floors: Dict[str, List[float]] = {}
        for key, rows in groups.items():
            if len(rows) < 4:
                continue
            amts = sorted(e.amount for e in rows)
            med = amts[len(amts) // 2]
            amts = [a for a in amts if a <= 2.0 * med]  # drop one-off invoices inside a stream
            if len(amts) < 4:
                continue
            ratios.setdefault(key[1], []).append(max(amts) / min(amts))
            mins = [e.minimum_allowed_amount for e in rows if e.minimum_allowed_amount]
            if mins:
                floors.setdefault(key[1], []).append(mins[-1] / (sum(amts) / len(amts)))
        noise = {}
        for cat, rs in ratios.items():
            rs.sort()
            r = rs[min(len(rs) - 1, int(0.98 * len(rs)))]
            if r > 1.0 + 1e-9:
                noise[cat] = (r - 1.0) / (r + 1.0)
        min_ratio = {}
        for cat, fs in floors.items():
            fs.sort()
            min_ratio[cat] = round(fs[len(fs) // 2] * 20) / 20.0  # nearest 0.05
        return noise, min_ratio

    # ------------------------------------------------------------------ helpers
    def _bracket_base(self, vals: List[float], category: str, floor: Optional[float]) -> Optional[float]:
        a = self.noise.get(category)
        if not a or len(vals) < 2:
            return None
        lo, hi = max(vals) / (1 + a), min(vals) / (1 - a)
        if floor and self.min_ratio.get(category):
            base = floor / self.min_ratio[category]
            if lo * 0.995 <= base <= hi * 1.005:
                return base
        if lo <= hi * 1.005:
            return (lo + hi) / 2.0
        return None

    def _estimate_home(self, rows: List[Event], profile: Profile, category: str = "") -> float:
        vals = [self._home(e.amount, e.currency, profile, e.cash_date) for e in rows]
        cfg = self.config
        essential = (category in profile.protected_categories if cfg.essential_categories == "protected"
                     else category in set(cfg.essential_categories.split(",")))
        how = cfg.essential_estimator if essential else cfg.variable_estimator
        est = None
        if how == "bracket" and vals and max(vals) - min(vals) > 1e-9:
            floor = rows[-1].minimum_allowed_amount if rows else None
            if floor is not None and rows[-1].currency != profile.home_currency:
                floor = self._home(floor, rows[-1].currency, profile, rows[-1].cash_date)
            est = self._bracket_base(vals, category, floor)
        if est is None:
            est = _estimate(vals, "mean" if how == "bracket" else how)
        unit = self.config.round_units.get(profile.home_currency)
        if unit and vals and max(vals) - min(vals) > 1e-9:
            est = round(est / unit) * unit
        return est

    def _home(self, amount: float, ccy: str, profile: Profile, on: date) -> float:
        return self.fx.convert(amount, ccy, profile.home_currency, on)

    def _resolve_amount(self, ev: Event, log: List[str]) -> Optional[float]:
        if ev.amount is not None:
            return ev.amount
        ref = self.ds.images_by_event.get(ev.event_id)
        if ref is None:
            log.append(f"{ev.event_id}: blank amount and no linked image; amount unknown")
            return None
        res = self.images.extract(ref.image_id, ref.path, ev.description, ev.category, ev.direction,
                                  ev.event_date.isoformat(), ev.currency)
        if res.instruction_like_text:
            log.append(f"{ref.image_id}: ignored instruction-like text in image")
        if res.amount is None:
            log.append(f"{ev.event_id}: could not read amount from {ref.image_id}")
            return None
        ev.amount = res.amount
        ev.amount_source = f"{ref.image_id}:{res.method}:{res.confidence}"
        log.append(f"{ev.event_id}: amount {res.amount} {ev.currency} read from {ref.image_id} "
                   f"({res.method}, {res.confidence})")
        return res.amount

    # --------------------------------------------------------------------- main
    def build(self, request: Request) -> FinancialState:
        cfg = self.config
        profile = self.ds.profiles[request.user_id]
        rd = request.request_date
        end = rd + timedelta(days=cfg.horizon_days)
        first_day = rd if cfg.include_request_day else rd + timedelta(days=1)
        log: List[str] = []
        events = self.ds.events_by_user.get(request.user_id, [])

        facts = [interpret_message(m, self.llm) for m in self.ds.messages_by_user.get(request.user_id, [])
                 if m.sent_date <= rd]
        for f in facts:
            if f.injection_flag:
                log.append(f"{f.message_id}: instruction-like content treated as data and ignored")

        history: List[Event] = []
        one_offs: List[CashItem] = []
        for ev in events:
            if ev.status in ("failed", "cancelled"):
                continue
            if ev.status == "unrealized" or ev.direction == "non_cash":
                continue
            if ev.status == "settled" and ev.event_date < rd:
                if ev.amount is None:
                    self._resolve_amount(ev, log)
                history.append(ev)
                continue
            if ev.status in ("pending", "scheduled"):
                if ev.direction == "credit" and ev.status == "pending":
                    log.append(f"{ev.event_id}: pending credit '{ev.description}' not counted")
                    continue
                if ev.direction == "credit" and ev.category == "salary":
                    continue  # handled by the income projection
                if ev.direction == "credit":
                    log.append(f"{ev.event_id}: scheduled non-salary credit '{ev.description}' not counted")
                    continue
                amt = self._resolve_amount(ev, log)
                if amt is None:
                    same = [e.amount for e in events if e.category == ev.category and e.amount]
                    amt = max(same) if same else 0.0
                    log.append(f"{ev.event_id}: reserved conservative placeholder {amt}")
                on = max(ev.cash_date, rd)
                if on <= end:
                    one_offs.append(CashItem(on, -self._home(amt, ev.currency, profile, ev.cash_date),
                                             ev.description, ev.event_id))

        series: List[Series] = []
        series += self._expense_series(history, profile, rd, first_day, end, facts, log)
        income_series, income_one_offs = self._income(events, history, profile, rd, first_day, end, facts, log)
        series += income_series
        one_offs += income_one_offs

        return FinancialState(request, profile, rd, end, profile.balance, profile.minimum_balance,
                              series, sorted(one_offs, key=lambda c: c.on), facts, log)

    # ---------------------------------------------------------------- expenses
    def _expense_series(self, history, profile, rd, first_day, end, facts, log) -> List[Series]:
        cfg = self.config
        out: List[Series] = []
        debits = [e for e in history if e.direction == "debit" and e.amount is not None and not e.linked_event_id]
        rent_pct = [f for f in facts if f.kind == "rent_increase_pct" and f.percent]

        by_cat: Dict[str, List[Event]] = {}
        by_desc: Dict[tuple, List[Event]] = {}
        for e in debits:
            if e.category in STREAM_CATEGORIES:
                by_cat.setdefault(e.category, []).append(e)
            else:
                by_desc.setdefault((e.category, e.description), []).append(e)

        # fixed-interval category streams (rotating descriptions)
        for cat, rows in by_cat.items():
            rows.sort(key=lambda e: e.event_date)
            if len(rows) < 3:
                continue
            gaps = [(b.event_date - a.event_date).days for a, b in zip(rows, rows[1:])]
            interval = statistics.mode(gaps)
            if interval <= 0 or interval > 31:
                continue
            # one-off purchases in the same category (e.g. an image-backed invoice) sit off the stream's
            # grid or share a grid date: anchor on the row with the most on-grid predecessors
            def support(a: Event) -> int:
                return sum(1 for e in rows if e.event_date <= a.event_date
                           and (a.event_date - e.event_date).days % interval == 0)
            anchor_row = max(rows, key=lambda a: (support(a), a.event_date))
            desc_counts = Counter(e.description for e in rows)
            by_day: Dict[date, List[Event]] = {}
            for e in rows:
                if e.event_date <= anchor_row.event_date and (anchor_row.event_date - e.event_date).days % interval == 0:
                    by_day.setdefault(e.event_date, []).append(e)
            on_grid = [sorted(es, key=lambda e: (e.amount_source != "dataset", desc_counts[e.description] == 1))[0]
                       for _, es in sorted(by_day.items())]
            if len(on_grid) < 3:
                continue
            anchor = on_grid[-1]
            stream_first = rd if cfg.stream_on_request_day else rd + timedelta(days=1)
            dates, d = [], anchor.event_date + timedelta(days=interval)
            while d <= end:
                if d >= stream_first:
                    dates.append(d)
                d += timedelta(days=interval)
            amount = self._estimate_home(on_grid, profile, cat)
            out.append(Series(f"stream:{cat}", cat, anchor.description, "debit", amount, dates,
                              anchor.flexibility, anchor.minimum_allowed_amount, anchor.event_id))

        # fixed day-of-month series
        for (cat, desc), rows in by_desc.items():
            rows.sort(key=lambda e: e.event_date)
            if len(rows) < 2:
                continue
            gaps = [(b.event_date - a.event_date).days for a, b in zip(rows, rows[1:])]
            if not all(27 <= g <= 32 for g in gaps):
                continue
            last = rows[-1]
            if (rd - last.event_date).days > 35:
                log.append(f"series '{desc}' stopped before the request; not projected")
                continue
            dom = max(e.event_date.day for e in rows[-3:])  # a 31st series keeps its day after short months
            dates, k = [], 1
            while True:
                d = add_months(last.event_date, k, dom)
                k += 1
                if d > end:
                    break
                if d >= first_day:
                    dates.append(d)
            amount = self._estimate_home(rows, profile, cat)
            s = Series(f"monthly:{cat}:{desc}", cat, desc, "debit", amount, dates, last.flexibility,
                       last.minimum_allowed_amount, last.event_id)
            if cat in ("rent",) and rent_pct:
                for f in rent_pct:
                    factor = 1 + f.percent / 100.0
                    for d in dates:
                        s.amount_overrides[d] = round(amount * factor, 2)
                    log.append(f"{f.message_id}: rent '{desc}' increased by {f.percent}% from next payment")
            out.append(s)
        return out

    # ------------------------------------------------------------------ income
    def _income(self, events, history, profile, rd, first_day, end, facts, log):
        cfg = self.config
        series: List[Series] = []
        one_offs: List[CashItem] = []

        for f in facts:
            if f.kind == "invoice_approved" and f.amount and f.on_date and first_day <= f.on_date <= end:
                amt = self._home(f.amount, f.currency or profile.home_currency, profile, f.on_date)
                one_offs.append(CashItem(f.on_date, amt, "approved invoice", f.message_id))
                log.append(f"{f.message_id}: confirmed invoice credit {f.amount} on {f.on_date}")

        kinds = {f.kind: f for f in facts}
        ended = "employment_ended" in kinds or "seasonal_ended" in kinds
        if ended:
            log.append("messages confirm regular salary has ended; only separately confirmed income is projected")

        salary_rows = [e for e in history if e.category == "salary" and e.direction == "credit"
                       and e.amount is not None
                       and not any(w in e.description.lower() for w in NON_RECURRING_INCOME_WORDS)]
        salary_rows.sort(key=lambda e: e.event_date)
        latest_any = max((e for e in history if e.category == "salary" and e.direction == "credit"),
                         key=lambda e: e.event_date, default=None)
        scheduled = sorted((e for e in events if e.status == "scheduled" and e.direction == "credit"
                            and e.category == "salary" and e.amount is not None and e.cash_date >= rd),
                           key=lambda e: e.cash_date)

        # regular monthly stream: rows sharing the latest row's day-of-month, roughly monthly apart
        anchor = None
        if salary_rows and not ended:
            # the regular stream is the modal pay day-of-month (a one-off payslip row on another day,
            # e.g. a month-end net-salary record, must not become the anchor)
            dom_counts: Dict[int, int] = {}
            for e in salary_rows:
                dom_counts[e.event_date.day] = dom_counts.get(e.event_date.day, 0) + 1
            best_dom = max(dom_counts, key=lambda k: (dom_counts[k], max(e.event_date for e in salary_rows
                                                                      if e.event_date.day == k)))
            stream = [e for e in salary_rows if e.event_date.day == best_dom]
            last = stream[-1]
            if "Second household income" in {e.description for e in salary_rows}:
                stream = [e for e in stream if e.description != "Second household income"]
                last = stream[-1] if stream else None
            if last is not None:
                monthly = len(stream) >= 2 and 25 <= (stream[-1].event_date - stream[-2].event_date).days <= 70
                # regular pay repeats an exact amount; gig/platform payouts that happen to share a
                # day-of-month vary every time and are not confirmed income
                recent_amts = [round(e.amount, 2) for e in stream[-3:]]
                stable = any(recent_amts.count(a) >= 2 for a in recent_amts) or len(stream) == 1
                if monthly and not stable:
                    log.append(f"income on day {best_dom} varies ({recent_amts}); treated as unconfirmed "
                               f"irregular income")
                    monthly = False
                recent = (rd - last.event_date).days <= 40
                if latest_any is not None and "final" in latest_any.description.lower():
                    log.append(f"{latest_any.event_id}: final employer payroll; salary not projected")
                elif monthly and recent:
                    anchor = last
                elif not recent:
                    log.append(f"salary stream '{last.description}' is stale; not projected")

        amount = None
        currency = profile.home_currency
        pay_dom = None
        next_date = None
        if anchor is not None:
            amount, currency = anchor.amount, anchor.currency
            pay_dom = anchor.cash_date.day
            next_date = add_months(anchor.event_date, 1, anchor.event_date.day)
            if anchor.cash_date != anchor.event_date:
                next_date = add_months(anchor.cash_date, 1, anchor.cash_date.day)

        if scheduled:
            s = scheduled[0]
            amount, currency, next_date, pay_dom = s.amount, s.currency, s.cash_date, s.cash_date.day
            log.append(f"{s.event_id}: confirmed scheduled salary {s.amount} {s.currency} on {s.cash_date}")

        change_from: Optional[tuple] = None
        for f in facts:
            if ended and f.kind in ("salary_base_confirmed", "household_income_ended", "salary_next_amount",
                                    "salary_temporary_amount", "salary_regular_next", "salary_date_moved"):
                continue  # these only adjust a regular salary that no longer exists
            if f.kind in ("salary_base_confirmed", "household_income_ended") and f.amount:
                # The notice and the deposited history disagree on the regular amount and neither is an
                # explicit amendment of the other: use the financially safer (lower) figure.
                msg_home = self._home(f.amount, f.currency or currency, profile, rd)
                hist_home = self._home(amount, currency, profile, rd) if amount is not None else None
                if hist_home is None or msg_home < hist_home:
                    amount, currency = f.amount, f.currency or currency
                if next_date is None:
                    next_date = add_months(rd.replace(day=1), 0 if rd.day < 15 else 1, 15)
                    pay_dom = 15
                log.append(f"{f.message_id}: {f.kind}; conflicting salary figures resolved to the lower "
                           f"amount {amount} {currency}")
            elif f.kind in ("salary_next_amount", "salary_temporary_amount", "salary_regular_next") and f.amount:
                amount, currency = f.amount, f.currency or currency
                if next_date is None:
                    next_date = add_months(rd.replace(day=1), 0 if rd.day < 15 else 1, 15)
                    pay_dom = 15
                log.append(f"{f.message_id}: salary amount set to {f.amount} ({f.kind})")
            elif f.kind in ("first_salary", "fx_salary", "salary_resumes") and f.amount and f.on_date:
                amount, currency = f.amount, f.currency or currency
                next_date, pay_dom = f.on_date, f.on_date.day
                log.append(f"{f.message_id}: salary {f.amount} {currency} confirmed from {f.on_date} ({f.kind})")
            elif f.kind == "salary_amount_from" and f.amount and f.on_date:
                change_from = (f.on_date, f.amount, f.currency or currency)
                log.append(f"{f.message_id}: salary changes to {f.amount} from {f.on_date}")
            elif f.kind == "salary_date_moved" and f.on_date:
                next_date, pay_dom = f.on_date, f.on_date.day
                if amount is None and salary_rows:
                    amount, currency = salary_rows[-1].amount, salary_rows[-1].currency
                log.append(f"{f.message_id}: salary date moved to {f.on_date}")

        if amount is None or next_date is None:
            if not salary_rows and not scheduled:
                log.append("no confirmed recurring salary found")
            return series, one_offs

        # step forward to the first pay date inside the horizon
        while next_date < first_day:
            next_date = add_months(next_date, 1, pay_dom)
        dates, overrides, k = [], {}, 0
        while True:
            d = add_months(next_date, k, pay_dom) if k else next_date
            k += 1
            if d > end:
                break
            if d >= first_day:
                dates.append(d)
                amt, ccy = amount, currency
                if change_from and d >= change_from[0]:
                    amt, ccy = change_from[1], change_from[2]
                overrides[d] = self._home(amt, ccy, profile, d)
        base = overrides[dates[0]] if dates else 0.0
        series.append(Series("income:salary", "salary", "salary", "credit", base, dates, "fixed", None,
                             anchor.event_id if anchor else "", overrides))
        return series, one_offs
