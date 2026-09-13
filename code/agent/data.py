"""Dataset loading and validation.

Every CSV is read with the standard library so the core engine has no heavy
dependencies. Rows are converted into small typed records; malformed values
raise a DataError that names the file, row and column instead of being
silently coerced.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional


class DataError(ValueError):
    pass


def parse_date(value: str, where: str) -> date:
    try:
        return date.fromisoformat(value.strip()[:10])
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"{where}: invalid date {value!r}") from exc


def parse_amount(value: str, where: str, allow_blank: bool = False) -> Optional[float]:
    value = (value or "").strip()
    if value == "":
        if allow_blank:
            return None
        raise DataError(f"{where}: blank amount")
    try:
        return float(value)
    except ValueError as exc:
        raise DataError(f"{where}: invalid number {value!r}") from exc


def split_list(value: str) -> List[str]:
    return [v.strip() for v in (value or "").split("|") if v.strip()]


@dataclass
class Profile:
    user_id: str
    home_currency: str
    balance: float
    minimum_balance: float
    priorities: List[str]
    protected_categories: List[str]
    reducible_categories: List[str]
    stoppable_categories: List[str]
    payment_methods: List[str]
    max_installment_months: Optional[int]


@dataclass
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[float]
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    # filled during evidence resolution
    amount_source: str = "dataset"

    @property
    def cash_date(self) -> date:
        return self.settlement_date or self.event_date


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    expected: Dict[str, str] = field(default_factory=dict)


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float

    def schedule(self) -> List[tuple]:
        """Return [(date, amount)] for every payment in this offer."""
        from datetime import timedelta

        out = []
        for k in range(self.number_of_payments):
            step = (self.frequency_days or 0) * k
            out.append((self.first_payment_date + timedelta(days=step), self.payment_amount))
        return out


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: str
    source_type: str
    text: str

    @property
    def sent_date(self) -> date:
        return parse_date(self.sent_at, f"message {self.message_id}")


@dataclass
class ImageRef:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str
    path: Path


@dataclass
class Dataset:
    root: Path
    profiles: Dict[str, Profile]
    events_by_user: Dict[str, List[Event]]
    events_by_id: Dict[str, Event]
    rates: Dict[tuple, float]
    requests: List[Request]
    samples: List[Request]
    options_by_request: Dict[str, List[PaymentOption]]
    messages_by_user: Dict[str, List[Message]]
    images_by_event: Dict[str, ImageRef]
    images_by_user: Dict[str, List[ImageRef]]


def _rows(path: Path) -> List[dict]:
    if not path.exists():
        raise DataError(f"missing dataset file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _require(row: dict, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in row]
    if missing:
        raise DataError(f"{where}: missing columns {missing}")


def _load_requests(path: Path, with_labels: bool) -> List[Request]:
    out = []
    label_cols = [
        "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
        "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation",
    ]
    for i, r in enumerate(_rows(path), start=2):
        where = f"{path.name}:{i}"
        _require(r, ["request_id", "user_id", "request_date", "requested_amount"], where)
        req = Request(
            request_id=r["request_id"].strip(),
            user_id=r["user_id"].strip(),
            request_date=parse_date(r["request_date"], where),
            request_type=r.get("request_type", "other").strip() or "other",
            requested_amount=parse_amount(r["requested_amount"], where),
            desired_completion_date=parse_date(r["desired_completion_date"], where),
            allows_partial_payment=r.get("allows_partial_payment", "").strip().lower() == "true",
            request_text=r.get("request_text", ""),
        )
        if with_labels:
            req.expected = {c: r.get(c, "") for c in label_cols}
        out.append(req)
    return out


def load_dataset(root: Path) -> Dataset:
    root = Path(root)

    profiles: Dict[str, Profile] = {}
    for i, r in enumerate(_rows(root / "financial_profiles.csv"), start=2):
        where = f"financial_profiles.csv:{i}"
        mim = (r.get("max_installment_months") or "").strip()
        profiles[r["user_id"]] = Profile(
            user_id=r["user_id"],
            home_currency=r["home_currency"].strip(),
            balance=parse_amount(r["current_available_balance"], where),
            minimum_balance=parse_amount(r["minimum_balance_to_keep"], where),
            priorities=split_list(r.get("financial_priorities", "")),
            protected_categories=split_list(r.get("expense_categories_to_protect", "")),
            reducible_categories=split_list(r.get("expense_categories_user_is_willing_to_reduce", "")),
            stoppable_categories=split_list(r.get("expense_categories_user_is_willing_to_stop", "")),
            payment_methods=split_list(r.get("payment_methods_user_will_consider", "")),
            max_installment_months=int(float(mim)) if mim else None,
        )

    events_by_user: Dict[str, List[Event]] = {}
    events_by_id: Dict[str, Event] = {}
    for i, r in enumerate(_rows(root / "financial_events.csv"), start=2):
        where = f"financial_events.csv:{i}"
        sd = (r.get("settlement_date") or "").strip()
        ev = Event(
            event_id=r["event_id"],
            user_id=r["user_id"],
            event_type=r["event_type"],
            description=r["description"],
            category=r["category"],
            direction=r["direction"],
            amount=parse_amount(r["amount"], where, allow_blank=True),
            currency=r["currency"].strip(),
            event_date=parse_date(r["event_date"], where),
            settlement_date=parse_date(sd, where) if sd else None,
            status=r["status"].strip(),
            linked_event_id=(r.get("linked_event_id") or "").strip(),
            flexibility=(r.get("flexibility") or "fixed").strip(),
            minimum_allowed_amount=parse_amount(r.get("minimum_allowed_amount", ""), where, allow_blank=True),
        )
        if ev.event_id in events_by_id:
            raise DataError(f"{where}: duplicate event_id {ev.event_id}")
        events_by_id[ev.event_id] = ev
        events_by_user.setdefault(ev.user_id, []).append(ev)

    rates: Dict[tuple, float] = {}
    for i, r in enumerate(_rows(root / "exchange_rates.csv"), start=2):
        where = f"exchange_rates.csv:{i}"
        rates[(parse_date(r["rate_date"], where), r["from_currency"].strip(), r["to_currency"].strip())] = \
            parse_amount(r["rate"], where)

    options: Dict[str, List[PaymentOption]] = {}
    for i, r in enumerate(_rows(root / "request_payment_options.csv"), start=2):
        where = f"request_payment_options.csv:{i}"
        freq = (r.get("payment_frequency_days") or "").strip()
        opt = PaymentOption(
            payment_option_id=r["payment_option_id"],
            request_id=r["request_id"],
            payment_method=r["payment_method"].strip(),
            payment_amount=parse_amount(r["payment_amount"], where),
            number_of_payments=int(float(r["number_of_payments"])),
            first_payment_date=parse_date(r["first_payment_date"], where),
            frequency_days=int(float(freq)) if freq else None,
            financing_fee=parse_amount(r.get("financing_fee") or "0", where),
            total_payable_amount=parse_amount(r["total_payable_amount"], where),
        )
        options.setdefault(opt.request_id, []).append(opt)

    messages: Dict[str, List[Message]] = {}
    for r in _rows(root / "messages.csv"):
        m = Message(
            message_id=r["message_id"], user_id=r["user_id"], request_id=(r.get("request_id") or "").strip(),
            related_event_id=(r.get("related_event_id") or "").strip(), sent_at=r["sent_at"],
            source_type=r.get("source_type", ""), text=r.get("message_text", ""),
        )
        messages.setdefault(m.user_id, []).append(m)

    images_by_event: Dict[str, ImageRef] = {}
    images_by_user: Dict[str, List[ImageRef]] = {}
    for r in _rows(root / "images.csv"):
        ref = ImageRef(
            image_id=r["image_id"], user_id=r["user_id"], request_id=(r.get("request_id") or "").strip(),
            related_event_id=(r.get("related_event_id") or "").strip(),
            path=root / "media" / "images" / f"{r['image_id']}.png",
        )
        if ref.related_event_id:
            images_by_event[ref.related_event_id] = ref
        images_by_user.setdefault(ref.user_id, []).append(ref)

    for evs in events_by_user.values():
        evs.sort(key=lambda e: (e.event_date, e.event_id))

    samples_path = root / "sample_requests.csv"
    return Dataset(
        root=root,
        profiles=profiles,
        events_by_user=events_by_user,
        events_by_id=events_by_id,
        rates=rates,
        requests=_load_requests(root / "requests.csv", with_labels=False),
        samples=_load_requests(samples_path, with_labels=True) if samples_path.exists() else [],
        options_by_request=options,
        messages_by_user=messages,
        images_by_event=images_by_event,
        images_by_user=images_by_user,
    )
