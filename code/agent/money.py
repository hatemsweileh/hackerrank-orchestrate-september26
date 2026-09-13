"""Money helpers: rounding, formatting and dated currency conversion."""
from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Optional

EPS = 1e-6


def r2(x: float) -> float:
    """Round half-up to cents (avoids banker's rounding surprises)."""
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fmt_amount(x: float) -> str:
    """Plain numeric output: no thousands separator, at most 2 decimals."""
    v = r2(x)
    if abs(v - round(v)) < EPS:
        return str(int(round(v)))
    return f"{v:.2f}"


def fmt_money_text(x: float, currency: str) -> str:
    """Human text used in explanations, e.g. 'EUR 1,300' or 'EUR 996.60'."""
    v = r2(x)
    if abs(v - round(v)) < EPS:
        return f"{currency} {int(round(v)):,}"
    return f"{currency} {v:,.2f}"


class FxTable:
    """Fixed dated exchange rates.

    A conversion uses the rate row for the cash (settlement) date and the
    stated direction. If only the inverse pair exists on that date it is
    inverted; if the pair is missing on that date the most recent earlier
    rate for the pair is used. Missing rates raise instead of guessing.
    """

    def __init__(self, rates: Dict[tuple, float]):
        self.rates = rates
        self.by_pair: Dict[tuple, list] = {}
        for (d, f, t), rate in rates.items():
            self.by_pair.setdefault((f, t), []).append((d, rate))
        for v in self.by_pair.values():
            v.sort()

    def _nearest(self, d: date, f: str, t: str) -> Optional[float]:
        """Latest rate for the pair on or before d, else the earliest one after d."""
        rows = self.by_pair.get((f, t))
        if not rows:
            return None
        before = [rate for dd, rate in rows if dd <= d]
        return before[-1] if before else rows[0][1]

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on: date) -> float:
        """Stated direction first (exact date, then nearest date); the inverse pair only as a fallback,
        because published pairs are not exact reciprocals (USD->EUR 0.92 vs EUR->USD 1.09)."""
        if from_ccy == to_ccy:
            return amount
        rate = self.rates.get((on, from_ccy, to_ccy))
        if rate is None:
            rate = self._nearest(on, from_ccy, to_ccy)
        if rate is None:
            inverse = self.rates.get((on, to_ccy, from_ccy)) or self._nearest(on, to_ccy, from_ccy)
            rate = 1.0 / inverse if inverse else None
        if rate is None:
            raise KeyError(f"no exchange rate for {from_ccy}->{to_ccy} near {on}")
        return amount * rate
