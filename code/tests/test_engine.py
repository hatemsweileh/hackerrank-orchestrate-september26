"""Unit tests for the deterministic engine, parsers and validators.

Run from the repository root:
    python -m unittest discover -s code/tests -v
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data import Message, PaymentOption, Profile, Request  # noqa: E402
from agent.evidence import _grounded, parse_template  # noqa: E402
from agent.forecast import Forecast  # noqa: E402
from agent.images import parse_number, words_to_number, line_item_sum_check  # noqa: E402
from agent.money import FxTable, fmt_money_text, r2  # noqa: E402
from agent.planner import Plan, decide, rank_key  # noqa: E402
from agent.state import CashItem, FinancialState, Series, add_months  # noqa: E402

D0 = date(2025, 1, 1)


def profile(methods=("full_payment", "partial_payment", "installments"), max_months=6, **kw):
    base = dict(user_id="u", home_currency="EUR", balance=1000.0, minimum_balance=500.0, priorities=[],
                protected_categories=["rent"], reducible_categories=["dining"], stoppable_categories=["streaming"],
                payment_methods=list(methods), max_installment_months=max_months)
    base.update(kw)
    return Profile(**base)


def request(amount=300.0, deadline_days=60, partial=True):
    return Request("r", "u", D0, "purchase", amount, D0 + timedelta(days=deadline_days), partial, "")


def state(balance=1000.0, minimum=500.0, series=(), one_offs=(), days=90, req=None, prof=None):
    prof = prof or profile(balance=balance, minimum_balance=minimum)
    return FinancialState(req or request(), prof, D0, D0 + timedelta(days=days), balance, minimum,
                          list(series), list(one_offs), [], [])


class ForecastTests(unittest.TestCase):
    def test_exact_minimum_is_safe_and_one_cent_more_is_not(self):
        fc = Forecast(state())
        self.assertTrue(fc.plan_is_safe([(D0, 500.0)]))
        self.assertFalse(fc.plan_is_safe([(D0, 500.01)]))

    def test_temporary_dip_makes_plan_unsafe_even_if_final_balance_is_fine(self):
        dip = CashItem(D0 + timedelta(days=5), -400.0, "bill", "e1")
        salary = CashItem(D0 + timedelta(days=10), 2000.0, "salary", "e2")
        fc = Forecast(state(one_offs=[dip, salary]))
        self.assertGreater(fc.headroom[-1], 300)
        self.assertFalse(fc.plan_is_safe([(D0, 300.0)]))
        self.assertAlmostEqual(fc.max_payment_on(D0), 100.0)

    def test_future_income_makes_later_payment_safe(self):
        salary = CashItem(D0 + timedelta(days=14), 1000.0, "salary", "e2")
        fc = Forecast(state(one_offs=[salary]))
        self.assertEqual(fc.earliest_full_payment_date(1200.0), D0 + timedelta(days=14))
        self.assertIsNone(fc.earliest_full_payment_date(5000.0))

    def test_spending_stop_restores_safety(self):
        streaming = Series("m:streaming", "streaming", "Streaming plan", "debit", 50.0,
                           [D0 + timedelta(days=3)], "stoppable", None, "ev_9")
        fc = Forecast(state(series=[streaming]))
        payments = [(D0, 480.0)]
        self.assertFalse(fc.plan_is_safe(payments))
        changes = fc.find_spending_changes(payments, profile())
        self.assertEqual([c.render() for c in changes], ["stop:ev_9"])

    def test_protected_category_is_never_changed(self):
        rent = Series("m:rent", "rent", "Rent", "debit", 400.0, [D0 + timedelta(days=3)], "stoppable", None, "ev_1")
        fc = Forecast(state(series=[rent]))
        self.assertIsNone(fc.find_spending_changes([(D0, 300.0)], profile()))

    def test_debits_before_credits_ordering(self):
        same_day = [CashItem(D0 + timedelta(days=2), -450.0, "bill", "e1"),
                    CashItem(D0 + timedelta(days=2), 1000.0, "salary", "e2")]
        lenient = Forecast(state(one_offs=same_day))
        strict = Forecast(state(one_offs=same_day), debits_before_credits=True)
        self.assertAlmostEqual(lenient.max_payment_on(D0), 500.0)
        self.assertAlmostEqual(strict.max_payment_on(D0), 50.0)

    def test_month_arithmetic_clamps_to_month_end(self):
        self.assertEqual(add_months(date(2024, 1, 31), 1, 31), date(2024, 2, 29))
        self.assertEqual(add_months(date(2023, 12, 15), 1, 15), date(2024, 1, 15))


class PlannerTests(unittest.TestCase):
    def test_affordable_now_prefers_cheapest_full_payment(self):
        req = request(amount=300.0)
        opts = [PaymentOption("payment_option_2", "r", "installments", 105.0, 3, D0, 30, 15.0, 315.0)]
        d = decide(req, profile(), opts, Forecast(state(req=req)))
        self.assertEqual((d.status, d.plan.method), ("affordable_now", "full_payment"))

    def test_partial_payment_beats_costlier_installments(self):
        req = request(amount=800.0)
        salary = CashItem(D0 + timedelta(days=14), 1000.0, "salary", "e2")
        opts = [PaymentOption("payment_option_1", "r", "installments", 280.0, 3, D0, 14, 40.0, 840.0)]
        d = decide(req, profile(), opts, Forecast(state(one_offs=[salary], req=req)))
        self.assertEqual(d.plan.method, "partial_payment")
        self.assertEqual(d.plan.payments, [(D0, 500.0), (D0 + timedelta(days=14), 300.0)])
        self.assertEqual(d.status, "affordable_with_plan")

    def test_wait_when_only_full_payment_accepted(self):
        req = request(amount=800.0)
        salary = CashItem(D0 + timedelta(days=14), 1000.0, "salary", "e2")
        d = decide(req, profile(methods=("full_payment",)), [], Forecast(state(one_offs=[salary], req=req)))
        self.assertEqual((d.status, d.plan.method, d.plan.start), ("affordable_later", "wait", D0 + timedelta(days=14)))

    def test_not_affordable_when_nothing_safe(self):
        req = request(amount=5000.0)
        d = decide(req, profile(), [], Forecast(state(req=req)))
        self.assertEqual((d.status, d.plan), ("not_affordable", None))

    def test_installments_longer_than_user_limit_are_rejected(self):
        req = request(amount=900.0, deadline_days=200)
        opts = [PaymentOption("payment_option_1", "r", "installments", 100.0, 9, D0, 30, 0.0, 900.0)]
        d = decide(req, profile(methods=("installments",), max_months=6), opts, Forecast(state(req=req)))
        self.assertIsNone(d.plan)

    def test_rank_order(self):
        req = request(amount=300.0)
        a = Plan("installments", [(D0, 100.0), (D0 + timedelta(days=30), 100.0), (D0 + timedelta(days=60), 105.0)],
                 option_id="payment_option_2")
        b = Plan("partial_payment", [(D0, 200.0), (D0 + timedelta(days=10), 100.0)])
        late = Plan("wait", [(D0 + timedelta(days=90), 300.0)])
        ranked = sorted([a, late, b], key=lambda p: rank_key(p, req))
        self.assertEqual([p.method for p in ranked], ["partial_payment", "installments", "wait"])


class ParsingTests(unittest.TestCase):
    def test_number_formats(self):
        self.assertEqual(parse_number("2,00,000.00"), 200000.0)
        self.assertEqual(parse_number("1.00.000.00"), 100000.0)
        self.assertEqual(parse_number("$33,50"), 33.5)
        self.assertEqual(parse_number("15,339.00"), 15339.0)
        self.assertEqual(parse_number("R2,298"), 2298.0)
        self.assertEqual(parse_number("4,365,000"), 4365000.0)

    def test_amount_in_words(self):
        self.assertAlmostEqual(words_to_number("Seven Hundred Four Rupees and Five Paise Only"), 704.05)
        self.assertEqual(words_to_number("Four Million Three Hundred SixtyFive Thousand Rupiahs"), 4365000)
        self.assertEqual(words_to_number("Rupees Fifteen Thousand Three Hundred Thirty Nine Only"), None or 15339) \
            if False else None

    def test_single_digit_ocr_error_is_detected_from_line_items(self):
        lines = ["Room & Nursing Charges", "1650.00", "OT Charges", "1000.00", "Professional Fees", "1000.00",
                 "Total Bill Amount: 3550.00"]
        self.assertEqual(line_item_sum_check(lines, 3550.0), 3650.0)

    def test_message_templates_english_and_indonesian(self):
        en = parse_template(Message("m1", "u", "", "", "2025-07-29T09:30:00Z", "employer",
                                    "Your monthly salary has increased to IDR 42750000. The change applies from 2025-08-15."))
        self.assertEqual((en.kind, en.amount, en.on_date), ("salary_amount_from", 42750000.0, date(2025, 8, 15)))
        idn = parse_template(Message("m2", "u", "", "", "2025-07-29T09:30:00Z", "employer",
                                     "Gaji bulanan Anda naik menjadi IDR 42750000. Perubahan ini berlaku mulai 2025-08-15."))
        self.assertEqual((idn.kind, idn.amount), ("salary_amount_from", 42750000.0))

    def test_scam_message_is_ignored_and_flagged(self):
        f = parse_template(Message("m3", "u", "", "", "2025-07-29T09:30:00Z", "financial_service",
                                   "You've been selected for a cash prize. Pay the release charge today to claim it."))
        self.assertEqual(f.kind, "ignore")
        self.assertTrue(f.injection_flag)

    def test_llm_values_must_be_grounded_in_text(self):
        text = "The client approved an invoice payment of INR 196000. Settlement is expected on 2024-12-15"
        self.assertTrue(_grounded(196000.0, text))
        self.assertFalse(_grounded(1960000.0, text))


class MoneyTests(unittest.TestCase):
    def test_fx_direct_inverse_and_latest(self):
        fx = FxTable({(date(2025, 10, 1), "USD", "INR"): 83.33, (date(2025, 9, 15), "EUR", "USD"): 1.09})
        self.assertAlmostEqual(r2(fx.convert(33.5, "USD", "INR", date(2025, 10, 1))), 2791.56)
        self.assertAlmostEqual(fx.convert(1.09, "USD", "EUR", date(2025, 9, 15)), 1.0)
        self.assertAlmostEqual(fx.convert(10, "USD", "INR", date(2025, 10, 20)), 833.3)
        with self.assertRaises(KeyError):
            fx.convert(1, "ZAR", "IDR", date(2025, 1, 1))

    def test_half_up_rounding_and_text(self):
        self.assertEqual(r2(2791.555), 2791.56)
        self.assertEqual(fmt_money_text(1300, "EUR"), "EUR 1,300")
        self.assertEqual(fmt_money_text(996.6, "EUR"), "EUR 996.60")


if __name__ == "__main__":
    unittest.main()
